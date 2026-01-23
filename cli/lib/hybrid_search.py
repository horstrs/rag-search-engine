import os

from .search_utils import load_movies
from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .gemini_integration import GeminiClient
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.TIMES_OVER_LIMIT = 500

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self, query: str, limit: int, debug: bool = False
    ) -> list[(dict, float)]:
        self.idx.load()
        return self.idx.bm25_search(query, limit, debug)

    def hybrid_score(
        self, bm25_score: float, semantic_score: float, alpha: float = 0.5
    ):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * self.TIMES_OVER_LIMIT)
        bm25_scores = [score for _, score in bm25_results]
        bm25_normalized = self.normalize(bm25_scores)

        weighted_results = {}

        for i, (movie, _) in enumerate(bm25_results):
            self._create_new_weighted_entry(
                weighted_results, movie, keyword_score=bm25_normalized[i]
            )
        semantic_results = self.semantic_search.search_chunks(
            query, limit * self.TIMES_OVER_LIMIT
        )
        semantic_scores = [result["score"] for result in semantic_results]
        semantic_normalized = self.normalize(semantic_scores)

        for i, result in enumerate(semantic_results):
            id = result["metadata"]["movie_idx"]
            semantic_score = semantic_normalized[i]
            weighted_entry = weighted_results.get(id)
            if not weighted_entry:
                self._create_new_weighted_entry(
                    weighted_results,
                    self.semantic_search.document_map[id],
                    semantic_score,
                )
            else:
                weighted_entry["semantic_score"] = semantic_score

        for entry in weighted_results.values():
            keyword_score = entry["keyword_score"]
            semantic_score = entry["semantic_score"]
            entry["hybrid_score"] = self.hybrid_score(
                keyword_score, semantic_score, alpha
            )

        return sorted(
            weighted_results.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )[:limit]

    def _create_new_weighted_entry(
        self,
        weighted_results: dict,
        movie: dict,
        keyword_score: float = 0,
        semantic_score: float = 0,
    ) -> None:
        id = movie["id"]
        weighted_results[id] = {
            "document": movie,
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "hybrid_score": 0,
        }

    def rrf_search(
        self, query: str, k: int, limit: int = 10, debug: bool = False
    ) -> list[(int, dict)]:
        bm25_results = self._bm25_search(query, limit * self.TIMES_OVER_LIMIT, debug)
        rrf_ranks = {}
        for i, (movie, _) in enumerate(bm25_results, 1):
            movie_id = movie["id"]
            self._create_rrf_entry(rrf_ranks, movie_id, movie)
            self._update_bm25_rank_and_score(rrf_ranks, movie_id, i, k)
        if debug:
            print("keyword results:")
            titles = [movie["title"] for movie, _ in bm25_results]
            print(titles)
            
        semantic_results = self.semantic_search.search_chunks(
            query, limit * self.TIMES_OVER_LIMIT
        )
        for i, sem_result in enumerate(semantic_results, 1):
            movie_id = sem_result["id"]
            if not rrf_ranks.get(movie_id):
                movie = self.idx.docmap[movie_id]
                self._create_rrf_entry(rrf_ranks, movie_id, movie)
            self._update_semantic_rank_and_score(rrf_ranks, movie_id, i, k)
        if debug:
            print("semantic results:")
            titles = [movie["title"] for movie in semantic_results]
            print(titles)
        sorted_rank = sorted(
            rrf_ranks.items(), key=lambda result: result[1]["rrf_score"], reverse=True
        )
        if debug:
            print("RRF Score Sorted Rank")
            titles = [doc["document"]["title"] for _, doc in sorted_rank]
            print(titles)
        return sorted_rank[:limit]

    def _create_rrf_entry(self, rrf_ranks: dict, movie_id: int, movie: dict) -> None:
        rrf_ranks[movie_id] = {
            "document": movie,
            "bm25_rank": 0,
            "semantic_rank": 0,
            "rrf_score": 0,
        }

    def _update_bm25_rank_and_score(
        self, rrf_ranks: dict, movie_id: int, rank: int, k: int
    ) -> None:
        rrf_ranks[movie_id]["bm25_rank"] = rank
        rrf_ranks[movie_id]["rrf_score"] += self._calculate_rrf(rank, k)

    def _update_semantic_rank_and_score(
        self, rrf_ranks: dict, movie_id: int, rank: int, k: int
    ) -> None:
        rrf_ranks[movie_id]["semantic_rank"] = rank
        rrf_ranks[movie_id]["rrf_score"] += self._calculate_rrf(rank, k)

    def _calculate_rrf(self, rank: int, k: int) -> float:
        return 1 / (k + rank)

    def normalize(self, scores: list[float]) -> list[float]:
        if not scores or len(scores) == 0:
            return []
        min_score = float(min(scores))
        max_score = float(max(scores))
        if min_score == max_score:
            return [1.0] * len(scores)

        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized_scores


def normalize_command(scores: list[float]) -> list[float]:
    movies = load_movies()
    search_instance = HybridSearch(movies)
    return search_instance.normalize(scores)


def weighted_search_command(query: str, alpha: float, limit: int) -> list[dict]:
    movies = load_movies()
    search_instance = HybridSearch(movies)
    return search_instance.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str,
    k: float,
    limit: int,
    enhance_method: str,
    rerank_method: str,
    debug: bool = False,
) -> list[dict]:
    match enhance_method:
        case "spell":
            gemini_client = GeminiClient()
            enhanced_query = gemini_client.fix_spelling(query)
            if enhanced_query != query:
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
        case "rewrite":
            gemini_client = GeminiClient()
            enhanced_query = gemini_client.rewrite_query(query)
            if enhanced_query != query:
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
        case "expand":
            gemini_client = GeminiClient()
            enhanced_query = gemini_client.expand_query(query)
            if enhanced_query != query:
                print(
                    f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query

    if rerank_method:
        original_limit = limit
        limit *= 5

    movies = load_movies()
    search_instance = HybridSearch(movies)
    results = search_instance.rrf_search(query, k, limit, debug)

    match rerank_method:
        case "individual":
            print(
                f"Reranking top {original_limit} results using {rerank_method} method..."
            )
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
            for _, result in results:
                gemini_client = GeminiClient()
                rerank_score = gemini_client.individual_rerank(
                    query, result["document"]
                )
                result["rerank_score"] = rerank_score

            results.sort(key=lambda movie: movie[1]["rerank_score"], reverse=True)
            results = results[:original_limit]
        case "batch":
            print(
                f"Reranking top {original_limit} results using {rerank_method} method..."
            )
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
            doc_list = [result["document"] for _, result in results]
            gemini_client = GeminiClient()
            reranked_ids = gemini_client.batch_rerank(query, doc_list)

            look_up = {item[0]: item for item in results}
            results = [
                (id, look_up[id][1] | {"rerank_rank": i})
                for i, id in enumerate(reranked_ids, 1)
            ]
            results = results[:original_limit]
        case "cross_encoder":
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            pairs_with_ids = [
                (doc_id, (query, f"{doc.get('title', '')} - {doc.get('document', '')}"))
                for doc_id, doc in results
            ]
            if debug:
                print("Results before cross_encoder:")
                titles = [doc["document"]["title"] for _, doc in results]
                print(titles)
            pairs = [item[1] for item in pairs_with_ids]
            # scores is a list of numbers, one for each pair
            scores = cross_encoder.predict(pairs)
            scored_pairs = list(zip(scores, pairs_with_ids))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            look_up = {item[0]: item for item in results}
            results = [
                (doc_id, look_up[doc_id][1] | {"cross_encoder_score": score})
                for score, (doc_id, (_, document)) in scored_pairs
            ]
            if debug:
                print("Results after cross_encoder:")
                titles = [doc["document"]["title"] for _, doc in results]
                print(titles)
            results = results[:original_limit]

    results = [result[1] for result in results]
    return results
