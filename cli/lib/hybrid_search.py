import os

from .search_utils import load_movies
from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[(dict, float)]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def hybrid_score(
        self, bm25_score: float, semantic_score: float, alpha: float = 0.5
    ):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        TIMES_OVER_LIMIT = 500
        bm25_results = self._bm25_search(query, limit * TIMES_OVER_LIMIT)
        bm25_scores = [score for _, score in bm25_results]
        bm25_normalized = self.normalize(bm25_scores)

        weighted_results = {}

        for i, (movie, _) in enumerate(bm25_results):
            self._create_new_weighted_entry(
                weighted_results, movie, keyword_score=bm25_normalized[i]
            )
        semantic_results = self.semantic_search.search_chunks(
            query, limit * TIMES_OVER_LIMIT
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

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        TIMES_OVER_LIMIT = 500
        bm25_results = self._bm25_search(query, limit * TIMES_OVER_LIMIT)

        rrf_ranks = {}
        for i, (movie, _) in enumerate(bm25_results, 1):
            movie_id = movie["id"]
            self._create_rrf_entry(rrf_ranks, movie_id, movie)
            self._update_bm25_rank_and_score(rrf_ranks, movie_id, i, k)

        semantic_results = self.semantic_search.search_chunks(
            query, limit * TIMES_OVER_LIMIT
        )

        for i, sem_result in enumerate(semantic_results, 1):
            movie_id = sem_result["id"]
            if not rrf_ranks.get(movie_id):
                movie = self.idx.docmap["movie_id"]
                self._create_rrf_entry(rrf_ranks, movie_id, movie)
            self._update_semantic_rank_and_score(rrf_ranks, movie_id, i, k)

        sorted_rank = sorted(
            rrf_ranks.values(), key=lambda result: result["rrf_score"], reverse=True
        )
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


def rrf_search_command(query: str, k: float, limit: int) -> list[dict]:
    movies = load_movies()
    search_instance = HybridSearch(movies)
    return search_instance.rrf_search(query, k, limit)
