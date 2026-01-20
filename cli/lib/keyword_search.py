from .text_processing import preprocess_text
from lib.search_utils import load_movies
from .inverted_index import InvertedIndex, BM25_K1, BM25_B

DEFAULT_SEARCH_LIMIT = 5


def build_command() -> None:
    movies = load_movies()
    inverted_index = InvertedIndex()
    inverted_index.build(movies)
    inverted_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    seen, results = set(), []
    query_tokens = preprocess_text(query)

    for token in query_tokens:
        token_ids = inverted_index.get_documents(token)
        for id in token_ids:
            if id in seen:
                continue
            seen.add(id)
            results.append(inverted_index.docmap[id])
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tfidf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(query: str, limit) -> list[dict, float]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)


def retrieve_movies_with_query_in_title(
    movies: dict, query_tokens: list[str], lenght_limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    search_result = []
    for movie in movies:
        title_tokens = preprocess_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            search_result.append(movie)
            if len(search_result) >= lenght_limit:
                break

    return search_result


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
