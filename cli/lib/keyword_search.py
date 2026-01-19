from .text_processing import preprocess_text
from lib.search_utils import load_movies
from .inverted_index import InvertedIndex

DEFAULT_SEARCH_LIMIT = 5


def build_command() -> None:
    movies = load_movies()
    inverted_index = InvertedIndex()
    inverted_index.build(movies)
    inverted_index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = preprocess_text(query)
        title_tokens = preprocess_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


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
