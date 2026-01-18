DEFAULT_SEARCH_LIMIT = 5


def retrieve_movies_with_query_in_title(
    movies: dict, query: str, lenght_limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    search_result = []
    for movie in movies:
        if query in movie["title"]:
            search_result.append(movie)
            if len(search_result) >= lenght_limit:
                break
    return search_result
