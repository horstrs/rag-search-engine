#!/usr/bin/env python3
from lib.keyword_search import retrieve_movies_with_query_in_title
from lib.search_utils import load_movies

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies = load_movies()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = retrieve_movies_with_query_in_title(movies, args.query)
            for i, movie in enumerate(result, 1):
                print(f"{i}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
