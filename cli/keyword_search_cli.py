#!/usr/bin/env python3
from lib.keyword_search import search_command, build_command, tf_command


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index cache for keyword search")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Return the count for a given term in a given document id"
    )
    tf_parser.add_argument(
        "doc_id", type=int, help="Document ID where the term will be counted"
    )
    tf_parser.add_argument(
        "term", type=str, help="Term to retrieve the frequency in the given Document ID"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "search":
            print(f"Searching for: {args.query}")
            result = search_command(args.query)
            for i, movie in enumerate(result, 1):
                print(f"{i}. ({movie['id']}) {movie['title']}")

        case "tf":
            result = tf_command(args.doc_id, args.term)
            print(
                f"The term {args.term} appears {result} times in Document ID {args.doc_id}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
