#!/usr/bin/env python3
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
)


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

    idf_parser = subparsers.add_parser(
        "idf", help="Return the inverse index frequency of a given term"
    )
    idf_parser.add_argument("term", help="The term to retrieve the idf")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Return the tf-idf fo a term in a given document id"
    )
    tfidf_parser.add_argument(
        "doc_id", type=int, help="Document ID to look for the term and calculate the tf"
    )
    tfidf_parser.add_argument(
        "term", type=str, help="The term to calculate both the tf and idf"
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
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
