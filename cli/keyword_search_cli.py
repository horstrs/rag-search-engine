#!/usr/bin/env python3
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command
)

from lib.inverted_index import BM25_K1, BM25_B

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

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Returns the BM25 IDF for a given term"
    )
    bm25_idf_parser.add_argument("term", help="The term to retrieve the BM25 idf")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Optional: limit the search to the top --limit hits",
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
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            print(f"Searching for: {args.query} using full BM25 scoring...")
            result = bm25_search_command(args.query, args.limit)
            for i, bm_result in enumerate(result, 1):
                movie, score = bm_result
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
