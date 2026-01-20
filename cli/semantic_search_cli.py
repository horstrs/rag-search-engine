#!/usr/bin/env python3
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_command,
)

import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser(
        "verify", help="Check if embedding model was loaded correctly"
    )

    subparsers.add_parser(
        "verify_embeddings", help="Check if the dataset embedding were loaded correctly"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Text to generate the embedding for"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a query"
    )
    embed_query_parser.add_argument(
        "query", type=str, help="Query to generate the embedding for"
    )

    search_parser = subparsers.add_parser(
        "search", help="Performa a semantical search for a query"
    )
    search_parser.add_argument(
        "query", type=str, help="Query to generate the embedding for"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Optional. Number of search hits to return",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split a given text in chunks of [--chunk-size]"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="Number of words in each chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Number of words that overlaps chunk boundaries",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split a given text in chunks of [--chunk-size]"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Number of sentences in each chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Number of words that overlaps chunk boundaries",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
