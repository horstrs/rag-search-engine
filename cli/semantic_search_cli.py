#!/usr/bin/env python3
from lib.semantic_search import verify_model, embed_text

import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser(
        "verify", help="Check if embedding model was loaded correctly"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Text to generate the embedding for"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
