import argparse

from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores. Highest score gets 1.0 and lowest gets 0.0",
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="List of scores to be normalized"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Perform a hybrid search, with configurable parameter to control weight of keyword vs semantic search",
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to be searched")

    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Constant that controls weight of keyword vs semantic. 1 = 100% Keyword, 0.5 = 50/50, 0.0 = 100% Semantic",
    )

    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of search hits to be returned",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Perform a hybrid search, using rrf scores in both types of searches to get the final result",
    )

    rrf_search_parser.add_argument("query", type=str, help="Query to be searched")

    rrf_search_parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        default=60,
        help="Constant that controls weight of the rank in each type of search. Lower K-value means higher hank results have more weight.",
    )

    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of search hits to be returned",
    )

    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            hits = weighted_search_command(args.query, args.alpha, args.limit)
            for i, hit in enumerate(hits, 1):
                print(f"{i}. {hit['document']['title']}")
                print(f"   Hybrid Score: {hit['hybrid_score']:.4f}")
                print(
                    f"   BM25: {hit['keyword_score']:.4f}, Semantic: {hit['semantic_score']:.4f}"
                )
                print(f"   {hit['document']['description'][:100]}")
        case "rrf-search":
            hits = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            for i, hit in enumerate(hits, 1):
                print(f"{i}. {hit['document']['title']}")
                print(f"   RRF Score: {hit['rrf_score']:.4f}")
                print(
                    f"   BM25 Rank: {hit['bm25_rank']}, Semantic Rank: {hit['semantic_rank']}"
                )
                print(f"   {hit['document']['description'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
