import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from lib.gemini_integration import GeminiClient


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_parser = subparsers.add_parser(
        "summarize",
        help="Performa an RRF search for a number of movies defined by parameter --limt. Then, generates a summary of the results. The summary is comprehensive, containing several key pieces of information about genre, plot, etc. for each movie",
    )
    rag_parser.add_argument("query", type=str, help="Search query for rrf search")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        const=5,
        help="Limit of movies to be summarized",
    )

    rag_parser = subparsers.add_parser(
        "citations",
        help="Performa an RRF search for a number of movies defined by parameter --limt. Then, generates an answer providing citations based on the search results",
    )
    rag_parser.add_argument("query", type=str, help="Search query for rrf search")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        const=5,
        help="Limit of movies to be summarized",
    )

    args = parser.parse_args()
    query = args.query
    movies = load_movies()
    hybrid_search_instance = HybridSearch(movies)
    gemini_client = GeminiClient()

    match args.command:
        case "rag":
            results = hybrid_search_instance.rrf_search(query, limit=5)
            results = [hits for _, hits in results]
            response = gemini_client.rag(query, results)
            print("Search Results:")
            titles = [entry["document"]["title"] for entry in results]
            for title in titles:
                print(f"  - {title}")
            print("\nRAG response:")
            print(response)
        case "summarize":
            results = hybrid_search_instance.rrf_search(query, limit=args.limit)
            results = [hits for _, hits in results]
            response = gemini_client.summarize(query, results)
            print("Search Results:")
            titles = [entry["document"]["title"] for entry in results]
            for title in titles:
                print(f"  - {title}")
            print("\nLLM Summary:")
            print(response)
        case "citations":
            results = hybrid_search_instance.rrf_search(query, limit=args.limit)
            results = [hits for _, hits in results]
            response = gemini_client.citations(query, results)
            print("Search Results:")
            titles = [entry["document"]["title"] for entry in results]
            for title in titles:
                print(f"  - {title}")
            print("\nLLM Answer:")
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
