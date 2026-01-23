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

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            movies = load_movies()
            hybrid_search_instance = HybridSearch(movies)
            gemini_client = GeminiClient()
            results = hybrid_search_instance.rrf_search(query, 5)
            results = [hits for _, hits in results]
            response = gemini_client.rag(query, results)
            print("Search Results:")
            titles = [entry["document"]["title"] for entry in results]
            for title in titles:
                print(f"  - {title}")
            print("\nRAG response:")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
