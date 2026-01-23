import argparse
from lib.search_utils import load_golden_dataset, load_movies
from lib.hybrid_search import HybridSearch

K_DEFAULT = 60


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    dataset = load_golden_dataset()
    movies = load_movies()
    hybrid_search_instance = HybridSearch(movies)
    print(f"k={limit}\n")
    for testcase in dataset:
        query = testcase["query"]
        test_results = hybrid_search_instance.rrf_search(query, K_DEFAULT, limit)
        fetched_movies = [movie["document"]["title"] for _, movie in test_results]
        relevant_retrieved = get_relevants_for_testcase(
            fetched_movies, testcase["relevant_docs"]
        )
        precision = len(relevant_retrieved) / len(test_results)
        recall = len(relevant_retrieved) / len(testcase["relevant_docs"])
        if precision == 0 and recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        log_results(
            query,
            limit,
            precision,
            recall,
            f1_score,
            fetched_movies,
            testcase["relevant_docs"],
        )


def get_relevants_for_testcase(
    fetched_movies: list[str], relevants: list[str]
) -> list[str]:
    relevants = [movie for movie in fetched_movies if movie in relevants]
    return relevants


def log_results(
    query: str,
    k: int,
    precision: float,
    recall: float,
    f1_score: float,
    retrieved: list[str],
    relevant: list[str],
) -> None:
    print(f"- Query: {query}")
    print(f"  - Precision@{k}: {precision:.4f}")
    print(f"  - Recall@{k}: {recall:.4f}")
    print(f"  - F1 Score: {f1_score:.4f}")
    print(f"  - Retrieved: {', '.join(retrieved)}")
    print(f"  - Relevant: {', '.join(relevant)}")


if __name__ == "__main__":
    main()
