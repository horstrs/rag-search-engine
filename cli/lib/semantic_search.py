import numpy as np
import os


from sentence_transformers import SentenceTransformer
from .search_utils import PROJECT_ROOT, load_movies

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_MOVIE_EMBEDDINGS = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def _initialize_docs(self, documents: list[dict]) -> list[str]:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        return [f"{d['title']}: {d['description']}" for d in documents]

    def _save(self) -> None:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

        with open(CACHE_MOVIE_EMBEDDINGS, "wb") as file:
            np.save(file, self.embeddings)

    def _load(self) -> None:
        if not os.path.exists(CACHE_MOVIE_EMBEDDINGS):
            raise FileNotFoundError(f"{CACHE_MOVIE_EMBEDDINGS} not found")

        with open(CACHE_MOVIE_EMBEDDINGS, "rb") as file:
            self.embeddings = np.load(file)

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        doc_list = self._initialize_docs(documents)
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        self._save()
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(CACHE_MOVIE_EMBEDDINGS):
            self._load()
            if len(self.embeddings) == len(documents):
                self._initialize_docs(documents)
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or text.isspace():
            raise ValueError("text for embeding can't be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        emb_query = self.generate_embedding(query)
        similarity_scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim_score = cosine_similarity(emb_query, doc_embedding)
            doc = self.documents[i]
            similarity_scores.append((sim_score, doc))
        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        search_hits = []
        for score, document in similarity_scores[:limit]:
            search_hits.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )
        return search_hits


def verify_model() -> None:
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def verify_embeddings() -> None:
    sem_search = SemanticSearch()
    documents = load_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str) -> None:
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> list[str]:
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int) -> None:
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    hits = search_instance.search(query, limit)
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']} (score: {hit['score']:.4f})")
        print(f"{hit['description'][:100]}... \n")
