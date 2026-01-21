import numpy as np
import re
import os
import json

from .semantic_search import SemanticSearch, cosine_similarity
from .search_utils import load_movies

SCORE_PRECISION = 10


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.CACHE_CHUNK_EMBEDDINGS = os.path.join(
            self.CACHE_DIR, "chunk_embeddings.npy"
        )
        self.CACHE_CHUNK_METADATA = os.path.join(
            self.CACHE_DIR, "chunk_metadata.json"
        )

    def _save_chunk(self, total_chunks) -> None:
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        with open(self.CACHE_CHUNK_EMBEDDINGS, "wb") as file:
            np.save(file, self.chunk_embeddings)
        with open(self.CACHE_CHUNK_METADATA, "w") as file:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": total_chunks},
                file,
                indent=2,
            )

    def _load_chunk(self) -> None:
        if not os.path.exists(self.CACHE_CHUNK_EMBEDDINGS):
            raise FileNotFoundError(f"{self.CACHE_CHUNK_EMBEDDINGS} not found")

        with open(self.CACHE_CHUNK_EMBEDDINGS, "rb") as file:
            self.chunk_embeddings = np.load(file)

        if not os.path.exists(self.CACHE_CHUNK_METADATA):
            raise FileNotFoundError(f"{self.CACHE_CHUNK_METADATA} not found")

        with open(self.CACHE_CHUNK_METADATA, "r") as file:
            data = json.load(file)
            self.chunk_metadata = data["chunks"]

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._initialize_docs(documents)
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents, 1):
            if not doc.get("description"):
                continue
            doc_chunks = semantic_chunk_text(doc.get("description"), 4, 1)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
                all_chunks.append(chunk)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        self._save_chunk(len(all_chunks))
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(self.CACHE_CHUNK_EMBEDDINGS):
            self._initialize_docs(documents)
            self._load_chunk()
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = super().generate_embedding(query)
        all_chunk_scores = self._compare_query_with_chunks(query_embedding)
        all_movies_scores = self._find_best_chunk_for_each_movie(all_chunk_scores)
        result = list(all_movies_scores.values())
        result.sort(key=lambda r: r["score"], reverse=True)
        return result[:limit]

    def _find_best_chunk_for_each_movie(self, all_chunk_scores) -> list[dict]:
        all_movies_scores = {}
        for chunk_score in all_chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            movie = self.document_map[movie_idx]
            if (
                movie_idx not in all_movies_scores
                or score > all_movies_scores[movie_idx]["score"]
            ):
                all_movies_scores[movie_idx] = {
                    "id": chunk_score["movie_idx"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "score": round(chunk_score["score"], SCORE_PRECISION),
                    "metadata": {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_score["chunk_idx"],
                    },
                }
        return all_movies_scores

    def _compare_query_with_chunks(self, query_embedding) -> list[dict]:
        all_chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[i]
            all_chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": similarity_score,
                }
            )

        return all_chunk_scores


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split(" ")
    return join_blocks_in_chunks(words, chunk_size, overlap)


def semantic_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text or len(text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = [s.strip() for s in sentences if s.strip()]
    return join_blocks_in_chunks(cleaned_sentences, chunk_size, overlap)


def join_blocks_in_chunks(
    blocks: list[str], chunk_size: int, overlap: int
) -> list[str]:
    all_chunks = []
    while True:
        chunk = " ".join(blocks[:chunk_size])
        all_chunks.append(chunk)
        if chunk_size >= len(blocks):
            break
        blocks = blocks[chunk_size - overlap :]
    return all_chunks


def embed_command() -> None:
    movies = load_movies()
    chunked_search_instance = ChunkedSemanticSearch()
    embeddings = chunked_search_instance.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int) -> None:
    movies = load_movies()
    chunked_search_instance = ChunkedSemanticSearch()
    chunked_search_instance.load_or_create_chunk_embeddings(movies)
    hits = chunked_search_instance.search_chunks(query, limit)
    for i, hit in enumerate(hits, 1):
        print(f"\n{i}. {hit['title']} (score: {hit['score']:.4f})")
        print(f"   {hit['document']}...")
