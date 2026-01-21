import numpy as np
import re
import os
import json

from .semantic_search import SemanticSearch, CACHE_DIR
from .search_utils import load_movies

CACHE_CHUNK_EMBEDDINGS = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CACHE_CHUNK_METADATA = os.path.join(CACHE_DIR, "chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def _save_chunk(self, total_chunks) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(CACHE_CHUNK_EMBEDDINGS, "wb") as file:
            np.save(file, self.chunk_embeddings)
        with open(CACHE_CHUNK_METADATA, "w") as file:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": total_chunks},
                file,
                indent=2,
            )

    def _load_chunk(self) -> None:
        if not os.path.exists(CACHE_CHUNK_EMBEDDINGS):
            raise FileNotFoundError(f"{CACHE_CHUNK_EMBEDDINGS} not found")

        with open(CACHE_CHUNK_EMBEDDINGS, "rb") as file:
            self.chunk_embeddings = np.load(file)

        if not os.path.exists(CACHE_CHUNK_METADATA):
            raise FileNotFoundError(f"{CACHE_CHUNK_METADATA} not found")

        with open(CACHE_CHUNK_METADATA, "r") as file:
            self.chunk_metadata = json.load(file)

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._initialize_docs(documents)
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            if not doc.get("description"):
                continue
            doc_chunks = semantic_chunk_text(doc.get("description"), 4, 1)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunk),
                    }
                )
                all_chunks.append(chunk)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        self._save_chunk(len(all_chunks))
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(CACHE_CHUNK_EMBEDDINGS):
            self._initialize_docs(documents)
            self._load()
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split(" ")
    return join_blocks_in_chunks(words, chunk_size, overlap)


def semantic_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return join_blocks_in_chunks(sentences, chunk_size, overlap)


def join_blocks_in_chunks(blocks: list[str], chunk_size: int, overlap: int) -> list[str]:
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
