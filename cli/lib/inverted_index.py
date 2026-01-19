import pickle
import os

from .text_processing import preprocess_text
from .search_utils import PROJECT_ROOT
from collections import defaultdict, Counter

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
CACHE_TERM_FREQ_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequency = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = preprocess_text(text)
        for token in tokenized_text:
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1

    def build(self, movie_lib: list[dict]) -> None:
        for movie in movie_lib:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

        with open(CACHE_INDEX_PATH, "wb") as file:
            pickle.dump(self.index, file)

        with open(CACHE_DOCMAP_PATH, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(CACHE_TERM_FREQ_PATH, "wb") as file:
            pickle.dump(self.term_frequency, file)

    def load(self) -> None:
        if not os.path.exists(CACHE_INDEX_PATH):
            raise FileNotFoundError(f"{CACHE_INDEX_PATH} not found")

        if not os.path.exists(CACHE_DOCMAP_PATH):
            raise FileNotFoundError(f"{CACHE_DOCMAP_PATH} not found")

        if not os.path.exists(CACHE_TERM_FREQ_PATH):
            raise FileNotFoundError(f"{CACHE_TERM_FREQ_PATH} not found")

        with open(CACHE_INDEX_PATH, "rb") as file:
            self.index = pickle.load(file)

        with open(CACHE_DOCMAP_PATH, "rb") as file:
            self.docmap = pickle.load(file)

        with open(CACHE_TERM_FREQ_PATH, "rb") as file:
            self.term_frequency = pickle.load(file)

    def get_documents(self, term: str) -> list[str]:
        query = term.lower()
        result = self.index.get(query, set())
        return sorted(list(result))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = preprocess_text(term)
        if len(token) != 1:
            raise ValueError("Only one token expected when getting token frequency")
        frequency = self.term_frequency[doc_id][term]
        return frequency
