import pickle
import os

from .text_processing import preprocess_text
from .search_utils import PROJECT_ROOT
from collections import defaultdict

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}

    def __add_document(self, doc_id: str, text: str) -> None:
        tokenized_text = preprocess_text(text)
        for token in tokenized_text:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[str]:
        query = term.lower()
        result = sorted(list(self.index[query]))
        return result

    def build(self, movie_lib: list[dict]) -> None:
        for movie in movie_lib:
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        with open(CACHE_INDEX_PATH, "wb") as file:
            pickle.dump(self.index, file)

        with open(CACHE_DOCMAP_PATH, "wb") as file:
            pickle.dump(self.docmap, file)
