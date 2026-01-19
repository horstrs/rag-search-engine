from lib.keyword_search import InvertedIndex  # or correct import path
from lib.search_utils import load_movies

idx = InvertedIndex()
idx.load()
idx.build(load_movies())
print(idx.term_frequency[424])  # should show a Counter with words
print(idx.get_tf(424, "bear"))  # should be > 0 if “bear” is in that movie
