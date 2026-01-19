from lib.keyword_search import InvertedIndex  # or correct import path
from lib.text_processing import preprocess_text

idx = InvertedIndex()
idx.load()

print("total docs:", len(idx.docmap))
token = preprocess_text("grizzly")
print("docs for 'grizzly':", idx.get_documents(token[0]))
print("count for 'grizzly':", len(idx.get_documents(token[0])))

print("docs for 'actor':", idx.get_documents("actor"))
print("count for 'actor':", len(idx.get_documents("actor")))


print(preprocess_text("Grizzly"))  # what is this?
print(preprocess_text("grizzly bear"))  # and this?
