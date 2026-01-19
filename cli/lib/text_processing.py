import string

from .search_utils import load_stopwords
from nltk.stem import PorterStemmer


def preprocess_text(query: str) -> list[str]:
    # Convert to lower text
    preprocess_text = query.lower()

    # Remove punctuation
    preprocess_text = preprocess_text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split query in tokens
    tokens = preprocess_text.split()
    tokenized_query = [t for t in tokens if t.strip()]

    # Remove stopwords
    stopwords = load_stopwords()
    tokenized_query = [t for t in tokenized_query if t not in stopwords]

    # Stem tokens
    stemmer = PorterStemmer()
    tokenized_query = [stemmer.stem(t) for t in tokenized_query]

    # Return preprocess_text
    return tokenized_query
