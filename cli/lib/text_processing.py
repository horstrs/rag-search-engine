import string
from .search_utils import load_stopwords

def preprocess_text(query: str) -> [str]:
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

    # Return preprocess_text
    return tokenized_query
