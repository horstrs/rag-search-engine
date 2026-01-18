import string


def preprocess_text(query: str) -> [str]:
    # Convert to lower text
    preprocess_text = query.lower()

    # Remove punctuation
    preprocess_text = preprocess_text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split query in tokens
    tokens = preprocess_text.split()
    preprocess_text = [t for t in tokens if t.strip()]

    # Return preprocess_text
    return preprocess_text
