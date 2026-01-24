import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "golden_dataset.json")


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as file:
        data = file.read()
    return data.split("\n")


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r") as file:
        data = json.load(file)
    return data["test_cases"]


def load_image(path) -> str:
    with open(path, "rb") as file:
        return file.read()