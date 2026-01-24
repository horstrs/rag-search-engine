import argparse
import os
import mimetypes

from lib.search_utils import load_image
from lib.gemini_integration import GeminiClient


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")

    parser.add_argument(
        "--image",
        type=str,
        help="Path of an image to be used as base for the query rewrite",
    )

    parser.add_argument(
        "--query", type=str, help="Search query to be rewritten based on the image"
    )

    args = parser.parse_args()
    if not os.path.exists(args.image):
        raise ValueError("File not found")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    image = load_image(args.image)

    query = args.query.strip()
    gemini_client = GeminiClient()

    response = gemini_client.rewrite_from_image(query, image, mime)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
        # case _:
        #     parser.print_help()


if __name__ == "__main__":
    main()
