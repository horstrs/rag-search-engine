import argparse
import os

from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_parser.add_argument(
        "image", type=str, help="Image path to be verified"
    )

    args = parser.parse_args()
    if not os.path.exists(args.image):
        raise ValueError("Image file not found")

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
