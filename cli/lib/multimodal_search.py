from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    def embed_image(self, image_path: str):
        loaded_image = Image.open(image_path)
        embed = self.model.encode([loaded_image])
        return embed[0]

def verify_image_embedding(image_path):
    search_instance = MultimodalSearch()
    embedding = search_instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")