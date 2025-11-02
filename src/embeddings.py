import torch
from typing import List
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    """Generate embeddings with Alibaba-NLP/gte-base-en-v1.5 (SentenceTransformer)."""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-base-en-v1.5") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query."""
        if not text:
            return []
        embedding = self.model.encode(
            text,
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embedding.cpu().tolist()