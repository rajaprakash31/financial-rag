"""Embeddings layer for managing local embedding models and caching."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingsManager:
    """Manages embeddings generation, caching, and model lifecycle."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        """
        Initialize embeddings manager.

        Args:
            model_name: HuggingFace model name for sentence-transformers
            cache_dir: Directory to cache embeddings model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def load_model(self) -> SentenceTransformer:
        """Load the embedding model (lazy loading)."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
            )
        return self.model

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (normalized)
        """
        model = self.load_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of shape (1, embedding_dim)
        """
        model = self.load_model()
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings from this model."""
        model = self.load_model()
        return model.get_sentence_embedding_dimension()

    def save_config(self, config_path: Path) -> None:
        """Save embedding configuration to a JSON file."""
        config = {
            "model_name": self.model_name,
            "embedding_dim": self.get_embedding_dim(),
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    @staticmethod
    def load_config(config_path: Path) -> dict:
        """Load embedding configuration from a JSON file."""
        return json.loads(config_path.read_text(encoding="utf-8"))
