"""Local vector database abstraction layer supporting FAISS and Chroma."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import faiss

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add embeddings and metadata to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[list[int], list[float]]:
        """Search for nearest neighbors."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for fast similarity search."""

    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimensionality of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add embeddings and metadata to the index."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata items")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[list[int], list[float]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding of shape (1, embedding_dim)
            top_k: Number of results to return

        Returns:
            Tuple of (indices, distances)
        """
        if query_embedding.shape[0] != 1:
            raise ValueError("Query embedding must have shape (1, embedding_dim)")

        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        return indices[0].tolist(), distances[0].tolist()

    def save(self, path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    def get_metadata(self) -> list[dict]:
        """Get all metadata."""
        return self.metadata


class ChromaVectorStore(VectorStore):
    """Chroma-based vector store for local vector database."""

    def __init__(self, collection_name: str = "investopedia", persist_dir: Optional[Path] = None):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            persist_dir: Directory for persistent storage
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed. Install it with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_dir = persist_dir or Path("chroma_data")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.persist_dir),
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.collection = None

    def _ensure_collection(self) -> None:
        """Ensure collection is initialized."""
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add embeddings and metadata to the collection."""
        self._ensure_collection()

        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata items")

        # Convert embeddings to list format for Chroma
        embeddings_list = embeddings.tolist()

        # Generate IDs based on source and chunk_id
        ids = []
        for item in metadata:
            doc_id = f"{item['source']}_{item['chunk_id']}"
            ids.append(doc_id)

        # Prepare documents and metadatas
        documents = [item["text"] for item in metadata]
        metadatas = []
        for item in metadata:
            meta_item = {
                "source": item["source"],
                "chunk_id": str(item["chunk_id"]),
            }
            if "chunk_hash" in item:
                meta_item["chunk_hash"] = item["chunk_hash"]
            metadatas.append(meta_item)

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[list[int], list[float]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return

        Returns:
            Tuple of (indices, distances)
        """
        self._ensure_collection()
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        # Chroma returns distances, we need to map back to indices
        distances = results["distances"][0]
        return list(range(len(distances))), distances

    def save(self, path: Path) -> None:
        """Persist Chroma collection to disk."""
        self._ensure_collection()
        self.client.persist()
        print(f"Chroma collection persisted to {self.persist_dir}")

    def load(self, path: Path) -> None:
        """Load Chroma collection from disk."""
        self._ensure_collection()

    def get_metadata(self) -> list[dict]:
        """Get all metadata from collection."""
        self._ensure_collection()
        all_items = self.collection.get()
        metadata = []
        for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
            item = {
                "text": doc,
                "source": meta["source"],
                "chunk_id": int(meta["chunk_id"]),
            }
            if "chunk_hash" in meta:
                item["chunk_hash"] = meta["chunk_hash"]
            metadata.append(item)
        return metadata


def create_vector_store(backend: str = "faiss", embedding_dim: Optional[int] = None, **kwargs) -> VectorStore:
    """
    Factory function to create a vector store.

    Args:
        backend: "faiss" or "chroma"
        embedding_dim: Required for FAISS backend
        **kwargs: Additional arguments for the backend

    Returns:
        VectorStore instance
    """
    if backend == "faiss":
        if embedding_dim is None:
            raise ValueError("embedding_dim is required for FAISS backend")
        return FAISSVectorStore(embedding_dim)
    elif backend == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
