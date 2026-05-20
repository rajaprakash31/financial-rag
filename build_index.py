import argparse
from pathlib import Path

from embeddings import EmbeddingsManager
from vector_store import create_vector_store


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_documents(source_dir: Path) -> list[dict[str, str]]:
    documents = []
    for path in sorted(source_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text)):
            documents.append({
                "source": str(path.name),
                "chunk_id": idx,
                "text": chunk,
            })
    return documents


def build_index(documents: list[dict[str, str]], model_name: str, index_dir: Path, backend: str = "faiss") -> None:
    """
    Build and save a vector index from documents.

    Args:
        documents: List of document chunks with metadata
        model_name: Embedding model name
        index_dir: Output directory for index
        backend: Vector store backend ("faiss" or "chroma")
    """
    # Initialize embeddings manager
    embeddings_mgr = EmbeddingsManager(model_name=model_name)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(documents)} chunks using {model_name}...")
    texts = [doc["text"] for doc in documents]
    embeddings = embeddings_mgr.embed_texts(texts, show_progress=True)
    
    # Create and populate vector store
    print(f"Creating {backend} vector store...")
    vector_store = create_vector_store(backend=backend, embedding_dim=embeddings_mgr.get_embedding_dim())
    vector_store.add(embeddings, documents)
    
    # Save index and metadata
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(index_dir)
    embeddings_mgr.save_config(index_dir / "embeddings_config.json")
    
    print(f"\n✓ Built index with {len(documents)} chunks.")
    print(f"✓ Saved {backend} index to {index_dir}")
    print(f"✓ Embedding dimension: {embeddings_mgr.get_embedding_dim()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a vector index from scraped Investopedia text files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing scraped .txt files.")
    parser.add_argument("--index-dir", default="indexes", help="Output directory for the index.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name for embeddings.")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"], 
                        help="Vector database backend to use.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    documents = load_documents(data_dir)
    if not documents:
        raise SystemExit(f"No text files found in {data_dir}. Run scrape_investopedia.py first.")

    build_index(documents, args.model, Path(args.index_dir), backend=args.backend)


if __name__ == "__main__":
    main()
