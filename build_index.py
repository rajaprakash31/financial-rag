import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from embeddings import EmbeddingsManager
from vector_store import create_vector_store


def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def load_documents(source_dir: Path, chunk_size: int = 250, overlap: int = 50) -> list[dict[str, str]]:
    documents = []
    for path in sorted(source_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            documents.append({
                "source": str(path.name),
                "chunk_id": idx,
                "text": chunk,
                "chunk_hash": compute_chunk_hash(chunk),
            })
    return documents


def build_index(
    documents: list[dict[str, str]],
    model_name: str,
    index_dir: Path,
    backend: str = "faiss",
    description: str | None = None,
    tags: list[str] | None = None,
    use_cases: list[str] | None = None,
    source_dir: Path | None = None,
    chunk_size: int = 250,
    overlap: int = 50,
) -> None:
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

    manifest = {
        "name": index_dir.name,
        "description": description or f"Search index for {index_dir.name}",
        "use_cases": use_cases or [],
        "tags": tags or [],
        "backend": backend,
        "model_name": model_name,
        "embedding_dim": embeddings_mgr.get_embedding_dim(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source_dir": str(source_dir) if source_dir is not None else None,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "documentation.txt").write_text(manifest["description"], encoding="utf-8")

    print(f"\n✓ Built index with {len(documents)} chunks.")
    print(f"✓ Saved {backend} index to {index_dir}")
    print(f"✓ Embedding dimension: {embeddings_mgr.get_embedding_dim()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a named vector index from scraped text files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing scraped .txt files.")
    parser.add_argument("--index-root", default="indexes", help="Root directory for named indexes.")
    parser.add_argument("--index-name", default="default", help="Name of the index to create.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name for embeddings.")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"], help="Vector database backend to use.")
    parser.add_argument("--description", default=None, help="Human-readable description of the index.")
    parser.add_argument("--tags", default=None, help="Comma-separated tags for this index.")
    parser.add_argument("--use-cases", default=None, help="Comma-separated use cases for this index.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.glob("*.txt")):
        raise SystemExit(f"No text files found in {data_dir}. Run scrape_investopedia.py first.")

    target_dir = Path(args.index_root) / args.index_name
    documents = load_documents(data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not documents:
        raise SystemExit(f"No document chunks found in {data_dir}.")

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()] if args.tags else []
    use_cases = [case.strip() for case in args.use_cases.split(",") if case.strip()] if args.use_cases else []

    build_index(
        documents,
        args.model,
        target_dir,
        backend=args.backend,
        description=args.description,
        tags=tags,
        use_cases=use_cases,
        source_dir=data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
