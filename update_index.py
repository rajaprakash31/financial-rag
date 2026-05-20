import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from embeddings import EmbeddingsManager
from vector_store import create_vector_store, FAISSVectorStore, ChromaVectorStore


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_new_documents(
    source_dir: Path,
    existing_ids: set[str],
    existing_hashes: set[str],
    existing_doc_hashes: dict[str, str],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for path in sorted(source_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        for chunk_id, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunk_hash = compute_chunk_hash(chunk)
            doc_id = f"{path.name}::{chunk_id}"

            if chunk_hash in existing_hashes:
                continue

            if doc_id in existing_ids:
                existing_hash = existing_doc_hashes.get(doc_id)
                if existing_hash and existing_hash != chunk_hash:
                    print(f"Updated chunk detected: {doc_id}")

            documents.append({
                "source": path.name,
                "chunk_id": chunk_id,
                "text": chunk,
                "chunk_hash": chunk_hash,
            })

    return documents


def get_existing_index_info(index_dir: Path, backend: str, embedding_dim: int):
    if not index_dir.exists():
        return [], set(), set(), {}

    if backend == "faiss":
        store = FAISSVectorStore(embedding_dim)
        store.load(index_dir)
    else:
        store = ChromaVectorStore(persist_dir=index_dir)
        store.load(index_dir)

    metadata = store.get_metadata()
    existing_ids = set()
    existing_hashes = set()
    existing_doc_hashes: dict[str, str] = {}

    for item in metadata:
        doc_id = f"{item['source']}::{item['chunk_id']}"
        existing_ids.add(doc_id)
        chunk_hash = item.get("chunk_hash")
        if chunk_hash:
            existing_hashes.add(chunk_hash)
            existing_doc_hashes[doc_id] = chunk_hash

    return metadata, existing_ids, existing_hashes, existing_doc_hashes


def load_or_create_store(index_dir: Path, backend: str, embedding_dim: int):
    if backend == "faiss":
        store = FAISSVectorStore(embedding_dim)
        if index_dir.exists() and (index_dir / "index.faiss").exists():
            store.load(index_dir)
    else:
        store = ChromaVectorStore(persist_dir=index_dir)
        if index_dir.exists():
            store.load(index_dir)
    return store


def load_index_manifest(index_dir: Path) -> dict[str, any] | None:
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def update_index(
    data_dir: Path,
    index_dir: Path,
    model_name: str,
    backend: str,
    chunk_size: int,
    overlap: int,
    description: str | None = None,
    tags: list[str] | None = None,
    use_cases: list[str] | None = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    config_path = index_dir / "embeddings_config.json"
    manifest = load_index_manifest(index_dir)

    if manifest is not None:
        if manifest["model_name"] != model_name:
            raise ValueError(
                f"Embedding model mismatch: index uses {manifest['model_name']} "
                f"but update requested {model_name}"
            )
        if manifest["backend"] != backend:
            raise ValueError(
                f"Backend mismatch: index uses {manifest['backend']} "
                f"but update requested {backend}"
            )
        embedding_dim = manifest["embedding_dim"] if "embedding_dim" in manifest else EmbeddingsManager(model_name=model_name).get_embedding_dim()
    else:
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config["model_name"] != model_name:
                raise ValueError(
                    f"Embedding model mismatch: index uses {config['model_name']} "
                    f"but update requested {model_name}"
                )
            embedding_dim = config["embedding_dim"]
        else:
            embeddings_mgr = EmbeddingsManager(model_name=model_name)
            embedding_dim = embeddings_mgr.get_embedding_dim()
            embeddings_mgr.save_config(config_path)

    if manifest is None:
        manifest = {
            "name": index_dir.name,
            "description": description or f"Search index for {index_dir.name}",
            "use_cases": use_cases or [],
            "tags": tags or [],
            "backend": backend,
            "model_name": model_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "source_dir": str(data_dir),
            "last_updated": "",
        }

    _, existing_ids, existing_hashes, existing_doc_hashes = get_existing_index_info(
        index_dir, backend, embedding_dim
    )
    new_documents = load_new_documents(
        data_dir,
        existing_ids,
        existing_hashes,
        existing_doc_hashes,
        chunk_size,
        overlap,
    )
    if not new_documents:
        print("No new document chunks found to ingest.")
        return

    embeddings_mgr = EmbeddingsManager(model_name=model_name)
    texts = [doc["text"] for doc in new_documents]
    print(f"Embedding {len(new_documents)} new chunks...")
    embeddings = embeddings_mgr.embed_texts(texts, show_progress=True)

    store = load_or_create_store(index_dir, backend, embedding_dim)
    store.add(embeddings, new_documents)
    store.save(index_dir)

    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "documentation.txt").write_text(manifest["description"], encoding="utf-8")

    print(f"Added {len(new_documents)} new chunks to the {backend} index at {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally update the local Investopedia RAG index.")
    parser.add_argument("--data-dir", default="data", help="Directory containing new .txt files.")
    parser.add_argument("--index-dir", default="indexes", help="Directory containing the index.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"], help="Vector store backend.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks.")
    args = parser.parse_args()

    update_index(
        data_dir=Path(args.data_dir),
        index_dir=Path(args.index_dir),
        model_name=args.model,
        backend=args.backend,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
