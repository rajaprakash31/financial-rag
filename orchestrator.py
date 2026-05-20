import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from embeddings import EmbeddingsManager
from index_registry import load_all_manifests, load_index_manifest
from vector_store import FAISSVectorStore, ChromaVectorStore


def build_routing_text(manifest: dict[str, any]) -> str:
    fields = [
        manifest.get("name", ""),
        manifest.get("description", ""),
        " ".join(manifest.get("use_cases", [])),
        " ".join(manifest.get("tags", [])),
    ]
    return "\n".join([field for field in fields if field])


def score_indexes(query: str, manifests: list[dict[str, any]], router_model: str) -> list[tuple[dict[str, any], float]]:
    router = EmbeddingsManager(model_name=router_model)
    query_embedding = router.embed_query(query)
    scores = []

    for manifest in manifests:
        document = build_routing_text(manifest)
        if not document:
            continue
        doc_embedding = router.embed_texts([document], show_progress=False)
        score = float(np.dot(query_embedding[0], doc_embedding[0]))
        scores.append((manifest, score))

    return sorted(scores, key=lambda item: item[1], reverse=True)


def get_vector_store(manifest: dict[str, any], embedding_dim: int):
    backend = manifest.get("backend", "faiss")
    if backend == "faiss":
        store = FAISSVectorStore(embedding_dim)
        store.load(Path(manifest["index_dir"]))
        return store
    if backend == "chroma":
        store = ChromaVectorStore(persist_dir=Path(manifest["index_dir"]))
        store.load(Path(manifest["index_dir"]))
        return store
    raise ValueError(f"Unsupported backend: {backend}")


def query_index(query: str, manifest: dict[str, any], top_k: int = 3) -> list[dict[str, any]]:
    index_dir = Path(manifest["index_dir"])
    embedding_dim = manifest.get("embedding_dim")
    if embedding_dim is None:
        config_path = index_dir / "embeddings_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        embedding_dim = config["embedding_dim"]

    embeddings_mgr = EmbeddingsManager(model_name=manifest["model_name"])
    query_embedding = embeddings_mgr.embed_query(query)
    vector_store = get_vector_store(manifest, embedding_dim)
    indices, distances = vector_store.search(query_embedding, top_k=top_k)
    metadata = vector_store.get_metadata()

    results = []
    for idx, distance in zip(indices, distances):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        results.append({
            "source": item.get("source"),
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text"),
            "similarity": float(distance),
        })
    return results


def route_and_query(
    query: str,
    index_root: Path,
    router_model: str = "all-MiniLM-L6-v2",
    top_k: int = 3,
    use_llm: bool = False,
    llm_model_path: Optional[str] = None,
) -> None:
    manifests = load_all_manifests(index_root)
    if not manifests:
        raise SystemExit(f"No index manifests found under {index_root}")

    ranked = score_indexes(query, manifests, router_model)
    if not ranked:
        raise SystemExit("No suitable index found for the query.")

    best_manifest, best_score = ranked[0]
    print(f"Selected index: {best_manifest['name']} (score={best_score:.4f})")
    print(f"Description: {best_manifest.get('description', 'No description')}\n")

    results = query_index(query, best_manifest, top_k=top_k)
    print("\n=== Retrieved Passages ===")
    for i, item in enumerate(results, 1):
        print(f"{i}. source={item['source']} chunk_id={item['chunk_id']} similarity={item['similarity']:.4f}")
        print(item['text'][:800].replace('\n', ' '))
        print()

    if use_llm:
        if not llm_model_path:
            raise ValueError("llm_model_path must be provided when use_llm is True")
        prompt = build_prompt(query, [item["text"] for item in results])
        from query_rag import generate_with_llm
        answer = generate_with_llm(llm_model_path, prompt)
        print("=== Generated Answer ===")
        print(answer)


def build_prompt(query: str, passages: list[str]) -> str:
    prompt = (
        "Use the documents below to answer the question accurately. "
        "If the answer is not contained in the context, say you are not sure.\n\n"
    )
    for i, passage in enumerate(passages, 1):
        prompt += f"[Document {i}] {passage}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a query to the best local index and return results.")
    parser.add_argument("--query", required=True, help="Question to ask.")
    parser.add_argument("--index-root", default="indexes", help="Root directory containing named indexes.")
    parser.add_argument("--router-model", default="all-MiniLM-L6-v2", help="Embedding model to use for routing.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve from selected index.")
    parser.add_argument("--llm-model", default=None, help="Path to a local llama.cpp model for generation.")
    args = parser.parse_args()

    route_and_query(
        query=args.query,
        index_root=Path(args.index_root),
        router_model=args.router_model,
        top_k=args.top_k,
        use_llm=bool(args.llm_model),
        llm_model_path=args.llm_model,
    )


if __name__ == "__main__":
    main()
