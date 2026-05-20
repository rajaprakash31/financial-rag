import argparse
import json
from pathlib import Path

from embeddings import EmbeddingsManager
from vector_store import create_vector_store, FAISSVectorStore, ChromaVectorStore

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def build_prompt(query: str, passages: list[str]) -> str:
    prompt = (
        "Use the documents below to answer the question accurately. "
        "If the answer is not contained in the context, say you are not sure.\n\n"
    )
    for i, passage in enumerate(passages, 1):
        prompt += f"[Document {i}] {passage}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    return prompt


def generate_with_llm(model_path: str, prompt: str, max_tokens: int = 256) -> str:
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed. Install it from requirements.txt.")
    llm = Llama(model_path=model_path, n_threads=4)
    response = llm(prompt=prompt, max_tokens=max_tokens)
    return response["choices"][0]["text"].strip()


def query_rag(
    query: str,
    index_dir: Path,
    model_name: str,
    backend: str = "faiss",
    top_k: int = 3,
    llm_model_path: str = None,
) -> None:
    """
    Query the RAG system.

    Args:
        query: User query
        index_dir: Directory containing the index
        model_name: Embedding model name
        backend: Vector store backend ("faiss" or "chroma")
        top_k: Number of results to retrieve
        llm_model_path: Optional path to local LLM for generation
    """
    # Load embeddings configuration
    config_path = index_dir / "embeddings_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Embeddings config not found at {config_path}. Run build_index.py first.")
    
    config = EmbeddingsManager.load_config(config_path)
    embedding_dim = config["embedding_dim"]
    
    # Initialize embeddings manager
    embeddings_mgr = EmbeddingsManager(model_name=model_name)
    
    # Initialize vector store
    print(f"Loading {backend} vector store from {index_dir}...")
    if backend == "faiss":
        vector_store = FAISSVectorStore(embedding_dim)
        vector_store.load(index_dir)
    elif backend == "chroma":
        vector_store = ChromaVectorStore(persist_dir=index_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Generate query embedding
    print(f"Embedding query: {query}")
    query_embedding = embeddings_mgr.embed_query(query)
    
    # Search vector store
    indices, distances = vector_store.search(query_embedding, top_k=top_k)
    metadata_list = vector_store.get_metadata()
    
    # Display results
    print("\n" + "="*80)
    print("RETRIEVED PASSAGES:")
    print("="*80 + "\n")
    
    passages = []
    for rank, (idx, distance) in enumerate(zip(indices, distances), 1):
        if idx < 0 or idx >= len(metadata_list):
            continue
        item = metadata_list[idx]
        snippet = item["text"].replace("\n", " ")
        print(f"{rank}. source={item['source']} chunk_id={item['chunk_id']} similarity={distance:.4f}")
        print(f"   {snippet[:800]}\n")
        passages.append(snippet)
    
    # Generate answer with LLM if provided
    if llm_model_path:
        prompt = build_prompt(query, passages)
        print("="*80)
        print("GENERATING ANSWER WITH LOCAL LLM...")
        print("="*80 + "\n")
        answer = generate_with_llm(llm_model_path, prompt)
        print("ANSWER:\n")
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local Investopedia RAG index.")
    parser.add_argument("--query", required=True, help="Question to ask.")
    parser.add_argument("--index-dir", default="indexes", help="Directory containing the index.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name for query embeddings.")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "chroma"],
                        help="Vector database backend to use.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve.")
    parser.add_argument("--llm-model", default=None, help="Path to a local llama.cpp model file for generation.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise SystemExit(f"Index directory not found at {index_dir}. Run build_index.py first.")

    query_rag(
        query=args.query,
        index_dir=index_dir,
        model_name=args.model,
        backend=args.backend,
        top_k=args.top_k,
        llm_model_path=args.llm_model,
    )


if __name__ == "__main__":
    main()
