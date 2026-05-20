"""Demo script to visualize vectors and explain the RAG pipeline."""

import json
import numpy as np
from pathlib import Path
from embeddings import EmbeddingsManager
from vector_store import FAISSVectorStore

# Load the index
print("=" * 80)
print("RAG PIPELINE WALKTHROUGH - With Sample Vectors")
print("=" * 80)

index_dir = Path("indexes")

# 1. Load embeddings config
print("\n1️⃣  EMBEDDINGS CONFIGURATION")
print("-" * 80)
config = EmbeddingsManager.load_config(index_dir / "embeddings_config.json")
print(f"Model: {config['model_name']}")
print(f"Vector Dimension: {config['embedding_dim']}")
print(f"What is this? Each text chunk gets converted into a {config['embedding_dim']}-dimensional vector")
print(f"The model captures semantic meaning: similar texts have similar vectors.")

# 2. Load metadata
print("\n2️⃣  METADATA (Text Chunks + Their Sources)")
print("-" * 80)
with open(index_dir / "metadata.json") as f:
    metadata = json.load(f)

print(f"Total chunks indexed: {len(metadata)}\n")
for i, chunk in enumerate(metadata[:3]):  # Show first 3
    print(f"Chunk #{chunk['chunk_id']} from {chunk['source'][:50]}...")
    print(f"Preview: {chunk['text'][:150]}...")
    print()

# 3. Load vector store and show actual vectors
print("\n3️⃣  SAMPLE VECTORS (Actual Embeddings)")
print("-" * 80)
embedding_dim = config["embedding_dim"]
vector_store = FAISSVectorStore(embedding_dim)
vector_store.load(index_dir)

# The FAISS index stores the actual vectors internally
# We can search to see what vectors look like
sample_query = "What is equity?"
embeddings_mgr = EmbeddingsManager(model_name=config["model_name"])
query_embedding = embeddings_mgr.embed_query(sample_query)

print(f"Query: '{sample_query}'")
print(f"Query vector shape: {query_embedding.shape} (1 query × {embedding_dim} dimensions)")
print(f"\nFirst 20 dimensions of query vector:")
print(query_embedding[0][:20])
print(f"... (384 dimensions total)")
print(f"\nVector statistics:")
print(f"  Min value: {query_embedding[0].min():.6f}")
print(f"  Max value: {query_embedding[0].max():.6f}")
print(f"  Mean value: {query_embedding[0].mean():.6f}")
print(f"  L2 Norm: {np.linalg.norm(query_embedding[0]):.6f} (normalized = ~1.0)")

# 4. Show similarity search
print("\n4️⃣  SIMILARITY SEARCH (Matching Query to Vectors)")
print("-" * 80)
indices, distances = vector_store.search(query_embedding, top_k=3)

print(f"Top 3 similar chunks to '{sample_query}':\n")
for rank, (idx, score) in enumerate(zip(indices, distances), 1):
    chunk = metadata[idx]
    print(f"{rank}. Score: {score:.4f}")
    print(f"   Source: {chunk['source']}")
    print(f"   Preview: {chunk['text'][:120]}...")
    print()

# 5. Show production architecture
print("\n5️⃣  PRODUCTION ARCHITECTURE FOR RAG")
print("=" * 80)

architecture = """
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION RAG SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

INDEXING PIPELINE (Offline - Run Once):
─────────────────────────────────────────
  Raw Documents → Chunking → Embeddings Generation → Vector Store
       (data/)      ├─ 250 word chunks        (384 dims each)      (index/)
                    ├─ 50 word overlap
                    └─ Metadata tracking

DATA FILES CREATED:
  ├─ index.faiss         (Binary FAISS index - stores vectors)
  ├─ metadata.json       (Maps vectors to source chunks)
  └─ embeddings_config.json (Model info for reproducibility)

QUERY PIPELINE (Real-time - On Every Request):
───────────────────────────────────────────────
  User Query → Embed Query → Vector Search → Retrieve Top-K → (Optional) LLM
     ↓           ↓              ↓              ↓              ↓
  "What is    Same model    FAISS index    Return          Generate
   equity?"   as indexing   similarity     passages        answer


KEY METRICS FOR PRODUCTION:
─────────────────────────────
  • Embedding dimension: 384 (smaller = faster, larger = more accurate)
  • Model: all-MiniLM-L6-v2 (lightweight, ~22M params)
  • Chunk size: 250 words (balance between context & granularity)
  • Vector index type: FAISS IndexFlatIP (Inner Product for cosine similarity)
  • Total vectors: {total_chunks}
  • Memory usage: ~{memory_mb:.1f}MB (in-memory indexing)

SCALING CONSIDERATIONS:
───────────────────────
  ✓ Small scale (< 100K vectors):  FAISS is sufficient
  ✓ Medium scale (100K - 1M):      Use HNSW indices for faster search
  ✓ Large scale (> 1M):            Use Pinecone, Weaviate, or Milvus
  ✓ Real-time updates:             Switch to Chroma or Weaviate
  ✓ Hybrid search:                 Combine vector + keyword search
  ✓ Multi-modal:                   Use cross-encoder models


COST & PERFORMANCE IN PRODUCTION:
──────────────────────────────────
  Indexing:
    • Time: ~2s per 100K tokens (single GPU)
    • Storage: ~1.5MB per 100K tokens (vectors only)
    
  Query (Search):
    • Latency: <10ms per query (FAISS)
    • Cost: Negligible (no API calls, runs locally)
    
  LLM Generation (if used):
    • Local llama.cpp: FREE, ~2-5 tokens/sec
    • API (OpenAI): $0.003 per 1K tokens
""".format(
    total_chunks=len(metadata),
    memory_mb=len(metadata) * 384 * 4 / (1024 * 1024),  # Rough estimate
)

print(architecture)

# 6. Show how to extend
print("\n6️⃣  NEXT STEPS FOR PRODUCTION")
print("=" * 80)
steps = """
1. SCRAPING:
   • Expand scraping to more domains (BeautifulSoup, Selenium)
   • Implement rate limiting and caching
   • Handle pagination and dynamic content

2. PREPROCESSING:
   • Clean HTML/Markdown properly
   • Remove duplicates
   • Language detection
   • Filter low-quality content

3. CHUNKING STRATEGIES:
   • Semantic chunking (split at sentence boundaries)
   • Hierarchical chunking (nested chunks)
   • Dynamic chunk sizing based on content

4. EMBEDDINGS:
   • Fine-tune model on domain data
   • Use larger models for better quality (all-mpnet-base-v2)
   • Cache embeddings to avoid recomputation

5. VECTOR DATABASE:
   • Add persistence (Chroma, Weaviate)
   • Implement incremental indexing
   • Add metadata filtering
   • Monitor search quality metrics

6. SEARCH & RANKING:
   • Hybrid search (dense + sparse/BM25)
   • Re-ranking with cross-encoders
   • Diversity-aware retrieval
   • Query expansion

7. LLM INTEGRATION:
   • Prompt engineering for better answers
   • Chain-of-thought reasoning
   • Multi-step retrieval
   • Citation tracking

8. MONITORING & EVAL:
   • Track retrieval metrics (precision, recall, MAP)
   • Monitor answer quality
   • A/B test different models
   • User feedback loops
"""
print(steps)

print("=" * 80)
print("✓ Index successfully built and analyzed!")
print("=" * 80)
