# QUICK START & SUMMARY

## What You've Built

A **local RAG system** that runs entirely on your Mac without any cloud APIs:

```
📄 Documents (Investopedia)
    ↓
🔪 Chunking (250-word overlaps)
    ↓
🧠 Embeddings (384-dimensional vectors)
    ↓
📊 Vector Database (FAISS)
    ↓
🔍 Similarity Search (<1ms per query)
    ↓
💬 LLM Integration (optional, local llama.cpp)
    ↓
📝 User Gets Answers
```

---

## Key Components Explained

### 1. **embeddings.py** - Text to Vectors
```python
from embeddings import EmbeddingsManager

mgr = EmbeddingsManager("all-MiniLM-L6-v2")
vector = mgr.embed_query("What is equity?")
# Returns: [0.042, -0.019, ..., -0.002]  (1 × 384)

# Metadata: What does each dimension mean?
# → Unknown! But collectively they capture semantic meaning
```

**What it does:**
- Loads pre-trained sentence transformer model
- Converts text into numerical vectors
- Normalizes vectors for cosine similarity
- Caches models locally

---

### 2. **vector_store.py** - Storing & Searching Vectors
```python
from vector_store import FAISSVectorStore

store = FAISSVectorStore(embedding_dim=384)
store.add(embeddings, metadata)

# Search
indices, scores = store.search(query_vector, top_k=3)
# Returns: [0, 10, 7]  (chunk IDs) and [0.750, 0.709, 0.695]  (similarity scores)
```

**Backends available:**
- **FAISS** (current) - Fast, in-memory, < 100K vectors
- **Chroma** - Persistent, real-time updates, < 1M vectors
- **Pinecone** - Managed cloud, >1M vectors, paid

---

### 3. **build_index.py** - Creating the Index
```bash
.venv/bin/python build_index.py \
  --data-dir data \
  --index-dir indexes \
  --backend faiss \
  --model all-MiniLM-L6-v2
```

**Process:**
1. Load all `.txt` files from `data/`
2. Split into overlapping chunks (250 words, 50-word overlap)
3. Generate embeddings for each chunk
4. Create FAISS index for fast search
5. Save to `indexes/`:
   - `index.faiss` (18 KB) - Binary index
   - `metadata.json` (19 KB) - Chunk content
   - `embeddings_config.json` (62 B) - Model info

**Our result:**
```
1 article → 12 chunks → 12 vectors → 0.0MB index
```

---

### 4. **query_rag.py** - Querying the System
```bash
.venv/bin/python query_rag.py \
  --query "What is equity?" \
  --backend faiss \
  --top-k 3
```

**Flow:**
```
Query: "What is equity?"
    ↓
Embed: [0.042, -0.019, ..., -0.002]  (1 × 384)
    ↓
Search FAISS index: O(log n) → <1ms
    ↓
Top 3 results with similarity scores
    ↓
Display passages
    ↓
(Optional) Generate answer with local LLM
```

---

### 5. **scrape_investopedia.py** - Getting Data
```bash
.venv/bin/python scrape_investopedia.py \
  https://www.investopedia.com/terms/e/equity.asp \
  https://www.investopedia.com/terms/s/stock.asp
```

**Saves to:** `data/investopedia-*.txt`

---

## Sample Vector Data

### Your First Query
```
Query: "What is equity?"

Vector representation (384 dims):
[ 0.0427, -0.0191, -0.1166, -0.0012, -0.1152, -0.0723,
  0.0541, -0.0336,  0.0949, -0.0172,  0.0022,  0.0237,
  0.0159, -0.0384, -0.0185, -0.0341,  0.0259,  0.0288,
 -0.1143, -0.0025, ... (364 more dimensions)]

Vector properties:
  - L2 Norm: 1.0 (normalized)
  - Min: -0.162
  - Max: +0.142
  - Mean: -0.001
```

### Similarity Scores
```
Query: "What is equity?"

Top 3 matches:
1. Chunk 0: "Equity: Meaning, How It Works..."
   Score: 0.7501 ← Cosine similarity
   
2. Chunk 10: "total liabilities from total assets..."
   Score: 0.7094
   
3. Chunk 7: "Home equity is roughly comparable..."
   Score: 0.6948
```

**Interpretation:**
- 0.75 = Good match (75% similarity)
- 0.70 = Good match (70% similarity)
- 0.50 = Fair match (50% similarity)
- 0.00 = No match (0% similarity)

---

## Files Created

```
financial-rag/
├── embeddings.py              # Embedding manager
├── vector_store.py            # Vector DB abstraction (FAISS/Chroma)
├── build_index.py             # Create index from documents
├── query_rag.py               # Query the system
├── scrape_investopedia.py     # Web scraper
│
├── demo_vectors.py            # Visualize vectors & architecture
├── BUILD_INDEX_GUIDE.md       # Deep dive into build_index.py
├── PRODUCTION_GUIDE.md        # How to scale to production
│
├── data/                       # Raw documents
│   └── investopedia-*.txt
│
├── indexes/                    # Vector index
│   ├── index.faiss
│   ├── metadata.json
│   └── embeddings_config.json
│
├── models/                     # Cached embedding models
│   └── [downloaded from HF]
│
└── requirements.txt            # Python dependencies
```

---

## Commands Cheat Sheet

```bash
# Setup (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Scrape data
.venv/bin/python scrape_investopedia.py https://...

# Build index
.venv/bin/python build_index.py \
  --data-dir data \
  --index-dir indexes \
  --backend faiss

# Query
.venv/bin/python query_rag.py \
  --query "What is equity?" \
  --top-k 3

# View vectors & architecture
.venv/bin/python demo_vectors.py
```

---

## Production Scaling Path

### **Now (Dev)**
- ✓ FAISS in-memory index
- ✓ Local embeddings
- ✓ Single machine
- Vectors: 12

### **Phase 1 (10K vectors)**
- Switch to GPU for faster embeddings
- Add caching layer
- Monitor retrieval quality

### **Phase 2 (100K vectors)**
- Migrate to Chroma (persistent)
- Implement incremental updates
- Add metadata filtering

### **Phase 3 (1M+ vectors)**
- Use Pinecone / Weaviate / Milvus
- Distributed indexing
- Real-time updates
- Multi-region deployment

---

## Key Insights for Production

### **1. Vector Quality**
```
Better embeddings → Better retrieval → Better answers
Trade-off: Speed vs Quality
- all-MiniLM-L6-v2: Fast (2s) but lower quality
- all-mpnet-base-v2: Slower (5s) but higher quality
```

### **2. Chunk Size**
```
Optimal: 200-500 words with 50-word overlap
- Smaller: More precise but fragmented
- Larger: More context but less precise
```

### **3. Search Speed**
```
FAISS:    <1ms  per query (in-memory)
Chroma:   ~10ms per query (disk I/O)
Pinecone: ~100ms per query (network)
```

### **4. Consistency**
```
CRITICAL: Use SAME model for indexing AND querying
- Different models = Different embedding dimensions
- Dimension mismatch = Search fails
```

---

## Learning Resources

### **Understanding Vectors**
- Each dimension captures different semantic features
- Similar texts have similar vectors
- Normalized vectors can use cosine similarity
- Vector space = semantic space

### **FAISS Basics**
```
IndexFlatIP:     Exact search, ~O(n) time
IndexIVF:        Approximate, ~O(log n) time
IndexHNSW:       Graph-based, ~O(1) time
```

### **Embedding Models**
- `all-MiniLM-L6-v2` - 384 dims, fast
- `all-mpnet-base-v2` - 768 dims, quality
- `e5-base` - 768 dims, best general
- Fine-tune for domain-specific tasks

---

## What's Next?

✅ **You now understand:**
- How embeddings work
- How vector similarity search works
- How to build a RAG system
- How to scale to production

🚀 **Next steps:**
1. Scrape more financial data
2. Experiment with different chunk sizes
3. Try different embedding models
4. Add local LLM for answer generation
5. Set up monitoring & evaluation
6. Deploy as API (FastAPI)

---

## Questions?

**Q: Why 384 dimensions?**  
A: Model choice. Trade-off between speed and quality.

**Q: Can I use different embedding models?**  
A: Yes, but reindex if switching models.

**Q: How many vectors can FAISS handle?**  
A: <1M comfortably. >1M needs distributed setup.

**Q: Is this production-ready?**  
A: Yes for <100K vectors. Scale differently for >1M.

**Q: Can I add/update documents?**  
A: Use Chroma instead of FAISS for real-time updates.

---

Good luck building your RAG system! 🚀
