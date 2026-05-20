# RAG BUILD INDEX SCRIPT - COMPLETE EXPLANATION

## What `build_index.py` Does

This script creates a **searchable vector database** from text documents. Here's the complete flow:

---

## 📊 STEP-BY-STEP BREAKDOWN

### **Step 1: Load Raw Documents**
```python
documents = load_documents(data_dir)  # Reads .txt files from data/
```

- Reads all `.txt` files from the `data/` directory
- Each file is treated as a complete document about a financial term
- Returns a list of text strings

**Output:**
```
1 document loaded:
- investopedia-www-investopedia-com-terms-e-equity-asp.txt (15KB)
```

---

### **Step 2: Chunk Documents into Smaller Pieces**
```python
def chunk_text(text, chunk_size=250, overlap=50):
```

**Why chunk?**
- Embeddings work better on focused content (250 words)
- Overlap (50 words) preserves context between chunks
- Allows more precise retrieval

**Example:**
```
Original text: "Equity is the remaining value of an asset..."
                ↓
Split into overlapping chunks:
  Chunk 0: [words 0-250]
  Chunk 1: [words 200-450]  ← overlaps by 50 words
  Chunk 2: [words 400-650]  ← overlaps by 50 words
  ...
```

**Result for our data:**
```
1 document → 12 chunks of 250 words each
Total metadata: 12 records
```

---

### **Step 3: Generate Embeddings**

The **EmbeddingsManager** converts each chunk into a 384-dimensional vector:

```python
embeddings_mgr = EmbeddingsManager(model_name="all-MiniLM-L6-v2")
embeddings = embeddings_mgr.embed_texts(texts)  # Returns shape (12, 384)
```

**What is an embedding?**
- A numerical representation of text meaning
- Similar texts → similar vectors → close in space
- Generated using a pre-trained neural network

**Model Details:**
```
Name: all-MiniLM-L6-v2
- 6 transformer layers
- 22M parameters
- Trained on 1B sentence pairs
- ~90MB model size
- Output: 384 dimensions
```

**Sample Vector (First 20 of 384 dimensions):**
```
[ 0.0427, -0.0191, -0.1166, -0.0012, -0.1152, -0.0723,
  0.0541, -0.0336,  0.0949, -0.0172,  0.0022,  0.0237,
  0.0159, -0.0384, -0.0185, -0.0341,  0.0259,  0.0288,
 -0.1143, -0.0025, ... ]
```

**Vector Properties:**
- Normalized (L2 norm = 1.0)
- Values range from -0.16 to +0.14
- Can be compared using cosine similarity

---

### **Step 4: Create Vector Index (FAISS)**

```python
vector_store = create_vector_store(backend="faiss", embedding_dim=384)
vector_store.add(embeddings, documents)
```

**FAISS = Facebook AI Similarity Search**

Creates an indexed data structure for fast similarity search:
```
Raw embeddings (12 × 384):    Direct search = O(n) ~ 12 ops
              ↓
        FAISS Index      Indexed search = O(log n) ~ 3-4 ops
```

**Index Type: IndexFlatIP**
- **Flat**: No compression, exact distances
- **IP**: Inner Product (cosine similarity for normalized vectors)
- Good for: < 100K vectors (local development)

---

### **Step 5: Save Index Files**

```
indexes/
├── index.faiss              (Binary FAISS index)
├── metadata.json            (Chunk content + sources)
└── embeddings_config.json   (Model info)
```

**File Contents:**

**embeddings_config.json** (62 bytes):
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "embedding_dim": 384
}
```

**metadata.json** (19 KB):
```json
[
  {
    "source": "investopedia-www-investopedia-com-terms-e-equity-asp.txt",
    "chunk_id": 0,
    "text": "Equity: Meaning, How It Works... [full 250-word chunk]"
  },
  {
    "source": "investopedia-www-investopedia-com-terms-e-equity-asp.txt",
    "chunk_id": 1,
    "text": "[next 250-word chunk with overlap]"
  },
  ...
]
```

**index.faiss** (18 KB):
```
Binary format containing:
- 12 vectors of 384 dimensions each
- FAISS index structure
- Similarity search acceleration
```

---

## 🔍 HOW QUERIES USE THIS INDEX

When you run `query_rag.py --query "What is equity?"`:

### **Step 1: Embed the Query**
```
Query: "What is equity?"
        ↓
    EmbeddingsManager (same model)
        ↓
    Query vector: [0.0427, -0.0191, ..., -0.0025]  (1 × 384)
```

### **Step 2: Search the Index**
```python
indices, distances = vector_store.search(query_embedding, top_k=3)
```

Returns:
```
Top 3 Results:
1. Chunk ID 0, Similarity Score: 0.7501
2. Chunk ID 6, Similarity Score: 0.7094
3. Chunk ID 3, Similarity Score: 0.6948
```

### **Step 3: Retrieve Original Text**
```python
for idx in indices:
    passage = metadata[idx]["text"]
    print(passage)
```

### **Step 4: Generate Answer (Optional)**
```
Retrieved passages + Question
              ↓
        Local LLM (llama.cpp)
              ↓
        Answer: "Equity represents..."
```

---

## 📈 VECTOR SPACE VISUALIZATION

Imagine a 384-dimensional space where:

```
                EQUITY-RELATED VECTORS
                    (clustered together)
                        
    ●●● ← Chunks about equity
    ●●●     ownership & shares
    ●●●
    
[Query vector for "What is equity?"]  ← Gets added here
           ↓
    ●●● ← Closest matches
    ●●●     returned
```

**Similarity Score** = How close vectors are (0 to 1)
- 1.0 = Perfect match
- 0.75 = Good match
- 0.50 = Partial match
- 0.0 = No similarity

---

## 🏭 PRODUCTION CONSIDERATIONS

### **1. Performance**
```
Our setup (12 vectors):
  Indexing time: ~5 seconds
  Query time: <1ms
  Memory: ~20MB (vectors + index)

Scale up to 1M vectors:
  Indexing time: ~5 minutes (GPU)
  Query time: <10ms (FAISS with GPU)
  Memory: ~1.5GB
```

### **2. Chunk Size Trade-offs**
```
Smaller chunks (100 words):
  ✓ More precise retrieval
  ✗ More fragmented context
  ✗ Slower indexing

Larger chunks (500 words):
  ✓ More complete context
  ✗ Less precise matches
  ✗ Slower search
```

### **3. Embedding Model Trade-offs**
```
Faster models (all-MiniLM-L6-v2):
  384 dims, 22M params
  ✓ Fast indexing
  ✓ Low memory
  ✗ Lower quality

Larger models (all-mpnet-base-v2):
  768 dims, 110M params
  ✓ Higher quality
  ✗ Slower
  ✗ More memory
```

### **4. Vector Database Options**

| Database | Vectors | Real-time Updates | Cost | Latency |
|----------|---------|------------------|------|---------|
| **FAISS** | <100K | No | Free | <1ms |
| **Chroma** | <1M | Yes | Free | 10ms |
| **Pinecone** | >1M | Yes | $0.40/M vectors | 100ms |
| **Weaviate** | Unlimited | Yes | $0/month (open) | 50ms |

---

## 💡 KEY LEARNINGS FOR PRODUCTION

1. **Embedding Consistency**: Use the SAME model for indexing & querying
2. **Normalization**: Vectors should be normalized (L2 norm = 1)
3. **Similarity Metric**: Cosine similarity for text (inner product on normalized)
4. **Scalability**: Switch from FAISS to HNSW/Pinecone at 100K+ vectors
5. **Incremental Updates**: Use Chroma/Weaviate for document updates
6. **Monitoring**: Track embedding quality & retrieval metrics

---

## 🚀 HOW TO RUN

```bash
# Build index
.venv/bin/python build_index.py \
  --data-dir data \
  --index-dir indexes \
  --backend faiss \
  --model all-MiniLM-L6-v2

# Query the index
.venv/bin/python query_rag.py \
  --query "What is equity?" \
  --backend faiss \
  --top-k 3

# See actual vectors
.venv/bin/python demo_vectors.py
```

---

## 📚 WHAT YOU NOW HAVE

✅ **12 text chunks** from Investopedia  
✅ **12 embedding vectors** (384 dimensions each)  
✅ **FAISS index** for fast similarity search  
✅ **Metadata mapping** (vector ↔ original text)  
✅ **Embedding configuration** (model + dimensions)  

This is the foundation for a production RAG system! 🎯
