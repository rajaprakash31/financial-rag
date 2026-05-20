# financial-rag

A local **Retrieval-Augmented Generation (RAG)** system for finance research running entirely on macOS.

## Features

- 🏠 **100% Local** – No cloud APIs required
- 🔍 **Vector Search** – FAISS or Chroma for similarity search
- 🧠 **Embeddings Layer** – Local sentence-transformers for text embeddings
- 📚 **Web Scraping** – Scrape Investopedia articles automatically
- 🤖 **Optional LLM** – Integrate a local llama.cpp model for answer generation
- ⚙️ **Flexible** – Switch between vector databases and embedding models

## Project Structure

```
financial-rag/
├── embeddings.py           # Embeddings manager and caching
├── vector_store.py         # Vector database abstraction (FAISS/Chroma)
├── build_index.py          # Create embeddings and index
├── query_rag.py            # Search and retrieve
├── scrape_investopedia.py  # Scrape web content
├── requirements.txt        # Python dependencies
├── models/                 # Cached embedding models
├── data/                   # Scraped documents
└── indexes/                # Vector indexes
```

## Setup (macOS)

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` – For local text embeddings
- `faiss-cpu` – Fast vector similarity search (default)
- `chromadb` – Alternative persistent vector database (optional)
- `requests` & `beautifulsoup4` – Web scraping
- `llama-cpp-python` – Local LLM inference (optional)
- `torch` – ML framework

### 3. Create working directories

```bash
mkdir -p data indexes models
```

## Workflow

### Step 1: Scrape Investopedia Articles

```bash
python scrape_investopedia.py https://www.investopedia.com/terms/r/roi.asp
python scrape_investopedia.py https://www.investopedia.com/terms/e/equity.asp
```

Articles are saved as `.txt` files in `data/`.

### Step 2: Build Embeddings & Vector Index

```bash
# Using FAISS (default, faster)
python build_index.py --data-dir data --index-dir indexes --backend faiss

# Or using Chroma (persistent, queryable)
python build_index.py --data-dir data --index-dir indexes --backend chroma
```

This:
1. Chunks documents into overlapping segments
2. Generates embeddings using a local model
3. Builds a vector index for fast retrieval
4. Saves embeddings config and metadata

### Step 3: Query the RAG System

```bash
# Basic retrieval
python query_rag.py --query "What is the difference between stocks and bonds?"

# With top-k results
python query_rag.py --query "What is ROI?" --top-k 5

# Using Chroma backend
python query_rag.py --query "Define dividend" --backend chroma

# With local LLM for answer generation
python query_rag.py --query "What is ROI?" --llm-model /path/to/model.gguf
```

## Configuration

### Embedding Models

By default, uses `all-MiniLM-L6-v2` (fast, ~384 dimensions). Other options:

```bash
# Larger, slower but more accurate
python build_index.py --model all-mpnet-base-v2

# Smaller, faster
python build_index.py --model all-MiniLM-L12-v2
```

Models are automatically downloaded and cached in `models/`.

### Vector Databases

#### FAISS (Default)
- ✅ Fast similarity search
- ✅ Memory efficient
- ⚠️ Limited to the data at build time

```bash
python build_index.py --backend faiss
python query_rag.py --backend faiss
```

#### Chroma
- ✅ Persistent storage
- ✅ Add/update documents dynamically
- ⚠️ Slightly slower than FAISS

```bash
python build_index.py --backend chroma
python query_rag.py --backend chroma
```

### Document Chunking

Edit in `build_index.py`:

```python
# Default: 250 words per chunk, 50-word overlap
chunk_text(text, chunk_size=250, overlap=50)
```

## Performance Tips

1. **Use GPU acceleration** (if available):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Use smaller embedding models** for faster indexing on older Macs:
   ```bash
   python build_index.py --model sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Batch queries** in your application to amortize model loading.

## Local LLM Integration

To generate answers instead of just retrieving passages:

1. Download a `llama.cpp` compatible model (e.g., from HuggingFace):
   ```bash
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf
   ```

2. Query with LLM:
   ```bash
   python query_rag.py --query "What is ROI?" --llm-model Mistral-7B-Instruct-v0.1.Q4_K_M.gguf
   ```

## Architecture

```
User Query
    ↓
[Embeddings Manager] → Encode query with local model
    ↓
[Vector Store] → Search (FAISS or Chroma)
    ↓
[Retrieved Passages] → Ranked by similarity
    ↓
[LLM (optional)] → Generate contextual answer
    ↓
Answer
```

## Notes

- ✅ Fully local – runs offline after initial model download
- ✅ No API keys or rate limits
- ⚠️ First run downloads embedding models (~500MB total)
- ⚠️ Always respect Investopedia's terms of use before scraping
- ⚠️ For production, add error handling, caching, and rate limiting

## Troubleshooting

### Models not caching?
Check `models/` directory. Models cache automatically in `~/.cache/huggingface` by default.

### Out of memory on M1/M2 Mac?
Use a smaller model:
```bash
python build_index.py --model sentence-transformers/all-MiniLM-L6-v2
```

### FAISS not working?
Install CPU version:
```bash
pip install faiss-cpu
```

## License

MIT

