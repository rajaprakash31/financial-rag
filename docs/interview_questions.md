# Interview Questions and Answers for the Multi-Index RAG Framework

## 1. What problem does your multi-index framework solve?

The framework addresses the need to support multiple independent search indexes for different domains or use cases. Instead of forcing all data into a single index with one set of chunking, tokenization, and embedding settings, each index can be optimized for its own data and usage pattern. This improves relevance, maintainability, and scalability.

## 2. Why use a separate orchestrator and index registry rather than one big index?

A separate orchestrator centralizes routing logic and keeps retrieval concerns separate from indexing. The index registry stores metadata and documentation for each index, so the orchestrator can choose the best index based on query intent. This avoids the complexity and poor relevance of one giant mixed-domain index.

## 3. How does the orchestrator decide which index to query?

The orchestrator reads each index manifest and builds a routing representation from index name, description, tags, and use cases. It embeds the user query with a routing model and scores that query against the documentation embeddings of each index. The highest scoring index is selected for retrieval.

## 4. What metadata is stored in `index_manifest.json`?

Each manifest includes:
- `name`
- `description`
- `use_cases`
- `tags`
- `backend`
- `model_name`
- `embedding_dim`
- `chunk_size`
- `overlap`
- `source_dir`
- `last_updated`

This information is used for routing, compatibility checks, and documentation.

## 5. How do you handle different chunk sizes or embedding models across indexes?

Each named index can be created with its own chunk size, overlap, and embedding model via `build_index.py`. The index manifest stores these settings. When querying or updating, the framework reads the manifest and uses the correct model and metadata, so each index remains independent.

## 6. What is deduplication and how does it work?

Deduplication prevents identical text chunks from being indexed multiple times. The framework computes a SHA-256 hash for each chunk and stores that hash in metadata. During incremental updates, if a chunk hash already exists in the target index, the chunk is skipped. This reduces duplicate content and keeps the index cleaner.

## 7. How does the system handle modified documents?

The update process compares both `source::chunk_id` and chunk hash. If the same chunk position exists but the text hash changed, the system detects the modification and logs an updated chunk. The new version is appended. In future iterations, this can be extended to replace or delete outdated vectors.

## 8. Why support both FAISS and Chroma?

FAISS is fast and efficient for local similarity search, while Chroma offers persistent, document-based storage and easier incremental updates. Supporting both gives flexibility: use FAISS for speed and Chroma for a more database-like experience.

## 9. What are the tradeoffs between FAISS and Chroma?

- FAISS: faster queries, efficient memory use, but less metadata flexibility and harder to perform complex updates.
- Chroma: persistent storage, better metadata handling, easier incremental operations, but potentially slower and more resource-intensive.

## 10. How do you ensure query embeddings match index embeddings?

Each index manifest stores the embedding model name and dimension. The framework checks this before query time. The query is embedded with the same model used to build the index, ensuring semantic compatibility.

## 11. Why not use a single shared tokenizer and chunking approach?

Different domains may require different granularity. Legal contracts need larger chunks and less overlap; finance glossary definitions need smaller chunks. A flat one-size-fits-all approach reduces relevance and forces suboptimal preprocessing on every dataset.

## 12. How would you add a new backend like Redis or Milvus?

Implement a new `VectorStore` subclass in `vector_store.py` with the same interface: `add`, `search`, `save`, `load`, and `get_metadata`. Then update the factory `create_vector_store()` and any index build/update logic to support the new backend.

## 13. How do you evaluate whether the orchestrator chose the right index?

You can measure routing accuracy by logging which index was selected and comparing it against ground truth or user feedback. Metrics include click-through rate, retrieval precision, and whether the selected index returned relevant passages. A/b tests between semantic routing and rule-based routing can also help.

## 14. How would you handle index versioning or rollback?

Store versioned snapshots of each named index folder, e.g. `indexes/<name>_v1`, `indexes/<name>_v2`, or use a commit-style manifest with timestamps. Maintain metadata for each version and implement a rollback script that restores a previous index folder and manifest.

## 15. What are the operational concerns for local LLM integration?

Key concerns are model file management, disk space, GPU/CPU usage, and loading latency. Use small models for study, keep a clear model path in `--llm-model`, and avoid using embedding models as LLM models. Also ensure the LLM is optional so retrieval still works without it.

## 16. How would you extend this for production?

- add error handling and retry logic
- add monitoring and logs for routing accuracy
- add index health checks and manifest validation
- add a web service/REST API for the orchestrator
- add model and index metadata versioning
- add support for more backends and distributed storage
