# Implementation Starting Point

Start by turning the current CLI RAG flow into a backend chat flow.

## Recommended Order

1. Build or normalize indexes under `indexes/<index_name>/`.
   Each index should have `index.faiss`, `metadata.json`, `embeddings_config.json`, and `index_manifest.json`.

2. Add strong index manifests.
   The orchestrator routes better when each manifest has a clear `name`, `description`, `tags`, and `use_cases`.

3. Put the decision service in front of retrieval.
   Not every marketplace chat request should hit vector search. Greetings, ambiguous questions, live market data, and deterministic calculations should use other routes.

4. Expose one backend endpoint.
   The marketplace UI should call `POST /chat`; it should not know whether the backend used retrieval, direct answer, or a tool call.

5. Store every request and response.
   Keep route, selected index, citations, latency, and final answer. This becomes the dataset for debugging, product analytics, evaluation, and future router training.

6. Add offline evaluations.
   Create examples for `retrieve`, `direct_answer`, `tool_call`, and `clarify` so routing quality does not regress.

## Request / Response Storage

The sample `chat_orchestrator.py` uses SQLite and creates:

- `conversation_id`
- `tenant_id`
- `user_message`
- `route`
- `intent`
- `selected_index`
- `retrieved_sources_json`
- `prompt`
- `answer`
- `latency_ms`
- `decision_json`
- `created_at`

For production, move the same schema to Postgres or an analytics warehouse. Keep raw prompts only if your privacy policy allows it. If not, store redacted prompts, hashes, or sampled records.

## Local Demo

Install dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
python chat_orchestrator.py --message "What is equity?"
```

With a local LLM:

```bash
python chat_orchestrator.py \
  --message "What is equity and how do I calculate it?" \
  --llm-model /path/to/model.gguf
```

The interaction log is written to:

```text
data/chat_interactions.sqlite
```

## FastAPI Shape

Once `chat_orchestrator.py` works locally, wrap it with an API server:

```python
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from chat_orchestrator import MarketplaceChatOrchestrator

app = FastAPI()
orchestrator = MarketplaceChatOrchestrator(index_root=Path("indexes"))


class ChatRequest(BaseModel):
    conversation_id: str
    tenant_id: str
    message: str
    llm_model_path: str | None = None


@app.post("/chat")
def chat(request: ChatRequest):
    return orchestrator.handle_message(
        conversation_id=request.conversation_id,
        tenant_id=request.tenant_id,
        message=request.message,
        llm_model_path=request.llm_model_path,
    )
```

For hosted LLMs, replace `_generate_answer()` in `chat_orchestrator.py` with your provider call and keep the rest of the flow unchanged.
