# Marketplace Chat Architecture for Financial RAG

This architecture exposes the financial RAG service through a marketplace chat UI while keeping retrieval optional. The orchestration layer decides whether a request needs vector context, a deterministic service, clarification, or a direct LLM response.

## High-Level Architecture

```mermaid
flowchart LR
    U[Marketplace User] --> UI[Marketplace Chat UI]
    UI --> API[Chat API Gateway]

    API --> AUTH[Auth, Tenant, Rate Limit]
    AUTH --> SAFE[Input Guardrails<br/>PII, abuse, prompt-injection scan]
    SAFE --> ORCH[Conversation Orchestrator]

    ORCH --> MEM[Session Memory<br/>chat history, user profile]
    ORCH --> DECIDE[Decision Service<br/>intent, topic, risk, retrieval need]

    DECIDE -->|Needs clarification| CLARIFY[Ask Follow-up Question]
    CLARIFY --> UI

    DECIDE -->|General chat / small talk| DIRECT[Direct LLM Answer]
    DIRECT --> FORMAT[Response Formatter]

    DECIDE -->|Known deterministic action| TOOLS[Financial Tools / APIs<br/>pricing, calculators, account data]
    TOOLS --> TOOLCTX[Tool Result Context]
    TOOLCTX --> LLM[LLM Response Generator]

    DECIDE -->|Financial knowledge needed| ROUTER[Index Router]
    ROUTER --> REG[Index Registry<br/>index_manifest.json files]

    subgraph Financial Knowledge Indexes
        I1[Investopedia Concepts Index]
        I2[Filings / Reports Index]
        I3[Marketplace FAQ Index]
        I4[Policy / Compliance Index]
    end

    REG --> ROUTER
    ROUTER -->|select index or indexes| RETRIEVE[Vector Retrieval<br/>FAISS / Chroma top-k]
    RETRIEVE --> RERANK[Rerank + Threshold Filter]
    RERANK --> CTX[Grounding Context<br/>chunks, sources, similarity]
    CTX --> LLM

    MEM --> LLM
    ORCH --> LLM
    LLM --> FORMAT
    FORMAT --> CITED[Final Chat Response<br/>answer, citations, disclaimers]
    CITED --> UI

    ORCH --> OBS[Logs, Metrics, Traces<br/>route, latency, retrieval hits]
    LLM --> OBS
    RETRIEVE --> OBS
```

## Runtime Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Marketplace Chat UI
    participant API as Chat API
    participant Orch as Orchestrator
    participant Decide as Decision Service
    participant Router as Index Router
    participant Vector as Vector Store
    participant LLM

    User->>UI: Ask finance question
    UI->>API: POST /chat
    API->>Orch: normalized message + tenant context
    Orch->>Decide: classify intent and retrieval need

    alt Retrieval not needed
        Decide-->>Orch: direct_answer
        Orch->>LLM: prompt with conversation context only
        LLM-->>Orch: answer
    else Retrieval needed
        Decide-->>Orch: retrieve(topic, index candidates, top_k)
        Orch->>Router: score indexes using manifests
        Router-->>Orch: selected index or indexes
        Orch->>Vector: embed query and search top-k chunks
        Vector-->>Orch: chunks + metadata + similarity
        Orch->>LLM: prompt with retrieved context and citations
        LLM-->>Orch: grounded answer
    else Clarification needed
        Decide-->>Orch: ask_clarifying_question
    end

    Orch->>UI: formatted response
    UI->>User: render chat answer
```

## Decision Service

The decision service should run before vector retrieval. It can be implemented as a small policy engine, a lightweight classifier, or an LLM router with strict JSON output.

Recommended routing outputs:

```json
{
  "route": "retrieve | direct_answer | tool_call | clarify | reject",
  "intent": "definition | comparison | calculation | market_data | account_help | small_talk | unsafe",
  "topic": "equity",
  "index_candidates": ["investopedia", "filings", "marketplace_faq"],
  "needs_retrieval": true,
  "top_k": 5,
  "confidence": 0.87,
  "reason": "User asks for a finance concept definition that should be grounded in indexed content."
}
```

Use vector retrieval when the user asks for:

- Financial concept explanations, such as equity, ROE, ETFs, bonds, liquidity, valuation, or risk.
- Questions requiring source-backed answers from indexed content.
- Marketplace policy, FAQ, compliance, or product documentation.
- Company filings, reports, or other indexed documents.

Skip vector retrieval when the user asks for:

- Greeting, small talk, or conversational follow-up that can be answered from chat memory.
- Formatting, summarization, or rewriting of text already provided in the chat.
- Deterministic calculations where a calculator/tool is more reliable than RAG.
- Live market prices or account-specific information, which should go to tools/APIs instead of static vectors.
- Ambiguous questions where the system should ask a follow-up before retrieving.

## Core Services

```mermaid
flowchart TB
    subgraph Marketplace
        UI[Chat Widget / Marketplace UI]
        SDK[Frontend SDK<br/>streaming, auth token, citations]
    end

    subgraph Chat Backend
        API[Chat API Gateway]
        ORCH[Orchestrator]
        DECIDE[Decision Service]
        PROMPT[Prompt Builder]
        FORMAT[Response Formatter]
    end

    subgraph RAG Layer
        REG[Index Registry]
        MANIFEST[Index Manifests]
        EMBED[Embeddings Manager]
        VECTOR[FAISS / Chroma Vector Store]
        RERANK[Reranker / Relevance Filter]
    end

    subgraph Model Layer
        LLM[LLM Provider<br/>local or hosted]
        GUARD[Output Guardrails]
    end

    subgraph Operations
        OBS[Observability]
        CACHE[Response / Retrieval Cache]
        EVAL[Offline Eval Set]
    end

    UI --> SDK --> API --> ORCH
    ORCH --> DECIDE
    DECIDE --> REG
    REG --> MANIFEST
    DECIDE --> EMBED
    EMBED --> VECTOR
    VECTOR --> RERANK
    RERANK --> PROMPT
    ORCH --> PROMPT
    PROMPT --> LLM
    LLM --> GUARD
    GUARD --> FORMAT
    FORMAT --> API --> SDK --> UI

    ORCH --> CACHE
    ORCH --> OBS
    VECTOR --> OBS
    FORMAT --> OBS
    EVAL --> DECIDE
    EVAL --> PROMPT
```

## API Shape

The marketplace UI only needs to know about the chat contract. The backend hides whether the answer came from direct LLM generation, tool calls, or vector retrieval.

```http
POST /chat
Content-Type: application/json

{
  "conversation_id": "conv_123",
  "tenant_id": "marketplace_a",
  "message": "What is equity and how do I calculate it?",
  "stream": true
}
```

Example response payload:

```json
{
  "conversation_id": "conv_123",
  "route": "retrieve",
  "answer": "Equity is the residual value of an asset after liabilities are subtracted...",
  "citations": [
    {
      "source": "investopedia-www-investopedia-com-terms-e-equity-asp.txt",
      "chunk_id": 2,
      "similarity": 0.82
    }
  ],
  "disclaimer": "This is educational information, not financial advice."
}
```

## Implementation Mapping to This Repo

- `orchestrator.py`: extend into the chat orchestrator. It already scores index manifests and routes to the best index.
- `index_registry.py`: keep as the registry for available indexes and their manifests.
- `vector_store.py`: keep as the vector abstraction for FAISS or Chroma retrieval.
- `query_rag.py`: reuse prompt construction and LLM generation, but move printing concerns behind an API-friendly return object.
- `indexes/*/index_manifest.json`: add strong descriptions, tags, use cases, tenant visibility, and freshness metadata so routing is reliable.
- `indexes/*/metadata.json`: continue storing chunk source, chunk id, text, and optional chunk hashes for citation support.

## Suggested Decision Logic

```mermaid
flowchart TD
    Q[Incoming Query] --> A{Unsafe or unsupported?}
    A -->|Yes| REJECT[Reject or safe-complete]
    A -->|No| B{Needs live/account data?}
    B -->|Yes| TOOL[Call approved financial API/tool]
    B -->|No| C{Ambiguous?}
    C -->|Yes| ASK[Ask clarification]
    C -->|No| D{Needs grounded financial knowledge?}
    D -->|No| DIRECT[Direct LLM response]
    D -->|Yes| E[Route to best index]
    E --> F[Retrieve top-k chunks]
    F --> G{Strong enough similarity?}
    G -->|No| ASK2[Ask clarification or answer uncertainty]
    G -->|Yes| H[Generate grounded answer]
    TOOL --> H
    DIRECT --> OUT[Format response]
    ASK --> OUT
    ASK2 --> OUT
    REJECT --> OUT
    H --> OUT
```

## Production Notes

- Add a confidence threshold after index routing and after vector search. Low confidence should trigger clarification or a cautious answer.
- Include source metadata in the LLM prompt and response so the marketplace UI can render citations.
- Keep financial disclaimers in the response formatter, not scattered across prompts.
- Log the selected route, selected index, similarity scores, latency, and whether citations were returned.
- Add evaluation cases for queries that should not retrieve, such as greetings, calculations, live market data, and ambiguous questions.
