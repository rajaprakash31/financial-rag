"""Backend-oriented chat orchestrator for marketplace RAG.

This module is intentionally small and dependency-light so it can be used from
FastAPI, Flask, a worker, or a CLI wrapper. It decides whether a message needs
retrieval, routes to the best index when needed, builds a prompt for the LLM,
and stores request/response metadata for later analysis.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from index_registry import load_all_manifests
from orchestrator import build_prompt, query_index, score_indexes


Route = Literal["retrieve", "direct_answer", "tool_call", "clarify", "reject"]


@dataclass
class Decision:
    route: Route
    intent: str
    topic: str | None = None
    index_candidates: list[str] = field(default_factory=list)
    needs_retrieval: bool = False
    top_k: int = 5
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ChatResponse:
    conversation_id: str
    tenant_id: str
    route: Route
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    selected_index: str | None = None
    latency_ms: int = 0


class InteractionStore:
    """Stores chat requests and responses for future analysis.

    SQLite is a good local starting point. In production, the same fields can
    move to Postgres, Snowflake, BigQuery, or an event stream.
    """

    def __init__(self, db_path: Path = Path("data/chat_interactions.sqlite")) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    route TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    selected_index TEXT,
                    retrieved_sources_json TEXT NOT NULL,
                    prompt TEXT,
                    answer TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_interactions_conversation
                ON chat_interactions (conversation_id, created_at)
                """
            )

    def save(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        user_message: str,
        response: ChatResponse,
        prompt: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_interactions (
                    conversation_id,
                    tenant_id,
                    user_message,
                    route,
                    intent,
                    selected_index,
                    retrieved_sources_json,
                    prompt,
                    answer,
                    latency_ms,
                    decision_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    tenant_id,
                    user_message,
                    response.route,
                    response.decision.get("intent", ""),
                    response.selected_index,
                    json.dumps(response.citations),
                    prompt,
                    response.answer,
                    response.latency_ms,
                    json.dumps(response.decision),
                ),
            )


class DecisionService:
    """First-pass router.

    Replace these heuristics later with a trained classifier or an LLM router
    that returns the same Decision JSON shape.
    """

    FINANCE_TERMS = {
        "equity",
        "stock",
        "bond",
        "etf",
        "dividend",
        "roe",
        "roi",
        "revenue",
        "liability",
        "asset",
        "valuation",
        "cash flow",
        "balance sheet",
        "income statement",
        "portfolio",
        "risk",
        "liquidity",
    }
    LIVE_DATA_TERMS = {"price", "quote", "today", "current", "latest", "market cap"}
    SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you"}

    def decide(self, message: str) -> Decision:
        normalized = message.lower().strip()

        if not normalized:
            return Decision(
                route="clarify",
                intent="empty",
                confidence=1.0,
                reason="Message is empty.",
            )

        if normalized in self.SMALL_TALK:
            return Decision(
                route="direct_answer",
                intent="small_talk",
                confidence=0.95,
                reason="Greeting or conversational message.",
            )

        if any(term in normalized for term in self.LIVE_DATA_TERMS):
            return Decision(
                route="tool_call",
                intent="market_data",
                confidence=0.8,
                reason="Question appears to require live or time-sensitive financial data.",
            )

        matched_terms = [term for term in self.FINANCE_TERMS if term in normalized]
        if matched_terms:
            return Decision(
                route="retrieve",
                intent="financial_knowledge",
                topic=matched_terms[0],
                index_candidates=["investopedia", "filings", "marketplace_faq"],
                needs_retrieval=True,
                top_k=5,
                confidence=0.85,
                reason="Question contains financial concepts that should be grounded in indexed content.",
            )

        if len(normalized.split()) <= 2:
            return Decision(
                route="clarify",
                intent="ambiguous",
                confidence=0.7,
                reason="Message is too short to route confidently.",
            )

        return Decision(
            route="direct_answer",
            intent="general",
            confidence=0.65,
            reason="No retrieval or tool requirement detected.",
        )


class MarketplaceChatOrchestrator:
    def __init__(
        self,
        *,
        index_root: Path = Path("indexes"),
        router_model: str = "all-MiniLM-L6-v2",
        store: InteractionStore | None = None,
        decision_service: DecisionService | None = None,
        min_similarity: float = 0.45,
    ) -> None:
        self.index_root = index_root
        self.router_model = router_model
        self.store = store or InteractionStore()
        self.decision_service = decision_service or DecisionService()
        self.min_similarity = min_similarity

    def handle_message(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        message: str,
        llm_model_path: str | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        prompt: str | None = None
        decision = self.decision_service.decide(message)

        if decision.route == "retrieve":
            response, prompt = self._handle_retrieval(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                message=message,
                decision=decision,
                llm_model_path=llm_model_path,
            )
        elif decision.route == "tool_call":
            response = self._tool_call_response(conversation_id, tenant_id, decision)
        elif decision.route == "clarify":
            response = ChatResponse(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                route="clarify",
                answer="Can you share a little more detail about what financial topic or company you mean?",
                decision=asdict(decision),
            )
        elif decision.route == "reject":
            response = ChatResponse(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                route="reject",
                answer="I cannot help with that request.",
                decision=asdict(decision),
            )
        else:
            response = self._direct_answer_response(conversation_id, tenant_id, decision)

        response.latency_ms = int((time.perf_counter() - started) * 1000)
        self.store.save(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_message=message,
            response=response,
            prompt=prompt,
        )
        return response

    def _handle_retrieval(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        message: str,
        decision: Decision,
        llm_model_path: str | None,
    ) -> tuple[ChatResponse, str | None]:
        manifests = load_all_manifests(self.index_root)
        if not manifests:
            fallback = ChatResponse(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                route="clarify",
                answer="I do not have a searchable finance index available yet. Build an index first, then try again.",
                decision=asdict(decision),
            )
            return fallback, None

        best_manifest, _score = score_indexes(message, manifests, self.router_model)[0]
        retrieved = query_index(message, best_manifest, top_k=decision.top_k)
        citations = [
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "similarity": item["similarity"],
            }
            for item in retrieved
        ]

        if not retrieved or retrieved[0]["similarity"] < self.min_similarity:
            response = ChatResponse(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                route="clarify",
                answer="I found weak matches in the finance index. Can you narrow the question or name the topic more directly?",
                citations=citations,
                decision=asdict(decision),
                selected_index=best_manifest.get("name"),
            )
            return response, None

        passages = [item["text"] for item in retrieved]
        prompt = build_prompt(message, passages)
        answer = self._generate_answer(prompt, llm_model_path, retrieved)
        response = ChatResponse(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            route="retrieve",
            answer=answer,
            citations=citations,
            decision=asdict(decision),
            selected_index=best_manifest.get("name"),
        )
        return response, prompt

    def _generate_answer(
        self,
        prompt: str,
        llm_model_path: str | None,
        retrieved: list[dict[str, Any]],
    ) -> str:
        if llm_model_path:
            from query_rag import generate_with_llm

            return generate_with_llm(llm_model_path, prompt)

        top = retrieved[0]
        return (
            "Retrieval succeeded, but no LLM model path was provided. "
            "Send the generated prompt to your hosted or local LLM to format the final answer. "
            f"Top source: {top['source']} chunk {top['chunk_id']}."
        )

    def _tool_call_response(
        self,
        conversation_id: str,
        tenant_id: str,
        decision: Decision,
    ) -> ChatResponse:
        return ChatResponse(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            route="tool_call",
            answer=(
                "This question appears to need live financial data. Route it to an approved market-data "
                "or account-data service, then pass the tool result to the LLM for formatting."
            ),
            decision=asdict(decision),
        )

    def _direct_answer_response(
        self,
        conversation_id: str,
        tenant_id: str,
        decision: Decision,
    ) -> ChatResponse:
        return ChatResponse(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            route="direct_answer",
            answer="I can help with financial concepts, indexed documents, or marketplace questions.",
            decision=asdict(decision),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the marketplace chat orchestrator once.")
    parser.add_argument("--message", required=True)
    parser.add_argument("--conversation-id", default="local_demo")
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--index-root", default="indexes")
    parser.add_argument("--llm-model", default=None)
    args = parser.parse_args()

    orchestrator = MarketplaceChatOrchestrator(index_root=Path(args.index_root))
    response = orchestrator.handle_message(
        conversation_id=args.conversation_id,
        tenant_id=args.tenant_id,
        message=args.message,
        llm_model_path=args.llm_model,
    )
    print(json.dumps(asdict(response), indent=2))


if __name__ == "__main__":
    main()
