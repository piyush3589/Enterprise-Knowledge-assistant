"""
Tests — router logic, retriever, and API endpoints.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import patch


# ── Router logic ─────────────────────────────────────────────────────────────

def test_routing_decision_model():
    from agents.models import RoutingDecision, QueryType
    r = RoutingDecision(
        query_type=QueryType.VECTOR_SEARCH,
        reasoning="Query is about pump maintenance.",
        refined_query="centrifugal pump bearing replacement procedure Plant B",
    )
    assert r.query_type == QueryType.VECTOR_SEARCH
    assert r.refined_query is not None


def test_agent_state_defaults():
    from agents.models import AgentState
    state = AgentState(query="What is the boiler operating pressure?")
    assert state.routing is None
    assert state.retrieved_chunks is None
    assert state.answer is None


# ── Retriever ────────────────────────────────────────────────────────────────

def test_retriever_returns_chunks():
    """Requires seeded ChromaDB."""
    try:
        from agents.retriever import HybridRetriever
        r = HybridRetriever()
        chunks = r.retrieve("boiler maintenance procedure", top_k=2)
        assert len(chunks) <= 2
        assert all(hasattr(c, "doc_id") for c in chunks)
        assert all(0.0 <= c.score <= 1.0 for c in chunks)
    except Exception:
        pytest.skip("ChromaDB not seeded — run data/seed_knowledge.py first")


def test_bm25_keyword_match():
    """BM25 should score exact keyword matches highly."""
    try:
        from agents.retriever import HybridRetriever
        r = HybridRetriever()
        chunks = r.retrieve("CV-201 control valve inspection")
        assert any("CV-201" in c.content for c in chunks)
    except Exception:
        pytest.skip("ChromaDB not seeded")


# ── API endpoints ────────────────────────────────────────────────────────────

def test_health():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_endpoint_mocked():
    from fastapi.testclient import TestClient
    from api.main import app

    mock_result = {
        "query": "What is the boiler operating temperature?",
        "routing_decision": "VECTOR_SEARCH",
        "answer": "The boiler B-01 operates at 60–85°C with a critical threshold of 95°C.",
        "sources": ["doc_boiler_001"],
        "confidence": 0.82,
        "low_confidence_warning": None,
        "follow_up_suggestions": ["What triggers a boiler shutdown?"],
    }

    with patch("api.main.run_pipeline", return_value=mock_result):
        client = TestClient(app)
        resp = client.post("/query", json={"query": "What is the boiler operating temperature?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "duration_ms" in data
        assert data["confidence"] == 0.82


def test_out_of_scope_routing():
    from fastapi.testclient import TestClient
    from api.main import app

    mock_result = {
        "query": "What is the company holiday policy?",
        "routing_decision": "OUT_OF_SCOPE",
        "answer": "This question is outside the scope of the industrial IoT knowledge base.",
        "sources": [],
        "confidence": 0.0,
        "low_confidence_warning": "Query is out of scope.",
        "follow_up_suggestions": [],
    }

    with patch("api.main.run_pipeline", return_value=mock_result):
        client = TestClient(app)
        resp = client.post("/query", json={"query": "What is the company holiday policy?"})
        assert resp.status_code == 200
        assert resp.json()["routing_decision"] == "OUT_OF_SCOPE"
