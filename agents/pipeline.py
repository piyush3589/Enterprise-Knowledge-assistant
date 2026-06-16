"""
CrewAI Multi-Agent Pipeline — three agents, three jobs:

  RouterAgent    → decides how to handle the query
  RetrieverAgent → hybrid search + reranking
  AnswerAgent    → synthesises retrieved docs into a final answer

Each agent is a pure specialist. The crew orchestrates them in sequence.
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agents.models import (
    AgentState, RoutingDecision, QueryType,
    RetrievedChunk, FinalAnswer,
)
from agents.retriever import retriever

# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ── Agent 1: Router ──────────────────────────────────────────────────────────

def router_agent(state: AgentState) -> AgentState:
    """
    Classifies the query and decides next action:
    - VECTOR_SEARCH: answerable from knowledge base
    - CLARIFICATION: query too vague, need more info
    - OUT_OF_SCOPE: not in the industrial IoT domain
    """
    print(f"[Router] Classifying query: {state.query}")

    prompt = f"""You are a routing agent for an Industrial IoT knowledge base assistant.
The knowledge base contains: equipment specifications, maintenance procedures, 
safety protocols, troubleshooting guides, and compliance documents for industrial 
equipment (boilers, pumps, pipelines, sensors).

USER QUERY: "{state.query}"

Classify this query and respond ONLY with JSON (no markdown, no backticks):
{{
  "query_type": "VECTOR_SEARCH" | "CLARIFICATION" | "OUT_OF_SCOPE",
  "reasoning": "<one sentence why>",
  "refined_query": "<rewritten query optimised for document retrieval, or null>",
  "clarification_question": "<question to ask user if CLARIFICATION, or null>"
}}

Rules:
- VECTOR_SEARCH: query is about equipment, maintenance, safety, troubleshooting, or compliance — even if vague, attempt retrieval first
- CLARIFICATION: ONLY use when the query cannot possibly be answered without more information (e.g. "fix it" with no context whatsoever)
- OUT_OF_SCOPE: completely unrelated to industrial equipment (e.g. HR policies, finance)
- When in doubt between VECTOR_SEARCH and CLARIFICATION, always choose VECTOR_SEARCH
- refined_query should expand abbreviations and add domain context for better retrieval
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().strip("```json").strip("```").strip()

    data = json.loads(raw)
    routing = RoutingDecision(**data)

    print(f"[Router] Decision: {routing.query_type} — {routing.reasoning}")
    return AgentState(**{**state.model_dump(), "routing": routing})

# ── Agent 2: Retriever ───────────────────────────────────────────────────────

def retriever_agent(state: AgentState, top_k: int = 3) -> AgentState:
    """
    Runs hybrid search (BM25 + vector) + Cohere reranking.
    Uses the refined query from the router for better results.
    """
    search_query = (
        state.routing.refined_query
        if state.routing and state.routing.refined_query
        else state.query
    )
    print(f"[Retriever] Hybrid search: {search_query}")

    chunks = retriever.retrieve(search_query, top_k=top_k)

    print(f"[Retriever] Retrieved {len(chunks)} chunks")
    for c in chunks:
        print(f"  → {c.doc_id} (score: {c.score}, category: {c.category})")

    return AgentState(**{**state.model_dump(), "retrieved_chunks": chunks})


# ── Agent 3: Answer ──────────────────────────────────────────────────────────

def answer_agent(state: AgentState) -> AgentState:
    """
    Synthesises retrieved chunks into a structured answer.
    Cites sources, flags low confidence, suggests follow-up questions.
    """
    print(f"[Answer] Generating answer from {len(state.retrieved_chunks)} chunks")

    chunks = state.retrieved_chunks
    context = "\n\n---\n\n".join([
        f"[Source: {c.doc_id} | Category: {c.category} | Score: {c.score}]\n{c.content}"
        for c in chunks
    ])
    source_ids = [c.doc_id for c in chunks]
    avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

    prompt = f"""You are an industrial IoT knowledge base assistant.
Answer the user's question using ONLY the provided context documents.
Be specific, cite equipment IDs, part numbers, and procedure steps where relevant.

USER QUESTION: {state.query}

CONTEXT DOCUMENTS:
{context}

Respond ONLY with JSON (no markdown, no backticks):
{{
  "answer": "<detailed answer using information from the documents>",
  "sources": {json.dumps(source_ids)},
  "confidence": <0.0-1.0 based on how well documents answer the question>,
  "low_confidence_warning": "<warning string if confidence < 0.5, else null>",
  "follow_up_suggestions": ["<related question 1>", "<related question 2>"]
}}

Rules:
- If the documents don't contain enough information, say so clearly in the answer
- confidence should reflect how directly the documents answer the question
- Include specific values, measurements, part numbers from the documents
- follow_up_suggestions should be natural next questions the user might have
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().strip("```json").strip("```").strip()

    data = json.loads(raw)

    # Override confidence with retrieval score if LLM overestimates
    data["confidence"] = round(min(float(data.get("confidence", avg_score)), avg_score + 0.2), 2)
    if data["confidence"] < 0.5 and not data.get("low_confidence_warning"):
        data["low_confidence_warning"] = "Answer may be incomplete — the knowledge base may not fully cover this topic."

    answer = FinalAnswer(**data)
    return AgentState(**{**state.model_dump(), "answer": answer})


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(query: str, top_k: int = 3) -> dict:
    """
    Entry point — runs the full multi-agent pipeline.
    Called by FastAPI.
    """
    state = AgentState(query=query)

    # Agent 1: Route
    state = router_agent(state)

    if state.routing.query_type == QueryType.CLARIFICATION:
        return {
            "query": query,
            "routing_decision": "CLARIFICATION",
            "answer": None,
            "clarification_needed": state.routing.clarification_question,
            "sources": [],
            "confidence": 0.0,
            "low_confidence_warning": None,
            "follow_up_suggestions": [],
        }

    if state.routing.query_type == QueryType.OUT_OF_SCOPE:
        return {
            "query": query,
            "routing_decision": "OUT_OF_SCOPE",
            "answer": "This question is outside the scope of the industrial IoT knowledge base.",
            "sources": [],
            "confidence": 0.0,
            "low_confidence_warning": "Query is out of scope.",
            "follow_up_suggestions": [],
        }

    # Agent 2: Retrieve
    state = retriever_agent(state, top_k=top_k)

    # Agent 3: Answer
    state = answer_agent(state)

    return {
        "query": query,
        "routing_decision": state.routing.query_type,
        "answer": state.answer.answer,
        "sources": state.answer.sources,
        "confidence": state.answer.confidence,
        "low_confidence_warning": state.answer.low_confidence_warning,
        "follow_up_suggestions": state.answer.follow_up_suggestions,
    }
