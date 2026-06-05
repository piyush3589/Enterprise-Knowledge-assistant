"""
Pydantic models — typed inputs/outputs for every agent stage.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    VECTOR_SEARCH = "VECTOR_SEARCH"       # answerable from knowledge base
    CLARIFICATION = "CLARIFICATION"       # question too vague
    OUT_OF_SCOPE   = "OUT_OF_SCOPE"       # not in the knowledge base


class RoutingDecision(BaseModel):
    query_type: QueryType
    reasoning: str = Field(description="Why the router made this decision")
    refined_query: Optional[str] = Field(
        default=None,
        description="Rewritten query optimised for retrieval (if VECTOR_SEARCH)"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user (if CLARIFICATION)"
    )


class RetrievedChunk(BaseModel):
    doc_id: str
    content: str
    score: float
    category: str
    equipment: str
    location: str


class FinalAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    sources: list[str] = Field(description="Document IDs used to generate this answer")
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence_warning: Optional[str] = Field(
        default=None,
        description="Warning if confidence is below 0.5"
    )
    follow_up_suggestions: list[str] = Field(
        default=[],
        description="Related questions the user might want to ask"
    )


class AgentState(BaseModel):
    """State passed between all agent nodes."""
    query: str
    routing: Optional[RoutingDecision] = None
    retrieved_chunks: Optional[list[RetrievedChunk]] = None
    answer: Optional[FinalAnswer] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    query: str
    routing_decision: str
    answer: Optional[str]
    sources: list[str]
    confidence: float
    low_confidence_warning: Optional[str]
    follow_up_suggestions: list[str]
    duration_ms: int
