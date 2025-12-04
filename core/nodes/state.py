from __future__ import annotations
from typing import Any, Dict, Optional, Literal, List
from typing_extensions import TypedDict, NotRequired
from ..connections.postgresql import PostgreSQLConfig
from ..utils.models import ModelConfig
from ..template_matching.template_repo import Template

available_moves = Literal["rag", "template", "trainer", "clarify", "executor"]

class WisbiState(TypedDict):
    db_config: PostgreSQLConfig

    moves_done: List[available_moves]
    current_move: Optional[available_moves]

    # LLM and EMbedding
    llm_config: ModelConfig
    embedder_config: ModelConfig

    # Trainer Node and RAG
    query_text: Optional[str]
    query_embedding: Optional[List[float]]
    
    # RAG Node
    rag_confidence: NotRequired[Optional[float]]
    rag_results: NotRequired[Optional[str]]
    
    # Trainer Node
    table_id: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    group_id: NotRequired[Optional[str]]
    top_k: NotRequired[Optional[int]]
    search_results: NotRequired[Optional[str]]

    # Template matching
    template: Optional[Template]
    template_score: Optional[float]
    
    # Human approval and clarification
    human_approved: NotRequired[Optional[bool]]
    human_explanation: NotRequired[Optional[str]]  # Human explanation/feedback to restart workflow
    tool_results: NotRequired[Optional[str]]  # Results/risk assessment for human review
    intent: NotRequired[Optional[str]]  # Intent type (e.g., "trade", "query", etc.)
    user_profile: NotRequired[Optional[Dict[str, Any]]]  # User profile information
    clarification_requested: NotRequired[Optional[bool]]  # Flag to indicate clarification is needed