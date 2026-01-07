from __future__ import annotations
from typing import Any, Dict, Optional, Literal, List
from typing_extensions import TypedDict, NotRequired
from ..connections.postgresql import PostgreSQLConfig
from ..utils.models import ModelConfig
from ..template_matching.models import Template

available_moves = Literal["template", "trainer", "clarify", "executor"]

class WisbiState(TypedDict):
    db_config: PostgreSQLConfig

    moves_done: List[available_moves]
    current_move: Optional[available_moves]

    # LLM and Embedding
    llm_config: ModelConfig
    embedder_config: ModelConfig

    # Trainer Node
    query_text: Optional[str]
    query_embedding: Optional[List[float]]
    table_id: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    group_id: NotRequired[Optional[str]]
    top_k: NotRequired[Optional[int]]
    search_results: NotRequired[Optional[str]]
    training_score: NotRequired[Optional[float]]  # Similarity score from training search (0.0 to 1.0)

    # Template matching
    template: Optional[Template]
    template_score: Optional[float]
    template_sql: NotRequired[Optional[str]]  # SQL command corresponding to the template
    top_templates: NotRequired[Optional[List[tuple[float, Template]]]]  # Top K templates with similarity scores
    
    # Human approval and clarification
    human_approved: NotRequired[Optional[bool]]
    human_explanation: NotRequired[Optional[str]]  # Human explanation/feedback to restart workflow
    tool_results: NotRequired[Optional[str]]  # Results/risk assessment for human review
    intent: NotRequired[Optional[str]]  # Intent type (e.g., "trade", "query", etc.)
    user_profile: NotRequired[Optional[Dict[str, Any]]]  # User profile information
    clarification_requested: NotRequired[Optional[bool]]  # Flag to indicate clarification is needed
    
    # Memory/Conversation history
    chat_id: NotRequired[Optional[str]]  # Current conversation ID
    turn_number: NotRequired[Optional[int]]  # Current turn number in chat
    chat_context: NotRequired[Optional[str]]  # Formatted conversation history for LLM
    execution_success: NotRequired[Optional[bool]]  # Whether query execution succeeded
    execution_error: NotRequired[Optional[str]]  # Error message if execution failed
    result_summary: NotRequired[Optional[str]]  # Summary of execution results
    result_count: NotRequired[Optional[int]]  # Number of rows returned
    generated_query: NotRequired[Optional[str]]  # Generated SQL/query
    sql: NotRequired[Optional[str]]  # Final SQL query to execute (set by executor_node)
    
    # Database schema
    db_structure: NotRequired[Optional[str]]  # Database schema structure string for SQL generation
    db_type: NotRequired[Optional[str]]  # Database type (postgresql, mysql, mongodb, etc.)