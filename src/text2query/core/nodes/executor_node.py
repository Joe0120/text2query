"""
Executor Node - Generates SQL using LLM based on training/template context
"""
from __future__ import annotations
from typing import Optional
import logging
import re

from .state import WisbiState
from ..utils.models import agenerate_chat
from ...llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ExecutorNode:
    """
    Executor node that generates SQL using LLM.
    
    This node:
    1. Collects context from training results (search_results) and/or template matching (template_sql)
    2. Builds a prompt using PromptBuilder with the collected context
    3. Calls LLM to generate the final SQL query
    4. Stores the result in state["sql"]
    
    Context priority:
    - If training_score >= 0.5 and search_results available: use training context
    - If template_sql available: include as additional context/hint
    - Falls back to LLM generation with just the user query and db_structure
    """
    
    def __init__(self, db_type: str = "postgresql"):
        """
        Initialize ExecutorNode.
        
        Args:
            db_type: Database type for prompt building (postgresql, mysql, mongodb, etc.)
        """
        self.db_type = db_type
        self.prompt_builder = PromptBuilder()
    
    async def __call__(self, state: WisbiState) -> WisbiState:
        """
        Execute the node: generate SQL using LLM with context from training/template.
        
        Args:
            state: Workflow state containing:
                - query_text: User's natural language query
                - llm_config: LLM configuration
                - db_structure: Database schema structure (optional)
                - search_results: Training context from TrainerNode (optional)
                - template_sql: SQL from template matching (optional)
                - training_score: Similarity score from training search (optional)
                - template_score: Similarity score from template matching (optional)
                
        Returns:
            Updated state with:
                - sql: Generated SQL query
                - current_move: "executor"
                - execution_success: True if generation succeeded
        """
        query_text = state.get("query_text")
        if not query_text:
            logger.warning("No query_text in state, skipping SQL generation")
            state["execution_success"] = False
            state["execution_error"] = "No query text provided"
            return state
        
        llm_config = state.get("llm_config")
        if not llm_config:
            logger.error("No llm_config in state, cannot generate SQL")
            state["execution_success"] = False
            state["execution_error"] = "LLM configuration not available"
            return state
        
        # Log confidence scores
        template_score = state.get("template_score")
        training_score = state.get("training_score")
        
        logger.info("=" * 60)
        logger.info("Executor Node - Generating SQL with LLM")
        logger.info("Confidence Scores:")
        if template_score is not None:
            logger.info(f"  Template Score: {template_score:.4f}")
        else:
            logger.info("  Template Score: None (no template match)")
        
        if training_score is not None:
            logger.info(f"  Training Score: {training_score:.4f}")
        else:
            logger.info("  Training Score: None (no training results)")
        logger.info("=" * 60)
        
        # Collect context
        training_context = self._get_training_context(state)
        additional_context = self._get_template_context(state)
        db_structure = state.get("db_structure", "")
        chat_context = state.get("chat_context")
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            question=query_text,
            db_structure=db_structure,
            db_type=self.db_type,
            chat_history=chat_context,
            training_context=training_context,
            additional_context=additional_context,
        )
        
        logger.debug(f"Generated prompt (first 500 chars): {prompt[:500]}...")
        
        try:
            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = await agenerate_chat(llm_config, messages)
            
            # Clean the response
            sql = self._clean_sql_response(response)
            
            logger.info(f"Generated SQL: {sql[:200]}..." if len(sql) > 200 else f"Generated SQL: {sql}")
            
            # Update state
            state["sql"] = sql
            state["generated_query"] = sql
            state["current_move"] = "executor"
            state["execution_success"] = True
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}", exc_info=True)
            state["execution_success"] = False
            state["execution_error"] = str(e)
            
            # Fall back to template SQL if available
            template_sql = state.get("template_sql")
            if template_sql:
                logger.info("Falling back to template SQL")
                state["sql"] = template_sql
                state["generated_query"] = template_sql
                state["execution_success"] = True
                state["execution_error"] = None
        
        return state
    
    def _get_training_context(self, state: WisbiState) -> Optional[str]:
        """
        Extract training context from state if available and score is sufficient.
        
        Args:
            state: Workflow state
            
        Returns:
            Training context string or None
        """
        search_results = state.get("search_results")
        training_score = state.get("training_score")
        
        if not search_results:
            return None
        
        # Only use training context if score >= 0.5 (same threshold as routing)
        if training_score is not None and training_score < 0.5:
            logger.info(f"Training score {training_score:.4f} < 0.5, not using training context")
            return None
        
        score_str = f"{training_score:.4f}" if training_score is not None else "N/A"
        logger.info(f"Using training context (score: {score_str})")
        return search_results
    
    def _get_template_context(self, state: WisbiState) -> Optional[str]:
        """
        Extract template context as additional hint.
        
        Args:
            state: Workflow state
            
        Returns:
            Template context string or None
        """
        template_sql = state.get("template_sql")
        template = state.get("template")
        template_score = state.get("template_score")
        
        if not template_sql:
            return None
        
        context_parts = []
        
        # Add template info
        if template:
            template_id = getattr(template, 'id', None) or template.get('id', 'unknown') if isinstance(template, dict) else 'unknown'
            template_desc = getattr(template, 'description', None) or template.get('description', '') if isinstance(template, dict) else ''
            context_parts.append(f"Template ID: {template_id}")
            if template_desc:
                context_parts.append(f"Template Description: {template_desc}")
        
        # Add template SQL as reference
        score_str = f"{template_score:.4f}" if template_score is not None else "N/A"
        context_parts.append(f"Reference SQL (similarity: {score_str}):")
        context_parts.append(template_sql)
        
        score_str = f"{template_score:.4f}" if template_score is not None else "N/A"
        logger.info(f"Using template context as hint (score: {score_str})")
        return "\n".join(context_parts)
    
    def _clean_sql_response(self, response: str) -> str:
        """
        Clean LLM response to extract pure SQL.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned SQL string
        """
        sql = response.strip()
        
        # Remove markdown code blocks
        if sql.startswith("```"):
            sql = re.sub(r'^```\w*\n?', '', sql)
            sql = re.sub(r'\n?```$', '', sql)
        
        # Remove common language identifiers
        sql = sql.replace("```sql", "").replace("```SQL", "")
        sql = sql.replace("```", "")
        
        # Strip whitespace
        sql = sql.strip()
        
        return sql


async def executor_node(state: WisbiState) -> WisbiState:
    """
    Functional wrapper for ExecutorNode for use in LangGraph.
    
    This is a convenience function that creates an ExecutorNode instance
    and executes it. For more control, use the ExecutorNode class directly.
    
    Args:
        state: Workflow state
        
    Returns:
        Updated state with generated SQL
    """
    # Determine db_type from state if possible
    db_config = state.get("db_config")
    db_type = "postgresql"  # Default
    
    if db_config:
        # Try to infer db_type from config class name
        config_class = type(db_config).__name__.lower()
        if "mysql" in config_class:
            db_type = "mysql"
        elif "mongodb" in config_class or "mongo" in config_class:
            db_type = "mongodb"
        elif "sqlite" in config_class:
            db_type = "sqlite"
        elif "sqlserver" in config_class:
            db_type = "sqlserver"
        elif "oracle" in config_class:
            db_type = "oracle"
        elif "trino" in config_class:
            db_type = "trino"
    
    node = ExecutorNode(db_type=db_type)
    return await node(state)

