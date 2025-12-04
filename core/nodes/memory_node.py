"""Memory Node - handles conversation history retrieval and storage"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import uuid
from ..memory.store import ConversationMemoryStore
from .state import WisbiState


class MemoryNode:
    """Memory Node for retrieving and storing conversation history
    
    This node:
    1. Retrieves conversation context before query generation
    2. Saves conversation messages after execution
    
    Usage:
        memory_store = await ConversationMemoryStore.initialize(...)
        memory_node = MemoryNode(memory_store)
        
        # Retrieve context
        state = await memory_node.retrieve_context(state)
        
        # After execution, save message
        state = await memory_node.save_message(state)
    """
    
    def __init__(self, memory_store: ConversationMemoryStore):
        """Initialize Memory Node
        
        Args:
            memory_store: ConversationMemoryStore instance
        """
        self.memory_store = memory_store
    
    async def retrieve_context(self, state: WisbiState) -> WisbiState:
        """Retrieve conversation context for current chat
        
        If chat_id is not set, creates a new chat_id.
        Retrieves recent conversation history.
        
        Args:
            state: Current workflow state
        
        Returns:
            WisbiState: Updated state with conversation context
        """
        chat_id = state.get("chat_id")
        
        # Create new chat_id if none exists
        if not chat_id:
            chat_id = str(uuid.uuid4())
            state["chat_id"] = chat_id
            state["turn_number"] = 1
        else:
            # Get next turn number for existing chat
            turn_number = await self.memory_store.get_next_turn_number(
                chat_id, state["user_id"]
            )
            state["turn_number"] = turn_number
            
            # Get recent context (last N messages)
            context = await self.memory_store.get_chat_context(
                chat_id, state["user_id"], context_window=10
            )
            state["chat_context"] = context
        
        return state
    
    async def save_message(self, state: WisbiState) -> WisbiState:
        """Save current conversation message to history
        
        Args:
            state: Current workflow state
        
        Returns:
            WisbiState: Updated state
        """
        chat_id = state.get("chat_id")
        if not chat_id:
            # No chat_id, skip saving
            return state
        
        turn_number = state.get("turn_number", 1)
        user_message = state.get("query_text")
        assistant_response = state.get("generated_query", "")
        
        if not user_message:
            # No user message, skip saving
            return state
        
        # Extract execution results
        result_summary = state.get("result_summary")
        result_count = state.get("result_count")
        
        # Extract workflow metadata
        strategy_used = state.get("current_move")
        confidence_score = state.get("template_score")
        
        # Build metadata
        metadata = {
            "template_id": state.get("template", {}).get("id") if state.get("template") else None,
        }
        
        # Save message
        await self.memory_store.save_message(
            chat_id=chat_id,
            turn_number=turn_number,
            user_id=state["user_id"],
            group_id=state.get("group_id", ""),
            user_message=user_message,
            assistant_response=assistant_response,
            result_summary=result_summary,
            result_count=result_count,
            strategy_used=strategy_used,
            confidence_score=confidence_score,
            metadata=metadata,
        )
        
        return state
    
    async def __call__(self, state: WisbiState) -> WisbiState:
        """Callable interface - retrieves context by default
        
        This allows the node to be used directly in the workflow.
        For saving, call save_message() explicitly after execution.
        
        Args:
            state: Current workflow state
        
        Returns:
            WisbiState: Updated state with context
        """
        return await self.retrieve_context(state)

