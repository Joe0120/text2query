"""Conversation Memory Store - manages conversation history storage"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import json

from ..connections.postgresql import PostgreSQLConfig
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_memory_ddl


class ConversationMemoryStore:
    """Manages conversation history in PostgreSQL
    
    This class handles storing and retrieving conversation history for ChatGPT-like
    interfaces, where each conversation (chat_id) maintains its own isolated memory.
    
    Usage:
        # Initialize instance (auto-creates tables) - call once at app startup
        store = await ConversationMemoryStore.initialize(
            postgres_config=PostgreSQLConfig(
                host="localhost",
                port=5432,
                database_name="your_db",
                username="user",
                password="pass",
            ),
            memory_schema="wisbi",
        )
    """
    
    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        memory_schema: str = "wisbi",
    ):
        """Initialize ConversationMemoryStore
        
        Note: Recommended to use ConversationMemoryStore.initialize() which
        automatically checks and creates required tables.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            memory_schema: Schema name for memory tables (default: "wisbi")
        """
        self.postgres_config = postgres_config
        self.memory_schema = memory_schema
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None
    
    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        memory_schema: str = "wisbi",
        auto_init_tables: bool = True,
    ) -> "ConversationMemoryStore":
        """Initialize ConversationMemoryStore instance and auto-setup tables
        
        This is the recommended initialization method, which automatically
        checks and creates required tables if they don't exist.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            memory_schema: Schema name for memory tables (default: "wisbi")
            auto_init_tables: Whether to auto-check and create tables (default: True)
        
        Returns:
            ConversationMemoryStore: Initialized instance
        """
        store = cls(postgres_config, memory_schema)
        
        if auto_init_tables:
            table_exists = await store.check_table_exists()
            
            if not table_exists:
                store.logger.info(f"Missing conversation_history table in schema '{memory_schema}'")
                store.logger.info("Creating memory tables...")
                
                success = await store.init_memory_tables()
                if success:
                    store.logger.info(f"Memory tables initialized successfully in schema '{memory_schema}'")
                else:
                    store.logger.warning("Failed to initialize memory tables")
            else:
                store.logger.info(f"Conversation history table already exists in schema '{memory_schema}'")
        
        return store
    
    def _get_adapter(self) -> PostgreSQLAdapter:
        """Get or create PostgreSQL adapter"""
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter
    
    # ============================================================================
    # Initialization methods
    # ============================================================================
    
    async def init_memory_tables(self) -> bool:
        """Initialize conversation history tables and indexes
        
        This method:
        1. Creates schema (if not exists)
        2. Creates conversation_history table (if not exists)
        3. Creates indexes
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            adapter = self._get_adapter()
            ddl_statements = get_memory_ddl(
                schema_name=self.memory_schema
            )
            
            for idx, ddl in enumerate(ddl_statements, 1):
                self.logger.debug(f"Executing DDL statement {idx}/{len(ddl_statements)}")
                result = await adapter.sql_execution(
                    ddl,
                    safe=False,  # DDL statements need safe=False
                    limit=None
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Failed to execute DDL statement {idx}: {error_msg}")
                    ddl_preview = ddl.strip()[:200].replace('\n', ' ')
                    self.logger.error(f"Failed DDL: {ddl_preview}...")
                    return False
            
            self.logger.info(f"Successfully set up memory tables in schema '{self.memory_schema}'")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error setting up memory tables: {e}")
            return False
    
    async def check_table_exists(self) -> bool:
        """Check if conversation_history table exists
        
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{self.memory_schema}' 
                AND table_name = 'conversation_history'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            return len(result.get("result", [])) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking table: {e}")
            return False
    
    # ============================================================================
    # Save conversation
    # ============================================================================
    
    async def save_message(
        self,
        chat_id: str,
        turn_number: int,
        user_id: str,
        group_id: str,
        user_message: str,
        assistant_response: str,
        result_summary: Optional[str] = None,
        result_count: Optional[int] = None,
        strategy_used: Optional[str] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Save a conversation message to history
        
        Args:
            chat_id: Conversation ID (UUID string)
            turn_number: Turn number within chat (1, 2, 3...)
            user_id: User ID
            group_id: Group ID
            user_message: User's natural language question
            assistant_response: Generated SQL/query
            result_summary: Summary of results
            result_count: Number of rows returned
            strategy_used: Strategy used ('template', 'trainer')
            confidence_score: Confidence score from strategy
            metadata: Additional metadata
        
        Returns:
            Optional[int]: Message ID if successful, None otherwise
        """
        try:
            adapter = self._get_adapter()
            
            md = metadata or {}
            metadata_json = json.dumps(md, ensure_ascii=False)
            
            insert_sql = f"""
                INSERT INTO {self.memory_schema}.conversation_history (
                    chat_id, turn_number, user_id, group_id,
                    user_message, assistant_response,
                    result_summary, result_count,
                    strategy_used, confidence_score,
                    metadata
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6,
                    $7, $8,
                    $9, $10,
                    $11::jsonb
                )
                RETURNING id
            """
            
            params = (
                chat_id,
                turn_number,
                user_id,
                group_id,
                user_message,
                assistant_response,
                result_summary,
                result_count,
                strategy_used,
                confidence_score,
                metadata_json,
            )
            
            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Saved conversation message: id={inserted_id}, chat_id={chat_id}, turn={turn_number}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to save message: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error saving message: {e}")
            return None
    
    # ============================================================================
    # Retrieve conversation history
    # ============================================================================
    
    async def get_chat_history(
        self,
        chat_id: str,
        user_id: str,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a chat
        
        Args:
            chat_id: Conversation ID
            user_id: User ID (for permission check)
            max_turns: Maximum number of turns to return (None = all)
        
        Returns:
            List[Dict]: List of messages ordered by turn_number
        """
        try:
            adapter = self._get_adapter()
            
            limit_clause = f"LIMIT {max_turns}" if max_turns else ""
            
            select_sql = f"""
                SELECT 
                    id, chat_id, turn_number,
                    user_id, group_id,
                    user_message, assistant_response,
                    result_summary, result_count,
                    strategy_used, confidence_score,
                    metadata, created_at
                FROM {self.memory_schema}.conversation_history
                WHERE chat_id = $1
                  AND user_id = $2
                  AND is_active = true
                ORDER BY turn_number ASC
                {limit_clause}
            """
            
            result = await adapter.sql_execution(
                select_sql, 
                params=(chat_id, user_id),
                safe=False,
                limit=None
            )
            
            if result.get("success"):
                columns = result.get("columns", [])
                rows = result.get("result", [])
                messages = [dict(zip(columns, row)) for row in rows]
                self.logger.info(f"Retrieved {len(messages)} messages for chat_id={chat_id}")
                return messages
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to get chat history: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error getting chat history: {e}")
            return []
    
    async def get_chat_context(
        self,
        chat_id: str,
        user_id: str,
        context_window: int = 10,
    ) -> str:
        """Get formatted conversation context for LLM
        
        Args:
            chat_id: Conversation ID
            user_id: User ID
            context_window: Number of recent messages to include
        
        Returns:
            str: Formatted context string
        """
        messages = await self.get_chat_history(chat_id, user_id, max_turns=context_window)
        
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            lines.append(f"User: {msg['user_message']}")
            lines.append(f"Assistant: {msg['assistant_response']}")
            if msg.get('result_summary'):
                lines.append(f"Result: {msg['result_summary']}")
            lines.append("")  # Blank line between turns
        
        return "\n".join(lines)
    
    async def get_next_turn_number(
        self,
        chat_id: str,
        user_id: str,
    ) -> int:
        """Get next turn number for a chat
        
        Args:
            chat_id: Conversation ID
            user_id: User ID
        
        Returns:
            int: Next turn number (1 if chat is new)
        """
        try:
            adapter = self._get_adapter()
            
            select_sql = f"""
                SELECT MAX(turn_number) as max_turn
                FROM {self.memory_schema}.conversation_history
                WHERE chat_id = $1
                  AND user_id = $2
                  AND is_active = true
            """
            
            result = await adapter.sql_execution(
                select_sql,
                params=(chat_id, user_id),
                safe=False,
                limit=None
            )
            
            if result.get("success") and result.get("result"):
                max_turn = result["result"][0][0]
                return (max_turn or 0) + 1
            else:
                return 1  # New chat
                
        except Exception as e:
            self.logger.exception(f"Error getting next turn number: {e}")
            return 1
    
    # ============================================================================
    # Connection management
    # ============================================================================
    
    async def close(self):
        """Close database connection
        
        In backend services, usually not needed as connection pool manages this.
        """
        if self._adapter:
            await self._adapter.close_pool()
            self.logger.debug("ConversationMemoryStore connection closed")

