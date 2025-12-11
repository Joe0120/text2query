"""Conversation history tables schema definitions"""

from typing import List


def get_memory_ddl(
    schema_name: str = "wisbi",
    table_name: str = "conversation_history",
) -> List[str]:
    """Return DDL statements for creating conversation history tables.
    
    Args:
        schema_name: Schema name (default: "wisbi")
        table_name: Table name (default: "conversation_history")
    
    Returns:
        List[str]: List of DDL statements
    """
    return [
        # Create schema
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
        
        # Conversation history table
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            id bigserial PRIMARY KEY,
            
            -- Conversation grouping (shared by multiple users)
            chat_id text NOT NULL,
            turn_number integer NOT NULL,
            
            -- Sender / tenant context
            -- user_id: the sender of this turn (not the "owner" of the chat)
            -- group_id: workspace / tenant / org identifier
            user_id text NOT NULL,
            group_id text NOT NULL DEFAULT '',
            
            -- Message content
            user_message text NOT NULL,
            assistant_response text NOT NULL,
            
            -- Execution results (optional)
            result_summary text,
            result_count integer,
            
            -- Workflow metadata
            strategy_used text,
            confidence_score float,
            
            -- Metadata
            metadata jsonb DEFAULT '{{}}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            is_active boolean NOT NULL DEFAULT true
        )
        """,
        
        # Create indexes
        f"CREATE INDEX IF NOT EXISTS {table_name}_chat_idx "
        f"ON {schema_name}.{table_name} (chat_id, turn_number)",
        
        # Per-user lookup index (still useful e.g. "recent chats for this user")
        f"CREATE INDEX IF NOT EXISTS {table_name}_user_idx "
        f"ON {schema_name}.{table_name} (user_id, group_id, chat_id, created_at DESC)",
        
        f"CREATE INDEX IF NOT EXISTS {table_name}_created_at_idx "
        f"ON {schema_name}.{table_name} (created_at DESC)",
    ]