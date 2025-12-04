"""Template matching tables schema definitions"""

from typing import List


def get_template_ddl(
    schema_name: str = "wisbi",
    embedding_dim: int = 768,
) -> List[str]:
    """Return DDL statements for creating template matching tables.
    
    Args:
        schema_name: Schema name (default: "wisbi")
        embedding_dim: Embedding vector dimension (default: 768)
    
    Returns:
        List[str]: List of DDL statements
    """
    return [
        # Create schema
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
        
        # Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # Templates table: stores templates with components and SQL
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.templates (
            id bigserial PRIMARY KEY,
            template_id text NOT NULL UNIQUE,
            description text,
            base_query text,
            dimensions jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            filters jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            metrics jsonb NOT NULL DEFAULT '[]'::jsonb,
            sql_command text,
            components_text text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            is_active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        
        # Create vector index for similarity search (using HNSW algorithm)
        f"CREATE INDEX IF NOT EXISTS templates_embedding_idx ON {schema_name}.templates USING hnsw (embedding vector_cosine_ops)",
        
        # Create indexes for faster lookups
        f"CREATE INDEX IF NOT EXISTS templates_template_id_idx ON {schema_name}.templates (template_id, is_active)",
        f"CREATE INDEX IF NOT EXISTS templates_base_query_idx ON {schema_name}.templates (base_query, is_active)",
    ]

