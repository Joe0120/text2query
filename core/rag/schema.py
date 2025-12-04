"""RAG template matching tables schema definitions"""

from typing import List


def get_template_matching_ddl(
    schema_name: str = "wisbi",
    table_name: str = "nlq_slot_templates",
    embedding_dim: int = 768,
) -> List[str]:
    """Return DDL statements for creating RAG template matching tables.
    
    Args:
        schema_name: Schema name (default: "wisbi")
        table_name: Table name (default: "nlq_slot_templates")
        embedding_dim: Embedding vector dimension (default: 768)
    
    Returns:
        List[str]: List of DDL statements
    """
    return [
        # Create schema
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
        
        # Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # Template matching table: stores slot templates and corresponding embeddings
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            id bigserial PRIMARY KEY,
            slots_json jsonb NOT NULL,
            template text NOT NULL,
            model text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            template_tsvector tsvector,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            is_active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        
        # Create vector index to accelerate similarity search (using HNSW algorithm)
        f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {schema_name}.{table_name} USING hnsw (embedding vector_cosine_ops)",
        
        # Create JSONB index to accelerate slots_json queries (optional, based on query needs)
        f"CREATE INDEX IF NOT EXISTS {table_name}_slots_json_idx ON {schema_name}.{table_name} USING gin (slots_json)",
        
        # Create GIN index for full-text search (BM25)
        f"CREATE INDEX IF NOT EXISTS {table_name}_template_tsvector_idx ON {schema_name}.{table_name} USING gin (template_tsvector)",
        
        # Create trigger function to automatically update tsvector when template changes
        f"""
        CREATE OR REPLACE FUNCTION {schema_name}.{table_name}_update_tsvector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.template_tsvector := to_tsvector('english', COALESCE(NEW.template, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create trigger to call the function
        f"""
        DROP TRIGGER IF EXISTS {table_name}_tsvector_trigger ON {schema_name}.{table_name};
        CREATE TRIGGER {table_name}_tsvector_trigger
        BEFORE INSERT OR UPDATE OF template ON {schema_name}.{table_name}
        FOR EACH ROW
        EXECUTE FUNCTION {schema_name}.{table_name}_update_tsvector();
        """,
    ]

