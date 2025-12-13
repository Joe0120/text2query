from typing import List


def get_template_matching_ddl(
    schema_name: str = "wisbi",
    table_name: str = "nlq_slot_templates",
    embedding_dim: int = 768,
) -> List[str]:
    """Return DDL statements to create template-matching table and indexes.
    
    Args:
        schema_name: Target schema (default "wisbi")
        table_name: Table name inside the schema (default "nlq_slot_templates")
        embedding_dim: Vector dimension for pgvector column
    
    Returns:
        List[str]: ordered DDL statements
    """
    return [
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            id bigserial PRIMARY KEY,
            slots_json jsonb NOT NULL,
            template text NOT NULL,
            model text,
            embedding vector({embedding_dim}) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {schema_name}.{table_name} USING hnsw (embedding vector_cosine_ops)",
    ]


