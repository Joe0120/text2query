"""RAG training tables schema definitions"""

from typing import List


def get_training_ddl(schema_name: str = "wisbi", embedding_dim: int = 768) -> List[str]:
    """返回創建 RAG training 表的 DDL 語句列表
    
    Args:
        schema_name: Schema 名稱（預設 "wisbi"）
        embedding_dim: Embedding 向量維度（預設 768）
    
    Returns:
        List[str]: DDL 語句列表
    """
    return [
        # 創建 schema
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}",
        
        # 啟用 pgvector 擴展
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # QnA 表：問答對訓練資料
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.qna (
            id bigserial PRIMARY KEY,
            user_id text NOT NULL,
            group_id text NOT NULL,
            table_id text NOT NULL,
            question text NOT NULL,
            answer_sql text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            is_active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        
        # SQL Examples 表：SQL 範例
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.sql_examples (
            id bigserial PRIMARY KEY,
            user_id text NOT NULL,
            group_id text NOT NULL,
            table_id text NOT NULL,
            content text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            is_active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        
        # Documentation 表：文件說明
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.documentation (
            id bigserial PRIMARY KEY,
            user_id text NOT NULL,
            group_id text NOT NULL,
            table_id text NOT NULL,
            title text,
            content text NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            is_active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        
        # 創建向量索引以加速相似度搜尋（使用 HNSW 演算法）
        f"CREATE INDEX IF NOT EXISTS qna_embedding_idx ON {schema_name}.qna USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS sql_examples_embedding_idx ON {schema_name}.sql_examples USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS documentation_embedding_idx ON {schema_name}.documentation USING hnsw (embedding vector_cosine_ops)",
        
        # 創建複合索引以加速查詢
        f"CREATE INDEX IF NOT EXISTS qna_user_group_idx ON {schema_name}.qna (user_id, group_id, is_active)",
        f"CREATE INDEX IF NOT EXISTS sql_examples_user_group_idx ON {schema_name}.sql_examples (user_id, group_id, is_active)",
        f"CREATE INDEX IF NOT EXISTS documentation_user_group_idx ON {schema_name}.documentation (user_id, group_id, is_active)",
        
    ]

