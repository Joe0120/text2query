"""Utility script to populate WISBI templates and training tables.

This script will:
- Ensure the template, training, and memory schemas/tables exist
- Populate the template store from `core/utils/templates/templates.yaml` (via TemplateRepo)
- Insert a few example RAG training items (QnA / SQL examples / docs)

Adjust the connection and model configs below as needed, then run:

    python populate_db.py
"""

from __future__ import annotations

import asyncio

from .core.connections.postgresql import PostgreSQLConfig
from .core.nodes.template_node import TemplateRepo
from .core.training.store import TrainingStore
from .core.utils.model_configs import ModelConfig
from .core.utils.models import aembed_text
from .adapters.sql.postgresql import PostgreSQLAdapter


# ---------------------------------------------------------------------------
# Configuration - KEEP IN SYNC with `test.py` as needed
# ---------------------------------------------------------------------------

DB_CONFIG = PostgreSQLConfig(
    host="localhost",
    port=5432,
    database_name="t2q_test",
    username="kennethzhang",
    password="wonyoung",
    schema="wisbi",
)

# Used for generating embeddings for templates and training data
EMBEDDER_CONFIG = ModelConfig(
    base_url="http://localhost:11434",
    endpoint="/api/embed",
    api_key="",
    model_name="qwen3-embedding:0.6b",
    provider="ollama",
)

# Example logical table/user identifiers for training data
TABLE_ID = "demo_table_id"
USER_ID = ""  # empty = globally visible
GROUP_ID = ""  # empty = globally visible


# ---------------------------------------------------------------------------
# Database clearing
# ---------------------------------------------------------------------------

async def clear_database(db_config: PostgreSQLConfig, schema_name: str = "wisbi") -> None:
    """Drop all tables in the specified schema to clear the database.
    
    This function queries pg_tables to find all tables in the schema
    and drops them with CASCADE to handle any dependencies.
    """
    adapter = PostgreSQLAdapter(db_config)
    
    try:
        # Query to get all tables in the schema
        query = """
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = $1
        """
        
        result = await adapter.sql_execution(
            query,
            safe=False,  # Need safe=False to query pg_tables system catalog
            params=[schema_name],
            limit=None
        )
        
        if not result.get("success"):
            print(f"Warning: Could not query tables in schema '{schema_name}': {result.get('error', 'Unknown error')}")
            return
        
        result_rows = result.get("result", [])
        columns = result.get("columns", [])
        
        if not result_rows:
            print(f"No tables found in schema '{schema_name}' - nothing to drop.")
            return
        
        # Extract table names from results
        # Result format: result is a list of lists, columns contains column names
        table_names = []
        if columns and "tablename" in columns:
            tablename_idx = columns.index("tablename")
            table_names = [row[tablename_idx] for row in result_rows]
        else:
            # Fallback: assume first column is tablename
            table_names = [row[0] for row in result_rows if row]
        
        if not table_names:
            print(f"No tables found in schema '{schema_name}' - nothing to drop.")
            return
        
        print(f"Dropping {len(table_names)} table(s) in schema '{schema_name}': {', '.join(table_names)}")
        
        # Drop each table with CASCADE to handle dependencies
        for table_name in table_names:
            drop_query = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}" CASCADE'
            drop_result = await adapter.sql_execution(
                drop_query,
                safe=False,  # DDL statements need safe=False
                limit=None
            )
            
            if not drop_result.get("success"):
                error_msg = drop_result.get("error", "Unknown error")
                print(f"Warning: Failed to drop table '{table_name}': {error_msg}")
            else:
                print(f"  Dropped table: {schema_name}.{table_name}")
        
        print(f"Successfully cleared database schema '{schema_name}'")
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise
    finally:
        await adapter.close_pool()


# ---------------------------------------------------------------------------
# Template population
# ---------------------------------------------------------------------------

async def populate_templates(db_config: PostgreSQLConfig, embedder_config: ModelConfig) -> None:
    """Populate the template store from templates.yaml if empty.

    Uses TemplateRepo.initialize(), which will:
    - Create the template tables (if missing)
    - Auto-populate from YAML if the store is empty
    """

    template_repo = TemplateRepo(
        db_config=db_config,
        llm_config=embedder_config,  # used for embeddings inside TemplateStore
        template_schema="wisbi",
        embedding_dim=1024,
    )

    await template_repo.initialize()
    await template_repo.close()


# ---------------------------------------------------------------------------
# Training data population
# ---------------------------------------------------------------------------

async def populate_training_data(
    db_config: PostgreSQLConfig,
    embedder_config: ModelConfig,
    table_id: str,
    user_id: str,
    group_id: str,
) -> None:
    """Insert a few example RAG training items into the training store."""

    store = await TrainingStore.initialize(
        postgres_config=db_config,
        training_schema="wisbi",
        embedding_dim=1024,
    )

    # Example QnA item
    qna_question = "Show me total sales last month"
    qna_answer_sql = f'SELECT date_trunc("month", order_date) AS month, SUM(amount) AS total_sales FROM "{table_id}" WHERE order_date >= current_date - interval \"1 month\" GROUP BY 1 ORDER BY 1;'
    qna_text_for_embedding = f"{qna_question} {qna_answer_sql}"
    qna_embedding = await aembed_text(embedder_config, qna_text_for_embedding)

    await store.insert_qna(
        table_id=table_id,
        question=qna_question,
        answer_sql=qna_answer_sql,
        embedding=qna_embedding,
        user_id=user_id,
        group_id=group_id,
        metadata={"source": "populate_db.py", "kind": "example"},
    )

    # Example SQL example item
    sql_example_content = (
        f'SELECT customer_id, SUM(amount) AS total_spent FROM "{table_id}" '
        "WHERE order_date >= current_date - interval '90 days' "
        "GROUP BY customer_id ORDER BY total_spent DESC LIMIT 10;"
    )
    sql_example_embedding = await aembed_text(embedder_config, sql_example_content)

    await store.insert_sql_example(
        table_id=table_id,
        content=sql_example_content,
        embedding=sql_example_embedding,
        user_id=user_id,
        group_id=group_id,
        metadata={"source": "populate_db.py", "kind": "example"},
    )

    # Example documentation item
    doc_title = "Sales fact table documentation"
    doc_content = (
        "This table stores sales transactions. Columns include: "
        "order_id (PK), order_date (timestamp), customer_id (text), "
        "amount (numeric), currency (text)."
    )
    doc_embedding_text = f"{doc_title} {doc_content}"
    doc_embedding = await aembed_text(embedder_config, doc_embedding_text)

    await store.insert_documentation(
        table_id=table_id,
        content=doc_content,
        embedding=doc_embedding,
        title=doc_title,
        user_id=user_id,
        group_id=group_id,
        metadata={"source": "populate_db.py", "kind": "example"},
    )

    await store.close()


# ---------------------------------------------------------------------------
# Display inserted data
# ---------------------------------------------------------------------------

async def print_all_inserted_data(db_config: PostgreSQLConfig, schema_name: str = "wisbi") -> None:
    """Query and print all inserted data from all tables."""
    adapter = PostgreSQLAdapter(db_config)
    
    try:
        # Query templates table
        print("\n" + "="*80)
        print("TEMPLATES TABLE")
        print("="*80)
        templates_query = f"""
            SELECT id, template_id, description, sql_command, components_text, 
                   metadata, is_active, created_at, updated_at
            FROM "{schema_name}".templates
            ORDER BY id
        """
        templates_result = await adapter.sql_execution(
            templates_query,
            safe=False,
            limit=None
        )
        
        if templates_result.get("success"):
            columns = templates_result.get("columns", [])
            rows = templates_result.get("result", [])
            
            if rows:
                print(f"\nFound {len(rows)} template(s):\n")
                for i, row in enumerate(rows, 1):
                    print(f"Template {i}:")
                    for col, val in zip(columns, row):
                        if val is not None:
                            if col == "components_text" and len(str(val)) > 200:
                                print(f"  {col}: {str(val)[:200]}...")
                            elif col == "sql_command" and len(str(val)) > 200:
                                print(f"  {col}: {str(val)[:200]}...")
                            else:
                                print(f"  {col}: {val}")
                    print()
            else:
                print("No templates found.")
        else:
            print(f"Error querying templates: {templates_result.get('error', 'Unknown error')}")
        
        # Query qna table
        print("\n" + "="*80)
        print("QNA TABLE")
        print("="*80)
        qna_query = f"""
            SELECT id, user_id, group_id, table_id, question, answer_sql, 
                   metadata, is_active, created_at, updated_at
            FROM "{schema_name}".qna
            ORDER BY id
        """
        qna_result = await adapter.sql_execution(
            qna_query,
            safe=False,
            limit=None
        )
        
        if qna_result.get("success"):
            columns = qna_result.get("columns", [])
            rows = qna_result.get("result", [])
            
            if rows:
                print(f"\nFound {len(rows)} QnA record(s):\n")
                for i, row in enumerate(rows, 1):
                    print(f"QnA {i}:")
                    for col, val in zip(columns, row):
                        if val is not None:
                            if col == "answer_sql" and len(str(val)) > 200:
                                print(f"  {col}: {str(val)[:200]}...")
                            else:
                                print(f"  {col}: {val}")
                    print()
            else:
                print("No QnA records found.")
        else:
            print(f"Error querying qna: {qna_result.get('error', 'Unknown error')}")
        
        # Query sql_examples table
        print("\n" + "="*80)
        print("SQL_EXAMPLES TABLE")
        print("="*80)
        sql_examples_query = f"""
            SELECT id, user_id, group_id, table_id, content, 
                   metadata, is_active, created_at, updated_at
            FROM "{schema_name}".sql_examples
            ORDER BY id
        """
        sql_examples_result = await adapter.sql_execution(
            sql_examples_query,
            safe=False,
            limit=None
        )
        
        if sql_examples_result.get("success"):
            columns = sql_examples_result.get("columns", [])
            rows = sql_examples_result.get("result", [])
            
            if rows:
                print(f"\nFound {len(rows)} SQL example(s):\n")
                for i, row in enumerate(rows, 1):
                    print(f"SQL Example {i}:")
                    for col, val in zip(columns, row):
                        if val is not None:
                            if col == "content" and len(str(val)) > 200:
                                print(f"  {col}: {str(val)[:200]}...")
                            else:
                                print(f"  {col}: {val}")
                    print()
            else:
                print("No SQL examples found.")
        else:
            print(f"Error querying sql_examples: {sql_examples_result.get('error', 'Unknown error')}")
        
        # Query documentation table
        print("\n" + "="*80)
        print("DOCUMENTATION TABLE")
        print("="*80)
        documentation_query = f"""
            SELECT id, user_id, group_id, table_id, title, content, 
                   metadata, is_active, created_at, updated_at
            FROM "{schema_name}".documentation
            ORDER BY id
        """
        documentation_result = await adapter.sql_execution(
            documentation_query,
            safe=False,
            limit=None
        )
        
        if documentation_result.get("success"):
            columns = documentation_result.get("columns", [])
            rows = documentation_result.get("result", [])
            
            if rows:
                print(f"\nFound {len(rows)} documentation record(s):\n")
                for i, row in enumerate(rows, 1):
                    print(f"Documentation {i}:")
                    for col, val in zip(columns, row):
                        if val is not None:
                            if col == "content" and len(str(val)) > 200:
                                print(f"  {col}: {str(val)[:200]}...")
                            else:
                                print(f"  {col}: {val}")
                    print()
            else:
                print("No documentation records found.")
        else:
            print(f"Error querying documentation: {documentation_result.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error printing inserted data: {e}")
        raise
    finally:
        await adapter.close_pool()


async def main() -> None:
    print("Clearing database...")
    await clear_database(DB_CONFIG, DB_CONFIG.schema or "wisbi")
    
    print("Populating templates...")
    await populate_templates(DB_CONFIG, EMBEDDER_CONFIG)

    print("Populating training data...")
    await populate_training_data(DB_CONFIG, EMBEDDER_CONFIG, TABLE_ID, USER_ID, GROUP_ID)

    print("Done.")
    
    # Print all inserted data
    print("\nDisplaying all inserted data...")
    await print_all_inserted_data(DB_CONFIG, DB_CONFIG.schema or "wisbi")

    

if __name__ == "__main__":
    asyncio.run(main())


