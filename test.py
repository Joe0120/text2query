"""End-to-end test harness for the Wisbi LangGraph workflow.

This script exercises the real LangGraph `StateGraph` defined in
`core/nodes/wisbiTree.py`, letting `WisbiWorkflow` internally
initialize all nodes and stores.

You must provide a working PostgreSQL instance and model endpoints
before running this script. All configuration values below are
placeholders and should be filled in with real values.

Run it with:
    python test.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from .core.connections.postgresql import PostgreSQLConfig
from .core.nodes.state import WisbiState
from .core.nodes.wisbiTree import WisbiWorkflow
from .core.utils.model_configs import ModelConfig


async def run_workflow() -> WisbiState:
    """Build and execute the Wisbi LangGraph workflow once."""
    db_config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database_name="t2q_test",
        username="kennethzhang",
        password="wonyoung",
        schema="wisbi",
    )

    llm_config = ModelConfig(
        base_url="http://localhost:11434",
        endpoint="/api/chat",
        api_key="",
        model_name="gemma3:1b",
        provider="ollama",
    )

    embedder_config = ModelConfig(
        base_url="http://localhost:11434",
        endpoint="/api/embed",
        api_key="",
        model_name="qwen3-embedding:0.6b",
        provider="ollama",
    )

    initial_state: WisbiState = {
        "db_config": db_config,
        "moves_done": [],
        "current_move": None,
        "llm_config": llm_config,
        "embedder_config": embedder_config,
        "query_text": "Show me total sales last month",
        # "user_id": "demo_user",
        # "group_id": "demo_group",
        "table_id": "demo_table_id",
        "top_k": 8,
    }

    workflow = WisbiWorkflow(
        db_config=db_config,
        template_llm_config=llm_config,
    )

    graph = await workflow.build()
    final_state: Any = await graph.ainvoke(initial_state)

    return final_state


def main() -> None:
    final_state = asyncio.run(run_workflow())
    print("Final state from workflow:\n")
    for key, value in final_state.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
