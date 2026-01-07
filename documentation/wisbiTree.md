# WisbiTree - WISBI Workflow Documentation

## Overview

**WisbiTree** (`wisbiTree.py`) implements the core WISBI (Workflow for Intelligent SQL/Business Intelligence) workflow engine for text2query. It orchestrates the process of converting natural language queries into SQL using a combination of **RAG-based training retrieval**, **template matching**, and **LLM-powered query generation**.

The workflow is built on **LangGraph's StateGraph** architecture, providing a flexible, state-machine-based approach to natural language to SQL translation.

---

## Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              WISBI Workflow                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────┐                                                    │
│   │ memory_retrieve_node│ ─────────────────────┐                             │
│   └─────────────────────┘                      │                             │
│                                                ▼                             │
│                                    ┌──────────────────────┐                  │
│                                    │   orchestrator_node  │                  │
│                                    └──────────────────────┘                  │
│                                               │                              │
│                          ┌────────────────────┼────────────────────┐         │
│                          │                    │                    │         │
│                          ▼                    ▼                    ▼         │
│              ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│              │  trainer_node   │  │  template_node  │  │  clarify_node   │   │
│              │  (RAG Search)   │  │  (Template RAG) │  │  (Placeholder)  │   │
│              └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                       │                    │                    │            │
│                       ▼                    │                    │            │
│         ┌──────────────────────┐           │                    │            │
│         │ score >= 0.5?        │           │                    │            │
│         │ Yes → executor_node  │───────────┼────────────────────┘            │
│         │ No  → template_node ─┼───────────┘                                 │
│         └──────────────────────┘                                             │
│                                                                              │
│                                    ┌──────────────────────┐                  │
│                                    │   executor_node      │                  │
│                                    │  (LLM SQL Gen)       │                  │
│                                    └──────────────────────┘                  │
│                                               │                              │
│                                               ▼                              │
│                                    ┌──────────────────────┐                  │
│                                    │  memory_save_node    │                  │
│                                    └──────────────────────┘                  │
│                                               │                              │
│                                               ▼                              │
│                               ┌───────────────────────────────┐              │
│                               │     need_human_approval?      │              │
│                               │  Yes → human_explanation_node │              │
│                               │  No  → END                    │              │
│                               └───────────────────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Flow Description

1. **Entry Point**: `memory_retrieve_node` - Retrieves conversation context (currently a passthrough stub)

2. **Orchestrator**: `orchestrator_node` - Initializes and tracks workflow moves, defaults to `trainer` first

3. **Training Path** (Priority): 
   - `trainer_node` performs RAG search against training data (QnA pairs, SQL examples, documentation)
   - If `training_score >= 0.5` → routes directly to `executor_node`
   - If `training_score < 0.5` → falls back to `template_node`

4. **Template Path** (Fallback):
   - `template_node` uses LLM to extract query components
   - Performs vector similarity search against stored templates
   - Always routes to `executor_node`

5. **Execution**: `executor_node` - Generates final SQL using LLM with collected context

6. **Memory**: `memory_save_node` - Saves conversation (currently a passthrough stub)

7. **Human Review** (Optional): 
   - `need_human_approval` checks if approval is required (e.g., trade operations)
   - `human_explanation_node` handles user interaction if needed

---

## WisbiState

The workflow uses a TypedDict-based state object that flows through all nodes:

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `db_config` | `PostgreSQLConfig` | Database connection configuration |
| `llm_config` | `ModelConfig` | LLM API configuration |
| `embedder_config` | `ModelConfig` | Embedding model configuration |
| `query_text` | `Optional[str]` | User's natural language query |
| `moves_done` | `List[available_moves]` | History of workflow moves |
| `current_move` | `Optional[available_moves]` | Current active move |

### Training Node Fields

| Field | Type | Description |
|-------|------|-------------|
| `query_embedding` | `Optional[List[float]]` | Embedding vector of the query |
| `table_id` | `Optional[str]` | Table identifier for scoped search |
| `user_id` | `Optional[str]` | User ID for personalized results |
| `group_id` | `Optional[str]` | Group ID for group-scoped results |
| `top_k` | `Optional[int]` | Number of results to retrieve (default: 8) |
| `search_results` | `Optional[str]` | Formatted training search results |
| `training_score` | `Optional[float]` | Best similarity score (0.0 to 1.0) |

### Template Matching Fields

| Field | Type | Description |
|-------|------|-------------|
| `template` | `Optional[Template]` | Best matching template |
| `template_score` | `Optional[float]` | Template similarity score |
| `template_sql` | `Optional[str]` | SQL from matched template |
| `top_templates` | `Optional[List[tuple]]` | Top K templates with scores |

### Execution Fields

| Field | Type | Description |
|-------|------|-------------|
| `sql` | `Optional[str]` | Final generated SQL query |
| `generated_query` | `Optional[str]` | Same as `sql` |
| `execution_success` | `Optional[bool]` | Whether generation succeeded |
| `execution_error` | `Optional[str]` | Error message if failed |
| `db_structure` | `Optional[str]` | Database schema for LLM context |
| `db_type` | `Optional[str]` | Database type (postgresql, mysql, etc.) |

### Human Approval Fields

| Field | Type | Description |
|-------|------|-------------|
| `human_approved` | `Optional[bool]` | Approval status |
| `human_explanation` | `Optional[str]` | User feedback for re-processing |
| `tool_results` | `Optional[str]` | Risk assessment for review |
| `intent` | `Optional[str]` | Query intent (e.g., "trade") |
| `clarification_requested` | `Optional[bool]` | Flag for clarification needed |

### Memory/Conversation Fields

| Field | Type | Description |
|-------|------|-------------|
| `chat_id` | `Optional[str]` | Conversation identifier |
| `turn_number` | `Optional[int]` | Current turn in conversation |
| `chat_context` | `Optional[str]` | Formatted conversation history |
| `result_summary` | `Optional[str]` | Summary of execution results |
| `result_count` | `Optional[int]` | Number of rows returned |

---

## Configuration

### WisbiWorkflow Initialization

The `WisbiWorkflow` class supports multiple initialization patterns:

#### Option 1: Config Objects (Recommended)

```python
from text2query.core.nodes.wisbiTree import WisbiWorkflow
from text2query.core.connections.postgresql import PostgreSQLConfig
from text2query.core.utils.model_configs import create_model_config

# Database configuration
db_config = PostgreSQLConfig(
    host="localhost",
    port=5432,
    database_name="mydb",
    username="user",
    password="password",
    schema="public"
)

# LLM configuration
llm_config = create_model_config(
    model_name="gpt-4",
    api_base="https://api.openai.com",
    apikey="sk-...",
    provider="openai"
)

# Embedder configuration
embedder_config = create_model_config(
    model_name="text-embedding-ada-002",
    api_base="https://api.openai.com",
    apikey="sk-...",
    provider="openai"
)

workflow = WisbiWorkflow(
    db_config=db_config,
    llm_config=llm_config,
    embedder_config=embedder_config,
)
```

#### Option 2: Individual Parameters

```python
workflow = WisbiWorkflow(
    db_config=db_config,
    # LLM parameters
    llm_name="gpt-4",
    llm_api_base="https://api.openai.com",
    llm_api_key="sk-...",
    llm_provider="openai",
    llm_temperature=0.2,
    llm_timeout=30,
    # Embedder parameters
    embed_model_name="text-embedding-ada-002",
    embed_api_base="https://api.openai.com",
    embed_api_key="sk-...",
    embed_provider="openai",
)
```

#### Option 3: Environment Variables

Set environment variables and omit `db_config`:

```bash
# Full connection string (highest priority)
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Or individual variables
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="mydb"
export DB_USER="user"
export DB_PASSWORD="password"
export DB_SCHEMA="public"
```

```python
# db_config will be loaded from environment
workflow = WisbiWorkflow(
    llm_config=llm_config,
    embedder_config=embedder_config,
)
```

### Supported Environment Variables

| Variable | Alternative | Description |
|----------|-------------|-------------|
| `DATABASE_URL` | `POSTGRES_CONNECTION_STRING` | Full connection string |
| `DB_HOST` | `POSTGRES_HOST` | Database host |
| `DB_PORT` | `POSTGRES_PORT` | Database port |
| `DB_NAME` | `POSTGRES_DB`, `POSTGRES_DATABASE` | Database name |
| `DB_USER` | `POSTGRES_USER` | Username |
| `DB_PASSWORD` | `POSTGRES_PASSWORD` | Password |
| `DB_SCHEMA` | `POSTGRES_SCHEMA` | Schema (default: "public") |
| `DB_SSL_MODE` | `POSTGRES_SSL_MODE` | SSL mode (default: "disable") |

### Workflow Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `template_schema` | `"wisbi"` | PostgreSQL schema for templates table |
| `training_schema` | `"wisbi"` | PostgreSQL schema for training tables |
| `memory_schema` | `"wisbi"` | PostgreSQL schema for conversation history |
| `embedding_dim` | `768` | Embedding vector dimension |

---

## Nodes Reference

### orchestrator_node

**Purpose**: Initializes workflow state and tracks execution history.

**Behavior**:
- Initializes `moves_done` list if not present
- Sets `current_move` to `"trainer"` if not set (training has priority)
- Tracks moves in history to prevent loops

### trainer_node (TrainerNode)

**Purpose**: RAG-based retrieval of training data (QnA pairs, SQL examples, documentation).

**Input Requirements**:
- `query_text` or `query_embedding` in state
- `table_id` (required) - scope for training data search
- `embedder_config` for embedding generation

**Output**:
- `search_results`: Formatted training context
- `training_score`: Best similarity score (0.0 - 1.0)
- `query_embedding`: Cached embedding for reuse

**Training Data Format**:
```
[QnA]
Q: What is the total revenue?
A: SELECT SUM(revenue) FROM sales;

[SQL EXAMPLE]
SELECT product_name, COUNT(*) FROM orders GROUP BY product_name;

[DOC] Revenue Calculation
Revenue is calculated as the sum of all sales...
```

### template_node (TemplateNode)

**Purpose**: LLM-powered component extraction and template matching.

**Process**:
1. Uses LLM to extract query components (base_query, dimensions, filters)
2. Searches template store using vector similarity
3. Returns top K matching templates with SQL

**Component Extraction Schema**:
```yaml
components:
  base_query: timeseries|snapshot|comparison|ranking|null
  dimensions:
    time: day|month|quarter|year|null
    org: single|multiple|all|null
    region: <string>|null
    industry: <string>|null
  filters:
    standard: {date_range: bool, org_filter: bool}
    business: {doc_type: bool, threshold: bool}
```

**Output**:
- `template`: Best matching Template object
- `template_score`: Similarity score
- `template_sql`: SQL from matched template
- `top_templates`: List of (score, template) tuples

### executor_node (ExecutorNode)

**Purpose**: Generates final SQL using LLM with context from training/template.

**Context Priority**:
1. Training context (if `training_score >= 0.5`)
2. Template SQL as additional hint
3. Database structure (`db_structure`)
4. Conversation history (`chat_context`)

**Output**:
- `sql`: Generated SQL query
- `generated_query`: Same as `sql`
- `execution_success`: Boolean success flag
- `execution_error`: Error message if failed

**Fallback**: If LLM generation fails, falls back to `template_sql` if available.

### human_explanation_node

**Purpose**: Handles human-in-the-loop interactions.

**Two Modes**:
1. **Clarification Mode** (`clarification_requested=True`):
   - Prompts user for explanation/feedback
   - Resets workflow to restart with enriched query

2. **Approval Mode** (default):
   - Displays risk assessment
   - Prompts for approval (y/n)
   - Sets `human_approved` flag

---

## Routing Logic

### route_by_current_move

Routes from orchestrator based on `current_move`:
- `"trainer"` → `trainer_node`
- `"template"` → `template_node`  
- `"clarify"` → `clarify_node`
- `"executor"` → `executor_node`
- Default → `trainer_node`

### reroute_based_on_confidence

Routes after `trainer_node` based on training results:

```
if no search_results:
    → template_node (fallback)
    
elif training_score is None but results exist:
    → executor_node (use training context)
    
elif training_score < 0.5:
    → template_node (low confidence fallback)
    
elif training_score >= 0.5:
    → executor_node (high confidence, use training)
```

**Threshold**: `0.5` similarity score determines training confidence.

### need_human_approval

Determines if human approval is required:

```
if human_approved is True/False:
    → END (already handled)
    
elif intent == "trade":
    → human_approval (high-risk operation)
    
elif tool_results exists:
    → human_approval (risk assessment present)
    
else:
    → END (no approval needed)
```

---

## Usage Examples

### Basic Usage

```python
import asyncio
from text2query.core.nodes.wisbiTree import WisbiWorkflow
from text2query.core.connections.postgresql import PostgreSQLConfig
from text2query.core.utils.model_configs import create_model_config

async def main():
    # Configure
    db_config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database_name="analytics",
        username="admin",
        password="secret"
    )
    
    llm_config = create_model_config(
        model_name="gpt-4",
        api_base="https://api.openai.com",
        apikey="sk-..."
    )
    
    embedder_config = create_model_config(
        model_name="text-embedding-ada-002",
        api_base="https://api.openai.com",
        apikey="sk-..."
    )
    
    # Create workflow
    workflow = WisbiWorkflow(
        db_config=db_config,
        llm_config=llm_config,
        embedder_config=embedder_config,
    )
    
    # Build graph (async)
    graph = await workflow.build()
    
    # Prepare initial state
    initial_state = {
        "db_config": db_config,
        "llm_config": llm_config,
        "embedder_config": embedder_config,
        "query_text": "Show me the total revenue by month for 2024",
        "table_id": "sales_data",
        "db_structure": """
            Table: sales
            Columns: id, product_id, revenue, sale_date
            
            Table: products
            Columns: id, name, category
        """,
    }
    
    # Execute workflow
    result = await graph.ainvoke(initial_state)
    
    # Get generated SQL
    print(f"Generated SQL: {result['sql']}")
    print(f"Training Score: {result.get('training_score')}")
    print(f"Template Score: {result.get('template_score')}")

asyncio.run(main())
```

### Using with Ollama (Local LLM)

```python
llm_config = create_model_config(
    model_name="llama2",
    api_base="http://localhost:11434",
    apikey="",  # Empty for Ollama
    provider="ollama"
)

embedder_config = create_model_config(
    model_name="nomic-embed-text",
    api_base="http://localhost:11434",
    apikey="",
    provider="ollama"
)
```

### Accessing the Compiled Graph

```python
# Build once
graph = await workflow.build()

# Subsequent access via property
same_graph = workflow.graph  # Raises error if not built

# Re-building returns cached graph
graph2 = await workflow.build()  # Returns same instance
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: db_config is required` | No database config provided | Set env vars or pass `db_config` |
| `ValueError: llm_config required` | Missing LLM configuration | Provide `llm_config` or individual params |
| `ValueError: embedder_config required` | Missing embedder config | Provide `embedder_config` or individual params |
| `RuntimeError: Graph not built` | Accessing `workflow.graph` before `build()` | Call `await workflow.build()` first |
| `ValueError: table_id is required` | TrainerNode missing scope | Set `table_id` in initial state |

### Logging

The workflow uses Python's `logging` module. Enable debug logging for detailed execution traces:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("text2query").setLevel(logging.DEBUG)
```

---

## API Reference

### WisbiWorkflow

```python
class WisbiWorkflow:
    def __init__(
        self,
        db_config: Optional[PostgreSQLConfig] = None,
        llm_config: Optional[ModelConfig] = None,
        embedder_config: Optional[ModelConfig] = None,
        *,
        # LLM individual params
        llm_name: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        llm_provider: Provider = "openai",
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = 0.2,
        llm_timeout: int = 30,
        # Embedder individual params
        embed_model_name: Optional[str] = None,
        embed_api_base: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_endpoint: Optional[str] = None,
        embed_provider: Provider = "openai",
        embed_max_tokens: Optional[int] = None,
        embed_temperature: Optional[float] = 0.2,
        embed_timeout: int = 30,
        # Workflow config
        template_schema: str = "wisbi",
        training_schema: str = "wisbi",
        memory_schema: str = "wisbi",
        embedding_dim: int = 768,
    ): ...
    
    async def build_nodes(self) -> None:
        """Build and initialize internal nodes (TemplateNode, TrainerNode)."""
    
    async def build(self) -> CompiledStateGraph:
        """Build and compile the workflow graph. Returns cached graph on subsequent calls."""
    
    @property
    def graph(self) -> CompiledStateGraph:
        """Access compiled graph. Raises RuntimeError if not built."""
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    base_url: str           # API base URL
    endpoint: str           # API endpoint path
    api_key: str            # Authentication key
    model_name: str         # Model identifier
    provider: Provider = "openai"  # "openai" or "ollama"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    timeout: int = 30
```

### create_model_config

```python
def create_model_config(
    model_name: str,
    api_base: str,
    apikey: str,
    endpoint: Optional[str] = None,  # Auto-set based on provider
    provider: Provider = "openai",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.2,
    timeout: int = 30,
) -> ModelConfig: ...
```

---

## Dependencies

- **LangGraph**: StateGraph workflow engine
- **PostgreSQL**: Backend for templates, training data, and memory
- **pgvector**: Vector similarity search extension
- **aiohttp/httpx**: Async HTTP for LLM calls
- **PyYAML**: Template loading

---

## See Also

- [`state.py`](../src/text2query/core/nodes/state.py) - WisbiState definition
- [`orchestrator.py`](../src/text2query/core/nodes/orchestrator.py) - Routing logic
- [`template_node.py`](../src/text2query/core/nodes/template_node.py) - Template matching
- [`trainer_node.py`](../src/text2query/core/nodes/trainer_node.py) - Training RAG
- [`executor_node.py`](../src/text2query/core/nodes/executor_node.py) - SQL generation
- [`human_approval.py`](../src/text2query/core/nodes/human_approval.py) - Human-in-the-loop

