# text2query

A Python library that converts natural language queries to database queries using LLM. Supports SQL databases (PostgreSQL, MySQL, SQLite, SQL Server, Oracle) and NoSQL databases (MongoDB) with async execution.

## Installation

```bash
pip install "text2query[all-db,openai]"

# Or install specific database drivers
pip install "text2query[postgresql-sync,openai]"
pip install "text2query[mysql,openai]"
pip install "text2query[mongodb,openai]"
```

## Quick Start

### 1. Create Config from Connection URL

```python
from text2query.core.connections import load_config_from_url
from text2query.adapters import create_adapter

# Automatically detect database type from URL
config = load_config_from_url("postgresql://user:pass@localhost:5432/mydb?schema=public")
adapter = create_adapter(config)  # Auto-creates PostgreSQLAdapter
```

Supported URL schemes:

| Database | Schemes |
|----------|---------|
| PostgreSQL | `postgresql://`, `postgres://` |
| MySQL | `mysql://` |
| MongoDB | `mongodb://`, `mongo://` |
| SQLite | `sqlite:///` |
| SQL Server | `sqlserver://`, `mssql://` |
| Oracle | `oracle://` |

### 2. Natural Language to SQL

```python
import asyncio
from text2query import load_config_from_url, create_adapter, Text2SQL, create_llm_config

async def main():
    # 1. Setup Database Adapter
    db_url = "postgresql://user:pass@localhost:5432/mydb"
    config = load_config_from_url(db_url)
    adapter = create_adapter(config)
    
    # Optional: Get schema for better query generation
    schema_str = await adapter.get_schema_str()

    # 2. Setup LLM Configuration (using LiteLLM)
    llm_config = create_llm_config(
        model_name="gpt-4o-mini",
        apikey="your-api-key",
        provider="openai"
    )

    # 3. Initialize Text2SQL
    t2s = Text2SQL(
        llm_config=llm_config,
        db_structure=schema_str,  # Optional: improves accuracy
        # db_type="postgresql",   # Optional: defaults to "postgresql"
        # adapter=adapter,        # Optional: enables query validation
    )

    # 4. Generate SQL from natural language
    sql = await t2s.generate_query("List all users older than 30")
    print(f"Generated SQL: {sql}")

    # 5. Execute query
    result = await adapter.sql_execution(sql)
    if result["success"]:
        print(result["result"])

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Using WisbiWorkflow (Advanced)

`WisbiWorkflow` provides an advanced LangGraph-based workflow with template matching, training data retrieval, and conversation memory.

```python
import asyncio
from text2query.core.nodes.wisbiTree import WisbiWorkflow
from text2query.core.connections.factory import load_config_from_url
from text2query.core.utils.model_configs import create_llm_config

async def main():
    # 1. Database configuration
    db_config = load_config_from_url("postgresql://user:pass@localhost:5432/mydb")
    
    # 2. LLM configuration
    llm_config = create_llm_config(
        model_name="gpt-4o-mini",
        apikey="your-api-key",
        provider="openai"
    )
    
    # 3. Embedding configuration (for template/training matching)
    embed_config = create_llm_config(
        model_name="text-embedding-3-small",
        apikey="your-api-key",
        provider="openai"
    )
    
    # 4. Initialize workflow
    workflow = WisbiWorkflow(
        db_config=db_config,
        llm_config=llm_config,
        embedder_config=embed_config,
        embedding_dim=1536,  # Must match your embedding model dimension
    )
    
    # 5. Build and run the graph
    graph = await workflow.build()
    
    result = await graph.ainvoke({
        "query_text": "How many users are there?",
        "table_id": "public.users",  # Target table
    })
    
    if "generated_query" in result:
        print(f"Generated SQL: {result['generated_query']}")
    
    if "execution_result" in result:
        print(f"Result: {result['execution_result']}")

if __name__ == "__main__":
    asyncio.run(main())
```

> **Note:** WisbiWorkflow requires additional tables in PostgreSQL for templates and training data. These are auto-created on first run.

### 4. Model Provider Configuration (via LiteLLM)

`text2query` uses [LiteLLM](https://github.com/BerriAI/litellm) to support 100+ LLM providers.

#### 🌟 Automatic Environment Detection (Recommended)
If you have set up standard environment variables (like `OPENAI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_API_KEY`), you can skip `apikey` and `api_base` parameters:

```python
# Automatically picks up GOOGLE_APPLICATION_CREDENTIALS from your environment
llm_config = create_llm_config(
    model_name="gemini-pro",
    provider="vertex_ai"
)
```

### Environment Variables Reference

Below are the environment variables supported by each provider. Create a `.env` file or export them in your shell:

#### Database
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
# Or individual components
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mydb
POSTGRES_USER=user
POSTGRES_PASSWD=password
```

#### OpenAI
```bash
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com  # Optional, defaults to OpenAI
OPENAI_MODEL_NAME=gpt-4o-mini           # Optional
```

#### Azure OpenAI
```bash
# LLM
AZURE_DEPLOYMENT_NAME=your-gpt-deployment
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Embedding (can be same or different resource)
AZURE_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment
AZURE_EMBEDDING_API_BASE=https://your-resource.openai.azure.com/
AZURE_EMBEDDING_API_KEY=your-key
```

#### GCP Vertex AI
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1  # Optional, defaults to us-central1
```

#### AWS Bedrock
```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION_NAME=us-east-1
```

#### Detailed Provider Examples

#### Azure OpenAI
Multiple parameters can be passed via `extra_kwargs`:
```python
llm_config = create_llm_config(
    model_name="your-deployment-name",
    api_base="https://your-endpoint.openai.azure.com/",
    apikey="your-azure-api-key",
    provider="azure",
    extra_kwargs={
        "api_version": "2024-02-15-preview"
    }
)
```

#### GCP Vertex AI (Gemini)
You can also pass credentials directly if not using environment variables:
```python
llm_config = create_llm_config(
    model_name="gemini-pro",
    provider="vertex_ai",
    extra_kwargs={
        "vertex_credentials": "/path/to/key.json"
    }
)
```

#### AWS Bedrock (Claude)
```python
llm_config = create_llm_config(
    model_name="anthropic.claude-v2",
    provider="bedrock" # Uses AWS_ACCESS_KEY_ID from env by default
)
```

#### Ollama (Local)
```python
llm_config = create_llm_config(
    model_name="llama3",
    api_base="http://localhost:11434",
    provider="ollama"
)
```

### Parameter Mapping Reference

`create_llm_config()` uses simplified parameter names that are mapped to LiteLLM internally:

| `create_llm_config` | LiteLLM Actual | Description |
|---------------------|----------------|-------------|
| `model_name` | `model` | Auto-prefixed with provider (e.g., `azure/gpt-4`) |
| `api_base` | `api_base` | API endpoint URL |
| `apikey` | `api_key` | API key for authentication |
| `provider` | — | Used to build model prefix, not passed to LiteLLM |
| `extra_kwargs` | Spread into kwargs | Provider-specific params (e.g., `api_version`, `vertex_project`) |

### Embedding Configuration

Embeddings also use LiteLLM via `aembed_text()`. Create embedding config the same way as LLM config:

```python
from text2query.core.utils.models import aembed_text
from text2query.core.utils.model_configs import create_llm_config

# OpenAI Embedding
embed_config = create_llm_config(
    model_name="text-embedding-3-small",
    apikey="your-api-key",
    provider="openai"
)

# Azure Embedding
embed_config = create_llm_config(
    model_name="your-embedding-deployment",
    api_base="https://your-resource.openai.azure.com/",
    apikey="your-key",
    provider="azure",
    extra_kwargs={"api_version": "2024-02-15-preview"}
)

# GCP Vertex AI Embedding
embed_config = create_llm_config(
    model_name="text-embedding-004",
    provider="vertex_ai",
    extra_kwargs={
        "vertex_project": "your-project-id",
        "vertex_location": "us-central1"
    }
)

# Generate embedding
embedding = await aembed_text(embed_config, "Your text here")
print(f"Dimension: {len(embedding)}")  # e.g., 1536 for OpenAI, 768 for GCP
```

> **⚠️ Note:** Different providers produce different embedding dimensions (e.g., OpenAI: 1536, GCP: 768). Ensure your vector database column matches the embedding dimension of your chosen provider.

### 4. Generate Chart.js Visualization

```python
from text2query.llm.chart_generator import ChartGenerator
from text2query.core.utils.model_configs import create_llm_config

llm_config = create_llm_config(
    model_name="gpt-4o-mini",
    apikey="your-api-key",
    provider="openai"
)

chart_gen = ChartGenerator(llm_config=llm_config)
chart_config = await chart_gen.generate_chart(
    question="Sales by department",
    sql_command=sql,
    columns=result["columns"],
    rows=result["result"]
)
# Returns Chart.js configuration object
```

## Supported Databases

| Database | Adapter | Config | `db_type` |
|----------|---------|--------|-----------|
| PostgreSQL | `PostgreSQLAdapter` | `PostgreSQLConfig` | `"postgresql"` |
| MySQL | `MySQLAdapter` | `MySQLConfig` | `"mysql"` |
| MongoDB | `MongoDBAdapter` | `MongoDBConfig` | `"mongodb"` |
| SQLite | `SQLiteAdapter` | `SQLiteConfig` | `"sqlite"` |
| SQL Server | `SQLServerAdapter` | `SQLServerConfig` | `"sqlserver"` |
| Oracle | `OracleAdapter` | `OracleConfig` | `"oracle"` |

## API Reference

### Connection Configuration

```python
from text2query.core.connections import (
    load_config_from_url,      # Create config from URL string
    load_config_from_dict,     # Create config from dictionary
    create_connection_config,  # Create config by type name
    PostgreSQLConfig,
    MySQLConfig,
    MongoDBConfig,
    SQLiteConfig,
    SQLServerConfig,
    OracleConfig,
)
```

### Adapters

```python
from text2query.adapters import (
    create_adapter,       # Auto-create adapter from config
    PostgreSQLAdapter,
    MySQLAdapter,
    SQLiteAdapter,
    SQLServerAdapter,
    OracleAdapter,
    MongoDBAdapter,
)
```

#### Adapter Methods

```python
# Get database type identifier
db_type = adapter.db_type  # "postgresql", "mysql", "sqlite", "oracle", "sqlserver", "mongodb"

# Get schema as CREATE TABLE statements
schema_str = await adapter.get_schema_str()                      # All tables
schema_str = await adapter.get_schema_str(tables=["t1", "t2"])   # Specific tables

# Get schema as structured list
schema_list = await adapter.get_schema_list()                    # All tables
schema_list = await adapter.get_schema_list(tables=["t1", "t2"]) # Specific tables

# Execute SQL query
result = await adapter.sql_execution(sql, safe=True, limit=100)
```

### Core Classes

```python
from text2query import (
    Text2SQL,           # Natural language to SQL converter
    QueryComposer,      # Query orchestration
    Text2Query,         # Legacy class
)

from text2query.llm.chart_generator import ChartGenerator  # Chart.js generator
from text2query.llm.prompt_builder import PromptBuilder    # Prompt templates
```

## CLI Usage

```bash
text2query "List all employees" -t sql -v
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/
ruff check src/
```
