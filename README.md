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
from llama_index.llms.openai import OpenAI
from text2query.core.connections import load_config_from_url
from text2query.adapters import create_adapter
from text2query.core.t2s import Text2SQL

async def main():
    # Setup
    config = load_config_from_url("postgresql://user:pass@localhost:5432/mydb")
    adapter = create_adapter(config)

    # Get database schema (all tables)
    schema_str = await adapter.get_schema_str()

    # Or filter specific tables only
    schema_str = await adapter.get_schema_str(tables=["users", "orders"])

    # Initialize LLM and Text2SQL
    llm = OpenAI(model="gpt-4o-mini", api_key="your-api-key")
    t2s = Text2SQL(
        llm=llm,
        db_structure=schema_str,
        db_type=adapter.db_type  # Auto-detect from adapter: "postgresql", "mysql", etc.
    )

    # Generate SQL from natural language
    sql = await t2s.generate_query("List all users older than 30")
    print(sql)

    # Execute query
    result = await adapter.sql_execution(sql)
    if result["success"]:
        print(result["result"])

asyncio.run(main())
```

### 3. Generate Chart.js Visualization

```python
from text2query.llm.chart_generator import ChartGenerator

chart_gen = ChartGenerator(llm=llm)
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
