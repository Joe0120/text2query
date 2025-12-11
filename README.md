# text2query

A Python library that converts natural language queries to database queries. Supports SQL databases (PostgreSQL, MySQL, SQLite, SQL Server, Oracle, Trino) and NoSQL databases (MongoDB) with async execution.

## Installation

```bash
# From Git URL
pip install "text2query @ git+https://your-repo-url@0.0.1"

# Local development
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import asyncio
from text2query import QueryComposer, Text2SQL
from text2query.core.connections import PostgreSQLConfig
from text2query.adapters.sql.postgresql import PostgreSQLAdapter

async def main():
    # Configure connection
    config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database_name="mydb",
        username="user",
        password="password"
    )

    # Create adapter and execute query
    adapter = PostgreSQLAdapter(config)
    result = await adapter.sql_execution("SELECT * FROM users LIMIT 10")

    if result["success"]:
        for row in result["result"]:
            print(row)

asyncio.run(main())
```

### Natural Language to SQL

```python
from llama_index.llms.openai import OpenAI
from text2query.core.t2s import Text2SQL

# Initialize LLM
llm = OpenAI(model="gpt-4", temperature=0)

# Create Text2SQL converter
t2s = Text2SQL(
    llm=llm,
    db_structure=db_structure,
    db_type="postgresql"
)

# Generate SQL from natural language
sql = await t2s.generate_query("Find all users older than 30")
```

### Cross-Database Query with Trino

```python
from text2query.core.connections import TrinoConfig
from text2query.adapters.sql.trino import TrinoAdapter

config = TrinoConfig(
    host="localhost",
    port=8080,
    username="trino",
    catalog="postgresql",
    schema="public"
)

adapter = TrinoAdapter(config)

# Query across PostgreSQL and MongoDB
result = await adapter.sql_execution("""
    SELECT e.name, e.department, s.salary
    FROM postgresql.public.employees e
    JOIN mongodb.hr.salaries s
        ON CAST(e.employee_id AS VARCHAR) = s.employee_id
    WHERE s.salary > 70000
""")
```

## Supported Databases

| Database | Direct | Via Trino | Adapter |
|----------|--------|-----------|---------|
| PostgreSQL | Yes | Yes | `PostgreSQLAdapter` |
| MySQL | Yes | Yes | `MySQLAdapter` |
| MongoDB | Yes | Yes | `MongoDBAdapter` |
| SQLite | Yes | No | `SQLiteAdapter` |
| SQL Server | Yes | Yes | `SQLServerAdapter` |
| Oracle | Yes | Yes | `OracleAdapter` |
| Trino | - | Yes | `TrinoAdapter` |

## Project Structure

```
text2query/
├── src/
│   └── text2query/
│       ├── __init__.py
│       ├── cli.py
│       ├── exceptions.py
│       ├── core/
│       │   ├── connections/    # Database configs
│       │   ├── query_composer.py
│       │   └── t2s.py          # Text-to-SQL
│       ├── adapters/
│       │   ├── sql/            # SQL adapters
│       │   └── nosql/          # NoSQL adapters
│       ├── llm/
│       └── utils/
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## API Reference

### Core Classes

- **`QueryComposer`** - Main orchestration class for managing multiple database adapters
- **`Text2SQL`** - Converts natural language to SQL using LLM
- **`BaseQueryComposer`** - Base class for all database adapters

### Connection Configs

```python
from text2query.core.connections import (
    PostgreSQLConfig,
    MySQLConfig,
    MongoDBConfig,
    SQLiteConfig,
    SQLServerConfig,
    OracleConfig,
    TrinoConfig,
)
```

### Adapters

```python
from text2query.adapters.sql import (
    PostgreSQLAdapter,
    MySQLAdapter,
    SQLiteAdapter,
    SQLServerAdapter,
    OracleAdapter,
    TrinoAdapter,
)
from text2query.adapters.nosql import MongoDBAdapter
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

