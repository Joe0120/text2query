"""
Database adapters for different query types
"""

from .sql.postgresql import PostgreSQLAdapter
from .sql.mysql import MySQLAdapter
from .sql.sqlite import SQLiteAdapter
from .sql.oracle import OracleAdapter
from .sql.sqlserver import SQLServerAdapter
from .nosql.mongodb import MongoDBAdapter

from text2query.core.connections import (
    PostgreSQLConfig,
    MySQLConfig,
    SQLiteConfig,
    OracleConfig,
    SQLServerConfig,
    MongoDBConfig,
    BaseConnectionConfig,
)


def create_adapter(config: BaseConnectionConfig):
    """
    Factory function to create appropriate adapter from config.

    Args:
        config: Database connection configuration

    Returns:
        Appropriate database adapter instance

    Example:
        >>> from text2query.core.connections import load_config_from_url
        >>> from text2query.adapters import create_adapter
        >>> config = load_config_from_url("postgresql://user:pass@localhost:5432/mydb")
        >>> adapter = create_adapter(config)
    """
    adapter_map = {
        PostgreSQLConfig: PostgreSQLAdapter,
        MySQLConfig: MySQLAdapter,
        SQLiteConfig: SQLiteAdapter,
        OracleConfig: OracleAdapter,
        SQLServerConfig: SQLServerAdapter,
        MongoDBConfig: MongoDBAdapter,
    }

    for config_class, adapter_class in adapter_map.items():
        if isinstance(config, config_class):
            return adapter_class(config)

    raise ValueError(f"No adapter found for config type: {type(config).__name__}")


__all__ = [
    "PostgreSQLAdapter",
    "MySQLAdapter",
    "SQLiteAdapter",
    "OracleAdapter",
    "SQLServerAdapter",
    "MongoDBAdapter",
    "create_adapter",
]
