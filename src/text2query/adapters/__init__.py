"""
Database adapters for different query types
"""

from .sql.postgresql import PostgreSQLAdapter
from .sql.mysql import MySQLAdapter
from .nosql.mongodb import MongoDBAdapter

__all__ = ["PostgreSQLAdapter", "MySQLAdapter", "MongoDBAdapter"]
