"""
Factory functions for creating connection configurations
"""

from typing import Dict, Any
from .base import BaseConnectionConfig
from .postgresql import PostgreSQLConfig
from .mysql import MySQLConfig
from .mongodb import MongoDBConfig
from .sqlite import SQLiteConfig
from .sqlserver import SQLServerConfig
from .oracle import OracleConfig


def create_connection_config(db_type: str, **kwargs) -> BaseConnectionConfig:
    """
    Factory function to create appropriate connection configuration
    
    Args:
        db_type: Database type (postgresql, mysql, mongodb, sqlite, sqlserver, oracle)
        **kwargs: Configuration parameters
        
    Returns:
        Appropriate connection configuration instance
    """
    config_map = {
        "postgresql": PostgreSQLConfig,
        "mysql": MySQLConfig,
        "mongodb": MongoDBConfig,
        "sqlite": SQLiteConfig,
        "sqlserver": SQLServerConfig,
        "oracle": OracleConfig,
    }
    
    if db_type not in config_map:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    config_class = config_map[db_type]
    return config_class(**kwargs)


def load_config_from_dict(config_dict: Dict[str, Any]) -> BaseConnectionConfig:
    """
    Load connection configuration from dictionary
    
    Args:
        config_dict: Dictionary containing database configuration
        
    Returns:
        Connection configuration instance
    """
    if "type" not in config_dict:
        raise ValueError("Configuration must specify 'type' field")
    
    db_type = config_dict.pop("type")
    return create_connection_config(db_type, **config_dict)


def load_config_from_url(connection_url: str) -> BaseConnectionConfig:
    """
    Load connection configuration from URL string
    
    Args:
        connection_url: Database connection URL
        
    Returns:
        Connection configuration instance
    """
    from urllib.parse import urlparse, parse_qs
    
    parsed = urlparse(connection_url)
    
    # Map URL schemes to database types
    scheme_map = {
        "postgresql": "postgresql",
        "postgres": "postgresql",
        "mysql": "mysql",
        "mongodb": "mongodb",
        "mongo": "mongodb",
        "sqlite": "sqlite",
        "sqlserver": "sqlserver",
        "mssql": "sqlserver",
        "oracle": "oracle",
    }
    
    if parsed.scheme not in scheme_map:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    
    db_type = scheme_map[parsed.scheme]
    
    # Extract basic components
    config_params = {
        "database_name": parsed.path.lstrip("/") if parsed.path else "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password
    }
    
    # Parse query parameters
    if parsed.query:
        query_params = parse_qs(parsed.query)
        for key, values in query_params.items():
            # Take the last value if multiple values for same key
            config_params[key] = values[-1] if values else None
    
    # Remove None values
    config_params = {k: v for k, v in config_params.items() if v is not None}
    
    return create_connection_config(db_type, **config_params)

