"""
text2query: A Python package for converting text to queries
"""

__version__ = "0.0.4.1"
__author__ = "Joe Lin"
__email__ = "joelin890120@gmail.com"

from .core.query_composer import QueryComposer, BaseQueryComposer
from .core.t2s import Text2SQL
from .exceptions import Text2QueryError, ConversionError

# Legacy import for backward compatibility
from .core.legacy_core import Text2Query

# Connection utilities
from .core.connections import (
    load_config_from_url,
    load_config_from_dict,
    create_connection_config,
)

# Adapter factory
from .adapters import create_adapter

__all__ = [
    # Core
    "QueryComposer",
    "BaseQueryComposer",
    "Text2SQL",
    "Text2Query",
    # Exceptions
    "Text2QueryError",
    "ConversionError",
    # Connection utilities
    "load_config_from_url",
    "load_config_from_dict",
    "create_connection_config",
    # Adapter factory
    "create_adapter",
]
