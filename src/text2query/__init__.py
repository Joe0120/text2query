"""
text2query: A Python package for converting text to queries
"""

__version__ = "0.0.1"
__author__ = "Joe Lin"
__email__ = "joelin890120@gmail.com"

from .core.query_composer import QueryComposer, BaseQueryComposer
from .core.t2s import Text2SQL
from .exceptions import Text2QueryError, ConversionError

# Legacy import for backward compatibility
from .core.legacy_core import Text2Query

__all__ = [
    "QueryComposer",
    "BaseQueryComposer",
    "Text2SQL",  # Text-to-SQL converter
    "Text2Query",  # Legacy
    "Text2QueryError",
    "ConversionError",
]
