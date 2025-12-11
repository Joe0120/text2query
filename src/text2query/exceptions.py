"""
Custom exceptions for text2query package
"""


class Text2QueryError(Exception):
    """Base exception for text2query package"""
    pass


class ConversionError(Text2QueryError):
    """Exception raised when text conversion fails"""
    pass


class ConfigurationError(Text2QueryError):
    """Exception raised when configuration is invalid"""
    pass
