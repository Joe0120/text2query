"""
Helper utility functions
"""

import re
import logging
from typing import Dict, Any, Optional, Union


def validate_text(text: str) -> bool:
    """
    Validate input text
    
    Args:
        text: Input text to validate
        
    Returns:
        True if text is valid, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    # Check if text is not empty after stripping whitespace
    if not text.strip():
        return False
    
    # Check for reasonable length (not too long)
    if len(text) > 10000:
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean and normalize input text
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove special characters that might interfere with parsing
    text = re.sub(r'[^\w\s\'"=<>%\-.,;:]', '', text)
    
    return text


def parse_config(config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
    """
    Parse configuration from various formats
    
    Args:
        config: Configuration as dict or JSON string
        
    Returns:
        Parsed configuration dictionary
    """
    if config is None:
        return {}
    
    if isinstance(config, dict):
        return config.copy()
    
    if isinstance(config, str):
        try:
            import json
            return json.loads(config)
        except json.JSONDecodeError:
            # If not valid JSON, treat as empty config
            return {}
    
    return {}


def format_query(query: str, style: str = "pretty") -> str:
    """
    Format query string for better readability
    
    Args:
        query: Query string to format
        style: Formatting style ("pretty", "compact", "minified")
        
    Returns:
        Formatted query string
    """
    if not isinstance(query, str):
        return ""
    
    if style == "minified":
        # Remove extra whitespace for minified format
        return re.sub(r'\s+', ' ', query.strip())
    
    elif style == "compact":
        # Basic formatting with minimal whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        return query
    
    elif style == "pretty":
        # Pretty format with proper indentation (basic SQL formatting)
        query = query.strip()
        
        # Add line breaks after major SQL keywords
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            query = re.sub(pattern, f'\n{keyword}', query, flags=re.IGNORECASE)
        
        # Add indentation
        lines = query.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if any(keyword in line.upper() for keyword in keywords):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f'  {line}')
        
        return '\n'.join(formatted_lines)
    
    return query


def get_logger(name: str = "text2query", level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger instance
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure logger if not already configured
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def extract_keywords(text: str) -> list[str]:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        
    Returns:
        List of extracted keywords
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time'
    }
    
    # Filter out stop words and short words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords
