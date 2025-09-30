"""
Core functionality for text2query package
"""

from typing import Dict, List, Optional, Any


class Text2Query:
    """
    Main class for converting text to queries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Text2Query converter
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
