"""
Query composition functionality
"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio
from .connections import BaseConnectionConfig


class BaseQueryComposer(ABC):
    """Abstract base class for query composers"""
    
    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        """
        Initialize query composer with connection configuration
        
        Args:
            config: Database connection configuration
        """
        self.config = config
        if config and not config.validate():
            raise ValueError("Invalid connection configuration")
    
    def get_connection_info(self) -> Optional[str]:
        """Get connection string for this composer"""
        return self.config.to_connection_string() if self.config else None
    
    def is_connected(self) -> bool:
        """Check if composer has valid connection configuration"""
        return self.config is not None and self.config.validate()
    
    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test the database connection asynchronously
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.config:
            return False, "No connection configuration available"
        
        return await self.config.test_connection()


class QueryComposer:
    """Main query composer that delegates to specific composers"""
    
    def __init__(self):
        self._composers: Dict[str, BaseQueryComposer] = {}
    
    def register_composer(self, query_type: str, composer: BaseQueryComposer):
        """
        Register a query composer for a specific type
        
        Args:
            query_type: Database type identifier (e.g., 'postgresql', 'mysql')
            composer: Query composer instance (should already be configured)
        """
        if not composer.is_connected():
            raise ValueError(f"Composer for {query_type} must be properly configured before registration")
        
        self._composers[query_type] = composer
    
    def get_composer(self, query_type: str) -> BaseQueryComposer:
        """Get a registered query composer"""
        if query_type not in self._composers:
            raise ValueError(f"No composer registered for query type: {query_type}")
        return self._composers[query_type]
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported query types"""
        return list(self._composers.keys())
    
    async def test_all_connections(self) -> Dict[str, Tuple[bool, str]]:
        """
        Test all registered database connections asynchronously
        
        Returns:
            Dictionary mapping database type to (success, message) tuples
        """
        tasks = []
        db_types = []
        
        # Create tasks for all connection tests
        for db_type in self.get_supported_types():
            composer = self._composers[db_type]
            tasks.append(composer.test_connection())
            db_types.append(db_type)
        
        # Execute all tests concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        results = {}
        for db_type, result in zip(db_types, results_list):
            if isinstance(result, Exception):
                results[db_type] = (False, f"Test failed with exception: {str(result)}")
            else:
                results[db_type] = result
        
        return results
    
    async def get_database_info(self, query_type: str) -> Dict[str, Any]:
        """
        Get database connection information asynchronously
        
        Args:
            query_type: Database type identifier
            
        Returns:
            Dictionary containing database information
        """
        if query_type not in self._composers:
            raise ValueError(f"No composer registered for query type: {query_type}")
        
        composer = self._composers[query_type]
        
        # Test connection asynchronously
        connection_test = await composer.test_connection()
        
        return {
            "type": query_type,
            "connected": composer.is_connected(),
            "connection_string": composer.get_connection_info(),
            "connection_test": {
                "success": connection_test[0],
                "message": connection_test[1]
            },
            "config": composer.config.__dict__ if composer.config else None
        }
    
    async def get_all_database_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all registered databases asynchronously
        
        Returns:
            Dictionary mapping database type to database information
        """
        tasks = []
        db_types = []
        
        # Create tasks for all database info requests
        for db_type in self.get_supported_types():
            tasks.append(self.get_database_info(db_type))
            db_types.append(db_type)
        
        # Execute all requests concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        results = {}
        for db_type, result in zip(db_types, results_list):
            if isinstance(result, Exception):
                results[db_type] = {
                    "type": db_type,
                    "error": str(result)
                }
            else:
                results[db_type] = result
        
        return results
