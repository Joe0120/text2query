"""
Trino connection configuration
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
from .base import BaseConnectionConfig


@dataclass
class TrinoConfig(BaseConnectionConfig):
    """Trino connection configuration
    
    Trino is a distributed SQL query engine designed to query large data sets
    distributed over one or more heterogeneous data sources.
    """
    
    host: str = "localhost"
    port: int = 8080
    username: str = "trino"
    catalog: str = "system"  # Trino catalog (e.g., postgresql, mongodb, mysql)
    schema: str = "default"  # Schema within catalog
    http_scheme: str = "http"  # http or https
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.validate():
            raise ValueError("Invalid Trino configuration")
    
    def to_connection_string(self) -> str:
        """
        Generate Trino connection string
        
        Returns:
            Trino connection URI
        """
        auth = f"{self.username}@" if self.username else ""
        return f"trino://{auth}{self.host}:{self.port}/{self.catalog}/{self.schema}"
    
    def validate(self) -> bool:
        """
        Validate Trino configuration
        
        Returns:
            True if configuration is valid
        """
        if not self.host:
            return False
        if not isinstance(self.port, int) or self.port <= 0:
            return False
        if not self.catalog:
            return False
        if self.http_scheme not in ["http", "https"]:
            return False
        return True
    
    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test Trino connection
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Test if host and port are reachable
        success, message = await self._test_host_port(self.host, self.port, self.timeout)
        
        if not success:
            return False, message
        
        # Try to establish actual Trino connection
        try:
            import trino
            
            conn = trino.dbapi.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                catalog=self.catalog,
                schema=self.schema,
                http_scheme=self.http_scheme,
            )
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] == 1:
                return True, f"Successfully connected to Trino at {self.host}:{self.port}"
            else:
                return False, "Trino connection test query failed"
                
        except ImportError:
            return False, "trino package not installed. Install with: pip install trino"
        except Exception as e:
            return False, f"Failed to connect to Trino: {str(e)}"
    
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get connection parameters for trino.dbapi.connect()
        
        Returns:
            Dictionary of connection parameters
        """
        params = {
            "host": self.host,
            "port": self.port,
            "user": self.username,
            "catalog": self.catalog,
            "schema": self.schema,
            "http_scheme": self.http_scheme,
        }
        
        # Add extra parameters
        params.update(self.extra_params)
        
        return params


