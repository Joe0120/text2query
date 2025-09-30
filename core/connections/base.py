"""
Base connection configuration class
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import socket
import asyncio


@dataclass
class BaseConnectionConfig(ABC):
    """Base class for database connection configurations"""
    
    database_name: str
    timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def to_connection_string(self) -> str:
        """Generate connection string from configuration"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the configuration"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test the database connection
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        pass
    
    async def _test_host_port(self, host: str, port: int, timeout: int = 5) -> Tuple[bool, str]:
        """
        Test if host and port are reachable asynchronously
        
        Args:
            host: Host address
            port: Port number
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Use asyncio to test connection
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            
            # Close the connection
            writer.close()
            await writer.wait_closed()
            
            return True, f"Successfully connected to {host}:{port}"
            
        except asyncio.TimeoutError:
            return False, f"Connection to {host}:{port} timed out after {timeout} seconds"
        except ConnectionRefusedError:
            return False, f"Cannot connect to {host}:{port} - Connection refused"
        except socket.gaierror as e:
            return False, f"DNS resolution failed for {host}: {str(e)}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

