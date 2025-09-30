"""
MongoDB connection configuration
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import asyncio
from .base import BaseConnectionConfig


@dataclass
class MongoDBConfig(BaseConnectionConfig):
    """MongoDB connection configuration"""
    
    host: str = "localhost"
    port: int = 27017
    username: Optional[str] = None
    password: Optional[str] = None
    auth_database: str = "admin"
    replica_set: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: str = "CERT_REQUIRED"  # CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
    read_preference: str = "primary"  # primary, secondary, nearest
    max_pool_size: int = 100
    
    def to_connection_string(self) -> str:
        """Generate MongoDB connection string"""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = ""
        
        conn_str = f"mongodb://{auth_part}{self.host}:{self.port}/{self.database_name}"
        
        # Add parameters
        params = []
        
        if self.username:
            params.append(f"authSource={self.auth_database}")
        
        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")
        
        if self.ssl:
            params.append("ssl=true")
            params.append(f"ssl_cert_reqs={self.ssl_cert_reqs}")
        
        params.extend([
            f"readPreference={self.read_preference}",
            f"maxPoolSize={self.max_pool_size}"
        ])
        
        # Add extra parameters
        for key, value in self.extra_params.items():
            params.append(f"{key}={value}")
        
        if params:
            conn_str += "?" + "&".join(params)
        
        return conn_str
    
    def validate(self) -> bool:
        """Validate MongoDB configuration"""
        if not self.host or not self.database_name:
            return False
        if not 1 <= self.port <= 65535:
            return False
        if self.read_preference not in ["primary", "secondary", "nearest"]:
            return False
        if self.ssl_cert_reqs not in ["CERT_NONE", "CERT_OPTIONAL", "CERT_REQUIRED"]:
            return False
        return True
    
    async def test_connection(self) -> Tuple[bool, str]:
        """Test MongoDB connection asynchronously"""
        # First validate configuration
        if not self.validate():
            return False, "Invalid MongoDB configuration"
        
        # Test basic connectivity
        host_test, host_msg = await self._test_host_port(self.host, self.port, self.timeout)
        if not host_test:
            return False, f"MongoDB connection failed: {host_msg}"
        
        # Try to import motor if available for async MongoDB connection test
        try:
            import motor.motor_asyncio
            
            # Build connection URI
            connection_uri = self.to_connection_string()
            
            # Create async client with timeout
            client = motor.motor_asyncio.AsyncIOMotorClient(
                connection_uri,
                serverSelectionTimeoutMS=self.timeout * 1000,
                connectTimeoutMS=self.timeout * 1000
            )
            
            # Test connection by getting server info
            server_info = await asyncio.wait_for(
                client.server_info(), 
                timeout=self.timeout
            )
            
            # Test database access
            db = client[self.database_name]
            await asyncio.wait_for(
                db.command('ping'), 
                timeout=self.timeout
            )
            
            client.close()
            
            version = server_info.get('version', 'Unknown')
            return True, f"MongoDB connection successful. Server version: {version}"
            
        except ImportError:
            # motor not available, try pymongo in executor
            try:
                import pymongo
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._test_mongodb_sync, 
                    host_msg
                )
                return result
                
            except ImportError:
                # No MongoDB drivers available, fallback to basic connectivity test
                return True, f"MongoDB host connectivity successful (motor/pymongo not available for full test): {host_msg}"
        
        except Exception as e:
            return False, f"MongoDB database connection failed: {str(e)}"
    
    def _test_mongodb_sync(self, host_msg: str) -> Tuple[bool, str]:
        """Synchronous MongoDB test for fallback"""
        try:
            import pymongo
            
            connection_uri = self.to_connection_string()
            
            client = pymongo.MongoClient(
                connection_uri,
                serverSelectionTimeoutMS=self.timeout * 1000,
                connectTimeoutMS=self.timeout * 1000
            )
            
            server_info = client.server_info()
            db = client[self.database_name]
            db.command('ping')
            client.close()
            
            version = server_info.get('version', 'Unknown')
            return True, f"MongoDB connection successful (via pymongo). Server version: {version}"
            
        except Exception as e:
            return False, f"MongoDB database connection failed: {str(e)}"

