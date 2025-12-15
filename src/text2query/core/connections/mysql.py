"""
MySQL connection configuration
"""

from typing import Tuple
from dataclasses import dataclass
import asyncio
from .base import BaseConnectionConfig


@dataclass 
class MySQLConfig(BaseConnectionConfig):
    """MySQL connection configuration"""
    
    host: str = "localhost"
    port: int = 3306
    username: str = ""
    password: str = ""
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"
    ssl_disabled: bool = False
    autocommit: bool = True
    
    def to_connection_string(self) -> str:
        """Generate MySQL connection string"""
        conn_str = f"mysql://{self.username}"
        if self.password:
            conn_str += f":{self.password}"
        conn_str += f"@{self.host}:{self.port}/{self.database_name}"
        
        # Add parameters
        params = [
            f"charset={self.charset}",
            f"collation={self.collation}",
            f"autocommit={'true' if self.autocommit else 'false'}"
        ]
        
        if self.ssl_disabled:
            params.append("ssl_disabled=true")
        
        # Add extra parameters
        for key, value in self.extra_params.items():
            params.append(f"{key}={value}")
        
        if params:
            conn_str += "?" + "&".join(params)
        
        return conn_str
    
    def validate(self) -> bool:
        """Validate MySQL configuration"""
        if not self.host or not self.database_name:
            return False
        if not 1 <= self.port <= 65535:
            return False
        return True
    
    async def test_connection(self) -> Tuple[bool, str]:
        """Test MySQL connection asynchronously"""
        # First validate configuration
        if not self.validate():
            return False, "Invalid MySQL configuration"
        
        # Test basic connectivity
        host_test, host_msg = await self._test_host_port(self.host, self.port, self.timeout)
        if not host_test:
            return False, f"MySQL connection failed: {host_msg}"
        
        # Try to import aiomysql if available for async database connection test
        try:
            import aiomysql
            
            # Build connection parameters
            conn_params = {
                'host': self.host,
                'port': self.port,
                'db': self.database_name,
                'user': self.username,
                'password': self.password,
                'charset': self.charset,
                'connect_timeout': min(self.timeout, 10),
                'autocommit': self.autocommit
            }
            
            # Remove empty values
            conn_params = {k: v for k, v in conn_params.items() if v}
            
            # Test actual database connection asynchronously
            conn = await asyncio.wait_for(
                aiomysql.connect(**conn_params), 
                timeout=self.timeout
            )
            
            cursor = await conn.cursor()
            await cursor.execute("SELECT VERSION();")
            version_row = await cursor.fetchone()
            version = version_row[0] if version_row else "Unknown"
            await cursor.close()
            conn.close()
            
            return True, f"MySQL connection successful. Server version: {version}"
            
        except ImportError:
            # aiomysql not available, try synchronous drivers in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._test_mysql_sync, 
                host_msg
            )
            return result
        
        except Exception as e:
            return False, f"MySQL database connection failed: {str(e)}"
    
    def _test_mysql_sync(self, host_msg: str) -> Tuple[bool, str]:
        """Synchronous MySQL test for fallback"""
        try:
            import mysql.connector
            
            conn_params = {
                'host': self.host,
                'port': self.port,
                'database': self.database_name,
                'user': self.username,
                'password': self.password,
                'charset': self.charset,
                'connection_timeout': min(self.timeout, 10),
                'autocommit': self.autocommit
            }
            
            # Remove empty values
            conn_params = {k: v for k, v in conn_params.items() if v}
            
            conn = mysql.connector.connect(**conn_params)
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return True, f"MySQL connection successful (via mysql.connector). Server version: {version}"
            
        except ImportError:
            try:
                import pymysql
                
                conn_params = {
                    'host': self.host,
                    'port': self.port,
                    'database': self.database_name,
                    'user': self.username,
                    'password': self.password,
                    'charset': self.charset,
                    'connect_timeout': min(self.timeout, 10),
                    'autocommit': self.autocommit
                }
                
                # Remove empty values
                conn_params = {k: v for k, v in conn_params.items() if v}
                
                conn = pymysql.connect(**conn_params)
                cursor = conn.cursor()
                cursor.execute("SELECT VERSION();")
                version = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                return True, f"MySQL connection successful (via PyMySQL). Server version: {version}"
                
            except ImportError:
                return True, f"MySQL host connectivity successful (aiomysql/mysql.connector/pymysql not available for full test): {host_msg}"
            except Exception as e:
                return False, f"MySQL database connection failed: {str(e)}"
        except Exception as e:
            return False, f"MySQL database connection failed: {str(e)}"

