"""
PostgreSQL connection configuration
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import asyncio
from .base import BaseConnectionConfig


@dataclass
class PostgreSQLConfig(BaseConnectionConfig):
    """PostgreSQL connection configuration"""
    
    # Optional DSN/connection string. When provided, it will be used directly.
    connection_string: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    schema: Optional[str] = "public"
    ssl_mode: str = "disable"  # disable, allow, prefer, require, verify-ca, verify-full
    
    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        # Prefer explicitly provided DSN/connection string if available
        if self.connection_string:
            return self.connection_string
        conn_str = f"postgresql://{self.username}"
        if self.password:
            conn_str += f":{self.password}"
        conn_str += f"@{self.host}:{self.port}/{self.database_name}"
        
        # Add SSL mode
        params = [f"sslmode={self.ssl_mode}"]
        if self.schema and self.schema != "public":
            # Include 'public' in search_path so extension types (like vector) are accessible
            params.append(f"options=-csearch_path={self.schema}, public")
        
        # Add extra parameters
        for key, value in self.extra_params.items():
            params.append(f"{key}={value}")
        
        if params:
            conn_str += "?" + "&".join(params)
        
        return conn_str
    
    def validate(self) -> bool:
        """Validate PostgreSQL configuration"""
        # If a DSN is provided, perform minimal validation and skip host/port checks
        if self.connection_string:
            # Basic sanity check
            return isinstance(self.connection_string, str) and len(self.connection_string.strip()) > 0
        if not self.host or not self.database_name:
            return False
        if not 1 <= self.port <= 65535:
            return False
        if self.ssl_mode not in ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]:
            return False
        return True
    
    async def test_connection(self) -> Tuple[bool, str]:
        """Test PostgreSQL connection asynchronously"""
        # First validate configuration
        if not self.validate():
            return False, "Invalid PostgreSQL configuration"
        
        # If DSN is provided, try to connect directly without separate host/port check
        if self.connection_string:
            try:
                import asyncpg
                conn = await asyncio.wait_for(
                    asyncpg.connect(self.connection_string),
                    timeout=self.timeout
                )
                version = await conn.fetchval("SELECT version();")
                schema_msg = ""
                if self.schema and self.schema != "public":
                    schema_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1);",
                        self.schema
                    )
                    if schema_exists:
                        schema_msg = f" Schema '{self.schema}' exists."
                    else:
                        await conn.close()
                        return False, "PostgreSQL connection successful but schema '{}' does not exist. Available schemas can be checked with: SELECT schema_name FROM information_schema.schemata;".format(self.schema)
                await conn.close()
                return True, f"PostgreSQL connection successful. Server version: {version[:50]}...{schema_msg}"
            except ImportError:
                try:
                    import psycopg2
                    conn = psycopg2.connect(self.connection_string)
                    cursor = conn.cursor()
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    schema_msg = ""
                    if self.schema and self.schema != "public":
                        cursor.execute(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = %s);",
                            (self.schema,)
                        )
                        schema_exists = cursor.fetchone()[0]
                        if schema_exists:
                            schema_msg = f" Schema '{self.schema}' exists."
                        else:
                            cursor.close()
                            conn.close()
                            return False, "PostgreSQL connection successful but schema '{}' does not exist. Available schemas can be checked with: SELECT schema_name FROM information_schema.schemata;".format(self.schema)
                    cursor.close()
                    conn.close()
                    return True, f"PostgreSQL connection successful (via psycopg2). Server version: {version[:50]}...{schema_msg}"
                except ImportError:
                    return False, "No PostgreSQL driver available (asyncpg/psycopg2) to verify DSN connectivity"
                except Exception as e:
                    return False, f"PostgreSQL database connection failed: {str(e)}"
            except Exception as e:
                return False, f"PostgreSQL database connection failed: {str(e)}"
        
        # Test basic connectivity when DSN not provided
        host_test, host_msg = await self._test_host_port(self.host, self.port, self.timeout)
        if not host_test:
            return False, f"PostgreSQL connection failed: {host_msg}"
        
        # Try to import asyncpg if available for async database connection test
        try:
            import asyncpg
            
            # Build connection URI
            conn_uri = self.to_connection_string()
            
            # Test actual database connection asynchronously
            conn = await asyncio.wait_for(
                asyncpg.connect(conn_uri), 
                timeout=self.timeout
            )
            
            # Get version
            version = await conn.fetchval("SELECT version();")
            
            # Check if schema exists (if not public)
            schema_msg = ""
            if self.schema and self.schema != "public":
                schema_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1);",
                    self.schema
                )
                if schema_exists:
                    schema_msg = f" Schema '{self.schema}' exists."
                else:
                    await conn.close()
                    return False, f"PostgreSQL connection successful but schema '{self.schema}' does not exist. Available schemas can be checked with: SELECT schema_name FROM information_schema.schemata;"
            
            await conn.close()
            
            return True, f"PostgreSQL connection successful. Server version: {version[:50]}...{schema_msg}"
            
        except ImportError:
            # asyncpg not available, try psycopg2 in executor
            try:
                import psycopg2
                
                # Run synchronous psycopg2 in thread executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._test_postgresql_sync
                )
                return result
                
            except ImportError:
                # No PostgreSQL drivers available, fallback to basic connectivity test
                return True, f"PostgreSQL host connectivity successful (asyncpg/psycopg2 not available for full test): {host_msg}"
        
        except Exception as e:
            return False, f"PostgreSQL database connection failed: {str(e)}"
    
    def _test_postgresql_sync(self) -> Tuple[bool, str]:
        """Synchronous PostgreSQL test for fallback"""
        try:
            import psycopg2
            
            conn_params = {
                'host': self.host,
                'port': self.port,
                'database': self.database_name,
                'user': self.username,
                'password': self.password,
                'connect_timeout': min(self.timeout, 10)
            }
            
            # Remove empty values
            conn_params = {k: v for k, v in conn_params.items() if v}
            
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Check if schema exists (if not public)
            schema_msg = ""
            if self.schema and self.schema != "public":
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = %s);",
                    (self.schema,)
                )
                schema_exists = cursor.fetchone()[0]
                if schema_exists:
                    schema_msg = f" Schema '{self.schema}' exists."
                else:
                    cursor.close()
                    conn.close()
                    return False, f"PostgreSQL connection successful but schema '{self.schema}' does not exist. Available schemas can be checked with: SELECT schema_name FROM information_schema.schemata;"
            
            cursor.close()
            conn.close()
            
            return True, f"PostgreSQL connection successful (via psycopg2). Server version: {version[:50]}...{schema_msg}"
            
        except Exception as e:
            return False, f"PostgreSQL database connection failed: {str(e)}"
