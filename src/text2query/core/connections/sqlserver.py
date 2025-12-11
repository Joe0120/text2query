"""SQL Server connection configuration"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import BaseConnectionConfig


@dataclass
class SQLServerConfig(BaseConnectionConfig):
    """SQL Server connection configuration"""
    
    host: str = "localhost"
    port: int = 1433
    username: str = "sa"
    password: str = ""
    charset: str = "utf8"
    autocommit: bool = True
    tds_version: str = "7.0"  # TDS protocol version
    appname: str = "text2query"
    
    def to_connection_string(self) -> str:
        """Generate SQL Server connection string"""
        conn_str = f"mssql+pymssql://{self.username}"
        
        if self.password:
            conn_str += f":{self.password}"
        
        conn_str += f"@{self.host}:{self.port}/{self.database_name}"
        
        # Add query parameters
        params = []
        if self.charset:
            params.append(f"charset={self.charset}")
        if self.tds_version:
            params.append(f"tds_version={self.tds_version}")
        
        if params:
            conn_str += "?" + "&".join(params)
        
        return conn_str
    
    def validate(self) -> bool:
        """Validate the configuration"""
        if not self.database_name:
            return False
        if not self.host:
            return False
        if self.port <= 0 or self.port > 65535:
            return False
        if not self.username:
            return False
        return True
    
    async def test_connection(self) -> tuple[bool, str]:
        """Test SQL Server connection"""
        import asyncio
        
        # 使用 aioodbc 進行異步連接測試
        try:
            import aioodbc
        except ImportError:
            # 如果沒有 aioodbc，使用 pymssql 在 executor 中運行
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._test_connection_sync)
        
        try:
            # 構建 DSN 連接字符串
            dsn = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.host},{self.port};"
                f"DATABASE={self.database_name};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )
            
            conn = await aioodbc.connect(dsn=dsn, timeout=self.timeout)
            
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT @@VERSION")
                version = await cursor.fetchone()
                version_str = version[0].split('\n')[0] if version else "Unknown"
            
            await conn.close()
            
            return True, f"SQL Server connection successful. Version: {version_str}"
            
        except Exception as e:
            return False, f"SQL Server connection failed: {str(e)}"
    
    def _test_connection_sync(self) -> tuple[bool, str]:
        """Synchronous connection test using pymssql"""
        try:
            import pymssql
            
            conn = pymssql.connect(
                server=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database_name,
                timeout=self.timeout,
                login_timeout=self.timeout
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()
            version_str = version[0].split('\n')[0] if version else "Unknown"
            
            cursor.close()
            conn.close()
            
            return True, f"SQL Server connection successful. Version: {version_str}"
            
        except Exception as e:
            return False, f"SQL Server connection failed: {str(e)}"


def create_sqlserver_config(
    host: str = "localhost",
    port: int = 1433,
    database_name: str = "master",
    username: str = "sa",
    password: str = "",
    **kwargs
) -> SQLServerConfig:
    """Factory function to create SQL Server configuration"""
    return SQLServerConfig(
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password=password,
        **kwargs
    )
