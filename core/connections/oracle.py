"""Oracle Database connection configuration"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .base import BaseConnectionConfig


@dataclass
class OracleConfig(BaseConnectionConfig):
    """Oracle Database connection configuration"""
    
    host: str = "localhost"
    port: int = 1521
    username: str = "system"
    password: str = ""
    service_name: str = "XE"  # Oracle service name (e.g., XE, ORCL)
    sid: str = ""  # 可選：使用 SID 而非 service_name
    encoding: str = "UTF-8"
    
    def to_connection_string(self) -> str:
        """Generate Oracle connection string"""
        # Oracle DSN format
        if self.sid:
            dsn = f"{self.host}:{self.port}/{self.sid}"
        else:
            dsn = f"{self.host}:{self.port}/{self.service_name}"
        
        # cx_Oracle/oracledb connection string format
        conn_str = f"{self.username}/{self.password}@{dsn}"
        return conn_str
    
    def get_dsn(self) -> str:
        """Get DSN string for connection"""
        if self.sid:
            return f"{self.host}:{self.port}/{self.sid}"
        else:
            return f"{self.host}:{self.port}/{self.service_name}"
    
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
        if not self.service_name and not self.sid:
            return False
        return True
    
    async def test_connection(self) -> tuple[bool, str]:
        """Test Oracle connection"""
        import asyncio
        
        # Oracle 目前主要使用 oracledb，暫無成熟的異步驅動
        # 使用 executor 運行同步版本
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._test_connection_sync)
    
    def _test_connection_sync(self) -> tuple[bool, str]:
        """Synchronous connection test using oracledb"""
        try:
            import oracledb
            
            dsn = self.get_dsn()
            
            # 如果 database_name 不是 system/sys，則連接到該 schema
            if self.database_name.upper() in ['SYSTEM', 'SYS', 'XE']:
                user = self.username
            else:
                user = self.database_name
            
            conn = oracledb.connect(
                user=user,
                password=self.password,
                dsn=dsn
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1")
            version = cursor.fetchone()
            version_str = version[0] if version else "Unknown"
            
            cursor.close()
            conn.close()
            
            return True, f"Oracle connection successful. Version: {version_str}"
            
        except Exception as e:
            return False, f"Oracle connection failed: {str(e)}"


def create_oracle_config(
    host: str = "localhost",
    port: int = 1521,
    database_name: str = "XE",
    username: str = "system",
    password: str = "",
    service_name: str = "XE",
    **kwargs
) -> OracleConfig:
    """Factory function to create Oracle configuration"""
    return OracleConfig(
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password=password,
        service_name=service_name,
        **kwargs
    )
