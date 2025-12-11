"""
SQLite connection configuration
"""

from typing import Tuple
from dataclasses import dataclass
import asyncio
from .base import BaseConnectionConfig


@dataclass
class SQLiteConfig(BaseConnectionConfig):
    """SQLite connection configuration"""
    
    file_path: str = ":memory:"  # ":memory:" for in-memory database
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"  # DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF
    synchronous: str = "NORMAL"  # OFF, NORMAL, FULL, EXTRA
    cache_size: int = -2000  # negative = KB, positive = pages
    
    def __post_init__(self):
        # SQLite doesn't use database_name in the same way
        if not hasattr(self, 'database_name') or not self.database_name:
            self.database_name = self.file_path
    
    def to_connection_string(self) -> str:
        """Generate SQLite connection string"""
        conn_str = f"sqlite:///{self.file_path}"
        
        # SQLite doesn't use URL parameters in the same way
        # These would typically be set via PRAGMA statements
        return conn_str
    
    def get_pragma_statements(self) -> list[str]:
        """Get SQLite PRAGMA statements for configuration"""
        pragmas = []
        
        if self.enable_foreign_keys:
            pragmas.append("PRAGMA foreign_keys = ON;")
        
        pragmas.extend([
            f"PRAGMA journal_mode = {self.journal_mode};",
            f"PRAGMA synchronous = {self.synchronous};",
            f"PRAGMA cache_size = {self.cache_size};"
        ])
        
        return pragmas
    
    def validate(self) -> bool:
        """Validate SQLite configuration"""
        if not self.file_path:
            return False
        if self.journal_mode not in ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]:
            return False
        if self.synchronous not in ["OFF", "NORMAL", "FULL", "EXTRA"]:
            return False
        return True
    
    async def test_connection(self) -> Tuple[bool, str]:
        """Test SQLite connection asynchronously"""
        # First validate configuration
        if not self.validate():
            return False, "Invalid SQLite configuration"
        
        # For in-memory database, test in executor
        if self.file_path == ":memory:":
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._test_sqlite_memory
            )
            return result
        
        # Test file database in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._test_sqlite_file
        )
        return result
    
    def _test_sqlite_memory(self) -> Tuple[bool, str]:
        """Test SQLite in-memory database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(":memory:", timeout=self.timeout)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()[0]
            
            # Apply PRAGMA settings
            for pragma in self.get_pragma_statements():
                cursor.execute(pragma)
            
            cursor.close()
            conn.close()
            
            return True, f"SQLite in-memory database connection successful. SQLite version: {version}"
            
        except Exception as e:
            return False, f"SQLite in-memory database connection failed: {str(e)}"
    
    def _test_sqlite_file(self) -> Tuple[bool, str]:
        """Test SQLite file database"""
        try:
            import os
            import sqlite3
            
            # Get directory path
            dir_path = os.path.dirname(os.path.abspath(self.file_path))
            
            # Check if directory exists or can be created
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    return False, f"Cannot create directory for SQLite database: {str(e)}"
            
            # Test SQLite connection
            conn = sqlite3.connect(self.file_path, timeout=self.timeout)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()[0]
            
            # Apply PRAGMA settings
            for pragma in self.get_pragma_statements():
                cursor.execute(pragma)
            
            cursor.close()
            conn.close()
            
            return True, f"SQLite connection successful. SQLite version: {version}"
            
        except Exception as e:
            return False, f"SQLite database connection failed: {str(e)}"

