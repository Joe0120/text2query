"""Async Trino query adapter for cross-database queries."""

from __future__ import annotations

import re
import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...core.connections import BaseConnectionConfig
from ...core.connections.trino import TrinoConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class TrinoAdapter(BaseQueryComposer):
    """Trino adapter for cross-database SQL queries.
    
    Trino enables querying data from multiple sources (PostgreSQL, MongoDB, MySQL, etc.)
    using standard SQL syntax.
    """

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        """
        Initialize Trino adapter
        
        Args:
            config: TrinoConfig instance with connection details
        """
        super().__init__(config)
        self.quote_char = '"'
        self.logger: logging.Logger = get_logger("text2query.adapters.trino")
        self._executor: Optional[ThreadPoolExecutor] = None
        
        if config and not isinstance(config, TrinoConfig):
            raise TypeError("TrinoAdapter requires TrinoConfig configuration")

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor for blocking operations."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=5)
        return self._executor

    async def sql_execution(
        self,
        sql_command: str,
        params: Optional[Sequence[Any]] = None,
        safe: bool = True,
        limit: Optional[int] = 100,
    ) -> Dict[str, Any]:
        """
        Execute SQL asynchronously via Trino
        
        Args:
            sql_command: SQL query to execute
            params: Query parameters (not yet supported by Trino adapter)
            safe: Whether to perform safety checks
            limit: Row limit for SELECT queries
            
        Returns:
            Dict with execution results
        """
        if not isinstance(sql_command, str) or not sql_command.strip():
            return {
                "success": False,
                "error": "SQL command must be a non-empty string",
                "sql_command": sql_command,
            }

        if safe:
            sql_to_run, error_msg = self.safe_sql_command(sql_command)
            if not sql_to_run:
                return {
                    "success": False,
                    "error": error_msg or "Unsafe SQL command detected",
                    "sql_command": sql_command,
                }
        else:
            sql_to_run = sql_command.strip()

        sql_to_run = self._apply_limit(sql_to_run, limit)

        try:
            # Execute in thread pool since trino client is blocking
            result = await asyncio.get_event_loop().run_in_executor(
                self._get_executor(),
                self._execute_sync,
                sql_to_run
            )
            return result
            
        except Exception as error:
            self.logger.error("Trino sql_execution failed: %s", error)
            return {
                "success": False,
                "error": str(error),
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
            }

    def _execute_sync(self, sql_command: str) -> Dict[str, Any]:
        """
        Execute SQL synchronously (called from executor)
        
        Args:
            sql_command: SQL to execute
            
        Returns:
            Result dictionary
        """
        try:
            import trino
            
            if not isinstance(self.config, TrinoConfig):
                raise RuntimeError("Invalid Trino configuration")
            
            # Create connection
            conn = trino.dbapi.connect(**self.config.get_connection_params())
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(sql_command)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch results
            result_rows = []
            if columns:
                rows = cursor.fetchall()
                result_rows = [self._convert_row(row) for row in rows]
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "columns": columns,
                "result": result_rows,
                "sql_command": sql_command,
                "metadata": {
                    "rows_returned": len(result_rows)
                }
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "trino package not installed. Install with: pip install trino",
                "sql_command": sql_command,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql_command": sql_command,
            }

    async def get_schema_str(self, catalogs: Optional[List[str]] = None) -> str:
        """
        獲取 Trino 中多個 catalog 的 schema 結構字符串
        
        Args:
            catalogs: 要查詢的 catalog 列表，例如 ["postgresql", "mongodb"]
                     如果為 None，則使用配置中的 catalog
        
        Returns:
            str: 包含所有表結構的字符串
        """
        try:
            if catalogs is None:
                if isinstance(self.config, TrinoConfig):
                    catalogs = [self.config.catalog]
                else:
                    catalogs = ["system"]
            
            schema_parts = []
            
            for catalog in catalogs:
                catalog_schema = await self._get_catalog_schema_str(catalog)
                if catalog_schema:
                    schema_parts.append(f"--- Catalog: {catalog} ---\n{catalog_schema}")
            
            return "\n\n".join(schema_parts)
            
        except Exception as error:
            self.logger.error("Failed to get schema string: %s", error)
            return ""

    async def _get_catalog_schema_str(self, catalog: str) -> str:
        """
        獲取單個 catalog 的 schema 字符串
        
        Args:
            catalog: Catalog 名稱
            
        Returns:
            Schema 字符串
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._get_executor(),
                self._get_catalog_schema_sync,
                catalog
            )
            return result
        except Exception as error:
            self.logger.error("Failed to get catalog schema for %s: %s", catalog, error)
            return ""

    def _get_catalog_schema_sync(self, catalog: str) -> str:
        """
        同步獲取 catalog schema（在執行緒池中運行）
        
        Args:
            catalog: Catalog 名稱
            
        Returns:
            Schema 字符串
        """
        try:
            import trino
            
            if not isinstance(self.config, TrinoConfig):
                return ""
            
            conn = trino.dbapi.connect(**self.config.get_connection_params())
            cursor = conn.cursor()
            
            # Get all schemas in catalog
            cursor.execute(f"SHOW SCHEMAS FROM {catalog}")
            schemas = [row[0] for row in cursor.fetchall()]
            
            # Filter out system schemas
            schemas = [s for s in schemas if s not in ["information_schema", "pg_catalog", "mysql", "sys"]]
            
            schema_parts = []
            
            for schema in schemas:
                # Get tables in schema
                try:
                    cursor.execute(f"SHOW TABLES FROM {catalog}.{schema}")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        # Get table structure
                        table_info = self._get_table_structure(cursor, catalog, schema, table)
                        if table_info:
                            schema_parts.append(table_info)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to get tables from {catalog}.{schema}: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            return "\n\n".join(schema_parts)
            
        except Exception as e:
            self.logger.error(f"Error getting catalog schema: {e}")
            return ""

    def _get_table_structure(self, cursor, catalog: str, schema: str, table: str) -> str:
        """
        獲取表結構信息
        
        Args:
            cursor: Trino cursor
            catalog: Catalog 名稱
            schema: Schema 名稱
            table: Table 名稱
            
        Returns:
            表結構字符串
        """
        try:
            full_table_name = f"{catalog}.{schema}.{table}"
            
            # Get column information
            cursor.execute(f"DESCRIBE {full_table_name}")
            columns = cursor.fetchall()
            
            # Build CREATE TABLE statement
            col_definitions = []
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                col_definitions.append(f'    "{col_name}" {col_type}')
            
            create_table = f'CREATE TABLE {full_table_name} (\n'
            create_table += ",\n".join(col_definitions)
            create_table += "\n);"
            
            return create_table
            
        except Exception as e:
            self.logger.error(f"Error getting table structure for {catalog}.{schema}.{table}: {e}")
            return ""

    async def get_schema_list(self, catalogs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        獲取結構化的資料庫 schema 信息
        
        Args:
            catalogs: 要查詢的 catalog 列表
        
        Returns:
            List[Dict]: 包含每個表的結構化信息
        """
        try:
            if catalogs is None:
                if isinstance(self.config, TrinoConfig):
                    catalogs = [self.config.catalog]
                else:
                    catalogs = ["system"]
            
            result = []
            
            for catalog in catalogs:
                catalog_tables = await self._get_catalog_schema_list(catalog)
                result.extend(catalog_tables)
            
            return result
            
        except Exception as error:
            self.logger.error("Failed to get schema list: %s", error)
            return []

    async def _get_catalog_schema_list(self, catalog: str) -> List[Dict[str, Any]]:
        """
        獲取單個 catalog 的結構化 schema
        
        Args:
            catalog: Catalog 名稱
            
        Returns:
            表列表
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._get_executor(),
                self._get_catalog_schema_list_sync,
                catalog
            )
            return result
        except Exception as error:
            self.logger.error("Failed to get catalog schema list for %s: %s", catalog, error)
            return []

    def _get_catalog_schema_list_sync(self, catalog: str) -> List[Dict[str, Any]]:
        """
        同步獲取 catalog schema list
        
        Args:
            catalog: Catalog 名稱
            
        Returns:
            表列表
        """
        try:
            import trino
            
            if not isinstance(self.config, TrinoConfig):
                return []
            
            conn = trino.dbapi.connect(**self.config.get_connection_params())
            cursor = conn.cursor()
            
            # Get all schemas in catalog
            cursor.execute(f"SHOW SCHEMAS FROM {catalog}")
            schemas = [row[0] for row in cursor.fetchall()]
            
            # Filter out system schemas
            schemas = [s for s in schemas if s not in ["information_schema", "pg_catalog", "mysql", "sys"]]
            
            result = []
            
            for schema in schemas:
                try:
                    cursor.execute(f"SHOW TABLES FROM {catalog}.{schema}")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        full_table_name = f"{catalog}.{schema}.{table}"
                        
                        # Get column information
                        cursor.execute(f"DESCRIBE {full_table_name}")
                        columns_raw = cursor.fetchall()
                        
                        columns = []
                        for col in columns_raw:
                            col_name = col[0]
                            col_type = col[1]
                            columns.append({col_name: col_type})
                        
                        result.append({
                            "type": "trino",
                            "catalog": catalog,
                            "schema": schema,
                            "table": table,
                            "full_path": full_table_name,
                            "columns": columns
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get tables from {catalog}.{schema}: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting catalog schema list: {e}")
            return []

    def safe_sql_command(self, sql_command: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Sanitize SQL command with security checks
        
        Args:
            sql_command: SQL to sanitize
            
        Returns:
            Tuple of (cleaned_sql, error_message)
        """
        # Basic cleaning
        sql_clean = sql_command.replace("`", '"').replace("\n", " ")
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()
        
        sql_upper = sql_clean.upper()
        
        # Forbidden keywords for safety
        forbidden_keywords = [
            "DROP", "ALTER", "TRUNCATE", "DELETE", "INSERT", "UPDATE",
            "CREATE", "GRANT", "REVOKE"
        ]
        
        # Check for forbidden keywords
        blocked_keywords = [kw for kw in forbidden_keywords if kw in sql_upper]
        if blocked_keywords:
            error_msg = f"Forbidden keywords detected: {', '.join(blocked_keywords)}"
            return None, error_msg
        
        return sql_clean, None

    def _apply_limit(self, sql_command: str, limit: Optional[int]) -> str:
        """Apply LIMIT clause to SELECT queries."""
        if limit is None or limit <= 0:
            return sql_command

        if not re.search(r"\bSELECT\b", sql_command, re.IGNORECASE):
            return sql_command

        limit_pattern = re.compile(r"LIMIT\s+\d+", re.IGNORECASE)
        if limit_pattern.search(sql_command):
            return limit_pattern.sub(f"LIMIT {limit}", sql_command)

        sql_no_semicolon = sql_command.rstrip().rstrip(";")
        return f"{sql_no_semicolon} LIMIT {limit};"

    def _convert_row(self, row: Sequence[Any]) -> List[Any]:
        """Convert database row to list."""
        return [self._convert_value(value) for value in row]

    def _convert_value(self, value: Any) -> Any:
        """Convert database values to Python types."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, (date, time, datetime)):
            return value.isoformat()
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors='ignore')
            except Exception:
                return value.hex()
        if isinstance(value, dict):
            import json
            return json.dumps(value)
        return str(value)

    async def close(self) -> None:
        """Close the thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

