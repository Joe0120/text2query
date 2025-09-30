"""Async SQLite query adapter."""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ...core.connections import BaseConnectionConfig
from ...core.connections.sqlite import SQLiteConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class SQLiteAdapter(BaseQueryComposer):
    """SQLite adapter that executes SQL asynchronously via sqlite3."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        super().__init__(config or SQLiteConfig())
        if not isinstance(self.config, SQLiteConfig):
            self.config = SQLiteConfig(file_path=self.config.database_name)
        self.quote_char = '"'
        self.logger: logging.Logger = get_logger("text2query.adapters.sqlite")

    # ============================================================================
    # 2. 公共 API 方法（核心功能）
    # ============================================================================

    async def sql_execution(
        self,
        sql_command: str,
        params: Optional[Sequence[Any]] = None,
        safe: bool = True,
        limit: Optional[int] = 100,
    ) -> Dict[str, Any]:
        """Execute SQL asynchronously and return a structured response."""

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
            args = self._normalize_params(params)
        except TypeError as error:
            return {
                "success": False,
                "error": str(error),
                "sql_command": sql_command,
            }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            execution = await loop.run_in_executor(
                None,
                self._execute_sql_sync,
                sql_to_run,
                args,
            )
        except Exception as error:
            self.logger.error("SQLite sql_execution failed: %s", error)
            return {
                "success": False,
                "error": str(error),
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
            }

        return {
            "success": True,
            "columns": execution["columns"],
            "result": execution["result"],
            "sql_command": sql_to_run,
            "ori_sql_command": sql_command,
            "metadata": execution["metadata"],
        }

    async def get_sql_struct_str(self) -> str:
        """
        獲取 SQLite 資料庫的 SQL 結構字符串
        
        Returns:
            str: 包含所有表結構的 CREATE TABLE 語句字符串
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            structures = await loop.run_in_executor(None, self._fetch_table_structures)
            return "\n\n".join(structures)
        except Exception as error:
            self.logger.error("Failed to get SQL struct string: %s", error)
            return ""

    # ============================================================================
    # 3. SQL 處理和安全檢查
    # ============================================================================

    def safe_sql_command(self, sql_command: str) -> tuple[Optional[str], Optional[str]]:
        """Sanitize SQL command with enhanced security and formatting
        
        Returns:
            tuple: (cleaned_sql, error_message) - cleaned_sql is None if validation fails
        """
        # 基本清理
        sql_clean = sql_command.replace("`", '"').replace("\n", " ")
        sql_clean = sql_clean.replace("    ", "")  # 移除多餘空格
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()
        
        if not sql_clean:
            return None, "SQL command must be a non-empty string"
        
        # UUID 格式轉換
        sql_clean = re.sub(r'[\' ]([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})[\' ]', r' "\g<1>" ', sql_clean)
        
        # 中文變體處理 (簡體/繁體)
        pattern = r'[`\"\']([^`\"\']+)[`\"\'] LIKE [`\"\']%([^%]+)%[`\"\']'
        for match in re.finditer(pattern, sql_clean):
            if self._contains_chinese(match.group(2)):
                traditional, simplified = self._convert_chinese_variant(match.group(2))
                ori_match = match.group()
                trad_match = ori_match.replace(match.group(2), traditional)
                simp_match = ori_match.replace(match.group(2), simplified)
                sql_clean = sql_clean.replace(ori_match, f'{trad_match} OR {simp_match}')

        sql_upper = sql_clean.upper()
        forbidden_keywords = [
            "SYSTEM", "ADMIN",
            "DROP", "ALTER", "TRUNCATE", "DELETE", "INSERT", "UPDATE",
            "CREATE", "GRANT", "REVOKE"
        ]

        # 檢查禁用關鍵字
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

    def _normalize_params(self, params: Optional[Sequence[Any]]) -> Union[Sequence[Any], Dict[str, Any], Tuple]:
        """Normalize query parameters."""
        if params is None:
            return ()
        if isinstance(params, dict):
            return params
        if isinstance(params, (list, tuple)):
            return tuple(params)
        if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
            return tuple(params)
        return (params,)

    def _extract_command(self, sql_upper: str) -> str:
        """Extract the command keyword from SQL."""
        parts = sql_upper.strip().split()
        return parts[0] if parts else ""

    def _is_resultset_command(self, keyword: str, sql_upper: str) -> bool:
        """Check if command returns a result set."""
        if keyword in {"SELECT", "PRAGMA", "EXPLAIN", "WITH"}:
            return True
        return " RETURNING " in sql_upper

    # ============================================================================
    # 4. 數據轉換和格式化
    # ============================================================================

    def _convert_row(self, row: Any) -> List[Any]:
        """Convert database row to list."""
        if isinstance(row, sqlite3.Row):
            return [self._convert_value(row[key]) for key in row.keys()]
        if isinstance(row, dict):
            return [self._convert_value(value) for value in row.values()]
        return [self._convert_value(value) for value in row]

    def _convert_value(self, value: Any) -> Any:
        """Convert database values to Python types with enhanced handling"""
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
                return value.decode("utf-8", errors="ignore")
            except Exception:  # pragma: no cover
                return value.hex()
        if isinstance(value, dict):
            import json
            return json.dumps(value)
        return str(value)

    # ============================================================================
    # 5. 中文處理輔助方法
    # ============================================================================

    def _contains_chinese(self, text: str) -> bool:
        """檢查文字是否包含中文字符"""
        chinese_pattern = re.compile(
            r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df"
            r"\U0002a700-\U0002b73f\U0002b740-\U0002b81f"
            r"\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebe0]"
        )
        return bool(re.search(chinese_pattern, text))

    def _convert_chinese_variant(self, text: str) -> tuple:
        """轉換中文變體（簡體/繁體）"""
        try:
            import opencc
            # 創建 OpenCC 轉換器
            simplified_to_traditional = opencc.OpenCC("s2t")  # 簡體轉繁體
            traditional_to_simplified = opencc.OpenCC("t2s")  # 繁體轉簡體
            traditional = simplified_to_traditional.convert(text)
            simplified = traditional_to_simplified.convert(text)
            return traditional, simplified
        except ImportError:
            # 如果沒有安裝 opencc，使用簡化版本
            self.logger.warning("opencc not installed, using simplified Chinese variant conversion")
            return text, text  # 返回原文本作為備用

    # ============================================================================
    # 6. SQLite 特定的同步執行方法
    # ============================================================================

    def _execute_sql_sync(
        self, 
        sql_command: str, 
        params: Union[Sequence[Any], Dict[str, Any], Tuple]
    ) -> Dict[str, Any]:
        """Execute SQL synchronously in executor."""
        connection = sqlite3.connect(
            self.config.file_path,
            timeout=self.config.timeout,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        metadata: Dict[str, Any] = {}

        try:
            self._apply_pragmas(cursor)
            sql_upper = sql_command.upper()
            command_keyword = self._extract_command(sql_upper)

            if self._is_resultset_command(command_keyword, sql_upper):
                cursor.execute(sql_command, params if params is not None else ())
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description] if cursor.description else []
                result_rows = [self._convert_row(row) for row in rows]
                metadata["rows_returned"] = len(result_rows)
            else:
                cursor.execute(sql_command, params if params is not None else ())
                connection.commit()
                columns = []
                result_rows = []
                metadata["rows_affected"] = cursor.rowcount if cursor.rowcount != -1 else None
                metadata["last_row_id"] = cursor.lastrowid

            return {"columns": columns, "result": result_rows, "metadata": metadata}

        finally:
            cursor.close()
            connection.close()

    def _fetch_table_structures(self) -> List[str]:
        """Fetch CREATE TABLE statements for all tables."""
        connection = sqlite3.connect(
            self.config.file_path,
            timeout=self.config.timeout,
            check_same_thread=False,
        )
        cursor = connection.cursor()
        structures: List[str] = []

        try:
            self._apply_pragmas(cursor)
            cursor.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%%'
                ORDER BY name;
                """
            )
            tables = cursor.fetchall()

            for name, create_sql in tables:
                # 總是使用 _reconstruct_table_sql 來確保格式一致
                structures.append(self._reconstruct_table_sql(cursor, name))

            return structures

        finally:
            cursor.close()
            connection.close()

    def _reconstruct_table_sql(self, cursor: sqlite3.Cursor, table_name: str) -> str:
        """Reconstruct CREATE TABLE statement from PRAGMA info."""
        cursor.execute(f"PRAGMA table_info({self._quote_identifier(table_name)});")
        columns = cursor.fetchall()
        if not columns:
            return f"-- No schema information available for table {table_name}"

        definitions = []
        for _, name, col_type, notnull, default, pk in columns:
            column_def = f"    {self._quote_identifier(name)} {col_type or 'TEXT'}"
            if notnull:
                column_def += " NOT NULL"
            if default is not None:
                column_def += f" DEFAULT {default}"
            if pk:
                column_def += " PRIMARY KEY"
            definitions.append(column_def)

        cursor.execute(f"PRAGMA index_list({self._quote_identifier(table_name)});")
        indexes = cursor.fetchall()
        index_sql = []
        for index in indexes:
            if index[2]:  # unique flag
                cursor.execute(f"PRAGMA index_info({self._quote_identifier(index[1])});")
                cols = cursor.fetchall()
                col_list = ", ".join(self._quote_identifier(col[2]) for col in cols)
                index_sql.append(f"    UNIQUE({col_list})")

        all_defs = definitions + index_sql
        joined = ",\n".join(all_defs)
        return f"CREATE TABLE {self._quote_identifier(table_name)} (\n{joined}\n);"

    def _apply_pragmas(self, cursor: sqlite3.Cursor) -> None:
        """Apply PRAGMA settings to connection."""
        if isinstance(self.config, SQLiteConfig):
            for pragma in self.config.get_pragma_statements():
                try:
                    cursor.execute(pragma)
                except sqlite3.DatabaseError as error:
                    self.logger.warning("Failed to apply PRAGMA %s: %s", pragma, error)

    def _quote_identifier(self, name: str) -> str:
        """Quote SQL identifier to prevent keyword conflicts."""
        escaped = name.replace('"', '""')
        return f'"{escaped}"'