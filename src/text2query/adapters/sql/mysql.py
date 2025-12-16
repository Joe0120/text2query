"""Async MySQL query adapter built on aiomysql."""

from __future__ import annotations

import re
import asyncio
import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.connections import BaseConnectionConfig
from ...core.connections.mysql import MySQLConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class MySQLAdapter(BaseQueryComposer):
    """MySQL adapter that executes SQL asynchronously via aiomysql."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "mysql"

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        super().__init__(config)
        self.quote_char = "`"
        self.logger: logging.Logger = get_logger("text2query.adapters.mysql")
        self._pool: Optional[Any] = None
        self._pool_lock: Optional[asyncio.Lock] = None

        if not self.config:
            self.config = MySQLConfig(database_name="default")

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
            pool = await self._get_pool()
        except Exception as exc:
            self.logger.error("Failed to acquire MySQL pool: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "sql_command": sql_command,
            }

        sql_upper = sql_to_run.upper()
        command_keyword = self._extract_command(sql_upper)

        columns: List[str] = []
        result_rows: List[List[Any]] = []
        metadata: Dict[str, Any] = {}

        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if self._is_resultset_command(command_keyword, sql_upper):
                        await cursor.execute(sql_to_run, args)
                        rows = await cursor.fetchall()
                        if cursor.description:
                            columns = [col[0] for col in cursor.description]
                        if rows:
                            result_rows = [self._convert_row(row) for row in rows]
                        metadata["rows_returned"] = len(result_rows)
                    else:
                        await cursor.execute(sql_to_run, args)
                        if not getattr(conn, "autocommit", True):
                            await conn.commit()
                        affected = cursor.rowcount if cursor.rowcount != -1 else None
                        metadata["rows_affected"] = affected
                        metadata["last_row_id"] = cursor.lastrowid

            return {
                "success": True,
                "columns": columns,
                "result": result_rows,
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
                "metadata": metadata,
            }

        except Exception as error:
            self.logger.error("MySQL sql_execution failed: %s", error)
            return {
                "success": False,
                "error": str(error),
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
            }

    async def _get_schema_info(self, tables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        獲取資料庫 schema 信息（核心方法）

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            Optional[Dict]: 包含資料庫和表信息的字典
                {
                    "database": "db_name",
                    "schema": "",  # MySQL 沒有 schema 概念
                    "tables": [
                        {
                            "table_name": "name",
                            "columns": [{column_info}, ...],
                            "primary_keys": [{pk_info}, ...],
                            "indexes": [{index_info}, ...]
                        },
                        ...
                    ]
                }
        """
        current_database = self._get_current_database()

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    all_tables = await self._get_database_tables(cursor, current_database)

                    if not all_tables:
                        return None

                    # 過濾表
                    if tables:
                        tables_set = set(tables)
                        all_tables = [t for t in all_tables if t['table_name'] in tables_set]
                        if not all_tables:
                            return None

                    tables_info = []
                    for table in all_tables:
                        table_name = table['table_name']

                        # 獲取列信息
                        columns_info = await self._get_table_columns(cursor, current_database, table_name)

                        # 獲取主鍵信息
                        primary_keys = await self._get_table_primary_keys(cursor, current_database, table_name)

                        # 獲取索引信息
                        indexes = await self._get_table_indexes(cursor, current_database, table_name)

                        tables_info.append({
                            "table_name": table_name,
                            "columns": columns_info,
                            "primary_keys": primary_keys,
                            "indexes": indexes
                        })

                    return {
                        "database": current_database,
                        "schema": "",  # MySQL 沒有 schema 概念
                        "tables": tables_info
                    }

        except Exception as error:
            self.logger.error("Failed to get schema info: %s", error)
            raise

    def _format_column_type(self, col: Dict) -> str:
        """格式化列類型"""
        col_type = col['data_type']
        if col['character_maximum_length']:
            return f"{col_type}({col['character_maximum_length']})"
        elif col['numeric_precision'] and col['numeric_scale']:
            return f"{col_type}({col['numeric_precision']},{col['numeric_scale']})"
        elif col['numeric_precision']:
            return f"{col_type}({col['numeric_precision']})"
        return col_type

    async def get_schema_list(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        獲取結構化的資料庫 schema 信息

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            List[Dict]: 包含每個表的結構化信息
                [{
                    "type": "mysql",
                    "database": "db_name",
                    "schema": "",  # MySQL 沒有 schema
                    "table": "table_name",
                    "full_path": "db_name.table_name",
                    "columns": [{"column_name": "type"}, ...]
                }]
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)
            if not schema_data:
                return []

            database_name = schema_data['database']
            result = []

            for table_info in schema_data['tables']:
                table_name = table_info['table_name']
                columns_info = table_info['columns']

                # 轉換為簡化的 {column_name: type} 格式
                columns = []
                for col in columns_info:
                    col_name = col['column_name']
                    col_type = self._format_column_type(col)
                    columns.append({col_name: col_type})

                result.append({
                    "type": "mysql",
                    "database": database_name,
                    "schema": "",
                    "table": table_name,
                    "full_path": f"{database_name}.{table_name}",
                    "columns": columns
                })

            return result

        except Exception as error:
            self.logger.error("Failed to get structured schema: %s", error)
            return []

    async def get_schema_str(self, tables: Optional[List[str]] = None) -> str:
        """
        獲取當前資料庫中所有表的 SQL 結構字符串

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            str: 包含所有表結構的 CREATE TABLE 語句字符串
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)
            if not schema_data or not schema_data.get('tables'):
                return ""

            table_structures = []
            for table_info in schema_data['tables']:
                table_sql = self._build_create_table(table_info)
                if table_sql:
                    table_structures.append(table_sql)

            return "\n\n".join(table_structures)

        except Exception as error:
            self.logger.error("Failed to get SQL struct string: %s", error)
            return ""

    def _build_create_table(self, table_info: Dict[str, Any]) -> str:
        """將結構化信息格式化為 CREATE TABLE 語句"""
        table_name = table_info['table_name']
        columns_full = table_info.get('columns', [])
        primary_keys = table_info.get('primary_keys', [])
        indexes = table_info.get('indexes', [])

        if not columns_full:
            return ""

        col_definitions = []

        # 處理列定義
        for column in columns_full:
            col_def = self._format_column_definition(column)
            col_definitions.append(col_def)

        # 添加主鍵約束
        if primary_keys:
            pk_columns = ", ".join([f"`{pk['column_name']}`" for pk in primary_keys])
            col_definitions.append(f"    PRIMARY KEY ({pk_columns})")

        # 添加唯一索引
        for index in indexes:
            if index['non_unique'] == 0 and index['key_name'] != 'PRIMARY':
                idx_columns = ", ".join([f"`{col}`" for col in index['columns']])
                col_definitions.append(f"    UNIQUE KEY `{index['key_name']}` ({idx_columns})")

        # 組合最終的 CREATE TABLE 語句
        create_table = f'CREATE TABLE `{table_name}` (\n'
        create_table += ",\n".join(col_definitions)
        create_table += "\n);"

        return create_table

    # ============================================================================
    # 3. 連線池管理
    # ============================================================================

    async def close_pool(self) -> None:
        """Close the connection pool if it was created."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is not None:
            return self._pool

        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()

        async with self._pool_lock:
            if self._pool is not None:
                return self._pool

            pool_kwargs = self._build_pool_kwargs()
            import aiomysql
            self._pool = await aiomysql.create_pool(**pool_kwargs)
            return self._pool

    def _build_pool_kwargs(self) -> Dict[str, Any]:
        """Build connection pool configuration."""
        if not self.config or not isinstance(self.config, BaseConnectionConfig):
            raise RuntimeError("MySQLAdapter requires a valid MySQL configuration")

        if isinstance(self.config, MySQLConfig):
            cfg = self.config
        else:
            cfg = MySQLConfig(
                database_name=self.config.database_name,
                timeout=self.config.timeout,
                extra_params=self.config.extra_params,
            )
            for attr in [
                "host",
                "port",
                "username",
                "password",
                "charset",
                "collation",
                "ssl_disabled",
                "autocommit",
            ]:
                if hasattr(self.config, attr):
                    setattr(cfg, attr, getattr(self.config, attr))

        min_size = int(cfg.extra_params.get("min_pool_size", 1))
        max_size = int(cfg.extra_params.get("max_pool_size", max(min_size, 5)))
        connect_timeout = float(cfg.extra_params.get("connect_timeout", min(cfg.timeout, 10)))
        pool_recycle = int(cfg.extra_params.get("pool_recycle", -1))

        pool_kwargs: Dict[str, Any] = {
            "host": cfg.host,
            "port": cfg.port,
            "user": cfg.username or None,
            "password": cfg.password or None,
            "db": cfg.database_name,
            "minsize": min_size,
            "maxsize": max_size,
            "connect_timeout": connect_timeout,
            "autocommit": cfg.autocommit,
            "charset": cfg.charset,
            "pool_recycle": pool_recycle,
        }

        if cfg.extra_params.get("unix_socket"):
            pool_kwargs["unix_socket"] = cfg.extra_params["unix_socket"]

        if not cfg.ssl_disabled and cfg.extra_params.get("ssl"):
            pool_kwargs["ssl"] = cfg.extra_params["ssl"]
        elif cfg.ssl_disabled:
            pool_kwargs["ssl"] = None

        for key, value in cfg.extra_params.items():
            if key in {
                "min_pool_size",
                "max_pool_size",
                "connect_timeout",
                "pool_recycle",
                "unix_socket",
                "ssl",
            }:
                continue
            pool_kwargs.setdefault(key, value)

        return pool_kwargs

    # ============================================================================
    # 4. SQL 處理和安全檢查
    # ============================================================================

    def safe_sql_command(self, sql_command: str) -> tuple[Optional[str], Optional[str]]:
        """Sanitize SQL command with enhanced security and formatting
        
        Returns:
            tuple: (cleaned_sql, error_message) - cleaned_sql is None if validation fails
        """
        # 基本清理
        sql_clean = sql_command.replace('"', '`').replace("\n", " ")
        sql_clean = sql_clean.replace("    ", "")  # 移除多餘空格
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()
        
        # UUID 格式轉換
        sql_clean = re.sub(r'[\' ]([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})[\' ]', r' "`\g<1>`" ', sql_clean)
        
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
            
        # 特別檢查系統表：只允許 SELECT 查詢
        system_schemas = ["INFORMATION_SCHEMA", "MYSQL.", "PERFORMANCE_SCHEMA", "SYS."]
        for schema_keyword in system_schemas:
            if schema_keyword in sql_upper:
                # 只允許以 SELECT 開頭的查詢
                if not sql_upper.strip().startswith("SELECT"):
                    error_msg = f"Only SELECT queries allowed for {schema_keyword}"
                    self.logger.warning(error_msg)
                    return None, error_msg
        
        # 資料庫權限檢查：只允許查詢配置的資料庫
        database_error = self._check_database_permission(sql_clean)
        if database_error:
            return None, database_error

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

    def _normalize_params(self, params: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
        """Normalize query parameters."""
        if params is None:
            return ()
        if isinstance(params, (list, tuple)):
            return tuple(params)
        if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
            return tuple(params)
        if isinstance(params, dict):
            raise TypeError("MySQLAdapter sql_execution only supports positional parameters")
        return (params,)

    def _extract_command(self, sql_upper: str) -> str:
        """Extract the command keyword from SQL."""
        parts = sql_upper.strip().split()
        return parts[0] if parts else ""

    def _is_resultset_command(self, keyword: str, sql_upper: str) -> bool:
        """Check if command returns a result set."""
        if keyword in {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH"}:
            return True
        return " RETURNING " in sql_upper

    def _check_database_permission(self, sql_command: str) -> Optional[str]:
        """檢查是否只訪問允許的資料庫
        
        Returns:
            str: 錯誤訊息，如果沒有錯誤則返回 None
        """
        current_database = self._get_current_database()
        sql_upper = sql_command.upper()
        
        # 尋找所有 database.table 模式
        database_table_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(database_table_pattern, sql_upper)
        
        for database_name, table_name in matches:
            database_name = database_name.lower()
            
            # 跳過表別名（通常是單字母或短名稱）
            if len(database_name) <= 2:
                continue
                
            # 允許系統資料庫
            if database_name in ["information_schema", "mysql", "performance_schema", "sys"]:
                continue
            
            # 檢查是否為配置的資料庫
            if database_name != current_database.lower():
                error_msg = f"Access denied: Cannot access database '{database_name}' with current database '{current_database}'"
                self.logger.warning(error_msg)
                return error_msg
        
        return None

    # ============================================================================
    # 5. 數據轉換和格式化
    # ============================================================================

    def _convert_row(self, row: Any) -> List[Any]:
        """Convert database row to list."""
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
    # 6. 中文處理輔助方法
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
    # 7. 表結構查詢輔助方法
    # ============================================================================

    def _get_current_database(self) -> str:
        """獲取當前使用的資料庫"""
        if isinstance(self.config, MySQLConfig):
            return self.config.database_name
        return "default"

    async def _get_database_tables(self, cursor, database: str) -> list:
        """獲取指定資料庫中的所有用戶表"""
        query = """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
              AND table_name NOT LIKE 'mysql_%%'
              AND table_name NOT LIKE 'sys_%%'
            ORDER BY table_name;
        """
        await cursor.execute(query, (database,))
        rows = await cursor.fetchall()
        return [{"table_name": row[0], "table_type": row[1]} for row in rows]

    async def _build_table_structure(self, cursor, database: str, table_name: str) -> str:
        """構建單個表的 CREATE TABLE 語句"""
        # 獲取列信息
        columns = await self._get_table_columns(cursor, database, table_name)
        if not columns:
            return ""
        
        # 獲取主鍵信息
        primary_keys = await self._get_table_primary_keys(cursor, database, table_name)
        
        # 獲取索引信息
        indexes = await self._get_table_indexes(cursor, database, table_name)
        
        # 構建 CREATE TABLE 語句
        col_definitions = []
        
        # 處理列定義
        for column in columns:
            col_def = self._format_column_definition(column)
            col_definitions.append(col_def)
        
        # 添加主鍵約束
        if primary_keys:
            pk_columns = ", ".join([f"`{pk['column_name']}`" for pk in primary_keys])
            col_definitions.append(f"    PRIMARY KEY ({pk_columns})")
        
        # 添加唯一索引
        for index in indexes:
            if index['non_unique'] == 0 and index['key_name'] != 'PRIMARY':
                idx_columns = ", ".join([f"`{col}`" for col in index['columns']])
                col_definitions.append(f"    UNIQUE KEY `{index['key_name']}` ({idx_columns})")
        
        # 組合最終的 CREATE TABLE 語句
        create_table = f'CREATE TABLE `{table_name}` (\n'
        create_table += ",\n".join(col_definitions)
        create_table += "\n);"
        
        return create_table

    async def _get_table_columns(self, cursor, database: str, table_name: str) -> list:
        """獲取表的列信息"""
        query = """
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                numeric_precision,
                numeric_scale,
                is_nullable,
                column_default,
                extra
            FROM
                information_schema.columns
            WHERE
                table_schema = %s
                AND table_name = %s
            ORDER BY
                ordinal_position;
        """
        await cursor.execute(query, (database, table_name))
        rows = await cursor.fetchall()
        return [
            {
                "column_name": row[0],
                "data_type": row[1],
                "character_maximum_length": row[2],
                "numeric_precision": row[3],
                "numeric_scale": row[4],
                "is_nullable": row[5],
                "column_default": row[6],
                "extra": row[7]
            }
            for row in rows
        ]

    async def _get_table_primary_keys(self, cursor, database: str, table_name: str) -> list:
        """獲取表的主鍵信息"""
        query = """
            SELECT
                kcu.column_name
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE
                tc.table_schema = %s
                AND tc.table_name = %s
                AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position;
        """
        await cursor.execute(query, (database, table_name))
        rows = await cursor.fetchall()
        return [{"column_name": row[0]} for row in rows]

    async def _get_table_indexes(self, cursor, database: str, table_name: str) -> list:
        """獲取表的索引信息"""
        query = """
            SELECT
                index_name,
                non_unique,
                GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns
            FROM
                information_schema.statistics
            WHERE
                table_schema = %s
                AND table_name = %s
                AND index_name != 'PRIMARY'
            GROUP BY
                index_name, non_unique
            ORDER BY
                index_name;
        """
        await cursor.execute(query, (database, table_name))
        rows = await cursor.fetchall()
        return [
            {
                "key_name": row[0],
                "non_unique": row[1],
                "columns": row[2].split(",") if row[2] else []
            }
            for row in rows
        ]

    def _format_column_definition(self, column: dict) -> str:
        """格式化列定義"""
        col_name = column['column_name']
        data_type = column['data_type']
        char_max_len = column['character_maximum_length']
        numeric_precision = column['numeric_precision']
        numeric_scale = column['numeric_scale']
        is_nullable = column['is_nullable']
        column_default = column['column_default']
        extra = column['extra']
        
        # MySQL 資料類型映射
        type_mapping = {
            'int': 'INT',
            'bigint': 'BIGINT',
            'smallint': 'SMALLINT',
            'tinyint': 'TINYINT',
            'varchar': 'VARCHAR',
            'char': 'CHAR',
            'text': 'TEXT',
            'longtext': 'LONGTEXT',
            'mediumtext': 'MEDIUMTEXT',
            'tinytext': 'TINYTEXT',
            'datetime': 'DATETIME',
            'timestamp': 'TIMESTAMP',
            'date': 'DATE',
            'time': 'TIME',
            'decimal': 'DECIMAL',
            'float': 'FLOAT',
            'double': 'DOUBLE',
            'json': 'JSON',
            'blob': 'BLOB',
            'longblob': 'LONGBLOB'
        }
        
        # 映射資料類型
        mapped_type = type_mapping.get(data_type.lower(), data_type.upper())
        
        # 處理類型和長度
        if mapped_type in ['VARCHAR', 'CHAR'] and char_max_len:
            type_spec = f"{mapped_type}({char_max_len})"
        elif mapped_type in ['DECIMAL', 'NUMERIC'] and numeric_precision:
            if numeric_scale and numeric_scale > 0:
                type_spec = f"{mapped_type}({numeric_precision},{numeric_scale})"
            else:
                type_spec = f"{mapped_type}({numeric_precision})"
        else:
            type_spec = mapped_type
        
        # 構建列定義
        col_def = f'    `{col_name}` {type_spec}'
        
        # 添加 NOT NULL 約束
        if is_nullable == 'NO':
            col_def += " NOT NULL"
        
        # 添加 AUTO_INCREMENT
        if extra and 'auto_increment' in extra.lower():
            col_def += " AUTO_INCREMENT"
        
        # 添加預設值
        if column_default is not None:
            default_str = str(column_default)
            if default_str.upper() in ['CURRENT_TIMESTAMP', 'NOW()']:
                col_def += f" DEFAULT {default_str.upper()}"
            else:
                col_def += f" DEFAULT '{default_str}'"
        
        return col_def
