"""Async PostgreSQL query adapter built on asyncpg."""

from __future__ import annotations

import re
import asyncio
import logging
# import asyncpg
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.connections import BaseConnectionConfig
from ...core.connections.postgresql import PostgreSQLConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class PostgreSQLAdapter(BaseQueryComposer):
    """PostgreSQL adapter that executes SQL asynchronously via asyncpg."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "postgresql"

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        super().__init__(config)
        self.quote_char = '"'
        self.logger: logging.Logger = get_logger("text2query.adapters.postgresql")
        self._pool: Optional[Any] = None
        self._pool_lock: Optional[asyncio.Lock] = None

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
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to acquire PostgreSQL pool: %s", exc)
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
                if self._is_resultset_command(command_keyword, sql_upper):
                    records = await conn.fetch(sql_to_run, *args)
                    if records:
                        columns = list(records[0].keys())
                        result_rows = [self._convert_record(record) for record in records]
                    metadata["rows_returned"] = len(result_rows)
                else:
                    status = await conn.execute(sql_to_run, *args)
                    affected = self._parse_status_rows(status)
                    metadata["rows_affected"] = affected
                    metadata["status"] = status

            return {
                "success": True,
                "columns": columns,
                "result": result_rows,
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
                "metadata": metadata,
            }

        except Exception as error:
            self.logger.error("PostgreSQL sql_execution failed: %s", error)
            return {
                "success": False,
                "error": str(error),
                "sql_command": sql_to_run,
                "ori_sql_command": sql_command,
            }

    async def get_schema_str(self, tables: Optional[List[str]] = None) -> str:
        """
        獲取當前 schema 中所有表的 SQL 結構字符串

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
                if table_info.get('create_sql'):
                    table_structures.append(table_info['create_sql'])

            return "\n\n".join(table_structures)

        except Exception as error:
            self.logger.error("Failed to get SQL struct string: %s", error)
            return ""

    async def get_schema_list(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        獲取結構化的資料庫 schema 信息

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            List[Dict]: 包含每個表的結構化信息
                [{
                    "type": "postgresql",
                    "database": "db_name",
                    "schema": "schema_name",
                    "table": "table_name",
                    "full_path": "db_name.schema_name.table_name",
                    "columns": [{"column_name": "type"}, ...]
                }]
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)
            if not schema_data:
                return []

            database_name = schema_data['database']
            schema_name = schema_data['schema']
            result = []

            for table_info in schema_data['tables']:
                table_name = table_info['table_name']
                columns_info = table_info['columns']

                # 轉換為簡化的 {column_name: type} 格式
                columns = []
                for col in columns_info:
                    col_name = col['column_name']
                    col_type = col['data_type']

                    # 添加長度或精度信息
                    if col['character_maximum_length']:
                        col_type = f"{col_type}({col['character_maximum_length']})"
                    elif col['numeric_precision'] and col['numeric_scale']:
                        col_type = f"{col_type}({col['numeric_precision']},{col['numeric_scale']})"
                    elif col['numeric_precision']:
                        col_type = f"{col_type}({col['numeric_precision']})"

                    columns.append({col_name: col_type})

                result.append({
                    "type": "postgresql",
                    "database": database_name,
                    "schema": schema_name,
                    "table": table_name,
                    "full_path": f"{database_name}.{schema_name}.{table_name}",
                    "columns": columns
                })

            return result

        except Exception as error:
            self.logger.error("Failed to get structured schema: %s", error)
            return []

    async def _get_schema_info(self, tables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        獲取資料庫 schema 信息（核心方法）

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            Optional[Dict]: 包含資料庫、schema 和表信息的字典
                {
                    "database": "db_name",
                    "schema": "schema_name",
                    "tables": [
                        {
                            "table_name": "name",
                            "columns": [{column_info}, ...],
                            "primary_keys": [{pk_info}, ...],
                            "create_sql": "CREATE TABLE ..."
                        },
                        ...
                    ]
                }
        """
        current_schema = self._get_current_schema()
        current_database = self.config.database_name

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                all_tables = await self._get_schema_tables(conn, current_schema)
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
                    columns_info = await self._get_table_columns(conn, current_schema, table_name)

                    # 獲取主鍵信息
                    primary_keys = await self._get_table_primary_keys(conn, current_schema, table_name)

                    # 構建 CREATE TABLE 語句
                    create_sql = await self._build_table_structure(conn, current_schema, table_name)

                    tables_info.append({
                        "table_name": table_name,
                        "columns": columns_info,
                        "primary_keys": primary_keys,
                        "create_sql": create_sql
                    })

                return {
                    "database": current_database,
                    "schema": current_schema,
                    "tables": tables_info
                }

        except Exception as error:
            self.logger.error("Failed to get schema info: %s", error)
            raise

    # ============================================================================
    # 3. 連線池管理
    # ============================================================================
    
    async def close_pool(self) -> None:
        """Close the connection pool if it was created."""
        if self._pool is not None:
            await self._pool.close()
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
            import asyncpg
            self._pool = await asyncpg.create_pool(**pool_kwargs)
            return self._pool

    def _build_pool_kwargs(self) -> Dict[str, Any]:
        """Build connection pool configuration."""
        if not self.config or not isinstance(self.config, BaseConnectionConfig):
            raise RuntimeError("PostgreSQLAdapter requires a valid PostgreSQL configuration")

        if isinstance(self.config, PostgreSQLConfig):
            cfg = self.config
        else:
            cfg = PostgreSQLConfig(
                database_name=self.config.database_name,
                timeout=self.config.timeout,
                extra_params=self.config.extra_params,
            )
            for attr in ["host", "port", "username", "password", "schema", "ssl_mode"]:
                if hasattr(self.config, attr):
                    setattr(cfg, attr, getattr(self.config, attr))

        min_size = int(cfg.extra_params.get("min_pool_size", 1))
        max_size = int(cfg.extra_params.get("max_pool_size", max(min_size, 5)))
        command_timeout = float(cfg.extra_params.get("command_timeout", cfg.timeout))

        # If a DSN/connection string is provided, prefer that over discrete params
        if getattr(cfg, "connection_string", None):
            pool_kwargs: Dict[str, Any] = {
                "dsn": cfg.connection_string,  # asyncpg supports dsn
                "min_size": min_size,
                "max_size": max_size,
                "timeout": cfg.timeout,
                "command_timeout": command_timeout,
                "server_settings": {
                    "application_name": "text2query_adapter",
                },
            }
        else:
            pool_kwargs = {
                "host": cfg.host,
                "port": cfg.port,
                "user": cfg.username or None,
                "password": cfg.password or None,
                "database": cfg.database_name,
                "min_size": min_size,
                "max_size": max_size,
                "timeout": cfg.timeout,
                "command_timeout": command_timeout,
                "server_settings": {
                    "application_name": "text2query_adapter",
                },
            }

        if cfg.schema and cfg.schema != "public":
            pool_kwargs["server_settings"]["search_path"] = cfg.schema

        if getattr(cfg, "ssl_mode", None):
            pool_kwargs["ssl"] = cfg.ssl_mode not in {"disable", "allow"}

        for key, value in cfg.extra_params.items():
            if key in {"min_pool_size", "max_pool_size", "command_timeout"}:
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
        sql_clean = sql_command.replace("`", '"').replace("\n", " ")
        sql_clean = sql_clean.replace("    ", "")  # 移除多餘空格
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()
        
        # UUID 格式轉換
        sql_clean = sql_command = re.sub(r'[\' ]([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})[\' ]', r' "\g<1>" ', sql_clean)
        
        # 中文變體處理 (簡體/繁體)
        pattern = r'[\"\']([^\"\']+)[\"\'] LIKE [\"\']%([^%]+)%[\"\']'
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
        system_schemas = ["INFORMATION_SCHEMA", "PG_CATALOG"]
        for schema_keyword in system_schemas:
            if schema_keyword in sql_upper:
                # 只允許以 SELECT 開頭的查詢
                if not sql_upper.strip().startswith("SELECT"):
                    error_msg = f"Only SELECT queries allowed for {schema_keyword}"
                    self.logger.warning(error_msg)
                    return None, error_msg
        
        # Schema 權限檢查：只允許查詢配置的 schema
        schema_error = self._check_schema_permission(sql_clean)
        if schema_error:
            return None, schema_error

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
            raise TypeError("PostgreSQLAdapter sql_execution only supports positional parameters")
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

    def _parse_status_rows(self, status: str) -> Optional[int]:
        """Parse affected rows from status string."""
        parts = status.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            try:
                return int(parts[-1])
            except ValueError:
                return None
        return None

    def _check_schema_permission(self, sql_command: str) -> Optional[str]:
        """檢查是否只訪問允許的 schema
        
        Returns:
            str: 錯誤訊息，如果沒有錯誤則返回 None
        """
        current_schema = self._get_current_schema()
        sql_upper = sql_command.upper()
        
        # 尋找所有 schema.table 模式，但排除表別名
        schema_table_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(schema_table_pattern, sql_upper)
        
        for schema_name, table_name in matches:
            schema_name = schema_name.lower()
            
            # 跳過表別名（通常是單字母或短名稱）
            if len(schema_name) <= 2:
                continue
                
            # 允許系統 schema
            if schema_name in ["information_schema", "pg_catalog"]:
                continue
            
            # 如果使用預設 public schema
            if current_schema == "public":
                # 允許明確的 public 前綴
                if schema_name == "public":
                    continue
                # 拒絕其他 schema
                error_msg = f"Access denied: Cannot access schema '{schema_name}' from public schema context"
                self.logger.warning(error_msg)
                return error_msg
            else:
                # 如果配置了特定 schema，只允許該 schema
                if schema_name != current_schema.lower():
                    error_msg = f"Access denied: Cannot access schema '{schema_name}' with current schema '{current_schema}'"
                    self.logger.warning(error_msg)
                    return error_msg
        
        return None

    # ============================================================================
    # 5. 數據轉換和格式化
    # ============================================================================
    
    def _convert_record(self, record) -> List[Any]:
        """Convert database record to list."""
        return [self._convert_value(value) for value in record.values()]

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
                return value.decode("utf-8", errors='ignore')
            except Exception:  # pragma: no cover - fallback path
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
        # 定義中文字符的範圍：基本中日韓統一表意文字、擴展A區、擴展B區、擴展C區、擴展D區、擴展E區、擴展F區
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebe0]')
        return bool(re.search(chinese_pattern, text))

    def _convert_chinese_variant(self, text: str) -> tuple:
        """轉換中文變體（簡體/繁體）"""
        try:
            import opencc
            # 創建 OpenCC 轉換器
            simplified_to_traditional = opencc.OpenCC('s2t')  # 簡體轉繁體
            traditional_to_simplified = opencc.OpenCC('t2s')  # 繁體轉簡體
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
    
    def _get_current_schema(self) -> str:
        """獲取當前使用的 schema"""
        if isinstance(self.config, PostgreSQLConfig) and self.config.schema:
            return self.config.schema
        return "public"

    async def _get_schema_tables(self, conn, schema: str) -> list:
        """獲取指定 schema 中的所有用戶表"""
        query = """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = $1
              AND table_type = 'BASE TABLE'
              AND table_name NOT LIKE 'pg_%'
              AND table_name NOT LIKE 'sql_%'
            ORDER BY table_name;
        """
        return await conn.fetch(query, schema)

    async def _build_table_structure(self, conn, schema: str, table_name: str) -> str:
        """構建單個表的 CREATE TABLE 語句"""
        # 獲取列信息
        columns = await self._get_table_columns(conn, schema, table_name)
        if not columns:
            return ""
        
        # 獲取主鍵信息
        primary_keys = await self._get_table_primary_keys(conn, schema, table_name)
        
        # 構建 CREATE TABLE 語句
        col_definitions = []
        
        # 處理列定義
        for column in columns:
            col_def = self._format_column_definition(column)
            col_definitions.append(col_def)
        
        # 添加主鍵約束
        if primary_keys:
            pk_columns = ", ".join([f'"{pk["column_name"]}"' for pk in primary_keys])
            col_definitions.append(f'    PRIMARY KEY ({pk_columns})')
        
        # 組合最終的 CREATE TABLE 語句
        create_table = f'CREATE TABLE "{table_name}" (\n'
        create_table += ",\n".join(col_definitions)
        create_table += "\n);"
        
        return create_table

    async def _get_table_columns(self, conn, schema: str, table_name: str) -> list:
        """獲取表的列信息"""
        query = """
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                numeric_precision,
                numeric_scale,
                is_nullable,
                column_default
            FROM
                information_schema.columns
            WHERE
                table_schema = $1
                AND table_name = $2
            ORDER BY
                ordinal_position;
        """
        return await conn.fetch(query, schema, table_name)

    async def _get_table_primary_keys(self, conn, schema: str, table_name: str) -> list:
        """獲取表的主鍵信息"""
        query = """
            SELECT
                kcu.column_name
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE
                tc.table_schema = $1
                AND tc.table_name = $2
                AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position;
        """
        return await conn.fetch(query, schema, table_name)

    def _format_column_definition(self, column: dict) -> str:
        """格式化列定義"""
        col_name = column['column_name']
        data_type = column['data_type']
        char_max_len = column['character_maximum_length']
        numeric_precision = column['numeric_precision']
        numeric_scale = column['numeric_scale']
        is_nullable = column['is_nullable']
        column_default = column['column_default']
        
        # PostgreSQL 資料類型映射
        type_mapping = {
            'integer': 'INT',
            'bigint': 'BIGINT',
            'smallint': 'SMALLINT',
            'character varying': 'VARCHAR',
            'text': 'TEXT',
            'timestamp without time zone': 'TIMESTAMP',
            'timestamp with time zone': 'TIMESTAMPTZ',
            'date': 'DATE',
            'time without time zone': 'TIME',
            'character': 'CHAR',
            'boolean': 'BOOLEAN',
            'numeric': 'NUMERIC',
            'decimal': 'DECIMAL',
            'real': 'REAL',
            'double precision': 'DOUBLE PRECISION',
            'uuid': 'UUID',
            'json': 'JSON',
            'jsonb': 'JSONB'
        }
        
        # 映射資料類型
        mapped_type = type_mapping.get(data_type, data_type.upper())
        
        # 處理類型和長度
        if mapped_type in ['VARCHAR', 'CHAR'] and char_max_len:
            type_spec = f"{mapped_type}({char_max_len})"
        elif mapped_type in ['NUMERIC', 'DECIMAL'] and numeric_precision:
            if numeric_scale and numeric_scale > 0:
                type_spec = f"{mapped_type}({numeric_precision},{numeric_scale})"
            else:
                type_spec = f"{mapped_type}({numeric_precision})"
        else:
            type_spec = mapped_type
        
        # 構建列定義
        col_def = f'    "{col_name}" {type_spec}'
        
        # 添加 NOT NULL 約束
        if is_nullable == 'NO':
            col_def += " NOT NULL"
        
        # 添加預設值
        if column_default and not column_default.strip().startswith("nextval("):
            col_def += f" DEFAULT {column_default.strip()}"
        
        return col_def
