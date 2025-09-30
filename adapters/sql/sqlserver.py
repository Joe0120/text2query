"""Async SQL Server query adapter built on aioodbc/pymssql."""

from __future__ import annotations

import re
import asyncio
import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.connections import BaseConnectionConfig
from ...core.connections.sqlserver import SQLServerConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class SQLServerAdapter(BaseQueryComposer):
    """SQL Server adapter that executes SQL asynchronously via aioodbc."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        super().__init__(config)
        self.quote_char = "["
        self.logger: logging.Logger = get_logger("text2query.adapters.sqlserver")
        self._pool: Optional[Any] = None
        self._pool_lock: Optional[asyncio.Lock] = None

        if not self.config:
            self.config = SQLServerConfig(database_name="master")

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
            self.logger.error("SQL Server sql_execution failed: %s", error)
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
        獲取當前資料庫中所有表的 SQL 結構字符串
        
        Returns:
            str: 包含所有表結構的 CREATE TABLE 語句字符串
        """
        current_database = self._get_current_database()
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            structures = await loop.run_in_executor(
                None, self._fetch_table_structures, current_database
            )
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
        sql_clean = sql_command.replace('"', '[').replace("`", "[").replace("\n", " ")
        sql_clean = sql_clean.replace("    ", "")  # 移除多餘空格
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()
        
        # UUID 格式轉換
        sql_clean = re.sub(
            r'[\' ]([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})[\' ]',
            r" '\g<1>' ",
            sql_clean
        )
        
        # 中文變體處理 (簡體/繁體)
        pattern = r'[\[\"\']([^\]\"\']+)[\]\"\'] LIKE [\[\"\']%([^%]+)%[\]\"\']'
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
            "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "XP_"
        ]
        
        # 檢查禁用關鍵字
        blocked_keywords = [kw for kw in forbidden_keywords if kw in sql_upper]
        if blocked_keywords:
            error_msg = f"Forbidden keywords detected: {', '.join(blocked_keywords)}"
            return None, error_msg
            
        # 特別檢查系統表：只允許 SELECT 查詢
        system_schemas = ["INFORMATION_SCHEMA", "SYS.", "MASTER.", "MSDB.", "TEMPDB."]
        for schema_keyword in system_schemas:
            if schema_keyword in sql_upper:
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
        """Apply TOP clause to SELECT queries (SQL Server style)."""
        if limit is None or limit <= 0:
            return sql_command

        if not re.search(r"\bSELECT\b", sql_command, re.IGNORECASE):
            return sql_command

        # Check if already has TOP clause
        if re.search(r"\bTOP\s+\d+\b", sql_command, re.IGNORECASE):
            return sql_command

        # Add TOP clause after SELECT
        sql_command = re.sub(
            r"\bSELECT\b",
            f"SELECT TOP {limit}",
            sql_command,
            count=1,
            flags=re.IGNORECASE
        )
        
        return sql_command

    def _normalize_params(self, params: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
        """Normalize query parameters."""
        if params is None:
            return ()
        if isinstance(params, (list, tuple)):
            return tuple(params)
        if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
            return tuple(params)
        if isinstance(params, dict):
            raise TypeError("SQLServerAdapter sql_execution only supports positional parameters")
        return (params,)

    def _extract_command(self, sql_upper: str) -> str:
        """Extract the command keyword from SQL."""
        parts = sql_upper.strip().split()
        return parts[0] if parts else ""

    def _is_resultset_command(self, keyword: str, sql_upper: str) -> bool:
        """Check if command returns a result set."""
        if keyword in {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH"}:
            return True
        return False

    def _check_database_permission(self, sql_command: str) -> Optional[str]:
        """檢查是否只訪問允許的資料庫
        
        Returns:
            str: 錯誤訊息，如果沒有錯誤則返回 None
        """
        current_database = self._get_current_database()
        sql_upper = sql_command.upper()
        
        # 尋找所有 database.schema.table 或 database..table 模式
        database_table_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.\.'
        matches = re.findall(database_table_pattern, sql_upper)
        
        for database_name in matches:
            database_name = database_name.lower()
            
            # 允許系統資料庫
            if database_name in ["master", "msdb", "tempdb", "model"]:
                continue
            
            # 檢查是否為配置的資料庫
            if database_name != current_database.lower():
                error_msg = f"Access denied: Cannot access database '{database_name}' with current database '{current_database}'"
                self.logger.warning(error_msg)
                return error_msg
        
        return None

    # ============================================================================
    # 4. 數據轉換和格式化
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
            simplified_to_traditional = opencc.OpenCC("s2t")
            traditional_to_simplified = opencc.OpenCC("t2s")
            traditional = simplified_to_traditional.convert(text)
            simplified = traditional_to_simplified.convert(text)
            return traditional, simplified
        except ImportError:
            self.logger.warning("opencc not installed, using simplified Chinese variant conversion")
            return text, text

    # ============================================================================
    # 6. SQL Server 特定的同步執行方法
    # ============================================================================

    def _execute_sql_sync(self, sql_command: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
        """Execute SQL synchronously using pymssql."""
        import pymssql
        
        if not isinstance(self.config, SQLServerConfig):
            raise RuntimeError("SQLServerAdapter requires a valid SQL Server configuration")
        
        cfg = self.config
        
        connection = pymssql.connect(
            server=cfg.host,
            port=cfg.port,
            user=cfg.username,
            password=cfg.password,
            database=cfg.database_name,
            timeout=cfg.timeout,
            login_timeout=cfg.timeout,
            charset=cfg.charset,
            tds_version=cfg.tds_version,
            appname=cfg.appname,
        )
        
        cursor = connection.cursor()
        metadata: Dict[str, Any] = {}

        try:
            sql_upper = sql_command.upper()
            command_keyword = self._extract_command(sql_upper)

            if self._is_resultset_command(command_keyword, sql_upper):
                cursor.execute(sql_command, params if params else ())
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description] if cursor.description else []
                result_rows = [self._convert_row(row) for row in rows]
                metadata["rows_returned"] = len(result_rows)
            else:
                cursor.execute(sql_command, params if params else ())
                connection.commit()
                columns = []
                result_rows = []
                metadata["rows_affected"] = cursor.rowcount if cursor.rowcount != -1 else None

            return {"columns": columns, "result": result_rows, "metadata": metadata}

        finally:
            cursor.close()
            connection.close()

    def _get_current_database(self) -> str:
        """獲取當前使用的資料庫"""
        if isinstance(self.config, SQLServerConfig):
            return self.config.database_name
        return "master"

    def _fetch_table_structures(self, database: str) -> List[str]:
        """Fetch CREATE TABLE statements for all tables."""
        import pymssql
        
        if not isinstance(self.config, SQLServerConfig):
            raise RuntimeError("SQLServerAdapter requires a valid SQL Server configuration")
        
        cfg = self.config
        
        connection = pymssql.connect(
            server=cfg.host,
            port=cfg.port,
            user=cfg.username,
            password=cfg.password,
            database=database,
            timeout=cfg.timeout,
            charset=cfg.charset,
        )
        
        cursor = connection.cursor()
        structures: List[str] = []

        try:
            # 獲取所有用戶表
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                  AND TABLE_CATALOG = %s
                  AND TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """, (database,))
            
            tables = cursor.fetchall()

            for (table_name,) in tables:
                table_sql = self._build_table_structure_sync(cursor, database, table_name)
                if table_sql:
                    structures.append(table_sql)

            return structures

        finally:
            cursor.close()
            connection.close()

    def _build_table_structure_sync(self, cursor, database: str, table_name: str) -> str:
        """構建單個表的 CREATE TABLE 語句"""
        # 獲取列信息
        cursor.execute("""
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                IS_NULLABLE,
                COLUMN_DEFAULT
            FROM
                INFORMATION_SCHEMA.COLUMNS
            WHERE
                TABLE_CATALOG = %s
                AND TABLE_NAME = %s
                AND TABLE_SCHEMA = 'dbo'
            ORDER BY
                ORDINAL_POSITION
        """, (database, table_name))
        
        columns = cursor.fetchall()
        if not columns:
            return ""
        
        # 獲取主鍵信息
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_CATALOG = %s
              AND TABLE_NAME = %s
              AND TABLE_SCHEMA = 'dbo'
              AND CONSTRAINT_NAME LIKE 'PK_%'
            ORDER BY ORDINAL_POSITION
        """, (database, table_name))
        
        primary_keys = [row[0] for row in cursor.fetchall()]
        
        # 構建 CREATE TABLE 語句
        col_definitions = []
        
        for col in columns:
            col_name, data_type, char_max_len, num_precision, num_scale, is_nullable, col_default = col
            col_def = self._format_column_definition_sync(
                col_name, data_type, char_max_len, num_precision, num_scale, is_nullable, col_default
            )
            col_definitions.append(col_def)
        
        # 添加主鍵約束
        if primary_keys:
            pk_columns = ", ".join([f"[{pk}]" for pk in primary_keys])
            col_definitions.append(f"    PRIMARY KEY ({pk_columns})")
        
        # 組合最終的 CREATE TABLE 語句
        create_table = f'CREATE TABLE [{table_name}] (\n'
        create_table += ",\n".join(col_definitions)
        create_table += "\n);"
        
        return create_table

    def _format_column_definition_sync(
        self,
        col_name: str,
        data_type: str,
        char_max_len: Optional[int],
        num_precision: Optional[int],
        num_scale: Optional[int],
        is_nullable: str,
        col_default: Optional[str]
    ) -> str:
        """格式化列定義"""
        # SQL Server 資料類型映射
        type_mapping = {
            'int': 'INT',
            'bigint': 'BIGINT',
            'smallint': 'SMALLINT',
            'tinyint': 'TINYINT',
            'varchar': 'VARCHAR',
            'nvarchar': 'NVARCHAR',
            'char': 'CHAR',
            'nchar': 'NCHAR',
            'text': 'TEXT',
            'ntext': 'NTEXT',
            'datetime': 'DATETIME',
            'datetime2': 'DATETIME2',
            'date': 'DATE',
            'time': 'TIME',
            'decimal': 'DECIMAL',
            'numeric': 'NUMERIC',
            'float': 'FLOAT',
            'real': 'REAL',
            'money': 'MONEY',
            'bit': 'BIT',
            'uniqueidentifier': 'UNIQUEIDENTIFIER'
        }
        
        mapped_type = type_mapping.get(data_type.lower(), data_type.upper())
        
        # 處理類型和長度
        if mapped_type in ['VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR'] and char_max_len:
            if char_max_len == -1:
                type_spec = f"{mapped_type}(MAX)"
            else:
                type_spec = f"{mapped_type}({char_max_len})"
        elif mapped_type in ['DECIMAL', 'NUMERIC'] and num_precision:
            if num_scale and num_scale > 0:
                type_spec = f"{mapped_type}({num_precision},{num_scale})"
            else:
                type_spec = f"{mapped_type}({num_precision})"
        else:
            type_spec = mapped_type
        
        # 構建列定義
        col_def = f'    [{col_name}] {type_spec}'
        
        # 添加 NOT NULL 約束
        if is_nullable == 'NO':
            col_def += " NOT NULL"
        
        # 添加預設值
        if col_default:
            col_def += f" DEFAULT {col_default}"
        
        return col_def
