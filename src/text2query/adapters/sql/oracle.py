"""Async Oracle query adapter built on oracledb."""

from __future__ import annotations

import re
import asyncio
import logging
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.connections import BaseConnectionConfig
from ...core.connections.oracle import OracleConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class OracleAdapter(BaseQueryComposer):
    """Oracle adapter that executes SQL asynchronously via oracledb."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "oracle"

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        super().__init__(config)
        self.quote_char = '"'
        self.logger: logging.Logger = get_logger("text2query.adapters.oracle")
        self._pool: Optional[Any] = None
        self._pool_lock: Optional[asyncio.Lock] = None

        if not self.config:
            self.config = OracleConfig(database_name="XE")

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
            self.logger.error("Oracle sql_execution failed: %s", error)
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

    async def validate_query(self, sql_command: str) -> Tuple[bool, str]:
        """Validate Oracle query by EXPLAINing it."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                self._validate_sql_sync,
                sql_command,
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def _validate_sql_sync(self, sql_command: str) -> None:
        """Execute EXPLAIN synchronously in executor for validation."""
        import oracledb
        cfg = self.config
        dsn = cfg.get_dsn()
        user = cfg.username if cfg.database_name.upper() in ['SYSTEM', 'SYS', 'XE'] else cfg.database_name
        connection = oracledb.connect(user=user, password=cfg.password, dsn=dsn)
        cursor = connection.cursor()
        try:
            # Oracle uses EXPLAIN PLAN FOR ...
            cursor.execute(f"EXPLAIN PLAN FOR {sql_command}")
        finally:
            cursor.close()
            connection.close()

    async def _get_schema_info(self, tables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        獲取資料庫 schema 信息（核心方法）

        Args:
            tables: 要獲取的表名列表，None 表示獲取所有表

        Returns:
            Optional[Dict]: 包含資料庫、schema 和表信息的字典
                {
                    "database": "",  # Oracle 沒有傳統的 database 概念
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

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        try:
            schema_data = await loop.run_in_executor(
                None, self._fetch_schema_info_sync, current_schema, tables
            )
            return schema_data
        except Exception as error:
            self.logger.error("Failed to get schema info: %s", error)
            raise

    def _fetch_schema_info_sync(self, schema: str, tables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """在執行器中同步獲取 schema 信息"""
        import oracledb

        if not isinstance(self.config, OracleConfig):
            raise RuntimeError("OracleAdapter requires a valid Oracle configuration")

        cfg = self.config
        dsn = cfg.get_dsn()

        # 確定連接的用戶
        if cfg.database_name.upper() in ['SYSTEM', 'SYS', 'XE']:
            user = cfg.username
        else:
            user = cfg.database_name

        connection = oracledb.connect(
            user=user,
            password=cfg.password,
            dsn=dsn
        )

        cursor = connection.cursor()

        try:
            # 獲取所有用戶表
            cursor.execute("""
                SELECT TABLE_NAME
                FROM USER_TABLES
                ORDER BY TABLE_NAME
            """)

            all_tables = cursor.fetchall()

            if not all_tables:
                return None

            # 過濾表
            if tables:
                tables_set = set(t.upper() for t in tables)  # Oracle 表名通常大寫
                all_tables = [(t,) for (t,) in all_tables if t.upper() in tables_set]
                if not all_tables:
                    return None

            tables_info = []
            for (table_name,) in all_tables:
                # 獲取列信息
                cursor.execute("""
                    SELECT
                        COLUMN_NAME,
                        DATA_TYPE,
                        DATA_LENGTH,
                        DATA_PRECISION,
                        DATA_SCALE,
                        NULLABLE,
                        DATA_DEFAULT
                    FROM
                        USER_TAB_COLUMNS
                    WHERE
                        TABLE_NAME = :table_name
                    ORDER BY
                        COLUMN_ID
                """, {"table_name": table_name})

                columns_info = cursor.fetchall()

                columns = []
                for col in columns_info:
                    col_name, data_type, data_len, data_prec, data_scale, nullable, data_default = col
                    columns.append({
                        "column_name": col_name,
                        "data_type": data_type,
                        "data_length": data_len,
                        "data_precision": data_prec,
                        "data_scale": data_scale,
                        "nullable": nullable,
                        "data_default": data_default
                    })

                # 獲取主鍵信息
                cursor.execute("""
                    SELECT cols.COLUMN_NAME
                    FROM USER_CONSTRAINTS cons
                    JOIN USER_CONS_COLUMNS cols ON cons.CONSTRAINT_NAME = cols.CONSTRAINT_NAME
                    WHERE cons.TABLE_NAME = :table_name
                      AND cons.CONSTRAINT_TYPE = 'P'
                    ORDER BY cols.POSITION
                """, {"table_name": table_name})

                primary_keys = [{"column_name": row[0]} for row in cursor.fetchall()]

                # 構建 CREATE TABLE 語句
                create_sql = self._build_table_structure_from_info(
                    table_name, columns, primary_keys
                )

                tables_info.append({
                    "table_name": table_name,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "create_sql": create_sql
                })

            return {
                "database": "",  # Oracle 沒有傳統的 database 概念
                "schema": schema,
                "tables": tables_info
            }

        finally:
            cursor.close()
            connection.close()

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
                    "type": "oracle",
                    "database": "",  # Oracle 沒有傳統的 database 概念
                    "schema": "schema_name",  # Oracle schema 相當於 database
                    "table": "table_name",
                    "full_path": "schema_name.table_name",
                    "columns": [{"column_name": "type"}, ...]
                }]
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)
            if not schema_data:
                return []

            schema_name = schema_data['schema']
            result = []

            for table_info in schema_data['tables']:
                table_name = table_info['table_name']
                columns_info = table_info['columns']

                # 轉換為簡化的 {column_name: type} 格式
                columns = []
                for col in columns_info:
                    col_name = col['column_name']
                    data_type = col['data_type']
                    data_len = col['data_length']
                    data_prec = col['data_precision']
                    data_scale = col['data_scale']

                    col_type = data_type
                    # 添加長度或精度信息
                    if data_type in ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR'):
                        col_type = f"{data_type}({data_len})"
                    elif data_type == 'NUMBER':
                        if data_prec and data_scale:
                            col_type = f"NUMBER({data_prec},{data_scale})"
                        elif data_prec:
                            col_type = f"NUMBER({data_prec})"
                    elif data_type in ('RAW', 'LONG RAW'):
                        col_type = f"{data_type}({data_len})"

                    columns.append({col_name: col_type})

                result.append({
                    "type": "oracle",
                    "database": "",  # Oracle 沒有傳統的 database 概念
                    "schema": schema_name,
                    "table": table_name,
                    "full_path": f"{schema_name}.{table_name}",
                    "columns": columns
                })

            return result

        except Exception as error:
            self.logger.error("Failed to get structured schema: %s", error)
            return []

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
        
        # UUID 格式轉換
        sql_clean = re.sub(
            r'[\' ]([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})[\' ]',
            r' "\g<1>" ',
            sql_clean
        )
        
        # 中文變體處理 (簡體/繁體)
        pattern = r'[`\"\']([^`\"\']+)[`\"\'] LIKE [<boltignore>`\"\']%([^%]+)%[`\"\']'
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
            "CREATE", "GRANT", "REVOKE", "EXECUTE"
        ]
        
        # 檢查禁用關鍵字
        blocked_keywords = [kw for kw in forbidden_keywords if kw in sql_upper]
        if blocked_keywords:
            error_msg = f"Forbidden keywords detected: {', '.join(blocked_keywords)}"
            return None, error_msg
            
        # 特別檢查系統表：只允許 SELECT 查詢
        system_schemas = ["SYS.", "SYSTEM.", "DBA_", "ALL_", "USER_"]
        for schema_keyword in system_schemas:
            if schema_keyword in sql_upper:
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
        """Apply ROWNUM clause to SELECT queries (Oracle style)."""
        if limit is None or limit <= 0:
            return sql_command

        if not re.search(r"\bSELECT\b", sql_command, re.IGNORECASE):
            return sql_command

        # 檢查是否已有 ROWNUM
        if re.search(r"\bROWNUM\b", sql_command, re.IGNORECASE):
            return sql_command

        # 使用傳統的 ROWNUM 方式（適用所有 Oracle 版本）
        # 移除結尾的分號
        sql_no_semicolon = sql_command.rstrip().rstrip(";")
        
        # 檢查是否有 WHERE 子句
        if re.search(r"\bWHERE\b", sql_no_semicolon, re.IGNORECASE):
            # 在 WHERE 後添加 ROWNUM
            sql_with_limit = re.sub(
                r"(\bWHERE\b)",
                f"\\1 ROWNUM <= {limit} AND",
                sql_no_semicolon,
                count=1,
                flags=re.IGNORECASE
            )
        else:
            # 沒有 WHERE，需要包裹成子查詢
            sql_with_limit = f"SELECT * FROM ({sql_no_semicolon}) WHERE ROWNUM <= {limit}"
        
        return sql_with_limit

    def _normalize_params(self, params: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
        """Normalize query parameters. Overrides base class to ensure tuple return."""
        return super()._normalize_params(params)

    def _extract_command(self, sql_upper: str) -> str:
        """Extract the command keyword from SQL."""
        parts = sql_upper.strip().split()
        return parts[0] if parts else ""

    def _is_resultset_command(self, keyword: str, sql_upper: str) -> bool:
        """Check if command returns a result set."""
        if keyword in {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH"}:
            return True
        return False

    def _check_schema_permission(self, sql_command: str) -> Optional[str]:
        """檢查是否只訪問允許的 schema
        
        Returns:
            str: 錯誤訊息，如果沒有錯誤則返回 None
        """
        current_schema = self._get_current_schema()
        sql_upper = sql_command.upper()
        
        # 尋找所有 schema.table 模式
        schema_table_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(schema_table_pattern, sql_upper)
        
        for schema_name, table_name in matches:
            schema_name = schema_name.upper()
            
            # 跳過表別名（通常是單字母或短名稱）
            if len(schema_name) <= 2:
                continue
                
            # 允許系統 schema
            if schema_name in ["SYS", "SYSTEM"] or schema_name.startswith("DBA_") or schema_name.startswith("ALL_") or schema_name.startswith("USER_"):
                continue
            
            # 檢查是否為配置的 schema
            if schema_name != current_schema.upper():
                error_msg = f"Access denied: Cannot access schema '{schema_name}' with current schema '{current_schema}'"
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

    # Methods _convert_value, _contains_chinese, _convert_chinese_variant 
    # are now inherited from BaseQueryComposer.

    # ============================================================================
    # 6. Oracle 特定的同步執行方法
    # ============================================================================

    def _execute_sql_sync(self, sql_command: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
        """Execute SQL synchronously using oracledb."""
        import oracledb
        
        if not isinstance(self.config, OracleConfig):
            raise RuntimeError("OracleAdapter requires a valid Oracle configuration")
        
        cfg = self.config
        dsn = cfg.get_dsn()
        
        # 確定連接的用戶
        if cfg.database_name.upper() in ['SYSTEM', 'SYS', 'XE']:
            user = cfg.username
        else:
            user = cfg.database_name
        
        connection = oracledb.connect(
            user=user,
            password=cfg.password,
            dsn=dsn
        )
        
        cursor = connection.cursor()
        metadata: Dict[str, Any] = {}

        try:
            sql_upper = sql_command.upper()
            command_keyword = self._extract_command(sql_upper)

            if self._is_resultset_command(command_keyword, sql_upper):
                if params:
                    cursor.execute(sql_command, params)
                else:
                    cursor.execute(sql_command)
                
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description] if cursor.description else []
                result_rows = [self._convert_row(row) for row in rows]
                metadata["rows_returned"] = len(result_rows)
            else:
                if params:
                    cursor.execute(sql_command, params)
                else:
                    cursor.execute(sql_command)
                connection.commit()
                columns = []
                result_rows = []
                metadata["rows_affected"] = cursor.rowcount if cursor.rowcount != -1 else None

            return {"columns": columns, "result": result_rows, "metadata": metadata}

        finally:
            cursor.close()
            connection.close()

    def _get_current_schema(self) -> str:
        """獲取當前使用的 schema"""
        if isinstance(self.config, OracleConfig):
            if self.config.database_name.upper() in ['SYSTEM', 'SYS', 'XE']:
                return self.config.username.upper()
            return self.config.database_name.upper()
        return "SYSTEM"

    def _build_table_structure_from_info(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],
        primary_keys: List[Dict[str, str]]
    ) -> str:
        """從結構化信息構建 CREATE TABLE 語句"""
        if not columns:
            return ""

        col_definitions = []

        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            data_length = col['data_length']
            data_precision = col['data_precision']
            data_scale = col['data_scale']
            nullable = col['nullable']
            data_default = col['data_default']

            col_def = self._format_column_definition_sync(
                col_name, data_type, data_length, data_precision,
                data_scale, nullable, data_default
            )
            col_definitions.append(col_def)

        # 添加主鍵約束
        if primary_keys:
            pk_columns = ", ".join([f'"{pk["column_name"]}"' for pk in primary_keys])
            col_definitions.append(f"    PRIMARY KEY ({pk_columns})")

        # 組合最終的 CREATE TABLE 語句
        create_table = f'CREATE TABLE "{table_name}" (\n'
        create_table += ",\n".join(col_definitions)
        create_table += "\n);"

        return create_table


    def _format_column_definition_sync(
        self,
        col_name: str,
        data_type: str,
        data_length: Optional[int],
        data_precision: Optional[int],
        data_scale: Optional[int],
        nullable: str,
        data_default: Optional[str]
    ) -> str:
        """格式化列定義"""
        # Oracle 資料類型映射
        type_mapping = {
            'NUMBER': 'NUMBER',
            'VARCHAR2': 'VARCHAR2',
            'CHAR': 'CHAR',
            'NVARCHAR2': 'NVARCHAR2',
            'NCHAR': 'NCHAR',
            'DATE': 'DATE',
            'TIMESTAMP': 'TIMESTAMP',
            'CLOB': 'CLOB',
            'BLOB': 'BLOB',
            'RAW': 'RAW',
            'LONG': 'LONG'
        }
        
        mapped_type = type_mapping.get(data_type.upper(), data_type.upper())
        
        # 處理類型和長度
        if mapped_type in ['VARCHAR2', 'NVARCHAR2', 'CHAR', 'NCHAR'] and data_length:
            type_spec = f"{mapped_type}({data_length})"
        elif mapped_type == 'NUMBER':
            if data_precision and data_scale is not None:
                if data_scale > 0:
                    type_spec = f"{mapped_type}({data_precision},{data_scale})"
                else:
                    type_spec = f"{mapped_type}({data_precision})"
            elif data_precision:
                type_spec = f"{mapped_type}({data_precision})"
            else:
                type_spec = mapped_type
        else:
            type_spec = mapped_type
        
        # 構建列定義
        col_def = f'    "{col_name}" {type_spec}'
        
        # 添加 NOT NULL 約束
        if nullable == 'N':
            col_def += " NOT NULL"
        
        # 添加預設值
        if data_default:
            default_str = data_default.strip()
            col_def += f" DEFAULT {default_str}"
        
        return col_def
