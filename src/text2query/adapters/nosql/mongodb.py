"""MongoDB query adapter bridging to legacy execution logic."""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from ...core.connections import BaseConnectionConfig
from ...core.connections.mongodb import MongoDBConfig
from ...core.query_composer import BaseQueryComposer
from ...utils.helpers import get_logger


class MongoDBAdapter(BaseQueryComposer):
    """MongoDB adapter that executes commands asynchronously via motor."""

    # ============================================================================
    # 1. 初始化和基本屬性
    # ============================================================================

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "mongodb"

    def __init__(self, config: Optional[BaseConnectionConfig] = None):
        if config is None:
            config = MongoDBConfig(database_name="default")
        super().__init__(config)
        self.logger: logging.Logger = get_logger("text2query.adapters.mongodb")
        self._client = None
        self._db = None

    # ============================================================================
    # 2. 公共 API 方法（核心功能）
    # ============================================================================

    async def sql_execution(
        self,
        command: Union[str, Dict[str, Any]],
        params: Optional[Sequence[Any]] = None,
        safe: bool = True,
        limit: Optional[int] = 1000,
    ) -> Dict[str, Any]:
        """
        超級安全的 MongoDB 命令執行器
        - 永遠不會 crash
        - 支援多種格式的命令
        - 自動轉換類型
        - 失敗時自動診斷

        Args:
            command: MongoDB 命令（字串或字典格式）
            params: Not used for MongoDB (kept for interface compatibility)
            safe: Whether to apply safety checks
            limit: Maximum number of results to return

        Returns:
            統一格式的結果字典
        """
        
        start_time = time.time()
        
        try:
            # 確保連接存在
            if self._client is None or self._db is None:
                await self._get_connection()
            
            # 1. 智能解析命令
            parsed = self._safe_parse_command(command)
            if not parsed["success"]:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[parsed["error"]]],
                    'sql_command': str(command)[:100],
                    'ori_sql_command': str(command)
                }
            
            # 2. 執行已解析的命令
            result = await self._safe_execute_parsed(parsed["data"], str(command))
            
            # 3. 如果執行失敗且是 aggregate 命令，進行詳細診斷
            if not result['success'] and parsed["data"].get('operation') == 'aggregate':
                self.logger.info("執行失敗，開始詳細診斷...")
                diagnostic_result = await self._diagnose_aggregate_failure(command, parsed["data"])
                
                # 如果診斷成功執行了查詢，返回診斷結果
                if diagnostic_result.get('success'):
                    execution_time = (time.time() - start_time) * 1000
                    diagnostic_result['execution_time'] = execution_time
                    return diagnostic_result
                else:
                    # 將診斷信息添加到錯誤結果中
                    result['diagnostic_info'] = diagnostic_result.get('diagnostic_info', '')
            
            execution_time = (time.time() - start_time) * 1000
            result['execution_time'] = execution_time
            self.logger.info(f"命令執行{'成功' if result['success'] else '失敗'}，耗時: {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"sql_execution 錯誤: {e}")
            self.logger.error(traceback.format_exc())
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'success': False,
                'columns': ['error', 'detail'],
                'result': [[str(e), traceback.format_exc()]],
                'sql_command': str(command)[:100],
                'ori_sql_command': str(command),
                'execution_time': execution_time
            }

    async def get_schema_str(self, tables: Optional[List[str]] = None) -> str:
        """
        獲取 MongoDB 資料庫結構字符串

        Args:
            tables: 要獲取的集合名稱列表，None 表示獲取所有集合

        Returns:
            str: 包含所有集合結構的描述字符串
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)

            if schema_data is None:
                return "MongoDB database has no collections"

            database_name = schema_data['database']
            collections_info = schema_data['collections']

            struct_parts = []
            struct_parts.append(f"MongoDB Connection:")
            struct_parts.append(f"Database: {database_name}")
            struct_parts.append("=" * 50)

            for collection_info in collections_info:
                collection_name = collection_info['collection_name']
                count = collection_info['document_count']
                fields = collection_info['fields']
                sample_docs = collection_info['sample_docs']
                error = collection_info.get('error')

                if error:
                    struct_parts.append(f"\nCollection: {collection_name}")
                    struct_parts.append(f"Analysis failed: {error}")
                    struct_parts.append("-" * 30)
                    continue

                struct_parts.append(f"\nCollection: {collection_name}")
                struct_parts.append(f"Document count: {count}")

                if fields:
                    struct_parts.append("Fields:")
                    for field_name, field_type in sorted(fields.items()):
                        struct_parts.append(f"  - {field_name}: {field_type}")

                    if sample_docs:
                        struct_parts.append("Sample data:")
                        for i, doc in enumerate(sample_docs[:2], 1):
                            serialized_doc = self._serialize_document_for_display(doc)
                            struct_parts.append(f"  Example {i}: {serialized_doc}")

                struct_parts.append("-" * 30)

            return "\n".join(struct_parts)

        except Exception as error:
            self.logger.error("Failed to get MongoDB struct string: %s", error)
            return f"Error retrieving MongoDB structure: {str(error)}"

    async def get_schema_list(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        獲取結構化的資料庫 schema 信息

        Args:
            tables: 要獲取的集合名稱列表，None 表示獲取所有集合

        Returns:
            List[Dict]: 包含每個集合的結構化信息
                [{
                    "type": "mongodb",
                    "database": "db_name",
                    "schema": "",  # MongoDB 沒有 schema
                    "table": "collection_name",  # collection 對應 table
                    "full_path": "db_name.collection_name",
                    "columns": [{"field_name": "type"}, ...]
                }]
        """
        try:
            schema_data = await self._get_schema_info(tables=tables)

            if schema_data is None:
                return []

            database_name = schema_data['database']
            collections_info = schema_data['collections']

            result = []
            for collection_info in collections_info:
                if collection_info.get('error'):
                    continue

                collection_name = collection_info['collection_name']
                fields = collection_info['fields']

                columns = [{field_name: field_type} for field_name, field_type in sorted(fields.items())]

                result.append({
                    "type": "mongodb",
                    "database": database_name,
                    "schema": "",  # MongoDB 沒有 schema 概念
                    "table": collection_name,
                    "full_path": f"{database_name}.{collection_name}",
                    "columns": columns
                })

            return result

        except Exception as error:
            self.logger.error("Failed to get structured schema: %s", error)
            return []

    async def _get_schema_info(self, tables: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        分析所有集合的 schema 信息（核心方法）

        Args:
            tables: 要獲取的集合名稱列表，None 表示獲取所有集合

        Returns:
            Optional[Dict]: 包含資料庫和集合信息的字典
                {
                    "database": "db_name",
                    "collections": [
                        {
                            "collection_name": "name",
                            "document_count": 100,
                            "fields": {"field_name": "type", ...},
                            "sample_docs": [{...}, ...],
                            "error": "error_message"  # 僅在錯誤時存在
                        },
                        ...
                    ]
                }
            如果沒有集合則返回 None
        """
        try:
            if self._client is None or self._db is None:
                await self._get_connection()

            connection_info = self._build_connection_info()
            database_name = connection_info['database']
            collections = connection_info.get('collections', [])

            if not collections:
                # 如果沒有指定集合，獲取所有集合
                collections = await self._db.list_collection_names()
                if not collections:
                    return None

            # 過濾集合
            if tables:
                tables_set = set(tables)
                collections = [c for c in collections if c in tables_set]
                if not collections:
                    return None

            collections_info = []

            for collection_name in collections:
                collection_data = {
                    "collection_name": collection_name,
                    "document_count": 0,
                    "fields": {},
                    "sample_docs": []
                }

                try:
                    # 獲取集合統計
                    count = await self._db[collection_name].count_documents({})
                    collection_data["document_count"] = count

                    if count > 0:
                        # 獲取樣本文檔來分析結構
                        cursor = self._db[collection_name].find({}).limit(3)
                        sample_docs = await cursor.to_list(length=3)
                        collection_data["sample_docs"] = sample_docs

                        if sample_docs:
                            # 收集所有字段並分析類型
                            fields = {}
                            all_fields = set()
                            for doc in sample_docs:
                                all_fields.update(doc.keys())

                            for field in all_fields:
                                # 分析字段類型
                                field_type = "unknown"
                                for doc in sample_docs:
                                    if field in doc and doc[field] is not None:
                                        value = doc[field]
                                        if isinstance(value, str):
                                            field_type = "string"
                                        elif isinstance(value, (int, float)):
                                            field_type = "number"
                                        elif isinstance(value, bool):
                                            field_type = "boolean"
                                        elif isinstance(value, list):
                                            field_type = "array"
                                        elif isinstance(value, dict):
                                            field_type = "object"
                                        elif isinstance(value, datetime):
                                            field_type = "date"
                                        break

                                fields[field] = field_type

                            collection_data["fields"] = fields

                except Exception as collection_error:
                    self.logger.warning(f"Failed to analyze collection {collection_name}: {collection_error}")
                    collection_data["error"] = str(collection_error)

                collections_info.append(collection_data)

            return {
                "database": database_name,
                "collections": collections_info
            }

        except Exception as error:
            self.logger.error("Failed to analyze collections schema: %s", error)
            raise

    async def close_conn(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self.logger.info("MongoDB connection closed")

    # ============================================================================
    # 3. 連線管理
    # ============================================================================

    async def _get_connection(self):
        """Get or create MongoDB connection."""
        if self._client is not None and self._db is not None:
            return self._client
        
        try:
            connection_info = self._build_connection_info()
            connection_string = connection_info['connection_string']
            database_name = connection_info['database']
            
            self.logger.info(f"Connecting to MongoDB: {database_name}")
            
            # 動態導入 motor
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError as exc:
                raise RuntimeError("MongoDB adapter requires motor package: pip install motor") from exc
            
            # 建立 motor 客戶端
            self._client = AsyncIOMotorClient(
                connection_string,
                maxPoolSize=10,
                minPoolSize=1,
                connectTimeoutMS=10000,
                serverSelectionTimeoutMS=10000
            )
            
            # 測試連接
            await self._client.admin.command('ping')
            self.logger.info("✅ MongoDB connection successful")
            
            # 獲取資料庫
            self._db = self._client[database_name]
            
            return self._client
            
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {e}")
            raise RuntimeError(f"MongoDB connection failed: {e}")

    def _build_connection_info(self) -> Dict[str, Any]:
        """Build connection information from config."""
        if not isinstance(self.config, BaseConnectionConfig):
            raise RuntimeError("MongoDBAdapter requires a valid MongoDB configuration")

        if isinstance(self.config, MongoDBConfig):
            cfg = self.config
        else:
            cfg = MongoDBConfig(
                database_name=self.config.database_name,
                timeout=self.config.timeout,
                extra_params=self.config.extra_params,
            )
            for attr in ["host", "port", "username", "password", "auth_database", "replica_set", "ssl", "ssl_cert_reqs", "read_preference", "max_pool_size"]:
                if hasattr(self.config, attr):
                    setattr(cfg, attr, getattr(self.config, attr))

        connection_info = {
            "connection_string": cfg.to_connection_string(),
            "database": cfg.database_name,
            "collections": cfg.extra_params.get("collections", []),
        }
        return connection_info

    # ============================================================================
    # 4. 診斷方法
    # ============================================================================

    async def _diagnose_aggregate_failure(self, original_command: str, parsed_data: Dict) -> Dict[str, Any]:
        """
        診斷聚合命令執行失敗的原因

        Args:
            original_command: 原始命令字串
            parsed_data: 解析後的命令數據

        Returns:
            診斷結果的字典格式
        """
        
        diagnostic_info = []
        diagnostic_info.append(f"診斷聚合命令失敗原因")
        diagnostic_info.append(f"原始命令: {original_command[:200]}...")
        
        try:
            collection_name = parsed_data.get('collection')
            pipeline = parsed_data.get('pipeline', [])
            
            diagnostic_info.append(f"集合: {collection_name}")
            diagnostic_info.append(f"Pipeline 階段數: {len(pipeline)}")
            
            # 檢查集合是否存在
            if collection_name:
                existing_collections = await self._db.list_collection_names()
                if collection_name not in existing_collections:
                    diagnostic_info.append(f"✗ 集合 {collection_name} 不存在")
                    return {
                        'success': False,
                        'columns': ['error'],
                        'result': [[f'集合 {collection_name} 不存在']],
                        'sql_command': str(parsed_data),
                        'ori_sql_command': original_command,
                        'diagnostic_info': '\n'.join(diagnostic_info)
                    }
                else:
                    diagnostic_info.append(f"✓ 集合存在")
            
            # 嘗試直接執行 pipeline
            try:
                collection = self._db[collection_name]
                
                # 逐步測試 pipeline
                for i, stage in enumerate(pipeline):
                    test_pipeline = pipeline[:i+1]
                    diagnostic_info.append(f"\n測試前 {i+1} 個階段:")
                    diagnostic_info.append(f"  {json.dumps(stage, default=str)[:100]}")
                    
                    try:
                        cursor = collection.aggregate(test_pipeline)
                        test_results = await cursor.to_list(length=1)
                        diagnostic_info.append(f"  ✓ 階段 {i+1} 執行成功")
                        
                        if i == len(pipeline) - 1:  # 最後一個階段
                            # 如果所有階段都成功，執行完整查詢
                            cursor = collection.aggregate(pipeline)
                            results = await cursor.to_list(length=None)
                            
                            if results:
                                # 收集欄位
                                all_fields = set()
                                for doc in results:
                                    if doc is not None:
                                        all_fields.update(doc.keys())
                                
                                columns = sorted(list(all_fields))
                                
                                # 轉換為表格
                                table_data = []
                                for doc in results:
                                    if doc is not None:
                                        row = []
                                        for col in columns:
                                            value = doc.get(col)
                                            row.append(self._serialize_value(value))
                                        table_data.append(row)
                                
                                diagnostic_info.append(f"\n✓ 診斷成功，查詢返回 {len(results)} 筆結果")
                                
                                return {
                                    'success': True,
                                    'columns': columns,
                                    'result': table_data,
                                    'sql_command': f"db.{collection_name}.aggregate({json.dumps(pipeline, default=str)})",
                                    'ori_sql_command': original_command,
                                    'diagnostic_info': '\n'.join(diagnostic_info)
                                }
                            else:
                                return {
                                    'success': True,
                                    'columns': [],
                                    'result': [],
                                    'sql_command': f"db.{collection_name}.aggregate({json.dumps(pipeline, default=str)})",
                                    'ori_sql_command': original_command,
                                    'diagnostic_info': '\n'.join(diagnostic_info)
                                }
                                
                    except Exception as stage_error:
                        diagnostic_info.append(f"  ✗ 階段 {i+1} 失敗: {str(stage_error)[:100]}")
                        
                        # 嘗試修復常見問題
                        if '$year' in str(stage) or '$month' in str(stage):
                            diagnostic_info.append(f"  檢測到日期操作，檢查 StartTime 欄位...")
                            
                            # 檢查 StartTime 欄位的類型
                            sample = await collection.find_one({'StartTime': {'$exists': True}})
                            if sample:
                                start_time_value = sample.get('StartTime')
                                diagnostic_info.append(f"  StartTime 類型: {type(start_time_value)}")
                                if not isinstance(start_time_value, datetime):
                                    diagnostic_info.append(f"  ✗ StartTime 不是日期類型，無法使用日期操作符")
                        
                        # 返回錯誤
                        return {
                            'success': False,
                            'columns': ['error', 'failed_stage', 'stage_index'],
                            'result': [[str(stage_error), json.dumps(stage, default=str), i+1]],
                            'sql_command': f"db.{collection_name}.aggregate({json.dumps(pipeline, default=str)})",
                            'ori_sql_command': original_command,
                            'diagnostic_info': '\n'.join(diagnostic_info)
                        }
                
            except Exception as exec_error:
                diagnostic_info.append(f"\n執行錯誤: {str(exec_error)}")
                
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[str(exec_error)]],
                    'sql_command': str(parsed_data),
                    'ori_sql_command': original_command,
                    'diagnostic_info': '\n'.join(diagnostic_info)
                }
                
        except Exception as diag_error:
            diagnostic_info.append(f"\n診斷過程錯誤: {str(diag_error)}")
            
            return {
                'success': False,
                'columns': ['error'],
                'result': [[f'診斷失敗: {str(diag_error)}']],
                'sql_command': str(parsed_data),
                'ori_sql_command': original_command,
                'diagnostic_info': '\n'.join(diagnostic_info)
            }

    # ============================================================================
    # 5. 命令解析和安全檢查
    # ============================================================================

    def _safe_parse_command(self, command: Union[str, Dict]) -> Dict[str, Any]:
        """智能解析各種格式的命令"""
        try:
            # 如果已經是字典
            if isinstance(command, dict):
                return {
                    "success": True,
                    "data": command,
                    "command_type": "json"
                }
            
            # 如果是字串
            if not isinstance(command, str):
                return {
                    "success": False,
                    "error": f"Unsupported command type: {type(command)}",
                    "command_type": "unknown"
                }
            
            command = command.strip()
            
            # 嘗試解析為 JSON
            if command.startswith('{') or command.startswith('['):
                try:
                    parsed = json.loads(command)
                    return {
                        "success": True,
                        "data": parsed,
                        "command_type": "json"
                    }
                except json.JSONDecodeError:
                    pass
            
            # 嘗試解析為 MongoDB 命令字串
            if command.startswith('db.'):
                parsed = self._safe_parse_mongo_string(command)
                if parsed["success"]:
                    return parsed
            
            return {
                "success": False,
                "error": "Unrecognized command format",
                "command_type": "unknown"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Parse error: {str(e)}",
                "command_type": "unknown"
            }

    def _safe_parse_mongo_string(self, command: str) -> Dict[str, Any]:
        """安全解析 MongoDB 命令字串"""
        try:
            # 找到 db.collection.method(...) 的結構
            main_command_match = re.match(r'db\.(\w+)\.(\w+)\((.*)\)', command)
            
            if not main_command_match:
                return {
                    "success": False,
                    "error": "Cannot parse MongoDB command format",
                    "command_type": "mongo_string"
                }
            
            collection = main_command_match.group(1)
            method = main_command_match.group(2)
            params_str = main_command_match.group(3)
            
            # 解析主命令參數
            if method in ['find', 'findOne']:
                parsed_params = self._parse_find_params(params_str)
                json_command = {
                    "operation": method,
                    "collection": collection,
                    "filter": parsed_params.get('filter', {}),
                    "projection": parsed_params.get('projection')
                }
            elif method == 'aggregate':
                params = self._parse_aggregate_pipeline(params_str)
                json_command = {
                    "operation": "aggregate",
                    "collection": collection,
                    "pipeline": params if isinstance(params, list) else [params] if params else []
                }
            elif method in ['countDocuments', 'count']:
                params = self._safe_parse_params(params_str)
                json_command = {
                    "operation": method,
                    "collection": collection,
                    "filter": params if params else {}
                }
            elif method == 'distinct':
                if params_str.startswith('"') or params_str.startswith("'"):
                    params = params_str.strip('"\'')
                    json_command = {
                        "operation": "distinct",
                        "collection": collection,
                        "field": params
                    }
                else:
                    params = self._safe_parse_params(params_str)
                    json_command = {
                        "operation": "distinct",
                        "collection": collection,
                        "params": params
                    }
            else:
                params = self._safe_parse_params(params_str)
                json_command = {
                    "operation": method,
                    "collection": collection,
                    "params": params
                }
            
            # 檢查是否有鏈式方法調用
            main_command_pattern = rf'db\.{re.escape(collection)}\.{re.escape(method)}\('
            main_match = re.search(main_command_pattern, command)
            
            if main_match:
                # 從主命令開始位置，找到對應的右括號
                start_pos = main_match.end() - 1
                paren_count = 0
                main_command_end = start_pos
                
                for i in range(start_pos, len(command)):
                    if command[i] == '(':
                        paren_count += 1
                    elif command[i] == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            main_command_end = i + 1
                            break
                
                remaining_command = command[main_command_end:]
                
                if remaining_command:
                    # 解析鏈式方法
                    chain_pattern = r'\.(\w+)\(([^)]*(?:\([^)]*\)[^)]*)*)\)'
                    chains = re.findall(chain_pattern, remaining_command)
                    
                    if chains:
                        json_command["options"] = {}
                        for chain_method, chain_params in chains:
                            if chain_method in ['sort', 'limit', 'skip', 'hint', 'projection']:
                                if chain_method == 'limit' or chain_method == 'skip':
                                    try:
                                        json_command["options"][chain_method] = int(chain_params.strip())
                                    except:
                                        json_command["options"][chain_method] = 0
                                else:
                                    parsed_chain_params = self._safe_parse_params(chain_params)
                                    json_command["options"][chain_method] = parsed_chain_params
            
            return {
                "success": True,
                "data": json_command,
                "command_type": "mongo_string"
            }
            
        except Exception as e:
            self.logger.error(f"Parse MongoDB command error: {str(e)}")
            return {
                "success": False,
                "error": f"Parse error: {str(e)}",
                "command_type": "mongo_string"
            }

    def _is_query_safe(self, query: Any) -> bool:
        """檢查查詢是否安全"""
        forbidden_operators = [
            '$where', '$function', '$accumulator', '$eval', 
            'mapReduce', 'system.', 'admin.', 'config.', 'local.',
            '$merge', '$out', 'dropDatabase', 'dropCollection',
            'renameCollection', 'createUser', 'dropUser',
            'grantRole', 'revokeRole', 'shutdown', 'fsync', 'replSetReconfig'
        ]
        
        def check_forbidden(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(forbidden in str(key) for forbidden in forbidden_operators):
                        self.logger.warning(f"Detected forbidden operation: {key}")
                        return False
                    if not check_forbidden(value):
                        return False
            elif isinstance(obj, list):
                for item in obj:
                    if not check_forbidden(item):
                        return False
            elif isinstance(obj, str):
                if any(forbidden in obj for forbidden in forbidden_operators):
                    self.logger.warning(f"Detected forbidden operation in string: {obj}")
                    return False
            return True
        
        return check_forbidden(query)

    # ============================================================================
    # 5. 命令執行方法
    # ============================================================================

    async def _safe_execute_parsed(self, command: Dict, ori_command: str) -> Dict:
        """執行已解析的命令 - 增強錯誤處理"""
        
        try:
            operation = command.get('operation', '').lower()
            collection_name = command.get('collection')
            
            if not collection_name:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['未指定集合名稱']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            # 檢查集合權限
            connection_info = self._build_connection_info()
            allowed_collections = connection_info.get('collections', [])
            if allowed_collections and collection_name not in allowed_collections:
                existing_collections = await self._db.list_collection_names()
                if collection_name not in existing_collections:
                    error_msg = f'集合 "{collection_name}" 不存在'
                else:
                    error_msg = f'集合 "{collection_name}" 存在但不在允許列表中'
                
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[error_msg]],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            # 檢查 collection 是否為 None
            if self._db is None:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['資料庫連接未建立']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            collection = self._db[collection_name]
            
            if collection is None:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[f'無法獲取集合: {collection_name}']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            # 根據操作類型執行
            if operation in ['find', 'findone', 'find_one']:
                return await self._safe_execute_find(collection, command, ori_command)
            elif operation in ['countdocuments', 'count_documents', 'count']:
                return await self._safe_execute_count(collection, command, ori_command)
            elif operation == 'aggregate':
                return await self._safe_execute_aggregate(collection, command, ori_command)
            elif operation == 'distinct':
                return await self._safe_execute_distinct(collection, command, ori_command)
            else:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[f'不支援的操作: {operation}']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
                
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"_safe_execute_parsed 錯誤: {e}")
            self.logger.error(error_trace)
            
            return {
                'success': False,
                'columns': ['error', 'traceback'],
                'result': [[str(e), error_trace]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    async def _safe_execute_find(self, collection, command: Dict, ori_command: str) -> Dict:
        """安全執行查找操作 - 支援 projection"""
        try:
            filter_doc = command.get('filter', command.get('params', {}))
            projection = command.get('projection')
            
            # 轉換特殊類型
            filter_doc = self._safe_convert_types(filter_doc)
            if projection:
                projection = self._safe_convert_types(projection)
            
            # 安全檢查
            if not self._is_query_safe(filter_doc):
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['查詢包含不允許的操作']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            operation = command.get('operation', '').lower()
            options = command.get('options', {})
            
            self.logger.info(f"執行 {operation}: filter={filter_doc}, projection={projection}")
            
            if operation in ['findone', 'find_one']:
                result = await collection.find_one(filter_doc, projection)
                if result:
                    columns = list(result.keys())
                    row = [self._serialize_value(result[col]) for col in columns]
                    return {
                        'success': True,
                        'columns': columns,
                        'result': [row],
                        'sql_command': f"db.{command['collection']}.findOne({json.dumps(filter_doc, default=str)}, {json.dumps(projection, default=str) if projection else ''})",
                        'ori_sql_command': ori_command
                    }
                else:
                    return {
                        'success': True,
                        'columns': [],
                        'result': [],
                        'sql_command': f"db.{command['collection']}.findOne({json.dumps(filter_doc, default=str)}, {json.dumps(projection, default=str) if projection else ''})",
                        'ori_sql_command': ori_command
                    }
            else:
                # find 操作
                cursor = collection.find(filter_doc, projection)
                
                # 應用選項
                if 'sort' in options:
                    cursor = cursor.sort(options['sort'])
                if 'skip' in options:
                    cursor = cursor.skip(options['skip'])
                if 'limit' in options:
                    cursor = cursor.limit(min(options['limit'], 10000))
                else:
                    cursor = cursor.limit(1000)
                
                results = await cursor.to_list(length=None)
                
                if results:
                    # 收集所有欄位
                    all_fields = set()
                    for doc in results:
                        all_fields.update(doc.keys())
                    columns = sorted(list(all_fields))
                    
                    # 轉換為表格
                    table_data = []
                    for doc in results:
                        row = [self._serialize_value(doc.get(col)) for col in columns]
                        table_data.append(row)
                    
                    # 構建包含鏈式方法的 sql_command
                    sql_command = f"db.{command['collection']}.find({json.dumps(filter_doc, default=str)}"
                    if projection:
                        sql_command += f", {json.dumps(projection, default=str)}"
                    
                    # 添加鏈式方法
                    if options:
                        if 'sort' in options:
                            sql_command += f".sort({json.dumps(options['sort'], default=str)})"
                        if 'skip' in options:
                            sql_command += f".skip({options['skip']})"
                        if 'limit' in options:
                            sql_command += f".limit({options['limit']})"
                    
                    return {
                        'success': True,
                        'columns': columns,
                        'result': table_data,
                        'sql_command': sql_command,
                        'ori_sql_command': ori_command
                    }
                else:
                    # 構建包含鏈式方法的 sql_command（空結果情況）
                    sql_command = f"db.{command['collection']}.find({json.dumps(filter_doc, default=str)}"
                    if projection:
                        sql_command += f", {json.dumps(projection, default=str)}"
                    
                    # 添加鏈式方法
                    if options:
                        if 'sort' in options:
                            sql_command += f".sort({json.dumps(options['sort'], default=str)})"
                        if 'skip' in options:
                            sql_command += f".skip({options['skip']})"
                        if 'limit' in options:
                            sql_command += f".limit({options['limit']})"
                    
                    return {
                        'success': True,
                        'columns': [],
                        'result': [],
                        'sql_command': sql_command,
                        'ori_sql_command': ori_command
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'columns': ['error'],
                'result': [[str(e)]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    async def _safe_execute_count(self, collection, command: Dict, ori_command: str) -> Dict:
        """安全執行計數操作"""
        try:
            filter_doc = command.get('filter', command.get('params', {}))
            filter_doc = self._safe_convert_types(filter_doc)
            
            # 安全檢查
            if not self._is_query_safe(filter_doc):
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['查詢包含不允許的操作']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            count = await collection.count_documents(filter_doc)
            
            return {
                'success': True,
                'columns': ['count'],
                'result': [[count]],
                'sql_command': f"db.{command['collection']}.countDocuments({json.dumps(filter_doc, default=str)})",
                'ori_sql_command': ori_command
            }
            
        except Exception as e:
            return {
                'success': False,
                'columns': ['error'],
                'result': [[str(e)]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    async def _safe_execute_aggregate(self, collection, command: Dict, ori_command: str) -> Dict:
        """安全執行聚合操作 - 支援自動攤平巢狀結果"""
        try:
            pipeline = command.get('pipeline', command.get('params', []))
            
            if not isinstance(pipeline, list):
                pipeline = [pipeline] if pipeline else []
            
            # 轉換特殊類型
            pipeline = self._safe_convert_types(pipeline)
            
            # 安全檢查
            if not self._is_query_safe(pipeline):
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['查詢包含不允許的操作']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            self.logger.info(f"執行聚合 pipeline: {json.dumps(pipeline, default=str)[:500]}")
            
            # 執行聚合
            try:
                cursor = collection.aggregate(pipeline)
                results = await cursor.to_list(length=None)
            except Exception as agg_error:
                self.logger.error(f"聚合執行錯誤: {agg_error}")
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[str(agg_error)]],
                    'sql_command': f"db.{command['collection']}.aggregate({json.dumps(pipeline, default=str)})",
                    'ori_sql_command': ori_command
                }
            
            self.logger.info(f"聚合返回 {len(results)} 筆結果")
            
            if results:
                # 攤平巢狀結構
                flattened_results = []
                for doc in results:
                    if doc is not None:
                        flattened_doc = self._flatten_document(doc)
                        flattened_results.append(flattened_doc)
                
                # 收集所有欄位
                all_fields = set()
                for doc in flattened_results:
                    all_fields.update(doc.keys())
                
                if not all_fields:
                    return {
                        'success': True,
                        'columns': [],
                        'result': [],
                        'sql_command': f"db.{command['collection']}.aggregate({json.dumps(pipeline, default=str)})",
                        'ori_sql_command': ori_command
                    }
                
                # 排序欄位，讓 _id 相關的欄位在前面
                columns = sorted(list(all_fields), key=lambda x: (not x.startswith('_id'), x))
                
                # 轉換為表格
                table_data = []
                for doc in flattened_results:
                    row = []
                    for col in columns:
                        value = doc.get(col)
                        row.append(self._serialize_value(value))
                    table_data.append(row)
                
                return {
                    'success': True,
                    'columns': columns,
                    'result': table_data,
                    'sql_command': f"db.{command['collection']}.aggregate({json.dumps(pipeline, default=str)})",
                    'ori_sql_command': ori_command
                }
            else:
                return {
                    'success': True,
                    'columns': [],
                    'result': [],
                    'sql_command': f"db.{command['collection']}.aggregate({json.dumps(pipeline, default=str)})",
                    'ori_sql_command': ori_command
                }
                
        except Exception as e:
            self.logger.error(f"_safe_execute_aggregate error: {e}")
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'columns': ['error', 'traceback'],
                'result': [[str(e), traceback.format_exc()]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    async def _safe_execute_distinct(self, collection, command: Dict, ori_command: str) -> Dict:
        """安全執行去重操作"""
        try:
            field = command.get('field', command.get('params'))
            filter_doc = command.get('filter', {})
            
            if not field:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['distinct 操作需要指定字段名']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            filter_doc = self._safe_convert_types(filter_doc)
            
            # 安全檢查
            if not self._is_query_safe(filter_doc):
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['查詢包含不允許的操作']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            distinct_values = await collection.distinct(field, filter_doc)
            
            # 轉換為表格格式
            return {
                'success': True,
                'columns': [field],
                'result': [[self._serialize_value(value)] for value in distinct_values],
                'sql_command': f"db.{command['collection']}.distinct(\"{field}\")",
                'ori_sql_command': ori_command
            }
            
        except Exception as e:
            return {
                'success': False,
                'columns': ['error'],
                'result': [[str(e)]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    # ============================================================================
    # 6. 數據轉換和格式化
    # ============================================================================

    def _serialize_value(self, value: Any) -> Any:
        """序列化值以確保可以安全返回"""
        if value is None:
            return None
        
        # 動態導入 bson 相關類型
        try:
            from bson import ObjectId
            if isinstance(value, ObjectId):
                return str(value)
        except ImportError:
            pass
        
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        elif isinstance(value, dict):
            return json.dumps(value, default=str)
        elif isinstance(value, list):
            return json.dumps(value, default=str)
        else:
            return value

    def _serialize_document_for_display(self, doc: Dict) -> str:
        """序列化文檔以便顯示"""
        try:
            # 限制顯示長度
            serialized = json.dumps(doc, default=str, ensure_ascii=False)
            if len(serialized) > 200:
                return serialized[:200] + "..."
            return serialized
        except Exception:
            return str(doc)[:200]

    def _flatten_document(self, doc: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """攤平巢狀的文檔結構"""
        items = []
        
        for key, value in doc.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # 遞迴處理巢狀字典
                items.extend(self._flatten_document(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                # 對於列表，保持原樣或根據需要處理
                if value and isinstance(value[0], dict):
                    # 如果是字典列表，轉換為 JSON 字串
                    items.append((new_key, json.dumps(value, default=str)))
                else:
                    items.append((new_key, value))
            else:
                items.append((new_key, value))
        
        return dict(items)

    # ============================================================================
    # 7. 參數解析輔助方法
    # ============================================================================

    def _parse_find_params(self, params_str: str) -> Dict:
        """
        解析 find 方法的參數（可能有 filter 和 projection 兩個參數）
        修正版：正確處理參數邊界
        """
        if not params_str:
            return {"filter": {}}

        try:
            # 方法1：嘗試直接解析整個字串作為 JSON 陣列
            try:
                # 如果整個參數是 [filter, projection] 格式
                params_list = json.loads(f"[{params_str}]")
                if len(params_list) >= 2:
                    return {
                        "filter": params_list[0],
                        "projection": params_list[1]
                    }
                elif len(params_list) == 1:
                    return {
                        "filter": params_list[0],
                        "projection": None
                    }
            except:
                pass

            # 方法2：使用括號匹配來分割參數
            depth = 0
            first_param_end = -1
            in_string = False
            escape_next = False

            for i, char in enumerate(params_str):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char in '{[(':
                        depth += 1
                    elif char in '}])':
                        depth -= 1
                        if depth == 0:
                            # 找到第一個參數的結尾
                            # 檢查後面是否還有參數
                            remaining = params_str[i+1:].strip()
                            if remaining and remaining[0] == ',':
                                first_param_end = i + 1
                                break

            if first_param_end > 0:
                # 有兩個參數
                first_param = params_str[:first_param_end].strip()

                # 找到第二個參數的結束位置
                remaining = params_str[first_param_end:].strip()
                if remaining.startswith(','):
                    remaining = remaining[1:].strip()

                # 找第二個參數的結束（可能後面還有 .sort() 等）
                second_param_end = -1
                depth = 0
                in_string = False
                escape_next = False

                for i, char in enumerate(remaining):
                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char in '{[(':
                            depth += 1
                        elif char in '}])':
                            depth -= 1
                            if depth == 0:
                                second_param_end = i + 1
                                break

                if second_param_end > 0:
                    second_param = remaining[:second_param_end]
                else:
                    second_param = remaining

                # 解析兩個參數
                filter_obj = self._safe_parse_params(first_param)
                projection_obj = self._safe_parse_params(second_param)

                return {
                    "filter": filter_obj,
                    "projection": projection_obj
                }
            else:
                # 只有一個參數（filter）
                # 但要確保不包含後面的 .sort() 等
                # 找到參數的真正結束位置
                param_end = -1
                depth = 0
                in_string = False
                escape_next = False

                for i, char in enumerate(params_str):
                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char in '{[(':
                            depth += 1
                        elif char in '}])':
                            depth -= 1
                            if depth == 0:
                                param_end = i + 1
                                break

                if param_end > 0 and param_end < len(params_str):
                    # 有額外的內容（可能是錯誤的解析）
                    actual_param = params_str[:param_end]
                else:
                    actual_param = params_str

                filter_obj = self._safe_parse_params(actual_param)
                return {
                    "filter": filter_obj,
                    "projection": None
                }

        except Exception as e:
            self.logger.error(f"解析 find 參數失敗: {e}")
            self.logger.error(f"參數字串: {params_str[:200]}")
            # 回退到簡單解析
            return {
                "filter": self._safe_parse_params(params_str),
                "projection": None
            }

    def _parse_aggregate_pipeline(self, pipeline_str: str) -> list:
        """專門解析 aggregate pipeline - 增強版"""
        try:
            # 先處理動態日期
            pipeline_str = self._handle_dynamic_dates(pipeline_str)
            
            # 替換單引號為雙引號（針對字段引用）
            
            # 智能替換單引號
            # 保護已經在雙引號內的內容
            protected_content = {}
            counter = 0
            
            def protect_double_quoted(match):
                nonlocal counter
                placeholder = f"__PROTECTED_{counter}__"
                protected_content[placeholder] = match.group(0)
                counter += 1
                return placeholder
            
            # 先保護雙引號內容
            pipeline_str = re.sub(r'"[^"]*"', protect_double_quoted, pipeline_str)
            
            # 現在可以安全地替換單引號為雙引號
            pipeline_str = pipeline_str.replace("'", '"')
            
            # 恢復保護的內容
            for placeholder, content in protected_content.items():
                pipeline_str = pipeline_str.replace(placeholder, content)
            
            # 處理 MongoDB 語法
            processed = self._preprocess_mongo_syntax_safe(pipeline_str)
            
            # 記錄處理過程
            self.logger.debug(f"原始 pipeline: {pipeline_str[:200]}...")
            self.logger.debug(f"處理後 pipeline: {processed[:200]}...")
            
            # 嘗試解析
            try:
                pipeline = json.loads(processed)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 解析失敗: {e}")
                self.logger.error(f"處理後的字串: {processed}")
                
                # 嘗試更激進的修復
                # 確保所有需要引號的地方都有引號
                processed = re.sub(r'(?<=[{,])\s*([a-zA-Z_$][a-zA-Z0-9_]*)\s*(?=:)', r'"\1"', processed)
                
                # 再次嘗試解析
                try:
                    pipeline = json.loads(processed)
                except json.JSONDecodeError as e2:
                    self.logger.error(f"第二次解析也失敗: {e2}")
                    # 返回一個基本的 pipeline
                    return [{"$limit": 10000}]  # 防止返回過多數據
            
            # 確保返回的是列表
            if not isinstance(pipeline, list):
                pipeline = [pipeline]
            
            # 添加安全限制（如果沒有 $limit）
            has_limit = any('$limit' in stage for stage in pipeline)
            if not has_limit and len(pipeline) > 0:
                # 檢查是否有潛在的大量數據操作
                has_group = any('$group' in stage for stage in pipeline)
                if not has_group:  # 如果沒有 group，添加限制
                    pipeline.append({"$limit": 10000})
            
            return pipeline
            
        except Exception as e:
            self.logger.error(f"解析 aggregate pipeline 失敗: {e}")
            self.logger.error(f"原始 pipeline: {pipeline_str}")
            self.logger.error(traceback.format_exc())
            # 返回一個安全的默認 pipeline
            return [{"$limit": 10000}]

    def _safe_parse_params(self, params_str: str) -> Any:
        """安全地解析參數字串 - 增強版"""
        if not params_str:
            return {}

        try:
            # 先處理特殊的日期語法
            params_str = self._handle_dynamic_dates(params_str)

            # 預處理 MongoDB 特殊語法
            params_str = self._preprocess_mongo_syntax_safe(params_str)

            # 嘗試 JSON 解析
            return json.loads(params_str)

        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON 解析失敗: {e}")
            self.logger.debug(f"嘗試解析的字串: {params_str[:500]}")

            # 處理特殊情況
            if params_str.startswith('"') and params_str.endswith('"'):
                return params_str[1:-1]
            elif params_str.startswith("'") and params_str.endswith("'"):
                return params_str[1:-1]

            # 嘗試修復常見問題
            try:
                # 修復 $currentDate 等特殊操作符
                fixed_str = self._fix_special_operators(params_str)
                return json.loads(fixed_str)
            except:
                pass

            # 嘗試使用 ast.literal_eval
            try:
                import ast
                # 替換 MongoDB 特殊語法
                safe_str = params_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                # 移除 $ 符號暫時
                safe_str = safe_str.replace('$', 'DOLLAR_')

                result = ast.literal_eval(safe_str)
                # 恢復 $ 符號
                return self._restore_dollar_signs(result)
            except:
                pass

            # 如果所有方法都失敗
            self.logger.warning(f"無法解析參數: {params_str[:200]}")
            return {}

    def _fix_special_operators(self, text: str) -> str:
        """修復特殊的 MongoDB 操作符"""
        # 處理 $currentDate
        text = re.sub(r'\$currentDate\s*:\s*\{\s*\}', '"$$NOW"', text)
        text = re.sub(r'\$currentDate\s*:\s*true', '"$$NOW"', text)

        # 確保所有的操作符都有引號
        operators = [
            'currentDate', 'now', 'literal', 'type', 'exists',
            'expr', 'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'nin',
            'and', 'or', 'not', 'nor', 'month', 'year', 'dayOfMonth',
            'dateFromString', 'dateFromParts', 'dateToParts'
        ]

        for op in operators:
            # 添加引號到操作符
            text = re.sub(rf'(?<!["\'])\$({op})(?!["\'])', rf'"$\1"', text, flags=re.IGNORECASE)

        # 修復嵌套的對象
        text = re.sub(r'(?<=[{,])\s*(?!")([a-zA-Z_][a-zA-Z0-9_]*)(?!")\s*(?=:)', r'"\1"', text)

        return text

    def _restore_dollar_signs(self, obj: Any) -> Any:
        """恢復 $ 符號（從 DOLLAR_ 轉換回來）"""
        if isinstance(obj, dict):
            return {
                self._restore_dollar_signs(key): self._restore_dollar_signs(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._restore_dollar_signs(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace('DOLLAR_', '$')
        else:
            return obj

    def _handle_dynamic_dates(self, text: str) -> str:
        """處理動態日期計算 - 增強版"""
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # 處理 $currentDate
        text = re.sub(r'\{\s*\$currentDate\s*:\s*\{\s*\}\s*\}', f'"{datetime.now().isoformat()}"', text)
        
        # 處理當年年初和年末
        text = re.sub(
            r'new\s+Date\s*\(\s*new\s+Date\s*\(\s*\)\s*\.\s*getFullYear\s*\(\s*\)\s*,\s*0\s*,\s*1\s*\)',
            f'ISODate("{current_year}-01-01T00:00:00.000Z")',
            text
        )
        
        text = re.sub(
            r'new\s+Date\s*\(\s*new\s+Date\s*\(\s*\)\s*\.\s*getFullYear\s*\(\s*\)\s*\+\s*1\s*,\s*0\s*,\s*1\s*\)',
            f'ISODate("{current_year + 1}-01-01T00:00:00.000Z")',
            text
        )
        
        # 處理 new Date() - 當前時間
        text = re.sub(r'new\s+Date\s*\(\s*\)', f'ISODate("{datetime.now().isoformat()}Z")', text)
        
        # 處理 $$NOW
        text = text.replace('$$NOW', f'ISODate("{datetime.now().isoformat()}Z")')
        
        return text
    
    def _preprocess_mongo_syntax_safe(self, text: str) -> str:
        """預處理 MongoDB 特殊語法 - 完整版"""
        # 保護字段引用
        field_refs = {}
        field_counter = 0
        
        def protect_field_refs(match):
            nonlocal field_counter
            field_ref = match.group(1) if match.lastindex else match.group(0)
            # 只保護真正的字段引用，不保護操作符
            if not any(field_ref.startswith(f'${op}') for op in ['expr', 'eq', 'month', 'year', 'currentDate', 'dateFromString']):
                placeholder = f"__FIELD_REF_{field_counter}__"
                field_refs[placeholder] = field_ref
                field_counter += 1
                return f'"{placeholder}"'
            return match.group(0)
        
        # 保護字段引用（改進的正則）
        text = re.sub(r'"(\$[a-z][a-zA-Z0-9_.]*)"', protect_field_refs, text)
        text = re.sub(r"'(\$[a-z][a-zA-Z0-9_.]*)'", protect_field_refs, text)
        
        # 修復：將單引號轉換為雙引號（針對字段名和字串值）
        # 保護已經在雙引號內的內容
        protected_content = {}
        counter = 0
        
        def protect_double_quoted(match):
            nonlocal counter
            placeholder = f"__PROTECTED_{counter}__"
            protected_content[placeholder] = match.group(0)
            counter += 1
            return placeholder
        
        # 先保護雙引號內容
        text = re.sub(r'"[^"]*"', protect_double_quoted, text)
        
        # 現在可以安全地替換單引號為雙引號
        text = text.replace("'", '"')
        
        # 恢復保護的內容
        for placeholder, content in protected_content.items():
            text = text.replace(placeholder, content)
        
        # 處理特殊類型 - 改進 ISODate 處理
        text = re.sub(r'ObjectId\s*\(\s*["\']([^"\']+)["\']\s*\)', r'{"$oid": "\1"}', text)
        
        # 改進 ISODate 處理，支援單引號和雙引號，以及沒有引號的情況
        text = re.sub(r'ISODate\s*\(\s*["\']([^"\']+)["\']\s*\)', r'{"$date": "\1"}', text)
        text = re.sub(r'ISODate\s*\(\s*([^"\')\s]+)\s*\)', r'{"$date": "\1"}', text)
        
        text = re.sub(r'new\s+Date\s*\(\s*["\']([^"\']+)["\']\s*\)', r'{"$date": "\1"}', text)
        
        # 處理 $currentDate
        text = re.sub(r'\{\s*\$currentDate\s*:\s*\{\s*\}\s*\}', '{"$date": "$$NOW"}', text)
        text = re.sub(r'\$currentDate\s*:\s*\{\s*\}', '"$date": "$$NOW"', text)
        
        # MongoDB 操作符列表（完整）
        all_operators = [
            # 比較操作符
            'eq', 'gt', 'gte', 'in', 'lt', 'lte', 'ne', 'nin',
            # 邏輯操作符
            'and', 'not', 'nor', 'or',
            # 元素操作符
            'exists', 'type',
            # 評估操作符
            'expr', 'jsonSchema', 'mod', 'regex', 'text', 'where',
            # 日期操作符
            'currentDate', 'dateFromString', 'dateFromParts', 'dateToParts', 
            'dateToString', 'dayOfMonth', 'dayOfWeek', 'dayOfYear', 'hour',
            'millisecond', 'minute', 'month', 'second', 'week', 'year',
            # 聚合操作符
            'addFields', 'bucket', 'bucketAuto', 'count', 'facet', 'group',
            'limit', 'lookup', 'match', 'project', 'skip', 'sort', 'unwind',
            # 數學操作符
            'abs', 'add', 'ceil', 'divide', 'exp', 'floor', 'ln', 'log',
            'log10', 'mod', 'multiply', 'pow', 'round', 'sqrt', 'subtract', 'trunc',
            # 字串操作符
            'concat', 'indexOfBytes', 'indexOfCP', 'ltrim', 'regexFind',
            'regexFindAll', 'regexMatch', 'replaceOne', 'replaceAll', 'rtrim',
            'split', 'strLenBytes', 'strLenCP', 'strcasecmp', 'substr',
            'substrBytes', 'substrCP', 'toLower', 'toString', 'toUpper', 'trim',
            # 陣列操作符
            'arrayElemAt', 'arrayToObject', 'concatArrays', 'filter', 'first',
            'in', 'indexOfArray', 'isArray', 'last', 'map', 'objectToArray',
            'range', 'reduce', 'reverseArray', 'size', 'slice', 'zip',
            # 其他
            'avg', 'sum', 'min', 'max', 'push', 'addToSet', 'cond', 'ifNull',
            'switch', 'let', 'literal'
        ]
        
        # 為操作符添加引號（更精確的處理）
        # 只處理在冒號前的操作符，避免處理字段引用
        for op in all_operators:
            # 只處理還沒有引號的操作符，且必須在冒號前
            pattern = rf'(?<!["\'])\$({op})(?!["\'])\s*:'
            replacement = r'"$\1":'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 處理字段引用（$開頭但不是操作符的）
        # 修復：只在冒號後處理字段引用，避免錯誤處理
        text = re.sub(r':\s*(?<!["\'])\$([a-zA-Z][a-zA-Z0-9_]*)(?!["\'])', r': "\$\1"', text)
        
        # 恢復字段引用
        for placeholder, field_ref in field_refs.items():
            text = text.replace(f'"{placeholder}"', f'"{field_ref}"')
        
        # 清理多餘的引號
        text = re.sub(r'""([^"]+)""', r'"\1"', text)
        
        return text

    def _safe_convert_types(self, obj: Any) -> Any:
        """安全地轉換特殊類型 - 增強版"""
        if isinstance(obj, dict):
            converted = {}
            for key, value in obj.items():
                # 處理特殊類型標記
                if key == '$date':
                    try:
                        if value == 'now':
                            converted = datetime.now()
                        elif isinstance(value, str):
                            # 嘗試多種日期格式
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ',
                                    '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d']:
                                try:
                                    converted = datetime.strptime(value, fmt)
                                    break
                                except:
                                    continue
                            else:
                                # ISO format (最後的回退)
                                try:
                                    converted = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                except:
                                    converted = value
                        else:
                            converted = value
                    except:
                        converted = value
                elif key == '$oid':
                    try:
                        # 動態導入 ObjectId
                        from bson import ObjectId
                        converted = ObjectId(value)
                    except:
                        converted = value
                elif key == '$numberLong':
                    try:
                        converted = int(value)
                    except:
                        converted = value
                elif key == '$numberDecimal':
                    try:
                        converted = float(value)
                    except:
                        converted = value
                else:
                    # 遞迴處理，但不改變字段引用
                    converted[key] = self._safe_convert_types(value)
            return converted
        elif isinstance(obj, list):
            return [self._safe_convert_types(item) for item in obj]
        elif isinstance(obj, str):
            # 不要自動轉換字串為日期，除非明確標記
            return obj
        else:
            return obj
