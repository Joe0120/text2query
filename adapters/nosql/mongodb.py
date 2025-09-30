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
        Execute MongoDB command asynchronously and return a structured response.
        
        Args:
            command: MongoDB command (string or dict format)
            params: Not used for MongoDB (kept for interface compatibility)
            safe: Whether to apply safety checks
            limit: Maximum number of results to return
            
        Returns:
            Structured response dictionary
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
            
            # 2. 應用安全檢查
            if safe and not self._is_query_safe(parsed["data"]):
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['Query contains forbidden operations']],
                    'sql_command': str(command)[:100],
                    'ori_sql_command': str(command)
                }
            
            # 3. 執行已解析的命令
            result = await self._safe_execute_parsed(parsed["data"], str(command), limit)
            
            execution_time = (time.time() - start_time) * 1000
            result['execution_time'] = execution_time
            result['metadata'] = result.get('metadata', {})
            result['metadata']['execution_time_ms'] = execution_time
            
            self.logger.info(f"MongoDB command executed {'successfully' if result['success'] else 'with errors'}, time: {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"MongoDB sql_execution error: {e}")
            self.logger.error(traceback.format_exc())
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'success': False,
                'columns': ['error', 'detail'],
                'result': [[str(e), traceback.format_exc()]],
                'sql_command': str(command)[:100],
                'ori_sql_command': str(command),
                'execution_time': execution_time,
                'metadata': {'execution_time_ms': execution_time}
            }

    async def get_sql_struct_str(self) -> str:
        """
        獲取 MongoDB 資料庫結構字符串
        
        Returns:
            str: 包含所有集合結構的描述字符串
        """
        try:
            if self._client is None or self._db is None:
                await self._get_connection()
            
            connection_info = self._build_connection_info()
            collections = connection_info.get('collections', [])
            
            if not collections:
                # 如果沒有指定集合，獲取所有集合
                collections = await self._db.list_collection_names()
                if not collections:
                    return "MongoDB database has no collections"
            
            struct_parts = []
            struct_parts.append(f"MongoDB Connection:")
            struct_parts.append(f"Database: {connection_info['database']}")
            struct_parts.append("=" * 50)
            
            for collection_name in collections:
                try:
                    # 獲取集合統計
                    count = await self._db[collection_name].count_documents({})
                    struct_parts.append(f"\nCollection: {collection_name}")
                    struct_parts.append(f"Document count: {count}")
                    
                    if count > 0:
                        # 獲取樣本文檔來分析結構
                        cursor = self._db[collection_name].find({}).limit(3)
                        sample_docs = await cursor.to_list(length=3)
                        
                        if sample_docs:
                            # 收集所有字段
                            all_fields = set()
                            for doc in sample_docs:
                                all_fields.update(doc.keys())
                            
                            struct_parts.append("Fields:")
                            for field in sorted(all_fields):
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
                                
                                struct_parts.append(f"  - {field}: {field_type}")
                            
                            # 添加示例數據
                            struct_parts.append("Sample data:")
                            for i, doc in enumerate(sample_docs[:2], 1):
                                # 序列化文檔以便顯示
                                serialized_doc = self._serialize_document_for_display(doc)
                                struct_parts.append(f"  Example {i}: {serialized_doc}")
                    
                    struct_parts.append("-" * 30)
                    
                except Exception as collection_error:
                    struct_parts.append(f"Collection {collection_name} analysis failed: {str(collection_error)}")
                    struct_parts.append("-" * 30)
            
            return "\n".join(struct_parts)
            
        except Exception as error:
            self.logger.error("Failed to get MongoDB struct string: %s", error)
            return f"Error retrieving MongoDB structure: {str(error)}"

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
    # 4. 命令解析和安全檢查
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

    async def _safe_execute_parsed(self, command: Dict, ori_command: str, limit: int) -> Dict:
        """執行已解析的命令"""
        try:
            operation = command.get('operation', '').lower()
            collection_name = command.get('collection')
            
            if not collection_name:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [['Collection name not specified']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            # 檢查集合權限
            connection_info = self._build_connection_info()
            allowed_collections = connection_info.get('collections', [])
            if allowed_collections and collection_name not in allowed_collections:
                existing_collections = await self._db.list_collection_names()
                if collection_name not in existing_collections:
                    error_msg = f'Collection "{collection_name}" does not exist'
                else:
                    error_msg = f'Collection "{collection_name}" exists but not in allowed list'
                
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[error_msg]],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            collection = self._db[collection_name]
            
            # 根據操作類型執行
            if operation in ['find', 'findone', 'find_one']:
                return await self._safe_execute_find(collection, command, ori_command, limit)
            elif operation in ['countdocuments', 'count_documents', 'count']:
                return await self._safe_execute_count(collection, command, ori_command)
            elif operation == 'aggregate':
                return await self._safe_execute_aggregate(collection, command, ori_command, limit)
            elif operation == 'distinct':
                return await self._safe_execute_distinct(collection, command, ori_command)
            else:
                return {
                    'success': False,
                    'columns': ['error'],
                    'result': [[f'Unsupported operation: {operation}']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
                
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"_safe_execute_parsed error: {e}")
            self.logger.error(error_trace)
            
            return {
                'success': False,
                'columns': ['error', 'traceback'],
                'result': [[str(e), error_trace]],
                'sql_command': str(command),
                'ori_sql_command': ori_command
            }

    async def _safe_execute_find(self, collection, command: Dict, ori_command: str, limit: int) -> Dict:
        """安全執行查找操作"""
        try:
            filter_doc = command.get('filter', command.get('params', {}))
            projection = command.get('projection')
            
            # 轉換特殊類型
            filter_doc = self._safe_convert_types(filter_doc)
            if projection:
                projection = self._safe_convert_types(projection)
            
            operation = command.get('operation', '').lower()
            options = command.get('options', {})
            
            self.logger.info(f"Executing {operation}: filter={filter_doc}, projection={projection}")
            
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
                    cursor = cursor.limit(min(options['limit'], limit))
                else:
                    cursor = cursor.limit(limit)
                
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
                    sql_command += ")"
                    
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
                    return {
                        'success': True,
                        'columns': [],
                        'result': [],
                        'sql_command': f"db.{command['collection']}.find({json.dumps(filter_doc, default=str)})",
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

    async def _safe_execute_aggregate(self, collection, command: Dict, ori_command: str, limit: int) -> Dict:
        """安全執行聚合操作"""
        try:
            pipeline = command.get('pipeline', command.get('params', []))
            
            if not isinstance(pipeline, list):
                pipeline = [pipeline] if pipeline else []
            
            # 轉換特殊類型
            pipeline = self._safe_convert_types(pipeline)
            
            # 添加安全限制
            has_limit = any('$limit' in stage for stage in pipeline)
            if not has_limit and len(pipeline) > 0:
                has_group = any('$group' in stage for stage in pipeline)
                if not has_group:
                    pipeline.append({"$limit": limit})
            
            self.logger.info(f"Executing aggregate pipeline: {json.dumps(pipeline, default=str)[:500]}")
            
            # 執行聚合
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            self.logger.info(f"Aggregate returned {len(results)} results")
            
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
                    'result': [['distinct operation requires field name']],
                    'sql_command': str(command),
                    'ori_sql_command': ori_command
                }
            
            filter_doc = self._safe_convert_types(filter_doc)
            
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
        """解析 find 方法的參數"""
        if not params_str:
            return {"filter": {}}
        
        try:
            # 嘗試直接解析整個字串作為 JSON 陣列
            try:
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
            
            # 簡化版本：只解析第一個參數作為 filter
            filter_obj = self._safe_parse_params(params_str)
            return {
                "filter": filter_obj,
                "projection": None
            }
                
        except Exception as e:
            self.logger.error(f"Parse find params failed: {e}")
            return {
                "filter": self._safe_parse_params(params_str),
                "projection": None
            }

    def _parse_aggregate_pipeline(self, pipeline_str: str) -> list:
        """解析 aggregate pipeline"""
        try:
            # 先處理動態日期
            pipeline_str = self._handle_dynamic_dates(pipeline_str)
            
            # 嘗試解析
            try:
                pipeline = json.loads(pipeline_str)
            except json.JSONDecodeError:
                # 返回一個基本的 pipeline
                return [{"$limit": 1000}]
            
            # 確保返回的是列表
            if not isinstance(pipeline, list):
                pipeline = [pipeline]
            
            # 添加安全限制
            has_limit = any('$limit' in stage for stage in pipeline)
            if not has_limit and len(pipeline) > 0:
                has_group = any('$group' in stage for stage in pipeline)
                if not has_group:
                    pipeline.append({"$limit": 1000})
            
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Parse aggregate pipeline failed: {e}")
            return [{"$limit": 1000}]

    def _safe_parse_params(self, params_str: str) -> Any:
        """安全地解析參數字串"""
        if not params_str:
            return {}
        
        try:
            # 先處理特殊的日期語法
            params_str = self._handle_dynamic_dates(params_str)
            
            # 嘗試 JSON 解析
            return json.loads(params_str)
            
        except json.JSONDecodeError:
            # 處理特殊情況
            if params_str.startswith('"') and params_str.endswith('"'):
                return params_str[1:-1]
            elif params_str.startswith("'") and params_str.endswith("'"):
                return params_str[1:-1]
            
            self.logger.warning(f"Cannot parse params: {params_str[:200]}")
            return {}

    def _handle_dynamic_dates(self, text: str) -> str:
        """處理動態日期計算"""
        current_year = datetime.now().year
        
        # 處理 $currentDate
        text = re.sub(r'\{\s*\$currentDate\s*:\s*\{\s*\}\s*\}', f'"{datetime.now().isoformat()}"', text)
        
        # 處理當年年初和年末
        text = re.sub(
            r'new\s+Date\s*\(\s*new\s+Date\s*\(\s*\)\s*\.\s*getFullYear\s*\(\s*\)\s*,\s*0\s*,\s*1\s*\)',
            f'ISODate("{current_year}-01-01T00:00:00.000Z")',
            text
        )
        
        # 處理 new Date() - 當前時間
        text = re.sub(r'new\s+Date\s*\(\s*\)', f'ISODate("{datetime.now().isoformat()}Z")', text)
        
        return text

    def _safe_convert_types(self, obj: Any) -> Any:
        """安全地轉換特殊類型"""
        if isinstance(obj, dict):
            converted = {}
            for key, value in obj.items():
                # 處理特殊類型標記
                if key == '$date':
                    try:
                        if value == 'now':
                            converted = datetime.now()
                        elif isinstance(value, str):
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
                else:
                    converted[key] = self._safe_convert_types(value)
            return converted
        elif isinstance(obj, list):
            return [self._safe_convert_types(item) for item in obj]
        else:
            return obj
