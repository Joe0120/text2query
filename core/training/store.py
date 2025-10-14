"""RAG Training Store - manages training data storage"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
import logging
import json

from ..connections.postgresql import PostgreSQLConfig
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_training_ddl


class TrainingStore:
    """Manages RAG training data in PostgreSQL
    
    這個類別負責管理 RAG (Retrieval-Augmented Generation) 訓練資料的儲存，
    包括問答對 (QnA)、SQL 範例 (sql_examples) 和文件說明 (documentation)。
    
    權限模型：
        - table_path: 必填，指定資料表路徑（例如：mysql.employees）
        - user_id: 可選，空字串 = 不限使用者
        - group_id: 可選，空字串 = 不限群組
        
    存取權限規則：
        1. user_id + group_id 都有值 → 只有該使用者在該群組下可存取
        2. user_id 有值, group_id 空 → 只有該使用者可存取（跨群組）
        3. user_id 空, group_id 有值 → 該群組所有成員可存取
        4. user_id + group_id 都空 → 所有人都可存取（全局資料）
    
    使用方式：
        # 初始化實例（自動檢查並建立表）- 應用啟動時執行一次
        store = await TrainingStore.initialize(
            postgres_config=PostgreSQLConfig(
                host="localhost",
                port=5432,
                database_name="your_db",
                username="user",
                password="pass",
            ),
            training_schema="wisbi",
            embedding_dim=768,
        )
        
        # 之後所有請求都重用這個 store 實例，不需要關閉
    """
    
    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        training_schema: str = "wisbi",
        embedding_dim: int = 768,
        embedder: Optional[Any] = None,
    ):
        """初始化 TrainingStore
        
        注意：建議使用 TrainingStore.initialize() 來建立實例，
        它會自動檢查並建立所需的表。
        
        Args:
            postgres_config: PostgreSQL 連線配置
            training_schema: RAG training 表要存放的 schema 名稱（預設 "wisbi"）
            embedding_dim: Embedding 向量維度（預設 768）
            embedder: LlamaIndex embedder 實例（可選），用於自動生成 embedding
        """
        self.postgres_config = postgres_config
        self.training_schema = training_schema
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None
    
    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        training_schema: str = "wisbi",
        embedding_dim: int = 768,
        embedder: Optional[Any] = None,
        auto_init_tables: bool = True,
    ) -> "TrainingStore":
        """初始化 TrainingStore 實例並自動設定表
        
        這是推薦的初始化方式，會自動檢查並建立所需的表（如果不存在）。
        
        Args:
            postgres_config: PostgreSQL 連線配置
            training_schema: RAG training 表要存放的 schema 名稱（預設 "wisbi"）
            embedding_dim: Embedding 向量維度（預設 768）
            embedder: LlamaIndex embedder 實例（可選），用於自動生成 embedding
            auto_init_tables: 是否自動檢查並建立表（預設 True）
        
        Returns:
            TrainingStore: 已初始化的實例
        
        Example:
            >>> from llama_index.embeddings.openai import OpenAIEmbedding
            >>> embedder = OpenAIEmbedding()
            >>> store = await TrainingStore.initialize(
            ...     postgres_config=PostgreSQLConfig(
            ...         host="localhost",
            ...         port=5432,
            ...         database_name="your_db",
            ...         username="user",
            ...         password="pass",
            ...     ),
            ...     training_schema="wisbi",
            ...     embedding_dim=768,
            ...     embedder=embedder,
            ... )
        """
        store = cls(postgres_config, training_schema, embedding_dim, embedder)
        
        if auto_init_tables:
            # 檢查表是否存在
            tables_exist = await store.check_tables_exist()
            
            # 如果有任何表不存在，就執行建立
            if not all(tables_exist.values()):
                missing_tables = [name for name, exists in tables_exist.items() if not exists]
                store.logger.info(f"Missing tables in schema '{training_schema}': {missing_tables}")
                store.logger.info("Creating training tables...")
                
                success = await store.init_training_tables()
                if success:
                    store.logger.info(f"Training tables initialized successfully in schema '{training_schema}'")
                else:
                    store.logger.warning("Failed to initialize some training tables")
            else:
                store.logger.info(f"All training tables already exist in schema '{training_schema}'")
        
        return store
    
    def _get_adapter(self) -> PostgreSQLAdapter:
        """獲取或創建 PostgreSQL adapter"""
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter
    
    # ============================================================================
    # 初始化方法
    # ============================================================================
    
    async def init_training_tables(self) -> bool:
        """初始化 RAG training 所需的表和索引
        
        這個方法會執行以下操作：
        1. 創建 schema（如果不存在）
        2. 啟用 pgvector 擴展（如果不存在）
        3. 創建 qna、sql_examples、documentation 表（如果不存在）
        4. 創建向量搜尋索引和複合索引
        
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        try:
            adapter = self._get_adapter()
            ddl_statements = get_training_ddl(
                schema_name=self.training_schema,
                embedding_dim=self.embedding_dim
            )
            
            for idx, ddl in enumerate(ddl_statements, 1):
                self.logger.debug(f"Executing DDL statement {idx}/{len(ddl_statements)}")
                result = await adapter.sql_execution(
                    ddl,
                    safe=False,  # DDL 語句需要關閉安全檢查
                    limit=None
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Failed to execute DDL statement {idx}: {error_msg}")
                    # 顯示失敗的 DDL（截斷顯示）
                    ddl_preview = ddl.strip()[:200].replace('\n', ' ')
                    self.logger.error(f"Failed DDL: {ddl_preview}...")
                    return False
            
            self.logger.info(f"Successfully set up training tables in schema '{self.training_schema}'")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error setting up training tables: {e}")
            return False
    
    async def check_tables_exist(self) -> Dict[str, bool]:
        """檢查 RAG training 表是否存在
        
        Returns:
            dict: {"qna": bool, "sql_examples": bool, "documentation": bool}
            
        Example:
            >>> tables_exist = await store.check_tables_exist()
            >>> if all(tables_exist.values()):
            ...     print("All tables exist")
            >>> else:
            ...     print(f"Missing tables: {[k for k, v in tables_exist.items() if not v]}")
        """
        try:
            adapter = self._get_adapter()
            schema = self.training_schema
            
            # 查詢指定 schema 中的表
            check_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}' 
                AND table_name IN ('qna', 'sql_examples', 'documentation')
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                self.logger.warning(f"Failed to check tables: {result.get('error', 'Unknown error')}")
                return {"qna": False, "sql_examples": False, "documentation": False}
            
            # 提取已存在的表名
            existing_tables = [row[0] for row in result.get("result", [])]
            
            return {
                "qna": "qna" in existing_tables,
                "sql_examples": "sql_examples" in existing_tables,
                "documentation": "documentation" in existing_tables,
            }
            
        except Exception as e:
            self.logger.exception(f"Error checking tables: {e}")
            return {"qna": False, "sql_examples": False, "documentation": False}
    
    async def check_schema_exists(self) -> bool:
        """檢查 training schema 是否存在
        
        Returns:
            bool: schema 存在返回 True，否則返回 False
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = '{self.training_schema}'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            return len(result.get("result", [])) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking schema: {e}")
            return False
    
    # ============================================================================
    # Embedding 生成輔助方法
    # ============================================================================
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本的 embedding 向量
        
        Args:
            text: 要生成 embedding 的文本
        
        Returns:
            List[float]: embedding 向量
        
        Raises:
            RuntimeError: 如果沒有提供 embedder
        """
        if self.embedder is None:
            raise RuntimeError(
                "Embedder not provided. Please initialize TrainingStore with an embedder "
                "or use the manual insert methods (insert_qna, insert_sql_example, insert_documentation) "
                "with pre-computed embeddings."
            )
        
        try:
            # LlamaIndex embedder 通常有 get_text_embedding 方法
            if hasattr(self.embedder, 'get_text_embedding'):
                embedding = await self.embedder.aget_text_embedding(text)
            elif hasattr(self.embedder, 'aget_text_embedding'):
                embedding = await self.embedder.aget_text_embedding(text)
            else:
                # 嘗試同步方法
                if hasattr(self.embedder, 'get_text_embedding'):
                    embedding = self.embedder.get_text_embedding(text)
                else:
                    raise AttributeError(
                        f"Embedder {type(self.embedder)} does not have get_text_embedding method"
                    )
            
            return embedding
            
        except Exception as e:
            self.logger.exception(f"Error generating embedding: {e}")
            raise
    
    # ============================================================================
    # INSERT 方法 - 新增訓練資料（統一介面）
    # ============================================================================
    
    async def insert_training_item(
        self,
        *,
        type: str,
        training_id: str,
        table_path: str,
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
        # QnA 特有
        question: Optional[str] = None,
        answer_sql: Optional[str] = None,
        # SQL Example/Documentation 特有
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[int]:
        """統一的訓練資料插入方法，自動生成 embedding
        
        這個方法簡化了訓練資料的插入流程，使用者只需指定 type 和相關欄位，
        系統會自動生成 embedding 並調用對應的底層方法。
        
        注意：使用此方法需要在初始化 TrainingStore 時提供 embedder。
        
        Args:
            type: 資料類型，必須是 "qna", "sql_example", "documentation" 之一
            training_id: 訓練批次 ID（UUID 字串）
            table_path: 資料表路徑（必填），例如：'mysql.employees'
            user_id: 使用者 ID（可選，預設空字串 = 不限使用者）
            group_id: 群組 ID（可選，預設空字串 = 不限群組）
            metadata: 額外的 metadata（JSON 格式）
            is_active: 是否啟用（預設 True）
            question: 問題文字（type="qna" 時必填）
            answer_sql: 答案 SQL（type="qna" 時必填）
            content: 內容（type="sql_example" 或 "documentation" 時必填）
            title: 標題（type="documentation" 時可選）
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        
        Raises:
            ValueError: 如果 type 不合法或缺少必要欄位
            RuntimeError: 如果沒有提供 embedder
        
        Example:
            >>> # 插入問答對
            >>> await store.insert_training_item(
            ...     type="qna",
            ...     training_id="550e8400-e29b-41d4-a716-446655440000",
            ...     table_path="mysql.employees",
            ...     question="查詢所有員工",
            ...     answer_sql="SELECT * FROM employees",
            ...     user_id="user_123",
            ...     group_id="group_A",
            ... )
            
            >>> # 插入 SQL 範例
            >>> await store.insert_training_item(
            ...     type="sql_example",
            ...     training_id="550e8400-...",
            ...     table_path="mysql.employees",
            ...     content="SELECT COUNT(*) FROM employees WHERE active = true",
            ... )
            
            >>> # 插入文件說明
            >>> await store.insert_training_item(
            ...     type="documentation",
            ...     training_id="550e8400-...",
            ...     table_path="mysql.employees",
            ...     title="員工表說明",
            ...     content="employees 表包含所有員工的基本資訊",
            ... )
        """
        # 驗證 type
        valid_types = {"qna", "sql_example", "documentation"}
        if type not in valid_types:
            raise ValueError(f"Invalid type '{type}'. Must be one of: {valid_types}")
        
        # 根據 type 組合文本用於生成 embedding
        if type == "qna":
            if not question or not answer_sql:
                raise ValueError("For type='qna', both 'question' and 'answer_sql' are required")
            text_for_embedding = f"{question} {answer_sql}"
        elif type == "sql_example":
            if not content:
                raise ValueError("For type='sql_example', 'content' is required")
            text_for_embedding = content
        else:  # documentation
            if not content:
                raise ValueError("For type='documentation', 'content' is required")
            title_part = f"{title} " if title else ""
            text_for_embedding = f"{title_part}{content}"
        
        # 生成 embedding
        try:
            embedding = await self._generate_embedding(text_for_embedding)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for type={type}: {e}")
            return None
        
        # 根據 type 調用對應的底層插入方法
        if type == "qna":
            return await self.insert_qna(
                training_id=training_id,
                table_path=table_path,
                question=question,
                answer_sql=answer_sql,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
        elif type == "sql_example":
            return await self.insert_sql_example(
                training_id=training_id,
                table_path=table_path,
                content=content,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
        else:  # documentation
            return await self.insert_documentation(
                training_id=training_id,
                table_path=table_path,
                content=content,
                embedding=embedding,
                title=title,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
    
    # ============================================================================
    # INSERT 方法 - 新增訓練資料（底層方法）
    # ============================================================================
    
    async def insert_qna(
        self,
        *,
        training_id: str,
        table_path: str,
        question: str,
        answer_sql: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入問答對訓練資料
        
        Args:
            training_id: 訓練批次 ID（UUID 字串）
            table_path: 資料表路徑（必填），例如：'mysql.employees'
            question: 問題文字
            answer_sql: 答案 SQL
            embedding: 向量 embedding（長度需符合 embedding_dim）
            user_id: 使用者 ID（可選，預設空字串 = 不限使用者）
            group_id: 群組 ID（可選，預設空字串 = 不限群組）
            metadata: 額外的 metadata（JSON 格式）
            is_active: 是否啟用（預設 True）
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        
        權限說明：
            - user_id="user_123", group_id="group_A" → 只有 user_123 在 group_A 可存取
            - user_id="user_123", group_id="" → 只有 user_123 可存取（跨所有群組）
            - user_id="", group_id="group_A" → group_A 的所有成員可存取
            - user_id="", group_id="" → 所有人可存取（全局公開資料）
        
        Example:
            >>> # 插入個人私有資料
            >>> id = await store.insert_qna(
            ...     training_id="550e8400-e29b-41d4-a716-446655440000",
            ...     table_path="mysql.employees",
            ...     question="查詢所有員工",
            ...     answer_sql="SELECT * FROM employees",
            ...     embedding=[0.1, 0.2, ...],  # 768 維向量
            ...     user_id="user_123",
            ...     group_id="group_A",
            ... )
        """
        try:
            adapter = self._get_adapter()
            
            # 準備 metadata（確保是 JSON 格式）
            md = metadata or {}
            md.setdefault("training_id", training_id)
            metadata_json = json.dumps(md, ensure_ascii=False)
            
            # 將 embedding 轉換為字串格式
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # 構建 INSERT SQL
            # 注意：使用 $$ 符號包圍文字以避免 SQL injection
            insert_sql = f"""
                INSERT INTO {self.training_schema}.qna (
                    user_id, group_id, table_path, training_id,
                    question, answer_sql, embedding, metadata, is_active
                ) VALUES (
                    '{user_id}', '{group_id}', '{table_path}', '{training_id}'::uuid,
                    $${question}$$, $${answer_sql}$$, '{embedding_str}'::vector, '{metadata_json}'::jsonb, {is_active}
                )
                RETURNING id
            """
            
            result = await adapter.sql_execution(insert_sql, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted QnA: id={inserted_id}, table_path={table_path}, training_id={training_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert QnA: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting QnA: {e}")
            return None
    
    async def insert_sql_example(
        self,
        *,
        training_id: str,
        table_path: str,
        content: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入 SQL 範例訓練資料
        
        Args:
            training_id: 訓練批次 ID（UUID 字串）
            table_path: 資料表路徑（必填）
            content: SQL 範例內容
            embedding: 向量 embedding
            user_id: 使用者 ID（可選）
            group_id: 群組 ID（可選）
            metadata: 額外的 metadata
            is_active: 是否啟用
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        """
        try:
            adapter = self._get_adapter()
            
            md = metadata or {}
            md.setdefault("training_id", training_id)
            metadata_json = json.dumps(md, ensure_ascii=False)
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            insert_sql = f"""
                INSERT INTO {self.training_schema}.sql_examples (
                    user_id, group_id, table_path, training_id,
                    content, embedding, metadata, is_active
                ) VALUES (
                    '{user_id}', '{group_id}', '{table_path}', '{training_id}'::uuid,
                    $${content}$$, '{embedding_str}'::vector, '{metadata_json}'::jsonb, {is_active}
                )
                RETURNING id
            """
            
            result = await adapter.sql_execution(insert_sql, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted SQL example: id={inserted_id}, table_path={table_path}, training_id={training_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert SQL example: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting SQL example: {e}")
            return None
    
    async def insert_documentation(
        self,
        *,
        training_id: str,
        table_path: str,
        content: str,
        embedding: List[float],
        title: Optional[str] = None,
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """插入文件說明訓練資料
        
        Args:
            training_id: 訓練批次 ID（UUID 字串）
            table_path: 資料表路徑（必填）
            content: 文件內容
            embedding: 向量 embedding
            title: 文件標題（可選）
            user_id: 使用者 ID（可選）
            group_id: 群組 ID（可選）
            metadata: 額外的 metadata
            is_active: 是否啟用
        
        Returns:
            Optional[int]: 成功返回插入的 id，失敗返回 None
        """
        try:
            adapter = self._get_adapter()
            
            md = metadata or {}
            md.setdefault("training_id", training_id)
            metadata_json = json.dumps(md, ensure_ascii=False)
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # 處理 title（可能為 NULL）
            title_sql = f"$${title}$$" if title else "NULL"
            
            insert_sql = f"""
                INSERT INTO {self.training_schema}.documentation (
                    user_id, group_id, table_path, training_id,
                    title, content, embedding, metadata, is_active
                ) VALUES (
                    '{user_id}', '{group_id}', '{table_path}', '{training_id}'::uuid,
                    {title_sql}, $${content}$$, '{embedding_str}'::vector, '{metadata_json}'::jsonb, {is_active}
                )
                RETURNING id
            """
            
            result = await adapter.sql_execution(insert_sql, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted documentation: id={inserted_id}, table_path={table_path}, training_id={training_id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert documentation: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting documentation: {e}")
            return None
    
    # ============================================================================
    # UPDATE 方法 - 更新訓練資料（Upsert 模式：先刪除再插入）
    # ============================================================================
    
    async def upsert_qna_by_training_id(
        self,
        *,
        training_id: str,
        table_path: str,
        question: str,
        answer_sql: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """更新問答對訓練資料（先刪除舊的，再插入新的）
        
        這個方法會先根據 training_id + table_path + user_id + group_id 刪除舊資料，
        然後插入新資料。適用於更新同一批次的訓練資料。
        
        Args:
            training_id: 訓練批次 ID（UUID 字串）
            table_path: 資料表路徑（必填）
            question: 問題文字
            answer_sql: 答案 SQL
            embedding: 向量 embedding
            user_id: 使用者 ID（可選）
            group_id: 群組 ID（可選）
            metadata: 額外的 metadata
            is_active: 是否啟用
        
        Returns:
            Optional[int]: 成功返回插入的新 id，失敗返回 None
        """
        try:
            adapter = self._get_adapter()
            
            # 先刪除舊的資料
            delete_sql = f"""
                DELETE FROM {self.training_schema}.qna
                WHERE training_id = '{training_id}'::uuid
                  AND table_path = '{table_path}'
                  AND user_id = '{user_id}'
                  AND group_id = '{group_id}'
            """
            
            delete_result = await adapter.sql_execution(delete_sql, safe=False, limit=None)
            
            if delete_result.get("success"):
                deleted_count = delete_result.get("metadata", {}).get("rows_affected", 0)
                if deleted_count > 0:
                    self.logger.info(f"Deleted {deleted_count} old QnA records for training_id={training_id}")
            
            # 再插入新的資料
            return await self.insert_qna(
                training_id=training_id,
                table_path=table_path,
                question=question,
                answer_sql=answer_sql,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
                
        except Exception as e:
            self.logger.exception(f"Error upserting QnA: {e}")
            return None
    
    async def upsert_sql_example_by_training_id(
        self,
        *,
        training_id: str,
        table_path: str,
        content: str,
        embedding: List[float],
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """更新 SQL 範例訓練資料（先刪除舊的，再插入新的）"""
        try:
            adapter = self._get_adapter()
            
            delete_sql = f"""
                DELETE FROM {self.training_schema}.sql_examples
                WHERE training_id = '{training_id}'::uuid
                  AND table_path = '{table_path}'
                  AND user_id = '{user_id}'
                  AND group_id = '{group_id}'
            """
            
            delete_result = await adapter.sql_execution(delete_sql, safe=False, limit=None)
            
            if delete_result.get("success"):
                deleted_count = delete_result.get("metadata", {}).get("rows_affected", 0)
                if deleted_count > 0:
                    self.logger.info(f"Deleted {deleted_count} old SQL examples for training_id={training_id}")
            
            return await self.insert_sql_example(
                training_id=training_id,
                table_path=table_path,
                content=content,
                embedding=embedding,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
                
        except Exception as e:
            self.logger.exception(f"Error upserting SQL example: {e}")
            return None
    
    async def upsert_documentation_by_training_id(
        self,
        *,
        training_id: str,
        table_path: str,
        content: str,
        embedding: List[float],
        title: Optional[str] = None,
        user_id: str = "",
        group_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Optional[int]:
        """更新文件說明訓練資料（先刪除舊的，再插入新的）"""
        try:
            adapter = self._get_adapter()
            
            delete_sql = f"""
                DELETE FROM {self.training_schema}.documentation
                WHERE training_id = '{training_id}'::uuid
                  AND table_path = '{table_path}'
                  AND user_id = '{user_id}'
                  AND group_id = '{group_id}'
            """
            
            delete_result = await adapter.sql_execution(delete_sql, safe=False, limit=None)
            
            if delete_result.get("success"):
                deleted_count = delete_result.get("metadata", {}).get("rows_affected", 0)
                if deleted_count > 0:
                    self.logger.info(f"Deleted {deleted_count} old documentation for training_id={training_id}")
            
            return await self.insert_documentation(
                training_id=training_id,
                table_path=table_path,
                content=content,
                embedding=embedding,
                title=title,
                user_id=user_id,
                group_id=group_id,
                metadata=metadata,
                is_active=is_active,
            )
                
        except Exception as e:
            self.logger.exception(f"Error upserting documentation: {e}")
            return None
    
    # ============================================================================
    # DELETE 方法 - 刪除訓練資料
    # ============================================================================
    
    async def delete_by_training_id(
        self,
        table: str,
        training_id: str,
        *,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        table_path: Optional[str] = None,
    ) -> int:
        """根據 training_id 刪除訓練資料
        
        可選擇性地加上 user_id, group_id, table_path 作為額外的過濾條件。
        
        Args:
            table: 表名（"qna", "sql_examples", "documentation"）
            training_id: 訓練批次 ID
            user_id: 可選的使用者 ID 過濾
            group_id: 可選的群組 ID 過濾
            table_path: 可選的資料表路徑過濾
        
        Returns:
            int: 刪除的行數
        
        Example:
            >>> # 刪除特定 training_id 的所有資料
            >>> deleted = await store.delete_by_training_id("qna", "550e8400-...")
            
            >>> # 刪除特定 training_id + user_id 的資料
            >>> deleted = await store.delete_by_training_id(
            ...     "qna", "550e8400-...", user_id="user_123"
            ... )
        """
        if table not in {"qna", "sql_examples", "documentation"}:
            self.logger.error(f"Invalid table name: {table}")
            return 0
        
        try:
            adapter = self._get_adapter()
            
            # 構建 WHERE 條件
            where_conditions = [f"training_id = '{training_id}'::uuid"]
            
            if user_id is not None:
                where_conditions.append(f"user_id = '{user_id}'")
            
            if group_id is not None:
                where_conditions.append(f"group_id = '{group_id}'")
            
            if table_path is not None:
                where_conditions.append(f"table_path = '{table_path}'")
            
            where_clause = " AND ".join(where_conditions)
            
            delete_sql = f"""
                DELETE FROM {self.training_schema}.{table}
                WHERE {where_clause}
            """
            
            result = await adapter.sql_execution(delete_sql, safe=False, limit=None)
            
            if result.get("success"):
                deleted_count = result.get("metadata", {}).get("rows_affected", 0)
                self.logger.info(f"Deleted {deleted_count} records from {table} with training_id={training_id}")
                return deleted_count
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to delete from {table}: {error_msg}")
                return 0
                
        except Exception as e:
            self.logger.exception(f"Error deleting from {table}: {e}")
            return 0
    
    # ============================================================================
    # SEARCH 方法 - 向量相似度搜尋
    # ============================================================================
    
    def _build_table_path_condition(self, table_path: Union[str, List[str]]) -> str:
        """構建 table_path 的 SQL 條件
        
        Args:
            table_path: 可以是：
                - 單一字串：'mysql.employees'
                - 通配符：'mysql.*'
                - 列表：['mysql.employees', 'mysql.groups']
        
        Returns:
            str: SQL WHERE 條件
        """
        if isinstance(table_path, list):
            # 列表模式：使用 IN
            paths = "', '".join(table_path)
            return f"table_path IN ('{paths}')"
        elif '*' in table_path:
            # 通配符模式：使用 LIKE
            pattern = table_path.replace('*', '%')
            return f"table_path LIKE '{pattern}'"
        else:
            # 單一字串：使用 =
            return f"table_path = '{table_path}'"
    
    async def search_qna(
        self,
        query_embedding: List[float],
        *,
        table_path: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的問答對訓練資料
        
        根據向量相似度搜尋，並根據權限過濾結果。
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_path: 資料表路徑，支援三種格式：
                - 單一字串：'mysql.employees'
                - 通配符：'mysql.*'（搜尋所有 mysql 開頭的表）
                - 列表：['mysql.employees', 'mysql.groups']（搜尋多個指定表）
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        Returns:
            List[Dict]: 搜尋結果列表，每筆包含所有欄位 + distance
        
        權限過濾邏輯：
            1. 如果提供 user_id + group_id → 可存取：
               - 該 user_id + group_id 的私有資料
               - 該 user_id 的跨群組資料 (group_id="")
               - 該 group_id 的群組共享資料 (user_id="")
               - 全局公開資料 (user_id="" + group_id="")
            
            2. 如果只提供 user_id → 可存取：
               - 該 user_id 的所有資料（不論 group_id）
               - 全局公開資料
            
            3. 如果只提供 group_id → 可存取：
               - 該 group_id 的群組共享資料
               - 全局公開資料
            
            4. 如果都不提供 → 只能存取：
               - 全局公開資料 (user_id="" + group_id="")
        
        Example:
            >>> # 搜尋單一表
            >>> results = await store.search_qna(
            ...     query_embedding=embedding,
            ...     table_path="mysql.employees"
            ... )
            
            >>> # 搜尋所有 mysql 表
            >>> results = await store.search_qna(
            ...     query_embedding=embedding,
            ...     table_path="mysql.*"
            ... )
            
            >>> # 搜尋多個指定表
            >>> results = await store.search_qna(
            ...     query_embedding=embedding,
            ...     table_path=["mysql.employees", "mysql.departments"]
            ... )
        """
        try:
            adapter = self._get_adapter()
            
            # 將 embedding 轉換為字串格式
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_path 條件
            table_path_condition = self._build_table_path_condition(table_path)
            
            # 構建權限過濾條件
            # 根據提供的 user_id 和 group_id 決定可存取的資料範圍
            if user_id and group_id:
                # 情境 1：有 user_id + group_id
                # 可存取：完全匹配、user+空group、空user+group、全空
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                # 情境 2：只有 user_id
                # 可存取：該 user 的所有資料 + 全局資料
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                # 情境 3：只有 group_id
                # 可存取：該 group 的共享資料 + 全局資料
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                # 情境 4：都沒有
                # 只能存取全局公開資料
                permission_condition = "(user_id = '' AND group_id = '')"
            
            # 構建完整的 SELECT 查詢
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_path, training_id,
                    question, answer_sql, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.qna
                WHERE {table_path_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                # 將結果轉換為字典列表
                columns = result.get("columns", [])
                rows = result.get("result", [])
                
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                self.logger.info(f"Found {len(results)} QnA results for table_path={table_path}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search QnA: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching QnA: {e}")
            return []
    
    async def search_sql_examples(
        self,
        query_embedding: List[float],
        *,
        table_path: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的 SQL 範例訓練資料
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_path: 資料表路徑，支援三種格式：
                - 單一字串：'mysql.employees'
                - 通配符：'mysql.*'
                - 列表：['mysql.employees', 'mysql.groups']
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        權限過濾邏輯與 search_qna 相同。
        """
        try:
            adapter = self._get_adapter()
            
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_path 條件
            table_path_condition = self._build_table_path_condition(table_path)
            
            # 構建權限過濾條件（與 search_qna 相同邏輯）
            if user_id and group_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                permission_condition = "(user_id = '' AND group_id = '')"
            
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_path, training_id,
                    content, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.sql_examples
                WHERE {table_path_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                columns = result.get("columns", [])
                rows = result.get("result", [])
                results = [dict(zip(columns, row)) for row in rows]
                self.logger.info(f"Found {len(results)} SQL example results for table_path={table_path}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search SQL examples: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching SQL examples: {e}")
            return []
    
    async def search_documentation(
        self,
        query_embedding: List[float],
        *,
        table_path: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 5,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """搜尋最相似的文件說明訓練資料
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_path: 資料表路徑，支援三種格式：
                - 單一字串：'mysql.employees'
                - 通配符：'mysql.*'
                - 列表：['mysql.employees', 'mysql.groups']
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 返回前 K 筆最相似的結果
            only_active: 是否只返回啟用的資料
        
        權限過濾邏輯與 search_qna 相同。
        """
        try:
            adapter = self._get_adapter()
            
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 構建 table_path 條件
            table_path_condition = self._build_table_path_condition(table_path)
            
            # 構建權限過濾條件（與 search_qna 相同邏輯）
            if user_id and group_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}' AND group_id = '{group_id}') OR
                        (user_id = '{user_id}' AND group_id = '') OR
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif user_id:
                permission_condition = f"""
                    (
                        (user_id = '{user_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            elif group_id:
                permission_condition = f"""
                    (
                        (user_id = '' AND group_id = '{group_id}') OR
                        (user_id = '' AND group_id = '')
                    )
                """
            else:
                permission_condition = "(user_id = '' AND group_id = '')"
            
            select_sql = f"""
                SELECT 
                    id, user_id, group_id, table_path, training_id,
                    title, content, metadata, is_active,
                    created_at, updated_at,
                    (embedding <=> '{embedding_str}'::vector) AS distance
                FROM {self.training_schema}.documentation
                WHERE {table_path_condition}
                  AND {permission_condition}
                  AND (NOT {only_active} OR is_active = TRUE)
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
            
            if result.get("success"):
                columns = result.get("columns", [])
                rows = result.get("result", [])
                results = [dict(zip(columns, row)) for row in rows]
                self.logger.info(f"Found {len(results)} documentation results for table_path={table_path}")
                return results
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search documentation: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Error searching documentation: {e}")
            return []
    
    async def search_all(
        self,
        query_embedding: List[float],
        *,
        table_path: Union[str, List[str]],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        top_k: int = 8,
        per_table_k: Optional[int] = None,
        only_active: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """搜尋所有類型的訓練資料（QnA, SQL範例, 文件）
        
        Args:
            query_embedding: 查詢的向量 embedding
            table_path: 資料表路徑，支援三種格式：
                - 單一字串：'mysql.employees'
                - 通配符：'mysql.*'
                - 列表：['mysql.employees', 'mysql.groups']
            user_id: 查詢者的使用者 ID（可選）
            group_id: 查詢者的群組 ID（可選）
            top_k: 總共返回的結果數量
            per_table_k: 每個表返回的結果數量（預設為 top_k // 3）
            only_active: 是否只返回啟用的資料
        
        Returns:
            Dict: {
                "qna": [...],
                "sql_examples": [...],
                "documentation": [...]
            }
        
        Example:
            >>> # 搜尋所有 mysql 表的所有類型資料
            >>> results = await store.search_all(
            ...     query_embedding=embedding,
            ...     table_path="mysql.*",
            ...     top_k=10
            ... )
        """
        per_k = per_table_k or max(2, top_k // 3)
        
        # 並行搜尋所有 3 個表
        qna_results = await self.search_qna(
            query_embedding,
            table_path=table_path,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        sql_results = await self.search_sql_examples(
            query_embedding,
            table_path=table_path,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        doc_results = await self.search_documentation(
            query_embedding,
            table_path=table_path,
            user_id=user_id,
            group_id=group_id,
            top_k=per_k,
            only_active=only_active,
        )
        
        return {
            "qna": qna_results,
            "sql_examples": sql_results,
            "documentation": doc_results,
        }
    
    # ============================================================================
    # 連線管理
    # ============================================================================
    
    async def close(self):
        """關閉資料庫連線
        
        在應用程式結束時應該呼叫這個方法來釋放資源。
        注意：在後端服務中，通常不需要手動呼叫，連線池會自動管理。
        
        Example:
            >>> store = await TrainingStore.initialize(...)
            >>> try:
            ...     # 使用 store
            ...     pass
            ... finally:
            ...     await store.close()
        """
        if self._adapter:
            await self._adapter.close_pool()
            self.logger.debug("TrainingStore connection closed")