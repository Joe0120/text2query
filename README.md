# text2query

一個用於將自然語言轉換為資料庫查詢的 Python 套件，支援多種資料庫類型並整合 LlamaIndex LLM。

## 功能特色

- 🤖 **Text-to-SQL/NoSQL**：將自然語言問題轉換為資料庫查詢
- 🔄 **異步執行**：完整的異步資料庫查詢支援
- 🗄️ **多資料庫支援**：PostgreSQL, MySQL, SQLite, MongoDB
- 🔌 **LLM 整合**：使用您自己的 LlamaIndex LLM 實例
- 🛡️ **安全檢查**：內建查詢安全驗證機制
- 📊 **結構分析**：自動分析資料庫結構供 LLM 參考
- 🧠 **RAG 訓練資料管理**：TrainingStore 模組支援向量相似度搜尋和權限控制

## 使用方式（作為 Git Submodule）

### 1. 添加 Submodule

在你的專案中添加 text2query 作為 submodule：

```bash
git submodule add https://github.com/Joe0120/text2query.git libs/text2query
git submodule update --init --recursive
```

### 2. 安裝依賴

```bash
pip install -r libs/text2query/requirements.txt
```

如需使用 Text-to-SQL 功能，還需安裝 LlamaIndex：

```bash
pip install llama-index llama-index-llms-openai  # 或其他 LLM 套件
```

如需使用 RAG 訓練資料功能，還需安裝 pgvector：

```bash
# PostgreSQL 需要安裝 pgvector 擴展
# Ubuntu/Debian: apt install postgresql-16-pgvector
# macOS: brew install pgvector
pip install pgvector
```

## 使用範例

### 基本查詢執行

#### PostgreSQL 範例

```python
from text2query.core.connections import PostgreSQLConfig
from text2query.adapters.sql.postgresql import PostgreSQLAdapter

# 建立連線配置
config = PostgreSQLConfig(
    database_name="mydb",
    host="localhost",
    port=5432,
    username="user",
    password="password"
)

# 建立 adapter
adapter = PostgreSQLAdapter(config)

# 測試連線
success, message = await adapter.test_connection()
print(f"連線狀態: {message}")

# 執行查詢
result = await adapter.sql_execution("SELECT * FROM users LIMIT 10")
print(result)
```

#### MongoDB 範例

```python
from text2query.core.connections import MongoDBConfig
from text2query.adapters.nosql.mongodb import MongoDBAdapter

# 建立連線配置
config = MongoDBConfig(
    host="localhost",
    port=27017,
    database_name="mydb",
    username="admin",
    password="password",
    auth_database="admin"
)

# 建立 adapter
adapter = MongoDBAdapter(config)

# 執行查詢（支援字串或字典格式）
result = await adapter.sql_execution('db.users.find({"age": {"$gt": 18}}).limit(10)')
print(result)
```

### RAG 訓練資料管理

TrainingStore 提供完整的 RAG 訓練資料管理功能，支援向量相似度搜尋和靈活的權限控制：

```python
from text2query.core.connections import PostgreSQLConfig
from text2query.core.training import TrainingStore
from llama_index.embeddings.openai import OpenAIEmbedding  # 或其他 embedder
import uuid

# 1. 初始化 TrainingStore 和 Embedder（應用啟動時執行一次）
# 創建 embedder（支援任何 LlamaIndex 相容的 embedder）
embedder = OpenAIEmbedding(
    model="text-embedding-3-small",
    dimensions=768,
)

# 初始化 TrainingStore 並傳入 embedder
store = await TrainingStore.initialize(
    postgres_config=PostgreSQLConfig(
        host="localhost",
        port=5432,
        database_name="your_db",
        username="user",
        password="password",
    ),
    training_schema="wisbi",  # 自訂 schema 名稱
    embedding_dim=768,        # 向量維度
    embedder=embedder,        # 傳入 embedder，自動生成 embedding
)

# 2. 新增訓練資料

# 新增問答對
qna_id = await store.insert_qna(
    table_id="employees",  # 必填：資料表 ID（字串）
    question="如何查詢所有員工？",
    answer_sql="SELECT * FROM employees",
    user_id="alice",    # 可選：使用者 ID
    group_id="sales",   # 可選：群組 ID
    metadata={"source": "manual", "type": "basic_query"},
)

# 新增 SQL 範例
sql_id = await store.insert_sql_example(
    table_id="employees",
    content="SELECT COUNT(*) FROM employees WHERE active = true",
    user_id="alice",
    group_id="sales",
    metadata={"type": "count_query"},
)

# 新增文件說明
doc_id = await store.insert_documentation(
    table_id="employees",
    title="員工表說明",
    content="employees 表包含所有員工的基本資訊",
    user_id="alice",
    group_id="sales",
    metadata={"type": "table_doc"},
)

# 3. 搜尋訓練資料（向量相似度搜尋 + 權限過濾）
query = "員工資料查詢"
query_embedding = embedder.get_text_embedding(query)

# 支援三種 table_path 格式：
# - 單一表：table_path="mysql.employees"
# - 通配符：table_path="mysql.*"（搜尋所有 mysql 表）
# - 列表：table_path=["mysql.employees", "mysql.departments"]

# 搜尋單一表
search_results = await store.search_all(
    query_embedding=[0.1, 0.2, ...],  # 查詢向量
    table_id="employees",
    user_id="alice",                  # 搜尋者身份
    group_id="sales",
    top_k=5,
)

# 搜尋所有 mysql 表
all_mysql_results = await store.search_all(
    query_embedding=query_embedding,
    table_path="mysql.*",  # 通配符
    user_id="alice",
    group_id="sales",
    top_k=10,
)

# 搜尋多個指定表
multi_results = await store.search_all(
    query_embedding=query_embedding,
    table_path=["mysql.employees", "mysql.departments"],  # 列表
    user_id="alice",
    group_id="sales",
    top_k=8,
)

# 結果格式：{"qna": [...], "sql_examples": [...], "documentation": [...]}
for table_name, results in search_results.items():
    print(f"{table_name}: {len(results)} 筆結果")

# 4. 更新訓練資料（Upsert 模式）
"""

# 5. 刪除訓練資料
"""
```

#### 傳統方式（手動提供 embedding）

如果你想手動管理 embedding，仍可使用傳統方法：

```python
# 不提供 embedder 的初始化
store = await TrainingStore.initialize(
    postgres_config=config,
    training_schema="wisbi",
    embedding_dim=768,
    # 不提供 embedder
)

# 手動生成 embedding 並插入
embedder = OpenAIEmbedding()
question = "如何查詢所有員工？"
answer_sql = "SELECT * FROM employees"
embedding = embedder.get_text_embedding(f"{question} {answer_sql}")

# 使用底層方法手動傳入 embedding
qna_id = await store.insert_qna(
    training_id=training_id,
    table_path="mysql.employees",
    question=question,
    answer_sql=answer_sql,
    embedding=embedding,  # 手動提供 embedding
    user_id="alice",
    group_id="sales",
)
```

#### 權限控制模型

TrainingStore 支援靈活的權限控制：

| user_id | group_id | 存取權限 | 使用情境 |
|---------|----------|----------|----------|
| "alice" | "sales" | 只有 alice 在 sales 群組可存取 | 個人私有資料（限定群組） |
| "alice" | "" | alice 在任何群組都可存取 | 個人跨群組資料 |
| "" | "sales" | sales 群組所有成員可存取 | 群組共享資料 |
| "" | "" | 所有人都可存取 | 全局公開資料 |

```python
# 權限搜尋範例
# Alice 在 sales 群組搜尋 → 可存取：個人私有 + 個人跨群組 + 群組共享 + 全局公開
alice_results = await store.search_qna(
    query_embedding=embedding,
    table_id="employees",
    user_id="alice",
    group_id="sales",
    top_k=10,
)

# 匿名使用者搜尋 → 只能存取：全局公開資料
public_results = await store.search_qna(
    query_embedding=embedding,
    table_id="employees",
    user_id=None,
    group_id=None,
    top_k=10,
)
```

### Text-to-SQL 功能

使用自然語言查詢資料庫：

```python
from text2query import Text2SQL
from text2query.adapters.nosql.mongodb import MongoDBAdapter
from text2query.core.connections import MongoDBConfig
from llama_index.llms.openai import OpenAI

# 1. 設置資料庫連線
config = MongoDBConfig(
    host="localhost",
    port=27017,
    database_name="employees",
    username="admin",
    password="password"
)
adapter = MongoDBAdapter(config)

# 2. 獲取資料庫結構
db_structure = await adapter.get_schema_str()

# 3. 設置 LLM（使用您自己的 LlamaIndex LLM）
llm = OpenAI(model="gpt-4", api_key="your-api-key")

# 4. 建立 Text2SQL 實例
t2s = Text2SQL(
    llm=llm,
    db_structure=db_structure,
    db_type="mongodb"  # 或 "postgresql", "mysql", "sqlite"
)

# 5. 使用自然語言生成查詢
query = await t2s.generate_query("每個部門有幾個主管？列出部門名稱和主管數量")
print(f"生成的查詢: {query}")

# 6. 執行查詢
result = await adapter.sql_execution(query)
print(f"查詢結果: {result}")
```

### 使用其他 LLM

Text2SQL 支援任何 LlamaIndex 相容的 LLM：

```python
# 使用 LiteLLM（支援多種 LLM 提供商）
from llama_index.llms.litellm import LiteLLM

llm = LiteLLM(
    model="openai/gpt-4",
    api_base="https://api.openai.com/v1",
    api_key="your-api-key",
    max_tokens=4096,
    temperature=0.1
)

# 使用 Anthropic Claude
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(
    model="claude-3-opus-20240229",
    api_key="your-api-key"
)

# 然後照常使用 Text2SQL
t2s = Text2SQL(llm=llm, db_structure=db_structure, db_type="postgresql")
```

### 進階功能

#### 自訂 Prompt 模板

```python
custom_template = """
給定以下資料庫結構：
{db_structure}

請根據以下問題生成 SQL 查詢：
{question}

注意事項：
- 使用標準 SQL 語法
- 確保查詢效率
- 添加適當的 LIMIT 限制
"""

t2s.set_custom_template(custom_template)
```

#### 使用對話歷史

```python
from llama_index.memory import ChatMemoryBuffer

# 建立記憶緩衝
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# 使用記憶建立 Text2SQL
t2s = Text2SQL(
    llm=llm,
    db_structure=db_structure,
    db_type="postgresql",
    chat_history=memory.memory
)

# 連續對話
query1 = await t2s.generate_query("顯示所有用戶")
query2 = await t2s.generate_query("只顯示前 10 個")  # 會參考之前的對話
```

## 支援的資料庫

- **SQL**:
  - PostgreSQL - 完整支援（含連線池）
  - MySQL - 完整支援
  - SQLite - 完整支援
  - SQL Server - 完整支援
  - Oracle - 完整支援

- **NoSQL**:
  - MongoDB - 完整支援（含 aggregate pipeline）

## API 文件

### TrainingStore

RAG 訓練資料管理類，支援向量相似度搜尋和權限控制。

**初始化：**
```python
store = await TrainingStore.initialize(
    postgres_config=PostgreSQLConfig(...),
    training_schema="wisbi",
    embedding_dim=768,
)
```

**主要方法：**

**新增資料：**
- `async insert_qna(table_id, question, answer_sql, embedding, user_id="", group_id="", metadata=None)`: 新增問答對
- `async insert_sql_example(table_id, content, embedding, user_id="", group_id="", metadata=None)`: 新增 SQL 範例
- `async insert_documentation(table_id, content, embedding, title=None, user_id="", group_id="", metadata=None)`: 新增文件說明

**搜尋資料：**
- `async search_qna(query_embedding, table_id, user_id=None, group_id=None, top_k=5)`: 搜尋問答對
- `async search_sql_examples(query_embedding, table_id, user_id=None, group_id=None, top_k=5)`: 搜尋 SQL 範例
- `async search_documentation(query_embedding, table_id, user_id=None, group_id=None, top_k=5)`: 搜尋文件說明
- `async search_all(query_embedding, table_id, user_id=None, group_id=None, top_k=8)`: 搜尋所有類型

**更新資料：**
（已移除 training_id 相關 upsert 介面）

**刪除資料：**
（已移除 training_id 相關刪除介面）

### Text2SQL

主要的 Text-to-SQL 轉換類。

**參數：**
- `llm` (Any): LlamaIndex LLM 實例
- `db_structure` (str): 資料庫結構字串（從 `adapter.get_schema_str()` 獲取）
- `chat_history` (Optional[Any]): ChatMemoryBuffer.memory 格式的對話歷史
- `db_type` (str): 資料庫類型（"postgresql", "mysql", "mongodb", "sqlite"）

**方法：**
- `async generate_query(question, include_history=True, stream_thinking=True, show_thinking=True)`: 生成資料庫查詢
- `update_db_structure(db_structure)`: 更新資料庫結構
- `set_custom_template(template)`: 設置自訂 prompt 模板
- `get_config()`: 獲取當前配置

### Database Adapters

所有 adapter 都繼承自 `BaseQueryComposer` 並提供以下方法：

- `async test_connection()`: 測試資料庫連線
- `async get_schema_str()`: 獲取資料庫結構字串
- `async sql_execution(command, params=None, safe=True, limit=1000)`: 執行查詢
- `async close_conn()`: 關閉連線

## 依賴項

核心依賴（在 `requirements.txt` 中）：
- `asyncpg` - PostgreSQL 異步驅動
- `aiomysql` - MySQL 異步驅動
- `motor` - MongoDB 異步驅動
- `pymongo` - MongoDB 驅動
- `dnspython` - DNS 解析（MongoDB）
- `opencc-python-reimplemented` - 繁簡轉換

Text-to-SQL 功能需要：
- `llama-index` - LLM 整合框架
- 任何 LlamaIndex 相容的 LLM 套件（如 `llama-index-llms-openai`）

RAG 訓練資料功能需要：
- `pgvector` - PostgreSQL 向量擴展
- PostgreSQL 資料庫（需安裝 pgvector 擴展）

## 安全性

本套件包含多層安全檢查：

1. **SQL Injection 防護**：參數化查詢和輸入驗證
2. **MongoDB 操作限制**：禁止危險操作（`$where`, `$function`, `dropDatabase` 等）
3. **查詢限制**：自動添加結果數量限制
4. **連線驗證**：配置驗證和連線測試
5. **權限控制**：TrainingStore 提供細粒度的資料存取控制

## 授權

本專案不含任何開源授權，僅供私人使用。
