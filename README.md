# text2query

一個用於將自然語言轉換為資料庫查詢的 Python 套件，支援多種資料庫類型並整合 LlamaIndex LLM。

## 功能特色

- 🤖 **Text-to-SQL/NoSQL**：將自然語言問題轉換為資料庫查詢
- 🔄 **異步執行**：完整的異步資料庫查詢支援
- 🗄️ **多資料庫支援**：PostgreSQL, MySQL, SQLite, MongoDB
- 🔌 **LLM 整合**：使用您自己的 LlamaIndex LLM 實例
- 🛡️ **安全檢查**：內建查詢安全驗證機制
- 📊 **結構分析**：自動分析資料庫結構供 LLM 參考

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
query = await t2s.generate_query(
    "每個部門有幾個主管？列出部門名稱和主管數量",
    stream_thinking=True,  # 顯示 LLM 思考過程
    show_thinking=True
)
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

## 安全性

本套件包含多層安全檢查：

1. **SQL Injection 防護**：參數化查詢和輸入驗證
2. **MongoDB 操作限制**：禁止危險操作（`$where`, `$function`, `dropDatabase` 等）
3. **查詢限制**：自動添加結果數量限制
4. **連線驗證**：配置驗證和連線測試

## 授權

本專案不含任何開源授權，僅供私人使用。