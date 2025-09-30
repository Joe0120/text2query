# text2query

一個用於將文字轉換為資料庫查詢的 Python 套件。

## 使用方式（作為 Git Submodule）

### 1. 添加 Submodule

在你的專案中添加 text2query 作為 submodule：

```bash
git submodule add <repository-url> libs/text2query
git submodule update --init --recursive
```

### 2. 安裝依賴

```bash
pip install -r libs/text2query/requirements.txt
```

### 3. 使用範例

```python
# 導入模組
from libs.text2query import QueryComposer
from libs.text2query.core.connections import PostgreSQLConfig
from libs.text2query.adapters.sql.postgresql import PostgreSQLAdapter

# 建立連線配置
config = PostgreSQLConfig(
    database_name="mydb",
    host="localhost",
    username="user",
    password="password"
)

# 建立 adapter 並註冊
composer = QueryComposer()
pg_adapter = PostgreSQLAdapter(config)
composer.register_composer("postgresql", pg_adapter)

# 執行查詢
result = await pg_adapter.sql_execution("SELECT * FROM users LIMIT 10")
```

## 支援的資料庫

- **SQL**: PostgreSQL, MySQL, SQLite
- **NoSQL**: MongoDB

## 核心功能

- 異步資料庫查詢執行
- 連線池管理（PostgreSQL）
- 連線配置驗證與測試
- 多資料庫類型支援

## 依賴項

所有依賴項已列在 `requirements.txt` 中：
- asyncpg - PostgreSQL 異步驅動
- aiomysql - MySQL 異步驅動
- motor/pymongo - MongoDB 驅動
- 其他相關套件

## 授權

MIT License