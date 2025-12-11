"""
Prompt builder for text-to-query conversion
"""

from typing import Dict, Optional


class PromptBuilder:
    """Build prompts for LLM to generate database queries"""

    # Default prompt templates for different database types
    POSTGRESQL_TEMPLATE = """你是一個 PostgreSQL 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。只返回 SQL 語句，不要任何其他解釋或格式標記。"""

    MYSQL_TEMPLATE = """你是一個 MySQL 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。只返回 SQL 語句，不要任何其他解釋或格式標記。"""

    MONGODB_TEMPLATE = """你是一個 MongoDB 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 MongoDB 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 MongoDB 查詢語句。使用 db.collection.method() 格式。只返回查詢語句，不要任何其他解釋或格式標記，不需要任何換行全部寫在一行就好。

範例格式：
- 查找：db.users.find({{"age": {{"$gt": 18}}}})
- 聚合：db.orders.aggregate([{{"$group": {{"_id": "$status", "count": {{"$sum": 1}}}}}}])
- 計數：db.products.countDocuments({{"price": {{"$lt": 100}}}})"""

    SQLITE_TEMPLATE = """你是一個 SQLite 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。只返回 SQL 語句，不要任何其他解釋或格式標記。"""

    SQLSERVER_TEMPLATE = """你是一個 SQL Server 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。注意 SQL Server 的特性：
- 使用 TOP N 而非 LIMIT N
- 使用方括號 [] 來引用標識符
- 只返回 SQL 語句，不要任何其他解釋或格式標記。"""

    ORACLE_TEMPLATE = """你是一個 Oracle 數據庫查詢專家。根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。注意 Oracle 的特性：
- 使用 ROWNUM 或 FETCH FIRST N ROWS ONLY 來限制結果
- 表名和列名通常為大寫
- 使用雙引號 "" 來引用標識符
- 只返回 SQL 語句，不要任何其他解釋或格式標記。"""

    TRINO_TEMPLATE = """你是一個 Trino 分散式 SQL 查詢專家。Trino 是一個能夠跨多個數據源（PostgreSQL、MongoDB、MySQL 等）進行查詢的分散式 SQL 引擎。

根據以下數據庫結構和用戶問題，生成相應的 SQL 查詢語句。

數據庫結構：
{db_structure}

用戶問題：{question}

請生成對應的 SQL 查詢語句。注意 Trino 的特性：
- 跨數據源查詢時使用完整路徑：catalog.schema.table（例如：postgresql.public.employees）
- 支援標準 SQL 語法
- 可以在同一個查詢中 JOIN 不同 catalog 的表（例如：postgresql 的表和 mongodb 的表）
- 使用 LIMIT N 來限制結果
- 使用雙引號 "" 來引用標識符
- 只返回 SQL 語句，不要任何其他解釋或格式標記。

範例格式：
- 單表查詢：SELECT * FROM postgresql.public.employees WHERE salary > 50000
- 跨數據源查詢：SELECT e.name, s.amount FROM postgresql.public.employees e JOIN mongodb.hr.salaries s ON e.employee_id = s.employee_id
- 聚合查詢：SELECT department, COUNT(*) as count FROM postgresql.public.employees GROUP BY department"""

    def __init__(self, custom_templates: Optional[Dict[str, str]] = None):
        """
        Initialize prompt builder

        Args:
            custom_templates: Custom prompt templates for different database types
                             Format: {"postgresql": "template...", "mongodb": "template..."}
        """
        self.templates = {
            "postgresql": self.POSTGRESQL_TEMPLATE,
            "mysql": self.MYSQL_TEMPLATE,
            "mongodb": self.MONGODB_TEMPLATE,
            "sqlite": self.SQLITE_TEMPLATE,
            "sqlserver": self.SQLSERVER_TEMPLATE,
            "oracle": self.ORACLE_TEMPLATE,
            "trino": self.TRINO_TEMPLATE,
        }

        if custom_templates:
            self.templates.update(custom_templates)

    def build_prompt(
        self,
        question: str,
        db_structure: str,
        db_type: str = "postgresql",
        chat_history: Optional[str] = None
    ) -> str:
        """
        Build prompt for LLM

        Args:
            question: User's natural language question
            db_structure: Database structure information
            db_type: Type of database (postgresql, mysql, mongodb, sqlite)
            chat_history: Optional chat history context

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If db_type is not supported
        """
        db_type_lower = db_type.lower()

        if db_type_lower not in self.templates:
            raise ValueError(
                f"Unsupported database type: {db_type}. "
                f"Supported types: {', '.join(self.templates.keys())}"
            )

        template = self.templates[db_type_lower]

        prompt = template.format(
            db_structure=db_structure,
            question=question
        )

        # Add chat history if provided
        if chat_history:
            prompt = f"對話歷史：\n{chat_history}\n\n{prompt}"

        return prompt

    def set_template(self, db_type: str, template: str) -> None:
        """
        Set custom template for a specific database type

        Args:
            db_type: Database type
            template: Template string with {db_structure} and {question} placeholders
        """
        self.templates[db_type.lower()] = template

    def get_template(self, db_type: str) -> str:
        """
        Get template for a specific database type

        Args:
            db_type: Database type

        Returns:
            Template string
        """
        return self.templates.get(db_type.lower(), self.POSTGRESQL_TEMPLATE)