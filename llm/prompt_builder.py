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
        }

        if custom_templates:
            self.templates.update(custom_templates)

    def build_prompt(
        self,
        question: str,
        db_structure: str,
        db_type: str = "postgresql",
        chat_history: Optional[str] = None,
        training_context: Optional[str] = None,
        additional_context: Optional[str] = None,
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

        # Add retrieved training context if provided
        if training_context:
            prompt = (
                f"{prompt}\n\n"
                f"以下是關於這些資料表的額外說明。它們可能會出現在「問題與 SQL 配對」、「文件說明」以及「常用 SQL」中，請將它們作為參考依據使用：\n"
                f"{training_context}"
            )

        if additional_context:
            prompt = (
                f"{prompt}\n\n"
                f"以下是根據使用者問題所產生的 SQL 查詢範例提示。"
                f"{additional_context}"
            )

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