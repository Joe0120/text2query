"""
Text-to-SQL/Query converter using HTTP request-based LLM calls
"""

from typing import Optional, Any, List, Dict
import re
import logging


class Text2SQL:
    """
    Convert natural language questions to database queries using HTTP request-based LLM calls

    This class uses ModelConfig and direct HTTP requests instead of LlamaIndex,
    making it more flexible and provider-agnostic.
    """

    def __init__(
        self,
        llm_config: Any,
        db_structure: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        db_type: str = "postgresql"
    ):
        """
        Initialize Text2SQL converter

        Args:
            llm_config: ModelConfig instance (from core.utils.model_configs)
            db_structure: Database structure information string
                         (from adapter.get_schema_str())
            chat_history: Optional list of chat messages in format
                         [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            db_type: Database type (postgresql, mysql, mongodb, sqlite, sqlserver, oracle)

        Example:
            >>> from text2query.core.utils.model_configs import ModelConfig
            >>> from text2query.core.t2s import Text2SQL
            >>>
            >>> llm_config = ModelConfig(
            ...     base_url="http://localhost:11434",
            ...     endpoint="/api/chat",
            ...     api_key="",
            ...     model_name="gemma3:1b",
            ...     provider="ollama"
            ... )
            >>> db_structure = await adapter.get_schema_str()
            >>> t2s = Text2SQL(llm_config=llm_config, db_structure=db_structure, db_type="mongodb")
            >>> query = await t2s.generate_query("每個部門有多少經理？")
        """
        self.llm_config = llm_config
        self.db_structure = db_structure
        self.chat_history = chat_history or []
        self.db_type = db_type.lower()
        self.logger = logging.getLogger("text2query.t2s")

        # Import prompt builder and model utilities
        from ..llm.prompt_builder import PromptBuilder
        from .utils.models import agenerate_chat
        
        self.prompt_builder = PromptBuilder()
        self.agenerate_chat = agenerate_chat

    async def generate_query(
        self,
        question: str,
        include_history: bool = True,
        stream_thinking: bool = False,
        show_thinking: bool = False,
        training_context: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Generate database query from natural language question

        Args:
            question: User's natural language question
            include_history: Whether to include chat history in prompt
            stream_thinking: Whether to stream the LLM thinking process (deprecated, always False)
            show_thinking: Whether to print the thinking process (deprecated, always False)
            training_context: Context from SQL Training Data
            additional_context: Additional context from additional data

        Returns:
            Generated database query string

        Raises:
            Exception: If query generation fails

        Example:
            >>> query = await t2s.generate_query("查詢所有用戶")
            >>> print(query)
            SELECT * FROM users
        """
        try:
            # Build prompt
            chat_history_text = None
            if include_history and self.chat_history:
                chat_history_text = self._format_chat_history()

            prompt = self.prompt_builder.build_prompt(
                question=question,
                db_structure=self.db_structure,
                db_type=self.db_type,
                chat_history=chat_history_text,
                training_context=training_context,
                additional_context=additional_context,
            )

            self.logger.info(f"Generating query for question: {question[:50]}..., with Prompt: {prompt[:100]}...")

            # Generate query using request-based approach
            generated_query = await self._generate_without_streaming(prompt)

            # Clean up the response
            generated_query = self._clean_query_response(generated_query)

            self.logger.info(f"Generated query: {generated_query}...")

            # Update chat history with this interaction
            if include_history:
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": generated_query})

            return generated_query

        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            raise Exception(f"Failed to generate query: {str(e)}") from e

    async def _generate_with_streaming(
        self,
        prompt: str,
        show_thinking: bool = True
    ) -> str:
        """
        Generate query with streaming output (shows thinking process)
        
        Note: Streaming is not currently supported in the request-based approach.
        This method falls back to non-streaming mode.

        Args:
            prompt: The prompt to send to LLM
            show_thinking: Whether to print thinking process to console

        Returns:
            Complete generated query string
        """
        self.logger.warning("Streaming not supported in request-based approach, using non-streaming mode")
        
        if show_thinking:
            print("\n🤔 LLM 思考過程:")
            print("-" * 60)

        try:
            response = await self._generate_without_streaming(prompt)
            
            if show_thinking:
                print(response)
                print("\n" + "-" * 60)
                print("✅ 思考完成\n")
            
            return response
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            if show_thinking:
                print(f"\n❌ 生成失敗: {e}\n")
            raise

    async def _generate_without_streaming(self, prompt: str) -> str:
        """
        Generate query without streaming (returns complete response at once)

        Args:
            prompt: The prompt to send to LLM

        Returns:
            Generated query string
        """
        # Prepare chat message in request format
        messages = [{"role": "user", "content": prompt}]

        # Call LLM using request-based approach
        response = await self.agenerate_chat(self.llm_config, messages)

        return response

    def _format_chat_history(self) -> str:
        """
        Format chat history from message list format

        Returns:
            Formatted chat history string
        """
        if not self.chat_history:
            return ""

        history_lines = []
        try:
            # Format last 10 messages (5 exchanges)
            recent_history = self.chat_history[-10:] if len(self.chat_history) > 10 else self.chat_history
            
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", str(msg))
                history_lines.append(f"{role}: {content}")

            return "\n".join(history_lines)

        except Exception as e:
            self.logger.warning(f"Failed to format chat history: {e}")
            return ""

    def _clean_query_response(self, query: str) -> str:
        """
        Clean up LLM response by removing markdown code blocks and extra whitespace

        Args:
            query: Raw query from LLM

        Returns:
            Cleaned query string
        """
        # Remove markdown code blocks
        if query.startswith("```"):
            # Remove code block markers
            query = re.sub(r'^```\w*\n', '', query)
            query = re.sub(r'\n```$', '', query)

        # Remove language identifiers
        query = query.replace("```sql", "").replace("```mongodb", "")
        query = query.replace("```json", "").replace("```", "")

        # Strip whitespace
        query = query.strip()

        return query

    def update_db_structure(self, db_structure: str) -> None:
        """
        Update database structure information

        Args:
            db_structure: New database structure string
        """
        self.db_structure = db_structure
        self.logger.info("Database structure updated")

    def update_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> None:
        """
        Update chat history

        Args:
            chat_history: New chat history as list of message dicts
                         Format: [{"role": "user", "content": "..."}, ...]
        """
        self.chat_history = chat_history or []
        self.logger.info("Chat history updated")

    def set_db_type(self, db_type: str) -> None:
        """
        Change database type

        Args:
            db_type: New database type (postgresql, mysql, mongodb, sqlite, sqlserver, oracle)
        """
        self.db_type = db_type.lower()
        self.logger.info(f"Database type changed to: {self.db_type}")

    def set_custom_template(self, template: str) -> None:
        """
        Set custom prompt template for current database type

        Args:
            template: Custom prompt template with {db_structure} and {question} placeholders
        """
        self.prompt_builder.set_template(self.db_type, template)
        self.logger.info(f"Custom template set for {self.db_type}")

    def get_config(self) -> dict:
        """
        Get current configuration

        Returns:
            Configuration dictionary
        """
        return {
            "db_type": self.db_type,
            "has_chat_history": len(self.chat_history) > 0,
            "chat_history_length": len(self.chat_history),
            "db_structure_length": len(self.db_structure),
            "llm_provider": self.llm_config.provider if hasattr(self.llm_config, 'provider') else "unknown",
            "llm_model": self.llm_config.model_name if hasattr(self.llm_config, 'model_name') else "unknown"
        }
