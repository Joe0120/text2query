"""
Text-to-SQL/Query converter using HTTP request-based LLM calls
"""

from typing import Optional, Any, List, Dict
import re
import asyncio
import logging


class Text2SQL:
    """
    Convert natural language questions to database queries using HTTP request-based LLM calls

    This class uses ModelConfig and direct HTTP requests instead of LlamaIndex,
    making it more flexible and provider-agnostic.
    """

    def __init__(
        self,
        llm_config: Any = None,
        db_structure: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        db_type: str = "postgresql",
        adapter: Any = None,
        llm: Any = None,  # For backward compatibility with some tests
    ):
        """
        Initialize Text2SQL converter

        Args:
            llm_config: ModelConfig instance (from core.utils.model_configs)
            db_structure: Database structure information string
            chat_history: Optional list of chat messages
            db_type: Database type
            adapter: Database adapter instance for validation
            llm: Legacy LLM instance (optional)
        """
        self.llm_config = llm_config
        self.db_structure = db_structure
        self.chat_history = chat_history or []
        self.db_type = db_type.lower()
        self.adapter = adapter
        self.llm = llm or llm_config
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
        validate: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generate database query from natural language question with retry logic

        Args:
            question: User's natural language question
            include_history: Whether to include chat history in prompt
            stream_thinking: (deprecated, always False)
            show_thinking: (deprecated, always False)
            training_context: Context from SQL Training Data
            additional_context: Additional context from additional data
            validate: Whether to validate the generated query via adapter
            max_retries: Maximum number of retries if generation or validation fails
            retry_delay: Delay between retries in seconds

        Returns:
            Generated database query string

        Raises:
            Exception: If query generation fails after all retries
        """
        last_error = ""
        chat_history_text = self._format_chat_history() if include_history else None

        for attempt in range(max_retries):
            try:
                # Add error feedback to prompt if this is a retry
                current_training_context = training_context
                if attempt > 0 and last_error:
                    feedback = f"\n\n[Previous Attempts Failed]\nError: {last_error}\nPlease fix the SQL query according to this error."
                    current_training_context = (current_training_context or "") + feedback

                prompt = self.prompt_builder.build_prompt(
                    question=question,
                    db_structure=self.db_structure,
                    db_type=self.db_type,
                    chat_history=chat_history_text,
                    training_context=current_training_context,
                    additional_context=additional_context,
                )

                self.logger.info(f"Attempt {attempt + 1}/{max_retries} generating query for: {question[:50]}...")
                
                # Support legacy llm.achat if provided in some tests, otherwise use request-based agenerate_chat
                if self.llm and hasattr(self.llm, 'achat'):
                    # LlamaIndex style
                    response_obj = await self.llm.achat([{"role": "user", "content": prompt}])
                    generated_query = response_obj.message.content
                else:
                    generated_query = await self._generate_without_streaming(prompt)

                # Clean up the response
                generated_query = self._clean_query_response(generated_query)
                self.logger.info(f"Generated query: {generated_query}")

                # Optional validation
                if validate and self.adapter:
                    is_valid, error_msg = await self.adapter.validate_query(generated_query)
                    if not is_valid:
                        self.logger.warning(f"Validation failed for query: {error_msg}")
                        last_error = error_msg
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            raise Exception(f"Validation failed: {error_msg}")

                # Update chat history with this interaction
                if include_history:
                    self.chat_history.append({"role": "user", "content": question})
                    self.chat_history.append({"role": "assistant", "content": generated_query})

                return generated_query

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to generate query after {max_retries} attempts: {last_error}") from e

        raise Exception(f"Failed to generate query after {max_retries} attempts")

    # Add query method for compatibility with some usage patterns
    async def query(self, question: str, **kwargs) -> str:
        """Alias for generate_query"""
        return await self.generate_query(question, **kwargs)

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
