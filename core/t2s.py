"""
Text-to-SQL/Query converter using LlamaIndex LLM
"""

from typing import Optional, Any
import re
import logging


class Text2SQL:
    """
    Convert natural language questions to database queries using LlamaIndex LLM

    This class is designed to work with user-provided LlamaIndex LLM instances,
    making it flexible and allowing users to control their own LLM configuration.
    """

    def __init__(
        self,
        llm: Any,
        db_structure: str,
        chat_history: Optional[Any] = None,
        db_type: str = "postgresql"
    ):
        """
        Initialize Text2SQL converter

        Args:
            llm: LlamaIndex LLM instance (e.g., OpenAI, Anthropic, etc.)
            db_structure: Database structure information string
                         (from adapter.get_sql_struct_str())
            chat_history: Optional ChatMemoryBuffer.memory for conversation context
            db_type: Database type (postgresql, mysql, mongodb, sqlite)

        Example:
            >>> from llama_index.llms.openai import OpenAI
            >>> from text2query.core.t2s import Text2SQL
            >>>
            >>> llm = OpenAI(model="gpt-4", api_key="your-key")
            >>> db_structure = await adapter.get_sql_struct_str()
            >>> t2s = Text2SQL(llm=llm, db_structure=db_structure, db_type="mongodb")
            >>> query = await t2s.generate_query("每個部門有多少經理？")
        """
        self.llm = llm
        self.db_structure = db_structure
        self.chat_history = chat_history
        self.db_type = db_type.lower()
        self.logger = logging.getLogger("text2query.t2s")

        # Import prompt builder
        from ..llm.prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder()

    async def generate_query(
        self,
        question: str,
        include_history: bool = True,
        stream_thinking: bool = False,
        show_thinking: bool = False
    ) -> str:
        """
        Generate database query from natural language question

        Args:
            question: User's natural language question
            include_history: Whether to include chat history in prompt
            stream_thinking: Whether to stream the LLM thinking process (deprecated, always False)
            show_thinking: Whether to print the thinking process (deprecated, always False)

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
                chat_history=chat_history_text
            )

            self.logger.info(f"Generating query for question: {question[:50]}...")

            # Always use non-streaming mode for clean async execution
            generated_query = await self._generate_without_streaming(prompt)

            # Clean up the response
            generated_query = self._clean_query_response(generated_query)

            self.logger.info(f"Generated query: {generated_query[:100]}...")

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

        Args:
            prompt: The prompt to send to LLM
            show_thinking: Whether to print thinking process to console

        Returns:
            Complete generated query string
        """
        from llama_index.core.llms import ChatMessage

        # Prepare chat message
        messages = [ChatMessage(role="user", content=prompt)]

        # Stream the response
        if show_thinking:
            print("\n🤔 LLM 思考過程:")
            print("-" * 60)

        full_response = ""

        try:
            # Use stream_chat for streaming response
            response_stream = await self.llm.astream_chat(messages)

            async for chunk in response_stream:
                # Try different ways to get the content from chunk
                chunk_text = None
                
                # Method 1: Try delta attribute (for some LLMs)
                if hasattr(chunk, 'delta') and chunk.delta:
                    chunk_text = chunk.delta
                # Method 2: Try message.content (for complete chunks)
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    # For complete message chunks, we need to get the delta
                    # by comparing with previous response
                    if full_response:
                        chunk_text = chunk.message.content[len(full_response):]
                    else:
                        chunk_text = chunk.message.content
                # Method 3: Direct content attribute
                elif hasattr(chunk, 'content') and chunk.content:
                    chunk_text = chunk.content
                # Method 4: String representation
                else:
                    chunk_text = str(chunk)
                
                if chunk_text:
                    full_response += chunk_text
                    if show_thinking:
                        print(chunk_text, end="", flush=True)

            if show_thinking:
                print("\n" + "-" * 60)
                print("✅ 思考完成\n")

        except Exception as e:
            # Catch all exceptions and try non-streaming fallback
            error_msg = str(e)
            self.logger.warning(f"Streaming failed ({type(e).__name__}: {error_msg}), trying non-streaming")
            
            if show_thinking:
                print(f"\n⚠️  串流模式失敗，切換到一般模式...")
                print("-" * 60)
            
            try:
                # Fallback: try non-streaming
                response = await self.llm.achat(messages)
                full_response = response.message.content
                if show_thinking:
                    print(full_response)
                    print("\n" + "-" * 60)
                    print("✅ 思考完成\n")
            except Exception as fallback_e:
                # If even non-streaming fails, raise the original error
                self.logger.error(f"Both streaming and non-streaming failed. Original error: {e}")
                raise e

        return full_response

    async def _generate_without_streaming(self, prompt: str) -> str:
        """
        Generate query without streaming (returns complete response at once)

        Args:
            prompt: The prompt to send to LLM

        Returns:
            Generated query string
        """
        from llama_index.core.llms import ChatMessage

        # Prepare chat message
        messages = [ChatMessage(role="user", content=prompt)]

        # Call LLM in chat mode
        response = await self.llm.achat(messages)

        return response.message.content

    def _format_chat_history(self) -> str:
        """
        Format chat history from ChatMemoryBuffer.memory format

        Returns:
            Formatted chat history string
        """
        if not self.chat_history:
            return ""

        history_lines = []
        try:
            # ChatMemoryBuffer.memory is typically a list of messages
            if hasattr(self.chat_history, 'get_all'):
                messages = self.chat_history.get_all()
            elif isinstance(self.chat_history, list):
                messages = self.chat_history
            else:
                return ""

            for msg in messages:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', str(msg))
                history_lines.append(f"{role}: {content}")

            return "\n".join(history_lines[-10:])  # Last 10 messages

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

    def update_chat_history(self, chat_history: Any) -> None:
        """
        Update chat history

        Args:
            chat_history: New chat history (ChatMemoryBuffer.memory format)
        """
        self.chat_history = chat_history
        self.logger.info("Chat history updated")

    def set_db_type(self, db_type: str) -> None:
        """
        Change database type

        Args:
            db_type: New database type (postgresql, mysql, mongodb, sqlite)
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
            "has_chat_history": self.chat_history is not None,
            "db_structure_length": len(self.db_structure),
            "llm_type": type(self.llm).__name__
        }