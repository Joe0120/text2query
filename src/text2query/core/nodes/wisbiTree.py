from __future__ import annotations
from typing import Any, Dict, Literal, Optional
import logging
import os

from langgraph.graph import END, StateGraph

from ..connections.postgresql import PostgreSQLConfig
from ..connections.factory import load_config_from_url
from ..template_matching.store import TemplateStore
from ..training.store import TrainingStore
from ..utils.model_configs import ModelConfig, create_model_config, Provider
from .executor_node import executor_node
from .human_approval import human_explanation_node, need_human_approval
from .orchestrator import (
    orchestrator_node,
    reroute_based_on_confidence,
    route_by_current_move,
)
from .state import WisbiState
from .template_node import TemplateNode
from .trainer_node import TrainerNode

logger = logging.getLogger(__name__)


def _load_db_config_from_env() -> Optional[PostgreSQLConfig]:
    """
    Load PostgreSQL configuration from environment variables.
    
    Supports the following environment variables (in order of precedence):
    1. DATABASE_URL - Full connection string (postgresql://user:pass@host:port/dbname)
    2. POSTGRES_CONNECTION_STRING - PostgreSQL-specific connection string
    3. Individual parameters:
       - DB_HOST or POSTGRES_HOST
       - DB_PORT or POSTGRES_PORT
       - DB_NAME or POSTGRES_DB or POSTGRES_DATABASE
       - DB_USER or POSTGRES_USER
       - DB_PASSWORD or POSTGRES_PASSWORD
       - DB_SCHEMA or POSTGRES_SCHEMA (optional, defaults to "public")
       - DB_SSL_MODE or POSTGRES_SSL_MODE (optional, defaults to "disable")
    
    Returns:
        PostgreSQLConfig if environment variables are found, None otherwise
    """
    # Try DATABASE_URL first (most common)
    database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_CONNECTION_STRING")
    if database_url:
        try:
            # Use the factory function to parse the URL
            config = load_config_from_url(database_url)
            if isinstance(config, PostgreSQLConfig):
                return config
            # If it's not PostgreSQL, log warning and fall through to individual env vars
            logger.warning(
                f"Database URL points to {type(config).__name__}, but PostgreSQL is expected. "
                "Falling back to individual environment variables."
            )
        except Exception as e:
            logger.warning(f"Failed to parse DATABASE_URL: {e}. Falling back to individual environment variables.")
    
    # Try individual environment variables
    # Support both generic DB_* and PostgreSQL-specific POSTGRES_* variables
    host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST")
    port_str = os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT")
    database_name = (
        os.getenv("DB_NAME") 
        or os.getenv("POSTGRES_DB") 
        or os.getenv("POSTGRES_DATABASE")
    )
    username = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    schema = os.getenv("DB_SCHEMA") or os.getenv("POSTGRES_SCHEMA")
    ssl_mode = os.getenv("DB_SSL_MODE") or os.getenv("POSTGRES_SSL_MODE", "disable")
    
    # Check if we have minimum required parameters
    if not database_name:
        return None
    
    # Build config with available parameters
    config_kwargs = {
        "database_name": database_name,
    }
    
    if host:
        config_kwargs["host"] = host
    
    if port_str:
        try:
            config_kwargs["port"] = int(port_str)
        except ValueError:
            logger.warning(f"Invalid port value: {port_str}, using default 5432")
    
    if username:
        config_kwargs["username"] = username
    
    if password is not None:  # Allow empty string
        config_kwargs["password"] = password
    
    if schema:
        config_kwargs["schema"] = schema
    
    if ssl_mode:
        config_kwargs["ssl_mode"] = ssl_mode
    
    return PostgreSQLConfig(**config_kwargs)


class WisbiWorkflow:
    """
    WISBI Workflow class that builds and manages the LangGraph StateGraph workflow.
    
    Flow:
    1. memory_retrieve_node → orchestrator_node (retrieves conversation context first)
    2. orchestrator_node → routes based on current_move (trainer_node is default/priority)
    3. trainer_node → (if score >= 0.5 and results available) → executor_node
                    → (otherwise) → template_node
    4. template_node → executor_node (always routes to executor)
    5. clarify_node → executor_node (currently does nothing, just passes through)
    6. executor_node → memory_save_node (saves conversation after execution)
    7. memory_save_node → need_human_approval → (human_explanation_node OR END)
    8. human_explanation_node → orchestrator (if explanation provided) → executor_node (if approved) OR END (if rejected)
    
    Usage:
        # Option 1: Using config objects
        workflow = WisbiWorkflow(
            db_config=db_config,
            llm_config=llm_config,
            embedder_config=embedder_config,
        )
        
        # Option 2: Using individual parameters
        workflow = WisbiWorkflow(
            db_config=db_config,
            llm_name="gpt-4",
            llm_api_base="https://api.openai.com",
            llm_api_key="sk-...",
            embed_model_name="text-embedding-ada-002",
            embed_api_base="https://api.openai.com",
            embed_api_key="sk-...",
        )
        
        graph = await workflow.build()
        result = await graph.ainvoke(initial_state)
    """
    
    def __init__(
        self,
        db_config: Optional[PostgreSQLConfig] = None,
        llm_config: Optional[ModelConfig] = None,
        embedder_config: Optional[ModelConfig] = None,
        *,
        # Individual LLM parameters (used if llm_config is not provided)
        llm_name: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        llm_provider: Provider = "openai",
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = 0.2,
        llm_timeout: int = 30,
        # Individual embedder parameters (used if embedder_config is not provided)
        embed_model_name: Optional[str] = None,
        embed_api_base: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_endpoint: Optional[str] = None,
        embed_provider: Provider = "openai",
        embed_max_tokens: Optional[int] = None,
        embed_temperature: Optional[float] = 0.2,
        embed_timeout: int = 30,
        # Workflow configuration
        template_schema: str = "wisbi",
        training_schema: str = "wisbi",
        memory_schema: str = "wisbi",
        embedding_dim: int = 768,
    ):
        """
        Initialize the WISBI workflow with configuration.
        
        Supports multiple initialization patterns:
        1. Config-based: Pass db_config, llm_config, and embedder_config directly
        2. Parameter-based: Pass individual parameters (llm_name, llm_api_base, etc.)
        3. Environment variables: Set DATABASE_URL or DB_* / POSTGRES_* environment variables
        
        Note: If both llm_config and individual LLM parameters are provided, llm_config takes precedence.
        Similarly, if both embedder_config and individual embedder parameters are provided, embedder_config takes precedence.
        For db_config, if not provided, the system will attempt to load from environment variables.
        
        Args:
            db_config: PostgreSQL connection configuration (optional if environment variables are set)
            llm_config: ModelConfig for LLM (optional if individual LLM params provided, takes precedence if both provided)
            embedder_config: ModelConfig for embeddings (optional if individual embedder params provided, takes precedence if both provided)
            
            # Individual LLM parameters (alternative to llm_config)
            llm_name: LLM model name (e.g., "gpt-4", "llama2")
            llm_api_base: Base URL for LLM API (e.g., "https://api.openai.com")
            llm_api_key: API key for LLM authentication
            llm_endpoint: API endpoint path (defaults based on provider)
            llm_provider: LLM provider type ("openai" or "ollama", default: "openai")
            llm_max_tokens: Maximum tokens for LLM responses
            llm_temperature: LLM temperature (default: 0.2)
            llm_timeout: LLM request timeout in seconds (default: 30)
            
            # Individual embedder parameters (alternative to embedder_config)
            embed_model_name: Embedding model name (e.g., "text-embedding-ada-002")
            embed_api_base: Base URL for embedding API
            embed_api_key: API key for embedding authentication
            embed_endpoint: API endpoint path (defaults based on provider)
            embed_provider: Embedding provider type ("openai" or "ollama", default: "openai")
            embed_max_tokens: Maximum tokens for embedding requests
            embed_temperature: Embedding temperature (default: 0.2)
            embed_timeout: Embedding request timeout in seconds (default: 30)
            
            # Workflow configuration
            template_schema: Schema name for templates table
            training_schema: Schema name for training tables
            memory_schema: Schema name for conversation history
            embedding_dim: Embedding vector dimension
        
        Raises:
            ValueError: If required configuration is missing (neither config object, individual parameters, nor environment variables provided)
        """
        # Validate and set db_config
        # Try to load from environment variables if not provided
        if db_config is None:
            db_config = _load_db_config_from_env()
        
        if db_config is None:
            raise ValueError(
                "db_config is required. Either provide db_config parameter, or set environment variables: "
                "DATABASE_URL (or POSTGRES_CONNECTION_STRING), or individual DB_* / POSTGRES_* variables."
            )
        self.db_config = db_config
        
        # Validate and create llm_config
        # llm_config takes precedence if both are provided
        if llm_config is not None:
            self.llm_config = llm_config
        elif llm_name and llm_api_base and llm_api_key is not None:
            # Parameter-based initialization - validate non-empty strings
            if not llm_name.strip():
                raise ValueError("llm_name cannot be empty")
            if not llm_api_base.strip():
                raise ValueError("llm_api_base cannot be empty")
            # llm_api_key can be empty string (e.g., for Ollama)
            self.llm_config = create_model_config(
                model_name=llm_name,
                api_base=llm_api_base,
                apikey=llm_api_key,
                endpoint=llm_endpoint,
                provider=llm_provider,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                timeout=llm_timeout,
            )
        else:
            raise ValueError(
                "Either llm_config or all of (llm_name, llm_api_base, llm_api_key) must be provided."
            )
        
        # Validate and create embedder_config
        # embedder_config takes precedence if both are provided
        if embedder_config is not None:
            self.embedder_config = embedder_config
        elif embed_model_name and embed_api_base and embed_api_key is not None:
            # Parameter-based initialization - validate non-empty strings
            if not embed_model_name.strip():
                raise ValueError("embed_model_name cannot be empty")
            if not embed_api_base.strip():
                raise ValueError("embed_api_base cannot be empty")
            # embed_api_key can be empty string (e.g., for Ollama)
            self.embedder_config = create_model_config(
                model_name=embed_model_name,
                api_base=embed_api_base,
                apikey=embed_api_key,
                endpoint=embed_endpoint,
                provider=embed_provider,
                max_tokens=embed_max_tokens,
                temperature=embed_temperature,
                timeout=embed_timeout,
            )
        else:
            raise ValueError(
                "Either embedder_config or all of (embed_model_name, embed_api_base, embed_api_key) must be provided."
            )
        self.template_schema = template_schema
        self.training_schema = training_schema
        self.memory_schema = memory_schema
        self.embedding_dim = embedding_dim

        # Nodes are built lazily via build_nodes()
        self.template_node: Optional[TemplateNode] = None
        self.trainer_node: Optional[TrainerNode] = None
        self._compiled_graph: Optional[Any] = None
    
    async def build_nodes(self) -> None:
        """
        Build and initialize all internal nodes and their backing stores.
        
        This method wires:
          - TemplateNode
          - TrainingStore / TrainerNode
        """
        if self.template_node and self.trainer_node:
            # Nodes already built
            return
        
        # Template store
        template_store = await TemplateStore.initialize(
            postgres_config=self.db_config,
            template_schema=self.template_schema,
            embedding_dim=self.embedding_dim,
        )
        # RAG training store
        training_store = await TrainingStore.initialize(
            postgres_config=self.db_config,
            training_schema=self.training_schema,
            embedding_dim=self.embedding_dim,
        )

        self.template_node = TemplateNode(template_store=template_store)
        self.trainer_node = TrainerNode(training_store=training_store)
    
    def _create_node_wrappers(self) -> Dict[str, Any]:
        """Create async wrapper functions for node classes."""

        if not self.template_node or not self.trainer_node:
            raise RuntimeError("Workflow nodes not initialized. Call await build() first.")

        async def template_node_wrapper(state: WisbiState) -> WisbiState:
            try:
                return await self.template_node(state)
            except Exception as e:
                logger.error(f"Error in template_node: {e}", exc_info=True)
                raise
        
        async def trainer_node_wrapper(state: WisbiState) -> WisbiState:
            try:
                logger.debug(f"Trainer node wrapper received state, training_score before: {state.get('training_score')}")
                result_state = await self.trainer_node(state)
                logger.debug(f"Trainer node wrapper returning state, training_score after: {result_state.get('training_score')}")
                return result_state
            except Exception as e:
                logger.error(f"Error in trainer_node: {e}", exc_info=True)
                raise
        
        async def memory_retrieve_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper - stub that passes through state"""
            return state
        
        async def memory_save_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper - stub that passes through state"""
            return state
        
        async def default_clarify_node(state: WisbiState) -> WisbiState:
            """Default clarify node - does nothing, just returns state"""
            # Currently does nothing, just passes through to executor_node
            return state
        
        return {
            "template_node": template_node_wrapper,
            "trainer_node": trainer_node_wrapper,
            "memory_retrieve_node": memory_retrieve_wrapper,
            "memory_save_node": memory_save_wrapper,
            "clarify_node": default_clarify_node,
            "executor_node": executor_node,
        }
    
    async def build(self) -> Any:
        """
        Builds and compiles the WISBI workflow graph.
        
        Returns:
            Compiled StateGraph ready to execute
        """
        if self._compiled_graph is not None:
            return self._compiled_graph

        # Lazily build nodes and their stores
        await self.build_nodes()
        
        node_wrappers = self._create_node_wrappers()
        
        # Create the workflow graph
        workflow = StateGraph(WisbiState)
        
        # Add all nodes
        workflow.add_node("orchestrator", orchestrator_node)
        workflow.add_node("template_node", node_wrappers["template_node"])
        workflow.add_node("trainer_node", node_wrappers["trainer_node"])
        workflow.add_node("memory_retrieve_node", node_wrappers["memory_retrieve_node"])
        workflow.add_node("memory_save_node", node_wrappers["memory_save_node"])
        workflow.add_node("clarify_node", node_wrappers["clarify_node"])
        workflow.add_node("executor_node", node_wrappers["executor_node"])
        workflow.add_node("human_explanation_node", human_explanation_node)
        
        workflow.set_entry_point("memory_retrieve_node")
        
        workflow.add_edge("memory_retrieve_node", "orchestrator")
        
        workflow.add_conditional_edges(
            "orchestrator",
            route_by_current_move,
            {
                "template_node": "template_node",
                "trainer_node": "trainer_node",
                "clarify_node": "clarify_node",
                "executor_node": "executor_node",
            }
        )
        
        # After trainer_node: if score not high or training is empty -> template_node, else -> executor_node
        workflow.add_conditional_edges(
            "trainer_node",
            reroute_based_on_confidence,
            {
                "template_node": "template_node",
                "executor_node": "executor_node",
            }
        )
        
        # After template_node: always go to executor_node
        workflow.add_edge("template_node", "executor_node")
        
        # clarify_node routes directly to executor_node (if used in future)
        workflow.add_edge("clarify_node", "executor_node")
        
        workflow.add_edge("executor_node", "memory_save_node")
        
        workflow.add_conditional_edges(
            "memory_save_node",
            need_human_approval,
            {
                "human_approval": "human_explanation_node",
                END: END,
            }
        )
        
        def route_after_human_interaction(state: WisbiState) -> Literal["orchestrator", "executor_node", END]:
            """Route after human approval/explanation"""
            if state.get("human_explanation"):
                state["human_explanation"] = None
                return "orchestrator"
            
            if state.get("human_approved") is True:
                return "executor_node"
            elif state.get("human_approved") is False:
                return END
            return END
        
        workflow.add_conditional_edges(
            "human_explanation_node",
            route_after_human_interaction,
            {
                "orchestrator": "orchestrator",
                "executor_node": "executor_node", 
                END: END, 
            }
        )
        
        self._compiled_graph = workflow.compile()
        return self._compiled_graph
    
    @property
    def graph(self) -> Any:
        """
        Property to access the compiled graph.
        
        Note:
            You must call `await workflow.build()` before accessing this property.
        
        Returns:
            Compiled StateGraph
        """
        if self._compiled_graph is None:
            raise RuntimeError("Graph not built. Call await workflow.build() first.")
        return self._compiled_graph

