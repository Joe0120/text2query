from __future__ import annotations
from typing import Any, Dict, Literal, Optional
import logging

from langgraph.graph import END, StateGraph

from ..connections.postgresql import PostgreSQLConfig
from ..template_matching.store import TemplateStore
from ..memory.store import ConversationMemoryStore
from ..training.store import TrainingStore
from ..utils.model_configs import ModelConfig
from .human_appoval import human_explanation_node, need_human_approval
from .memory_node import MemoryNode
from .orchestrator import (
    orchestrator_node,
    reroute_based_on_confidence,
    route_by_current_move,
)
from .state import WisbiState, available_moves
from .template_node import TemplateNode
from .trainer_node import TrainerNode

logger = logging.getLogger(__name__)


class WisbiWorkflow:
    """
    WISBI Workflow class that builds and manages the LangGraph StateGraph workflow.
    
    Flow:
    1. memory_retrieve_node → orchestrator_node (retrieves conversation context first)
    2. orchestrator_node → trainer_node (training is priority first)
    3. trainer_node → (if score not high or training is empty) → template_node
                    → (otherwise) → executor_node
    4. template_node → (if score is low) → clarify_node → executor_node
                    → (otherwise) → executor_node
    5. clarify_node → executor_node (currently does nothing, just passes through)
    6. executor_node → memory_save_node (saves conversation after execution)
    7. memory_save_node → need_human_approval → (human_explanation_node OR END)
    8. human_explanation_node → executor_node (if approved) OR END (if rejected)
    
    Usage:
        workflow = WisbiWorkflow(
            db_config=db_config,
            template_llm_config=llm_config,
        )
        graph = await workflow.build()
        result = await graph.ainvoke(initial_state)
    """
    
    def __init__(
        self,
        db_config: PostgreSQLConfig,
        llm_config: ModelConfig,
        embedder_config: ModelConfig,
        *,
        template_schema: str = "wisbi",
        training_schema: str = "wisbi",
        memory_schema: str = "wisbi",
        embedding_dim: int = 768,
        executor_node: Optional[Any] = None,  # TODO: Define executor node type
        clarify_node: Optional[Any] = None,   # TODO: Define clarify node type
    ):
        """
        Initialize the WISBI workflow with configuration.
        
        Args:
            db_config: PostgreSQL connection configuration
            llm_config: ModelConfig for template embeddings (TemplateStore)
            embedder_config: ModelConfig for template embeddings (TemplateStore)
            template_schema: Schema name for templates table
            training_schema: Schema name for training tables
            memory_schema: Schema name for conversation history
            embedding_dim: Embedding vector dimension
            executor_node: Executor node instance (optional, can be a simple function)
            clarify_node: Clarify node instance (optional, can be a simple function)
        """
        self.db_config = db_config
        self.llm_config = llm_config
        self.embedder_config = embedder_config
        self.template_schema = template_schema
        self.training_schema = training_schema
        self.memory_schema = memory_schema
        self.embedding_dim = embedding_dim

        # Nodes are built lazily via build_nodes()
        self.template_node: Optional[TemplateNode] = None
        self.trainer_node: Optional[TrainerNode] = None
        self.memory_node: Optional[MemoryNode] = None
        self.executor_node = executor_node
        self.clarify_node = clarify_node
        self._compiled_graph: Optional[Any] = None
    
    async def build_nodes(self) -> None:
        """
        Build and initialize all internal nodes and their backing stores.
        
        This method wires:
          - TemplateNode
          - TrainingStore / TrainerNode
          - ConversationMemoryStore / MemoryNode
        """
        if self.template_node and self.trainer_node and self.memory_node:
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

        # Conversation memory store
        memory_store = await ConversationMemoryStore.initialize(
            postgres_config=self.db_config,
            memory_schema=self.memory_schema,
        )

        self.template_node = TemplateNode(template_store=template_store)
        self.trainer_node = TrainerNode(training_store=training_store)
        self.memory_node = MemoryNode(memory_store=memory_store)
    
    def _create_node_wrappers(self) -> Dict[str, Any]:
        """Create async wrapper functions for node classes."""

        if not self.template_node or not self.trainer_node or not self.memory_node:
            raise RuntimeError("Workflow nodes not initialized. Call await build() first.")

        async def template_node_wrapper(state: WisbiState) -> WisbiState:
            try:
                return await self.template_node(state)
            except Exception as e:
                logger.error(f"Error in template_node: {e}", exc_info=True)
                raise
        
        async def trainer_node_wrapper(state: WisbiState) -> WisbiState:
            logger.debug(f"Trainer node wrapper received state, training_score before: {state.get('training_score')}")
            result_state = await self.trainer_node(state)
            logger.debug(f"Trainer node wrapper returning state, training_score after: {result_state.get('training_score')}")
            return result_state
        
        async def memory_retrieve_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper for retrieving context"""
            if self.memory_node:
                return await self.memory_node.retrieve_context(state)
            return state
        
        async def memory_save_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper for saving message"""
            # Stub: just pass through
            return state
        
        async def default_executor_node(state: WisbiState) -> WisbiState:
            """Default executor node - extracts SQL from state and marks execution complete"""
            # Log confidence scores
            template_score = state.get("template_score")
            training_score = state.get("training_score")
            
            logger.info("=" * 60)
            logger.info("Executor Node - Final Confidence Scores:")
            if template_score is not None:
                logger.info(f"  Template Score: {template_score:.4f} (similarity)")
            else:
                logger.info("  Template Score: None (no template match)")
            
            if training_score is not None:
                logger.info(f"  Training Score: {training_score:.4f} (similarity)")
            else:
                logger.info("  Training Score: None (no training results)")
            logger.info("=" * 60)
            
            # Extract SQL from state (template_sql is set by template_node if available)
            template_sql = state.get("template_sql")
            if template_sql:
                state["sql"] = template_sql
                logger.info(f"Using template SQL in executor_node (confidence: {template_score:.4f})")
            # If no template_sql, sqlgen.py will generate it as fallback
            
            state["current_move"] = "executor"
            state["execution_success"] = True
            return state
        
        async def default_clarify_node(state: WisbiState) -> WisbiState:
            """Default clarify node - does nothing, just returns state"""
            # Currently does nothing, just passes through to executor_node
            return state
        
        executor = self.executor_node if self.executor_node else default_executor_node
        clarify = self.clarify_node if self.clarify_node else default_clarify_node
        
        return {
            "template_node": template_node_wrapper,
            "trainer_node": trainer_node_wrapper,
            "memory_retrieve_node": memory_retrieve_wrapper,
            "memory_save_node": memory_save_wrapper,
            "clarify_node": clarify,
            "executor_node": executor,
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
                explanation = state.get("human_explanation")
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

