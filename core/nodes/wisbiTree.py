from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from langgraph.graph import StateGraph, END
from .state import WisbiState, available_moves
from .orchestrator import orchestrator_node, reroute_based_on_confidence, route_by_current_move
from .template_node import TemplateNode
from .trainer_node import TrainerNode
from .memory_node import MemoryNode
from .human_appoval import human_explanation_node, need_human_approval


class WisbiWorkflow:
    """
    WISBI Workflow class that builds and manages the LangGraph StateGraph workflow.
    
    Flow:
    1. memory_retrieve_node → orchestrator_node (retrieves conversation context first)
    2. orchestrator_node → routes to (trainer_node/template_node/clarify_node/executor_node)
    3. After trainer/template → reroute_based_on_confidence → (orchestrator for next node OR executor_node)
       - Training is priority first, template is fallback
    4. executor_node → memory_save_node (saves conversation after execution)
    5. memory_save_node → need_human_approval → (human_explanation_node OR END)
    6. clarify_node → human_explanation_node → (gets explanation) → orchestrator_node (restart workflow)
    7. human_explanation_node → executor_node (if approved) OR END (if rejected)
    
    Usage:
        workflow = WisbiWorkflow(
            template_node=template_node,
            trainer_node=trainer_node,
            memory_node=memory_node,
        )
        graph = workflow.build()
        result = await graph.ainvoke(initial_state)
    """
    
    def __init__(
        self,
        template_node: TemplateNode,
        trainer_node: TrainerNode,
        memory_node: Optional[MemoryNode] = None,
        executor_node: Optional[Any] = None,  # TODO: Define executor node type
        clarify_node: Optional[Any] = None,   # TODO: Define clarify node type
    ):
        """
        Initialize the WISBI workflow with required nodes.
        
        Args:
            template_node: TemplateNode instance
            trainer_node: TrainerNode instance
            memory_node: MemoryNode instance (optional, for conversation history)
            executor_node: Executor node instance (optional, can be a simple function)
            clarify_node: Clarify node instance (optional, can be a simple function)
        """
        self.template_node = template_node
        self.trainer_node = trainer_node
        self.memory_node = memory_node
        self.executor_node = executor_node
        self.clarify_node = clarify_node
        self._compiled_graph: Optional[Any] = None
    
    def _create_node_wrappers(self) -> Dict[str, Any]:
        """Create async wrapper functions for node classes."""
        async def template_node_wrapper(state: WisbiState) -> WisbiState:
            return await self.template_node(state)
        
        async def trainer_node_wrapper(state: WisbiState) -> WisbiState:
            return await self.trainer_node(state)
        
        async def memory_retrieve_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper for retrieving context"""
            if self.memory_node:
                return await self.memory_node.retrieve_context(state)
            return state
        
        async def memory_save_wrapper(state: WisbiState) -> WisbiState:
            """Memory node wrapper for saving message"""
            if self.memory_node:
                return await self.memory_node.save_message(state)
            return state
        
        async def default_executor_node(state: WisbiState) -> WisbiState:
            """Default executor node - can be replaced with actual implementation"""
            # TODO: Implement actual query execution
            state["current_move"] = "executor"
            return state
        
        async def default_clarify_node(state: WisbiState) -> WisbiState:
            """Default clarify node - sets flag to request human explanation"""
            state["clarification_requested"] = True
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
    
    def build(self) -> Any:
        """
        Builds and compiles the WISBI workflow graph.
        
        Returns:
            Compiled StateGraph ready to execute
        """
        if self._compiled_graph is not None:
            return self._compiled_graph
        
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
        
        workflow.add_conditional_edges(
            "trainer_node",
            reroute_based_on_confidence,
            {
                "orchestrator": "orchestrator",
                "executor_node": "executor_node",
            }
        )
        
        workflow.add_conditional_edges(
            "template_node",
            reroute_based_on_confidence,
            {
                "orchestrator": "orchestrator",
                "executor_node": "executor_node",
            }
        )
        
        workflow.add_edge("clarify_node", "human_explanation_node")
        
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
        Property to access the compiled graph. Builds it if not already built.
        
        Returns:
            Compiled StateGraph
        """
        return self.build()

