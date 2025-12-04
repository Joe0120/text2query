from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from langgraph.graph import StateGraph, END
from .state import WisbiState, available_moves
from .orchestrator import orchestrator_node, reroute_based_on_confidence, route_by_current_move
from .rag_node import RAGNode
from .template_node import TemplateNode
from .trainer_node import TrainerNode
from .human_appoval import human_explanation_node, need_human_approval


class WisbiWorkflow:
    """
    WISBI Workflow class that builds and manages the LangGraph StateGraph workflow.
    
    Flow:
    1. orchestrator_node → routes to (rag_node/template_node/trainer_node/clarify_node/executor_node)
    2. After rag/template/trainer → reroute_based_on_confidence → (orchestrator for next node OR executor_node)
    3. clarify_node → human_explanation_node → (gets explanation) → orchestrator_node (restart workflow)
    4. executor_node → need_human_approval → (human_approval OR END)
    5. human_approval → executor_node (if approved) OR END (if rejected)
    
    Usage:
        workflow = WisbiWorkflow(
            rag_node=rag_node,
            template_node=template_node,
            trainer_node=trainer_node,
        )
        graph = workflow.build()
        result = await graph.ainvoke(initial_state)
    """
    
    def __init__(
        self,
        rag_node: RAGNode,
        template_node: TemplateNode,
        trainer_node: TrainerNode,
        executor_node: Optional[Any] = None,  # TODO: Define executor node type
        clarify_node: Optional[Any] = None,   # TODO: Define clarify node type
    ):
        """
        Initialize the WISBI workflow with required nodes.
        
        Args:
            rag_node: RAGNode instance
            template_node: TemplateNode instance
            trainer_node: TrainerNode instance
            executor_node: Executor node instance (optional, can be a simple function)
            clarify_node: Clarify node instance (optional, can be a simple function)
        """
        self.rag_node = rag_node
        self.template_node = template_node
        self.trainer_node = trainer_node
        self.executor_node = executor_node
        self.clarify_node = clarify_node
        self._compiled_graph: Optional[Any] = None
    
    def _create_node_wrappers(self) -> Dict[str, Any]:
        """Create async wrapper functions for node classes."""
        async def rag_node_wrapper(state: WisbiState) -> WisbiState:
            return await self.rag_node(state)
        
        async def template_node_wrapper(state: WisbiState) -> WisbiState:
            return await self.template_node(state)
        
        async def trainer_node_wrapper(state: WisbiState) -> WisbiState:
            return await self.trainer_node(state)
        
        # Default executor node if not provided
        async def default_executor_node(state: WisbiState) -> WisbiState:
            """Default executor node - can be replaced with actual implementation"""
            # TODO: Implement actual query execution
            state["current_move"] = "executor"
            return state
        
        # Default clarify node if not provided
        async def default_clarify_node(state: WisbiState) -> WisbiState:
            """Default clarify node - sets flag to request human explanation"""
            # Set flag to indicate clarification is needed
            state["clarification_requested"] = True
            # Route to human_approval_node to get explanation
            return state
        
        executor = self.executor_node if self.executor_node else default_executor_node
        clarify = self.clarify_node if self.clarify_node else default_clarify_node
        
        return {
            "rag_node": rag_node_wrapper,
            "template_node": template_node_wrapper,
            "trainer_node": trainer_node_wrapper,
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
        workflow.add_node("rag_node", node_wrappers["rag_node"])
        workflow.add_node("template_node", node_wrappers["template_node"])
        workflow.add_node("trainer_node", node_wrappers["trainer_node"])
        workflow.add_node("clarify_node", node_wrappers["clarify_node"])
        workflow.add_node("executor_node", node_wrappers["executor_node"])
        workflow.add_node("human_explanation_node", human_explanation_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Orchestrator routes to appropriate node based on current_move
        workflow.add_conditional_edges(
            "orchestrator",
            route_by_current_move,
            {
                "rag_node": "rag_node",
                "template_node": "template_node",
                "trainer_node": "trainer_node",
                "clarify_node": "clarify_node",
                "executor_node": "executor_node",
            }
        )
        
        # After rag/template/trainer nodes, check confidence and reroute
        # reroute_based_on_confidence now directly returns "orchestrator" or "executor_node"
        workflow.add_conditional_edges(
            "rag_node",
            reroute_based_on_confidence,
            {
                "orchestrator": "orchestrator",  # Loop back to try next node
                "executor_node": "executor_node",  # Confidence high enough, proceed
            }
        )
        
        workflow.add_conditional_edges(
            "template_node",
            reroute_based_on_confidence,
            {
                "orchestrator": "orchestrator",  # Loop back to try next node
                "executor_node": "executor_node",  # Confidence high enough, proceed
            }
        )
        
        workflow.add_conditional_edges(
            "trainer_node",
            reroute_based_on_confidence,
            {
                "orchestrator": "orchestrator",  # Loop back (though trainer is last, so will go to executor)
                "executor_node": "executor_node",  # Always proceed after trainer
            }
        )
        
        # Clarify node goes to human_approval to get explanation
        workflow.add_edge("clarify_node", "human_approval")
        
        # Executor node: prepare and check if human approval is needed
        # If approval needed and not yet given, go to human_approval
        # If approval not needed or already approved, go to END
        workflow.add_conditional_edges(
            "executor_node",
            need_human_approval,
            {
                "human_approval": "human_approval",
                END: END,  # No approval needed or already approved/rejected, execution complete
            }
        )
        
        # Human approval routes based on context
        def route_after_human_interaction(state: WisbiState) -> Literal["orchestrator", "executor_node", END]:
            """Route after human approval/explanation"""
            # Check if we have a human explanation (from clarification flow)
            if state.get("human_explanation"):
                # Explanation received, restart workflow from beginning
                # Clear the explanation flag to prevent infinite loops
                explanation = state.get("human_explanation")
                state["human_explanation"] = None  # Clear after routing
                return "orchestrator"
            
            # Check if approval was given (from approval flow)
            if state.get("human_approved") is True:
                # Approved: go back to executor_node to actually execute
                return "executor_node"
            elif state.get("human_approved") is False:
                # Rejected: end workflow
                return END
            
            # Default: if no explanation and no approval decision, end workflow
            # This shouldn't normally happen, but prevents hanging
            return END
        
        workflow.add_conditional_edges(
            "human_approval",
            route_after_human_interaction,
            {
                "orchestrator": "orchestrator",  # Explanation received, restart workflow
                "executor_node": "executor_node",  # Approved, execute
                END: END,  # Rejected, end workflow
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

