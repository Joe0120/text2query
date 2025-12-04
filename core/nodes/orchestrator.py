from __future__ import annotations
from typing import Any, Dict, Optional, Literal, List
from .state import WisbiState, available_moves

def orchestrator_node(state: WisbiState) -> WisbiState:
    """
    Orchestrator node that initializes and tracks moves.
    Returns updated state. Routing is handled by conditional edges in the workflow.
    """
    text = state.get("query_text")
    if text is None:
        return state
    
    # Initialize moves_done if not present
    if "moves_done" not in state or state["moves_done"] is None:
        state["moves_done"] = []
    
    # Initialize current_move if not present - start with training (priority first)
    if state.get("current_move") is None:
        state["current_move"] = "trainer"
    
    # Track the move in moves_done (if not already tracked)
    current = state.get("current_move")
    if current and current not in state["moves_done"]:
        state["moves_done"].append(current)
    
    return state

def reroute_based_on_confidence(state: WisbiState) -> Literal["orchestrator", "executor_node"]:
    """
    Evaluates confidence from the current node and determines next route.
    Returns either "orchestrator" to try the next node, or "executor_node" to proceed.
    
    Flow: After a node (trainer/template) executes, this function checks
    confidence. If below threshold, routes back to orchestrator to try next node.
    Otherwise, routes to executor.
    
    Training is priority first, template is fallback.
    """
    reroute = False
    current = state.get("current_move")
    
    if current == "trainer":
        # Check if trainer found good results
        search_results = state.get("search_results")
        if search_results is None or search_results == "":
            # No results found, try template next
            reroute = True
        else:
            # Trainer found results, proceed to executor
            reroute = False
    elif current == "template":
        # Template is the fallback after trainer
        template_score = state.get("template_score")
        if template_score is None:
            # If score not set, assume low confidence and proceed to executor anyway
            reroute = False
        else:
            reroute = template_score < 0.5
    else:
        # Unknown move, default to executor
        state["current_move"] = "executor"
        return "executor_node"

    route_order = {"trainer": 0, "template": 1}
    routes = ["trainer", "template"]

    if reroute:
        current_idx = route_order.get(current, -1)
        if current_idx >= 0 and current_idx < len(routes) - 1:
            # Route to next node in sequence
            next_route = routes[current_idx + 1]
            state["current_move"] = next_route
            return "orchestrator"  # Go back to orchestrator to handle next node
        else:
            # Already at last route, proceed to executor
            state["current_move"] = "executor"
            return "executor_node"
    else:
        # Confidence is high enough, proceed to executor
        state["current_move"] = "executor"
        return "executor_node"
        

def route_by_current_move(state: WisbiState) -> Literal["template_node", "trainer_node", "clarify_node", "executor_node"]:
    """
    Conditional edge function that routes from orchestrator to the appropriate node
    based on current_move in state.
    """
    current_move = state.get("current_move")
    
    if current_move == "template":
        return "template_node"
    elif current_move == "trainer":
        return "trainer_node"
    elif current_move == "clarify":
        return "clarify_node"
    elif current_move == "executor":
        return "executor_node"
    else:
        # Default to trainer if current_move is None or invalid (training is priority first)
        return "trainer_node"