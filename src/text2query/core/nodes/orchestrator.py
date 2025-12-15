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

def reroute_based_on_confidence(state: WisbiState) -> Literal["template_node", "clarify_node", "executor_node"]:
    """
    Evaluates confidence from the current node and determines next route.
    
    Flow:
    - After trainer_node: if score not high or training is empty → template_node, else → executor_node
    - After template_node: if score is low → clarify_node, else → executor_node
    
    Training is priority first, template is fallback.
    """
    current = state.get("current_move")
    
    if current == "trainer":
        search_results = state.get("search_results")
        training_score = state.get("training_score")
        
        if search_results is None or search_results == "":
            state["current_move"] = "template"
            return "template_node"
        
        if training_score is not None and training_score < 0.7:
            state["current_move"] = "template"
            return "template_node"
        
        state["current_move"] = "executor"
        return "executor_node"
        
    elif current == "template":
        template_score = state.get("template_score")
        
        if template_score is None:
            state["current_move"] = "clarify"
            return "clarify_node"
        
        if template_score < 0.5:
            state["current_move"] = "clarify"
            return "clarify_node"
        
        state["current_move"] = "executor"
        return "executor_node"
    else:
        state["current_move"] = "executor"
        return "executor_node"
        

def route_by_current_move(state: WisbiState) -> Literal["template_node", "trainer_node", "clarify_node", "executor_node"]:
    """
    Routes from orchestrator to the appropriate node based on current_move in state.
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