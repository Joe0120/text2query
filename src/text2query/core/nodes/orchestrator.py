from __future__ import annotations
from typing import Any, Dict, Optional, Literal, List
import logging
from .state import WisbiState, available_moves

logger = logging.getLogger(__name__)

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
    - After trainer_node: if score >= 0.5 and training results available → executor_node (use training context)
                         if score < 0.5 or no results → template_node (fallback to template)
    - After template_node: always → executor_node
    
    Training is priority first, template is fallback.
    Uses training context when similarity >= 0.5.
    """
    current = state.get("current_move")
    
    if current == "trainer":
        search_results = state.get("search_results")
        training_score = state.get("training_score")
        
        logger.info(f"Routing decision after trainer_node:")
        logger.info(f"  Training score: {training_score}")
        logger.info(f"  Search results available: {search_results is not None and search_results != ''}")
        
        if search_results is None or search_results == "":
            logger.info("  → Routing to template_node (no training results)")
            state["current_move"] = "template"
            return "template_node"
        
        # Use training context if similarity >= 0.5 (lowered from 0.7)
        # If training_score is None but we have results, calculate similarity from best distance
        if training_score is None:
            # Try to extract similarity from search_results if available
            # This handles cases where training_score wasn't set but we have results
            logger.info("  → Training score is None, but search results available - will use training context")
            state["current_move"] = "executor"
            return "executor_node"
        
        if training_score < 0.5:
            logger.info(f"  → Routing to template_node (training score {training_score:.4f} < 0.5 threshold)")
            state["current_move"] = "template"
            return "template_node"
        
        # training_score >= 0.5, use training context
        logger.info(f"  → Routing to executor_node (training score {training_score:.4f} >= 0.5 threshold, using training context)")
        state["current_move"] = "executor"
        return "executor_node"
        
    elif current == "template":
        template_score = state.get("template_score")
        logger.info(f"Routing decision after template_node:")
        logger.info(f"  Template score: {template_score}")
        logger.info(f"  → Routing to executor_node")
        state["current_move"] = "executor"
        return "executor_node"
    else:
        # Default: go to executor
        logger.info(f"Routing decision (current_move={current}): → executor_node (default)")
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
    else:
        # Default to trainer_node if current_move is None or unexpected
        return "trainer_node"