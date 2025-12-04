from __future__ import annotations
from typing import Literal
from langgraph.graph import END
from .state import WisbiState


def human_explanation_node(state: WisbiState) -> WisbiState:
    """
    Human approval node that prompts for manual confirmation and explanation.
    
    If clarification is requested, prompts for explanation.
    Otherwise, prompts for approval/confirmation.
    After getting explanation, resets state to restart workflow.
    """
    clarification_requested = state.get("clarification_requested", False)
    
    if clarification_requested:
        # Clarification flow: get explanation from human
        print(f"\n{'='*60}")
        print(f"💬 需要人工說明 (Human Explanation Required)")
        
        # Display current query/context
        query_text = state.get("query_text", "N/A")
        print(f"當前查詢 (Current Query): {query_text}")
        
        # Display user information if available
        user_profile = state.get("user_profile")
        if user_profile:
            user_name = user_profile.get("name", "Unknown")
            print(f"用戶 (User): {user_name}")
        
        print(f"{'='*60}")
        
        # Prompt for explanation
        explanation = input("請提供說明或反饋 (Please provide explanation or feedback): ").strip()
        state["human_explanation"] = explanation
        
        if explanation:
            print(f"✅ 收到說明: {explanation[:50]}...")
            # Reset state to restart workflow from beginning
            # Clear intermediate results but keep original query
            state["moves_done"] = []
            state["current_move"] = None
            # Clear confidence scores to restart
            state["rag_confidence"] = None
            state["template_score"] = None
            # Update query text with explanation if needed (optional)
            # Or append explanation to query for better context
            if query_text:
                state["query_text"] = f"{query_text} {explanation}".strip()
            state["clarification_requested"] = False
            state["human_explanation"] = explanation
        else:
            print("⚠️  未提供說明，將繼續原流程")
    else:
        # Approval flow: prompt for approval
        print(f"\n{'='*60}")
        print(f"🤚 需要人工確認 (Human Approval Required)")
        
        # Display user information if available
        user_profile = state.get("user_profile")
        if user_profile:
            user_name = user_profile.get("name", "Unknown")
            print(f"用戶 (User): {user_name}")
        
        # Display tool results / risk assessment
        tool_results = state.get("tool_results")
        if tool_results:
            print(f"風險評估 (Risk Assessment):\n{tool_results}")
        else:
            query_text = state.get("query_text", "N/A")
            print(f"查詢內容 (Query): {query_text}")
        
        print(f"{'='*60}")
        
        # Prompt for approval
        user_input = input("是否批准？(y/n / Approve?): ").strip().lower()
        state["human_approved"] = user_input == 'y'
        
        if state["human_approved"]:
            print("✅ 人工批准 (Human Approved)")
        else:
            print("❌ 人工拒絕 (Human Rejected)")
    
    return state


def need_human_approval(state: WisbiState) -> Literal["human_approval", END]:
    """
    Determines if human approval is needed based on state.
    
    Returns:
        "human_approval" if approval is needed and not yet given,
        END if approval not needed, already approved, or rejected
    """
    # Check if already approved - executor has completed, go to END
    if state.get("human_approved") is True:
        return END
    
    # Check if explicitly rejected - end workflow
    if state.get("human_approved") is False:
        return END
    
    # Check intent - if it's a trade or high-risk operation, require approval
    intent = state.get("intent")
    if intent == "trade":
        return "human_approval"
    
    # Check if tool_results exist (indicating risk assessment was done)
    tool_results = state.get("tool_results")
    if tool_results:
        # If risk assessment exists, require approval
        return "human_approval"
    
    # Default: no approval needed, go to END
    return END

