from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import date
from ..rag.store import TemplateEmbeddingStore
from ..rag.slot_extractions import SlotExtraction
from .state import WisbiState

class RAGNode:
    """
    RAG Node - callable class that extracts slots from queries using request-based LLM calls
    and searches for similar templates using hybrid search (vector + BM25) with Reciprocal Rank Fusion.
    Sets rag_confidence based on the best match RRF score.
    """
    
    def __init__(
        self,
        template_store: TemplateEmbeddingStore,
        min_confidence: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize RAG Node.
        
        Args:
            template_store: TemplateEmbeddingStore instance
            min_confidence: Minimum confidence threshold for results
            rrf_k: RRF constant for Reciprocal Rank Fusion (default: 60)
        """
        self.template_store = template_store
        self.min_confidence = min_confidence
        self.rrf_k = rrf_k
        self.slot_extractor = SlotExtraction()
    
    async def __call__(self, state: WisbiState) -> WisbiState:
        """
        Extracts slots from query using request-based LLM, searches for similar templates
        using hybrid search (vector + BM25) with Reciprocal Rank Fusion, and sets rag_confidence in state.
        """
        query_text = state.get("query_text")
        if query_text is None:
            return state
        
        llm_config = state.get("llm_config")
        if llm_config is None:
            raise ValueError("llm_config is required in state for slot extraction")
        
        # Extract slots from query using request-based method
        context_date = date.today()  # Could be passed from state if needed
        slots = await self.slot_extractor.extract_slots(
            llm_config=llm_config,
            question=query_text,
            context_date=context_date,
        )
        
        # Always use hybrid search with RRF
        similar_templates = await self.template_store.search_hybrid(
            slots=slots,
            query_text=query_text,
            top_k=1,  # Get best match
            min_similarity=0.0,  # Get all results, we'll filter by confidence
            rrf_k=self.rrf_k,
        )
        
        # Extract rrf_score from hybrid search results
        if similar_templates and len(similar_templates) > 0:
            best_match = similar_templates[0]
            confidence = float(best_match.get("rrf_score", 0.0))
            state["rag_confidence"] = confidence
            state["rag_results"] = best_match.get("template", "")
            # Store additional metadata if available
            if "similarity" in best_match:
                state["rag_vector_score"] = best_match.get("similarity", 0.0)
            if "bm25_score" in best_match:
                state["rag_bm25_score"] = best_match.get("bm25_score", 0.0)
            if "rrf_score" in best_match:
                state["rag_rrf_score"] = best_match.get("rrf_score", 0.0)
        else:
            # No matches found, low confidence
            state["rag_confidence"] = 0.0
            state["rag_results"] = None
        
        return state

