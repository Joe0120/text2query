from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import date
from ..rag.slot_extractions import SlotExtraction
from ..rag.store import TemplateEmbeddingStore
from .state import WisbiState


class RAGNode:
    """
    RAG Node extracts slots from the query and searches for similar templates.
    Sets rag_confidence based on the best match similarity score.
    """
    
    def __init__(
        self,
        slot_extractor: SlotExtraction,
        template_store: TemplateEmbeddingStore,
        min_confidence: float = 0.5,
    ):
        self.slot_extractor = slot_extractor
        self.template_store = template_store
        self.min_confidence = min_confidence
    
    async def __call__(self, state: WisbiState) -> WisbiState:
        """
        Extracts slots from query, searches for similar templates,
        and sets rag_confidence in state.
        """
        query_text = state.get("query_text")
        if query_text is None:
            return state
        
        # Extract slots from query
        context_date = date.today()  # Could be passed from state if needed
        slots = await self.slot_extractor.extract_slots(
            question=query_text,
            context_date=context_date,
        )
        
        # Search for similar templates using the extracted slots
        similar_templates = await self.template_store.search_similar_templates(
            slots=slots,
            top_k=1,  # Get best match
            min_similarity=0.0,  # Get all results, we'll filter by confidence
        )
        
        # Calculate confidence from best match
        if similar_templates and len(similar_templates) > 0:
            best_match = similar_templates[0]
            confidence = float(best_match.get("similarity", 0.0))
            state["rag_confidence"] = confidence
            state["rag_results"] = best_match.get("template", "")
        else:
            # No matches found, low confidence
            state["rag_confidence"] = 0.0
            state["rag_results"] = None
        
        return state

