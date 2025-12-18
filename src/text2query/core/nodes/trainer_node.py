from curses import meta
from ..training.store import TrainingStore
from .state import WisbiState

from typing import Optional, List, Dict, Any
from ..utils.models import aembed_text
import logging

logger = logging.getLogger(__name__)

class TrainerNode:
    def __init__(self, training_store: TrainingStore):
        self.training_store = training_store
    
    def format_response(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        search_results schema (from TrainingStore.search_all):

            {
              "qna": [...],
              "sql_examples": [...],
              "documentation": [...]
            }
        """
        lines: List[str] = []

        # QnA
        for item in search_results.get("qna", []):
            q = item.get("question", "")
            a = item.get("answer_sql", "")
            lines.append(f"[QnA]\nQ: {q}\nA: {a}\n")

        # SQL examples
        for item in search_results.get("sql_examples", []):
            content = item.get("content", "")
            lines.append(f"[SQL EXAMPLE]\n{content}\n")

        # Documentation
        for item in search_results.get("documentation", []):
            title = item.get("title") or "Documentation"
            content = item.get("content", "")
            lines.append(f"[DOC] {title}\n{content}\n")

        return "\n".join(lines)

    async def __call__(self, state: WisbiState) -> WisbiState:
        
        query_embedding: Optional[List[float]] = state.get("query_embedding")
        if query_embedding is None:
            query_text = state.get("query_text")
            if query_text is None:
                return state
            query_embedding = await aembed_text(state["embedder_config"], query_text)
            state["query_embedding"] = query_embedding
        
        table_id = state.get("table_id")
        if table_id is None:
            raise ValueError("table_id is required in state for TrainerNode")
        
        search_results = await self.training_store.search_all(
            query_embedding=state["query_embedding"],
            table_id=table_id,
            user_id=state.get("user_id", None),
            group_id=state.get("group_id", None),
            top_k=state.get("top_k", 8),
        )

        # Log training search results with similarity scores/distances
        logger.info("Training search results (distance scores):")
        total_results = 0
        best_distance = None
        
        for table_name, results in search_results.items():
            count = len(results)
            total_results += count
            if count > 0:
                # Extract distances from results (if available)
                distances = [r.get("distance") for r in results if r.get("distance") is not None]
                if distances:
                    min_dist = min(distances)
                    max_dist = max(distances)
                    avg_dist = sum(distances) / len(distances)
                    if best_distance is None or min_dist < best_distance:
                        best_distance = min_dist
                    logger.info(f"  {table_name}: {count} results - distances: min={min_dist:.4f}, max={max_dist:.4f}, avg={avg_dist:.4f}")
                else:
                    logger.info(f"  {table_name}: {count} results (no distance scores available)")
        
        if best_distance is not None:
            # Convert distance to similarity (1 - distance for cosine distance)
            training_similarity = 1 - best_distance
            state["training_score"] = training_similarity
            logger.info(f"Best training match - distance: {best_distance:.4f}, similarity: {training_similarity:.4f}")
            logger.info(f"Setting training_score in state: {state.get('training_score')}")
        else:
            logger.info(f"Total training results: {total_results} (no distance scores to calculate similarity)")
            if total_results > 0:
                # Set a default score if we have results but no distances
                state["training_score"] = 0.5
                logger.info(f"Setting training_score to default 0.5 (results available but no distances)")
            else:
                state["training_score"] = 0.0
                logger.info(f"Setting training_score to 0.0 (no results)")
        
        state["search_results"] = self.format_response(search_results)
        logger.info(f"Trainer node returning state with training_score: {state.get('training_score')}, search_results length: {len(state.get('search_results', ''))}")
        return state

if __name__ == "__main__":
    async def main():
        from ..connections.postgresql import PostgreSQLConfig
        from ..utils.model_configs import ModelConfig
        from ..training.store import TrainingStore
        state = WisbiState(
            db_config=PostgreSQLConfig(
                host="...",
                port=...,
                database_name="...",
                username="...",
                password="...",
            ),
            llm_config=ModelConfig(
                model_name="...",
                base_url="...",
                endpoint="...",
                api_key="...",
                provider="...",
            ),
            embedder_config=ModelConfig(
                model_name="...",
                base_url="...",
                endpoint="...",
                api_key="...",
                provider="...",
            ),
            query_text="Why is the sky blue?",
        )
        training_store = await TrainingStore.initialize(
            postgres_config=state["db_config"],
            training_schema="...",
            embedding_dim=...,
        )

        trainer = TrainerNode(training_store)
        result = await trainer(state)
        print(result)
    import asyncio
    asyncio.run(main())