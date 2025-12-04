from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..connections.postgresql import PostgreSQLConfig
from ..utils.model_configs import ModelConfig
from ..utils.models import aembed_text
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_template_matching_ddl


class TemplateEmbeddingStore:
    """Async store for template matching, mirroring training store structure."""

    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        embedder_config: ModelConfig,
        *,
        template_schema: str = "wisbi",
        table_name: str = "nlq_slot_templates",
        embedding_dim: int = 768,
    ):
        self.postgres_config = postgres_config
        self.embedder_config = embedder_config
        self.template_schema = template_schema
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None

    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        embedder_config: ModelConfig,
        *,
        template_schema: str = "wisbi",
        table_name: str = "nlq_slot_templates",
        embedding_dim: int = 768,
        auto_init_tables: bool = True,
    ) -> "TemplateEmbeddingStore":
        store = cls(
            postgres_config,
            embedder_config,
            template_schema=template_schema,
            table_name=table_name,
            embedding_dim=embedding_dim,
        )
        if auto_init_tables:
            # Check if table exists
            table_exists = await store.check_table_exists()
            
            # If table doesn't exist, create it
            if not table_exists:
                store.logger.info(f"Table '{table_name}' not found in schema '{template_schema}'")
                store.logger.info("Creating template matching table...")
                
                success = await store.init_tables()
                if success:
                    store.logger.info(f"Template matching table initialized successfully in schema '{template_schema}'")
                else:
                    store.logger.warning("Failed to initialize template matching table")
            else:
                store.logger.info(f"Template matching table already exists in schema '{template_schema}'")
        
        return store

    def _get_adapter(self) -> PostgreSQLAdapter:
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter

    async def init_tables(self) -> bool:
        """Initialize template matching tables and indexes.
        
        This method performs the following operations:
        1. Create schema (if it doesn't exist)
        2. Enable pgvector extension (if it doesn't exist)
        3. Create template matching table (if it doesn't exist)
        4. Create vector search indexes
        
        Returns:
            bool: True on success, False on failure
        """
        try:
            adapter = self._get_adapter()
            ddl_statements = get_template_matching_ddl(
                schema_name=self.template_schema,
                table_name=self.table_name,
                embedding_dim=self.embedding_dim,
            )
            
            for idx, ddl in enumerate(ddl_statements, 1):
                self.logger.debug(f"Executing DDL statement {idx}/{len(ddl_statements)}")
                result = await adapter.sql_execution(
                    ddl,
                    safe=False,  # DDL statements need to disable safety checks
                    limit=None
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Failed to execute DDL statement {idx}: {error_msg}")
                    # Show failed DDL (truncated)
                    ddl_preview = ddl.strip()[:200].replace('\n', ' ')
                    self.logger.error(f"Failed DDL: {ddl_preview}...")
                    return False
            
            self.logger.info(f"Successfully set up template matching table in schema '{self.template_schema}'")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error setting up template matching tables: {e}")
            return False

    async def check_table_exists(self) -> bool:
        """Check if template matching table exists.
        
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{self.template_schema}' 
                AND table_name = '{self.table_name}'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            rows = result.get("result", []) or []
            return len(rows) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking table existence: {e}")
            return False
    
    async def check_schema_exists(self) -> bool:
        """Check if template schema exists.
        
        Returns:
            bool: True if schema exists, False otherwise
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = '{self.template_schema}'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            return len(result.get("result", [])) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking schema existence: {e}")
            return False

    async def close(self):
        if self._adapter:
            await self._adapter.close_pool()
            self.logger.debug("TemplateEmbeddingStore connection closed")

    # -------------------------------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _stable_slots_view(slots: Dict[str, Any]) -> Dict[str, Any]:
        entities = (slots or {}).get("entities") or {}
        modifiers = (slots or {}).get("modifiers") or {}
        intent = (slots or {}).get("intent") or None

        metric = entities.get("metric") if isinstance(entities, dict) else None
        granularity = entities.get("granularity") if isinstance(entities, dict) else None
        org = entities.get("org") if isinstance(entities, dict) else None
        org_present = bool(org) and isinstance(org, list) and len(org) > 0
        org_count = len(org) if isinstance(org, list) else 0

        time = entities.get("time") if isinstance(entities, dict) else None
        time_ref = None
        if isinstance(time, dict):
            if isinstance(time.get("reference"), str) and time.get("reference"):
                time_ref = str(time.get("reference"))
            elif time.get("type") == "absolute":
                time_ref = "absolute_range"
        comparison = bool(modifiers.get("comparison")) if isinstance(modifiers, dict) else False
        aggregation = modifiers.get("aggregation") if isinstance(modifiers, dict) else None

        return {
            "metric": metric,
            "granularity": granularity,
            "intent": intent,
            "org_present": bool(org_present),
            "org_count": int(org_count),
            "time_ref": time_ref,
            "comparison": comparison,
            "aggregation": aggregation,
        }

    @staticmethod
    def _get_stable_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
        """Get stable slots view for embedding generation.
        
        Args:
            slots: Dictionary containing entities, intent, and modifiers
            
        Returns:
            Dictionary with normalized slot values for embedding
        """
        return TemplateEmbeddingStore._stable_slots_view(slots or {})

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using request-based embedding method.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return await aembed_text(self.embedder_config, text)

    @staticmethod
    def _vector_literal(embedding: List[float]) -> str:
        return "[" + ",".join(map(str, embedding)) + "]"

    # -------------------------------------------------------------------------
    # Insert/query APIs
    # -------------------------------------------------------------------------
    async def insert_template(
        self,
        *,
        slots: Dict[str, Any],
        template: str,
        embedding: Optional[List[float]] = None,
        model_name: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a template with its slots and embedding.
        
        Args:
            slots: Dictionary containing entities, intent, and modifiers
            template: SQL template string
            embedding: Optional precomputed embedding vector
            model_name: Optional model name (defaults to embedder config or "custom")
            
        Returns:
            Inserted template ID or None on failure
        """
        if not template or not isinstance(slots, dict):
            raise ValueError("Invalid arguments: 'template' and 'slots' are required")

        payload = self._get_stable_slots(slots)
        if embedding is None:
            embedding = await self._generate_embedding(json.dumps(payload, sort_keys=True))
            inferred_model = model_name or self.embedder_config.model_name
        else:
            # Validate embedding dimension
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} doesn't match expected {self.embedding_dim}"
                )
            inferred_model = model_name or "custom"

        adapter = self._get_adapter()
        embedding_str = self._vector_literal(embedding)
        insert_sql = f"""
            INSERT INTO {self.template_schema}.{self.table_name} (slots_json, template, model, embedding)
            VALUES ($1::jsonb, $2, $3, $4::vector)
            RETURNING id
        """
        params = (json.dumps(payload, sort_keys=True), template, inferred_model, embedding_str)
        try:
            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            if result.get("success") and result.get("result"):
                return int(result["result"][0][0])
            self.logger.error(f"Failed to insert template: {result.get('error')}")
            return None
        except Exception as e:
            self.logger.exception(f"Error inserting template: {e}")
            return None

    async def search_similar_templates(
        self,
        *,
        slots: Dict[str, Any],
        top_k: int = 1,
        min_similarity: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Search for similar templates using slot-based embedding.
        
        Args:
            slots: Dictionary containing entities, intent, and modifiers
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of dictionaries with id, template, and similarity score
        """
        payload = self._get_stable_slots(slots)
        try:
            # Generate embedding from JSON string for consistency
            query_vec = await self._generate_embedding(json.dumps(payload, sort_keys=True))
        except Exception as e:
            self.logger.error(f"Failed generating embedding for slots: {e}")
            return []

        # Validate embedding dimension
        if len(query_vec) != self.embedding_dim:
            self.logger.error(
                f"Generated embedding dimension {len(query_vec)} doesn't match expected {self.embedding_dim}"
            )
            return []

        adapter = self._get_adapter()
        vec_str = self._vector_literal(query_vec)
        # Use parameterized query to prevent SQL injection
        select_sql = f"""
            SELECT id, template, (1 - (embedding <=> $1::vector)) AS similarity
            FROM {self.template_schema}.{self.table_name}
            WHERE is_active = true
            ORDER BY embedding <=> $1::vector ASC
            LIMIT $2
        """
        params = (vec_str, top_k)
        try:
            result = await adapter.sql_execution(select_sql, params=params, safe=False, limit=None)
            if not result.get("success"):
                self.logger.error(f"Failed to search templates: {result.get('error')}")
                return []
            columns = result.get("columns", [])
            rows = result.get("result", [])
            out: List[Dict[str, Any]] = []
            # Expect columns: id, template, similarity
            for row in rows:
                data = dict(zip(columns, row))
                sim = float(data.get("similarity") or 0.0)
                if sim < float(min_similarity):
                    continue
                out.append(
                    {
                        "id": int(data.get("id")),
                        "template": str(data.get("template") or ""),
                        "similarity": sim,
                    }
                )
            return out
        except Exception as e:
            self.logger.exception(f"Error searching templates: {e}")
            return []

    async def search_similar_by_embedding(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 1,
        min_similarity: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Search similar templates using a precomputed embedding (training-style).
        
        Args:
            query_embedding: Precomputed embedding vector
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of dictionaries with id, template, and similarity score
        """
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dim:
            self.logger.error(
                f"Query embedding dimension {len(query_embedding)} doesn't match expected {self.embedding_dim}"
            )
            return []
        
        adapter = self._get_adapter()
        vec_str = self._vector_literal(query_embedding)
        select_sql = f"""
            SELECT id, template, (1 - (embedding <=> $1::vector)) AS similarity
            FROM {self.template_schema}.{self.table_name}
            WHERE is_active = true
            ORDER BY embedding <=> $1::vector ASC
            LIMIT $2
        """
        params = (vec_str, top_k)
        try:
            result = await adapter.sql_execution(select_sql, params=params, safe=False, limit=None)
            if not result.get("success"):
                self.logger.error(f"Failed to search templates by embedding: {result.get('error')}")
                return []
            columns = result.get("columns", [])
            rows = result.get("result", [])
            out: List[Dict[str, Any]] = []
            for row in rows:
                data = dict(zip(columns, row))
                sim = float(data.get("similarity") or 0.0)
                if sim < float(min_similarity):
                    continue
                out.append(
                    {
                        "id": int(data.get("id")),
                        "template": str(data.get("template") or ""),
                        "similarity": sim,
                    }
                )
            return out
        except Exception as e:
            self.logger.exception(f"Error searching templates by embedding: {e}")
            return []

    async def search_by_bm25(
        self,
        query_text: str,
        *,
        top_k: int = 1,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search templates using BM25 full-text search.
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of dictionaries with id, template, and bm25_score
        """
        if not query_text or not query_text.strip():
            return []
        
        adapter = self._get_adapter()
        select_sql = f"""
            SELECT 
                id, 
                template,
                ts_rank_cd(template_tsvector, plainto_tsquery('english', $1)) AS bm25_score
            FROM {self.template_schema}.{self.table_name}
            WHERE is_active = true
              AND template_tsvector @@ plainto_tsquery('english', $1)
            ORDER BY bm25_score DESC
            LIMIT $2
        """
        params = (query_text.strip(), top_k)
        try:
            result = await adapter.sql_execution(select_sql, params=params, safe=False, limit=None)
            if not result.get("success"):
                self.logger.error(f"Failed to search templates by BM25: {result.get('error')}")
                return []
            columns = result.get("columns", [])
            rows = result.get("result", [])
            out: List[Dict[str, Any]] = []
            for row in rows:
                data = dict(zip(columns, row))
                score = float(data.get("bm25_score") or 0.0)
                if score < float(min_score):
                    continue
                out.append(
                    {
                        "id": int(data.get("id")),
                        "template": str(data.get("template") or ""),
                        "bm25_score": score,
                    }
                )
            return out
        except Exception as e:
            self.logger.exception(f"Error searching templates by BM25: {e}")
            return []

    async def search_hybrid(
        self,
        *,
        slots: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 1,
        min_similarity: float = 0.0,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and BM25 using Reciprocal Rank Fusion (RRF).
        
        RRF combines ranked results from multiple search methods by:
        - Assigning scores based on rank: 1 / (k + rank)
        - Summing scores across methods
        - Re-ranking by combined RRF score
        
        Args:
            slots: Dictionary containing entities, intent, and modifiers (for vector search)
            query_text: Text query for BM25 search (required for hybrid search)
            query_embedding: Precomputed embedding vector (alternative to slots)
            top_k: Number of top results to return
            min_similarity: Minimum RRF score threshold (0.0-1.0)
            rrf_k: RRF constant (default: 60, common value in literature)
            
        Returns:
            List of dictionaries with id, template, similarity, bm25_score, and rrf_score
        """
        # Get vector search results
        vector_results: List[Dict[str, Any]] = []
        if slots is not None:
            vector_results = await self.search_similar_templates(
                slots=slots,
                top_k=top_k * 3,  # Get more results for better RRF ranking
                min_similarity=0.0,  # Don't filter here, RRF will handle ranking
            )
        elif query_embedding is not None:
            vector_results = await self.search_similar_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k * 3,
                min_similarity=0.0,
            )
        
        # Get BM25 search results
        bm25_results: List[Dict[str, Any]] = []
        if query_text:
            bm25_results = await self.search_by_bm25(
                query_text=query_text,
                top_k=top_k * 3,
                min_score=0.0,
            )
        
        # Build rank maps for RRF calculation
        vector_ranks: Dict[int, int] = {}  # template_id -> rank (1-indexed)
        bm25_ranks: Dict[int, int] = {}  # template_id -> rank (1-indexed)
        
        # Store full result data
        all_results: Dict[int, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            template_id = result["id"]
            vector_ranks[template_id] = rank
            all_results[template_id] = {
                "id": template_id,
                "template": result.get("template", ""),
                "similarity": result.get("similarity", 0.0),
                "bm25_score": 0.0,
            }
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            template_id = result["id"]
            bm25_ranks[template_id] = rank
            if template_id in all_results:
                # Update existing result
                all_results[template_id]["bm25_score"] = result.get("bm25_score", 0.0)
            else:
                # New result from BM25 only
                all_results[template_id] = {
                    "id": template_id,
                    "template": result.get("template", ""),
                    "similarity": 0.0,
                    "bm25_score": result.get("bm25_score", 0.0),
                }
        
        # Calculate RRF scores
        rrf_scores: List[Dict[str, Any]] = []
        for template_id, result in all_results.items():
            rrf_score = 0.0
            
            # Add vector RRF contribution
            if template_id in vector_ranks:
                rank = vector_ranks[template_id]
                rrf_score += 1.0 / (rrf_k + rank)
            
            # Add BM25 RRF contribution
            if template_id in bm25_ranks:
                rank = bm25_ranks[template_id]
                rrf_score += 1.0 / (rrf_k + rank)
            
            result["rrf_score"] = rrf_score
            rrf_scores.append(result)
        
        # Sort by RRF score descending
        sorted_results = sorted(
            rrf_scores,
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Filter by min_similarity and return top_k
        filtered_results = [
            r for r in sorted_results
            if r["rrf_score"] >= min_similarity
        ][:top_k]
        
        return filtered_results


