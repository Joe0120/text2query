from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..connections.postgresql import PostgreSQLConfig
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_template_matching_ddl


class TemplateEmbeddingStore:
    """Async store for template matching, mirroring training store structure."""

    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        *,
        template_schema: str = "wisbi",
        table_name: str = "nlq_slot_templates",
        embedding_dim: int = 768,
        embedder: Optional[Any] = None,
    ):
        self.postgres_config = postgres_config
        self.template_schema = template_schema
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None

    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        *,
        template_schema: str = "wisbi",
        table_name: str = "nlq_slot_templates",
        embedding_dim: int = 768,
        embedder: Optional[Any] = None,
        auto_init_tables: bool = True,
    ) -> "TemplateEmbeddingStore":
        store = cls(
            postgres_config,
            template_schema=template_schema,
            table_name=table_name,
            embedding_dim=embedding_dim,
            embedder=embedder,
        )
        if auto_init_tables:
            exists = await store.check_table_exists()
            if not exists:
                ok = await store.init_tables()
                if ok:
                    store.logger.info(
                        f"Template-matching table initialized in schema '{template_schema}' as '{table_name}'"
                    )
                else:
                    store.logger.warning("Failed to initialize template-matching table")
        return store

    def _get_adapter(self) -> PostgreSQLAdapter:
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter

    async def init_tables(self) -> bool:
        try:
            adapter = self._get_adapter()
            ddls = get_template_matching_ddl(
                schema_name=self.template_schema,
                table_name=self.table_name,
                embedding_dim=self.embedding_dim,
            )
            for ddl in ddls:
                result = await adapter.sql_execution(ddl, safe=False, limit=None)
                if not result.get("success"):
                    self.logger.error(f"Failed to execute DDL: {result.get('error')}")
                    return False
            return True
        except Exception as e:
            self.logger.exception(f"Error initializing template-matching tables: {e}")
            return False

    async def check_table_exists(self) -> bool:
        try:
            adapter = self._get_adapter()
            sql = f"""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = '{self.template_schema}'
                  AND table_name = '{self.table_name}'
                LIMIT 1
            """
            result = await adapter.sql_execution(sql, safe=False, limit=None)
            if not result.get("success"):
                return False
            rows = result.get("result") or []
            return bool(rows)
        except Exception as e:
            self.logger.exception(f"Error checking template table existence: {e}")
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
    def _get_stable_slots(slots: Dict[str, Any]) -> str:
        return TemplateEmbeddingStore._stable_slots_view(slots or {})

    async def _generate_embedding(self, text: str) -> List[float]:
        if self.embedder is None:
            raise RuntimeError(
                "Embedder not provided. Initialize TemplateEmbeddingStore with an embedder "
                "or pass pre-computed embeddings."
            )
        # Prefer async method name aget_text_embedding, fallback to get_text_embedding
        if hasattr(self.embedder, "aget_text_embedding"):
            return await self.embedder.aget_text_embedding(text)
        if hasattr(self.embedder, "get_text_embedding"):
            return self.embedder.get_text_embedding(text)
        raise AttributeError("Embedder does not provide aget_text_embedding/get_text_embedding")

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
        if not template or not isinstance(slots, dict):
            raise ValueError("Invalid arguments: 'template' and 'slots' are required")

        payload = self._get_stable_slots(slots)
        if embedding is None:
            embedding = await self._generate_embedding(payload)
            inferred_model = model_name or getattr(self.embedder, "model", "unknown")
        else:
            inferred_model = model_name or "custom"

        adapter = self._get_adapter()
        embedding_str = self._vector_literal(embedding)
        insert_sql = f"""
            INSERT INTO {self.template_schema}.{self.table_name} (slots_json, template, model, embedding)
            VALUES ($1::jsonb, $2, $3, $4::vector)
            RETURNING id
        """
        params = (payload, template, inferred_model, embedding_str)
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
        payload = self._get_stable_slots(slots)
        try:
            query_vec = await self._generate_embedding(payload)
        except Exception as e:
            self.logger.error(f"Failed generating embedding for slots: {e}")
            return []

        adapter = self._get_adapter()
        vec_str = self._vector_literal(query_vec)
        select_sql = f"""
            SELECT id, template, (1 - (embedding <=> '{vec_str}'::vector)) AS similarity
            FROM {self.template_schema}.{self.table_name}
            ORDER BY embedding <=> '{vec_str}'::vector ASC
            LIMIT {top_k}
        """
        try:
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
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
        """Search similar templates using a precomputed embedding (training-style)."""
        adapter = self._get_adapter()
        vec_str = self._vector_literal(query_embedding)
        select_sql = f"""
            SELECT id, template, (1 - (embedding <=> '{vec_str}'::vector)) AS similarity
            FROM {self.template_schema}.{self.table_name}
            ORDER BY embedding <=> '{vec_str}'::vector ASC
            LIMIT {top_k}
        """
        try:
            result = await adapter.sql_execution(select_sql, safe=False, limit=None)
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


