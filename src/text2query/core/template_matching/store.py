"""Template Store - manages template storage and RAG-based retrieval"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import logging
import json

from ..connections.postgresql import PostgreSQLConfig
from ...adapters.sql.postgresql import PostgreSQLAdapter
from .schema import get_template_ddl
from .models import Template
from ..utils.models import ModelConfig, aembed_text, agenerate_chat


class TemplateStore:
    """Manages template storage in PostgreSQL with RAG-based retrieval
    
    This class handles storing and retrieving templates using vector similarity
    search based on component embeddings.
    
    Usage:
        store = await TemplateStore.initialize(
            postgres_config=PostgreSQLConfig(...),
            template_schema="wisbi",
            embedding_dim=768,
        )
    """
    
    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        template_schema: str = "wisbi",
        embedding_dim: int = 768,
    ):
        """Initialize TemplateStore
        
        Note: Recommended to use TemplateStore.initialize() which
        automatically checks and creates required tables.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            template_schema: Schema name for template tables (default: "wisbi")
            embedding_dim: Embedding vector dimension (default: 768)
        """
        self.postgres_config = postgres_config
        self.template_schema = template_schema
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None
    
    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        template_schema: str = "wisbi",
        embedding_dim: int = 768,
        auto_init_tables: bool = True,
    ) -> "TemplateStore":
        """Initialize TemplateStore instance and auto-setup tables
        
        This is the recommended initialization method, which automatically
        checks and creates required tables if they don't exist.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            template_schema: Schema name for template tables (default: "wisbi")
            embedding_dim: Embedding vector dimension (default: 768)
            auto_init_tables: Whether to auto-check and create tables (default: True)
        
        Returns:
            TemplateStore: Initialized instance
        """
        store = cls(postgres_config, template_schema, embedding_dim)
        
        if auto_init_tables:
            table_exists = await store.check_table_exists()
            
            if not table_exists:
                store.logger.info(f"Missing templates table in schema '{template_schema}'")
                store.logger.info("Creating template tables...")
                
                success = await store.init_template_tables()
                if success:
                    store.logger.info(f"Template tables initialized successfully in schema '{template_schema}'")
                else:
                    store.logger.warning("Failed to initialize template tables")
            else:
                store.logger.info(f"Templates table already exists in schema '{template_schema}'")
        
        return store
    
    def _get_adapter(self) -> PostgreSQLAdapter:
        """Get or create PostgreSQL adapter"""
        if self._adapter is None:
            self._adapter = PostgreSQLAdapter(self.postgres_config)
        return self._adapter
    
    # ============================================================================
    # Initialization methods
    # ============================================================================
    
    async def init_template_tables(self) -> bool:
        """Initialize template tables and indexes
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            adapter = self._get_adapter()
            ddl_statements = get_template_ddl(
                schema_name=self.template_schema,
                embedding_dim=self.embedding_dim,
            )
            
            for idx, ddl in enumerate(ddl_statements, 1):
                self.logger.debug(f"Executing DDL statement {idx}/{len(ddl_statements)}")
                result = await adapter.sql_execution(
                    ddl,
                    safe=False,  # DDL statements need safe=False
                    limit=None
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Failed to execute DDL statement {idx}: {error_msg}")
                    ddl_preview = ddl.strip()[:200].replace('\n', ' ')
                    self.logger.error(f"Failed DDL: {ddl_preview}...")
                    return False
            
            self.logger.info(f"Successfully set up template tables in schema '{self.template_schema}'")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error setting up template tables: {e}")
            return False
    
    async def check_table_exists(self) -> bool:
        """Check if templates table exists
        
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            adapter = self._get_adapter()
            check_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{self.template_schema}' 
                AND table_name = 'templates'
            """
            
            result = await adapter.sql_execution(check_query, safe=False, limit=None)
            
            if not result.get("success"):
                return False
            
            return len(result.get("result", [])) > 0
            
        except Exception as e:
            self.logger.exception(f"Error checking table: {e}")
            return False
    
    async def is_empty(self) -> bool:
        """Check if templates table is empty
        
        Returns:
            bool: True if table is empty, False otherwise
        """
        try:
            adapter = self._get_adapter()
            count_query = f"""
                SELECT COUNT(*) 
                FROM {self.template_schema}.templates
                WHERE is_active = true
            """
            
            result = await adapter.sql_execution(count_query, safe=False, limit=None)
            
            if not result.get("success"):
                return True
            
            rows = result.get("result", [])
            if rows and len(rows) > 0:
                count = rows[0][0]
                return count == 0
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error checking if store is empty: {e}")
            return True
    
    # ============================================================================
    # Embedding generation
    # ============================================================================
    
    def _components_to_text(self, components: Dict[str, Any]) -> str:
        """Convert components dictionary to text representation for embedding
        
        Args:
            components: Components dictionary with base_query, dimensions, filters, etc.
            
        Returns:
            str: Text representation of components
        """
        normalized = {
            "base_query": components.get("base_query"),
            "dimensions": components.get("dimensions") or {},
            "filters": components.get("filters") or {},
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    
    async def insert_template(
        self,
        template: Template,
        embedder_config: ModelConfig,
        sql_command: Optional[str] = None,
        components: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Insert a template into the store
        
        Args:
            template: Template object
            embedder_config: Embedder config for generating embeddings
            sql_command: SQL command associated with this template
            components: Components dictionary (if None, will be derived from template)
            metadata: Additional metadata
            
        Returns:
            Optional[int]: Template ID if successful, None otherwise
        """
        try:
            adapter = self._get_adapter()
            
            # Build components from template if not provided
            if components is None:
                components = {
                    "base_query": template.base_query,
                    "dimensions": template.dimensions,
                    "filters": template.filters,
                }
            
            # Convert components to text for embedding
            components_text = self._components_to_text(components)
            
            # Generate embedding
            embedding = await aembed_text(embedder_config, components_text)
            
            # Prepare data
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            insert_sql = f"""
                INSERT INTO {self.template_schema}.templates (
                    template_id, description,
                    sql_command, components_text, embedding, metadata
                ) VALUES (
                    $1, $2, $3,
                    $4, $5::vector, $6::jsonb
                )
                ON CONFLICT (template_id) 
                DO UPDATE SET
                    description = EXCLUDED.description,
                    sql_command = EXCLUDED.sql_command,
                    components_text = EXCLUDED.components_text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING id
            """
            
            params = (
                template.id,
                template.description,
                sql_command,
                components_text,
                embedding_str,
                metadata_json,
            )
            
            result = await adapter.sql_execution(insert_sql, params=params, safe=False, limit=None)
            
            if result.get("success") and result.get("result"):
                inserted_id = result["result"][0][0]
                self.logger.info(f"Inserted template: id={inserted_id}, template_id={template.id}")
                return int(inserted_id)
            else:
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to insert template: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error inserting template: {e}")
            return None
    
    # ============================================================================
    # Retrieve templates using RAG
    # ============================================================================
    
    async def find_similar_templates(
        self,
        components: Dict[str, Any],
        embedder_config: ModelConfig,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Tuple[float, Template]]:
        """Find top K similar templates using vector similarity
        
        Args:
            components: Components dictionary to search for
            embedder_config: Embedder config for generating embeddings
            top_k: Number of top templates to return (default: 3)
            min_similarity: Minimum similarity score threshold (default: 0.0)
            
        Returns:
            List[Tuple[float, Template]]: List of (similarity_score, template) tuples.
                The template.sql field contains the corresponding SQL command (if available).
        """
        try:
            adapter = self._get_adapter()
            
            # Convert components to text and generate embedding
            components_text = self._components_to_text(components)
            query_embedding = await aembed_text(embedder_config, components_text)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Use cosine similarity for vector search
            search_sql = f"""
                SELECT 
                    template_id, description,
                    sql_command, components_text,
                    1 - (embedding <=> $1::vector) as similarity
                FROM {self.template_schema}.templates
                WHERE is_active = true
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """
            
            params = (embedding_str, min_similarity, top_k)
            
            result = await adapter.sql_execution(search_sql, params=params, safe=False, limit=None)
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Failed to search templates: {error_msg}")
                return []
            
            templates: List[Tuple[float, Template]] = []
            columns = result.get("columns", [])
            rows = result.get("result", [])
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                similarity = float(row_dict["similarity"])
                sql_command = row_dict.get("sql_command")
                
                template = Template(
                    id=row_dict["template_id"],
                    description=row_dict["description"],
                    base_query=None,
                    dimensions={},
                    filters={},
                    metrics=[],
                    sql=sql_command,
                )
                
                templates.append((similarity, template))
            
            self.logger.info(f"Found {len(templates)} similar templates")
            return templates
            
        except Exception as e:
            self.logger.exception(f"Error finding similar templates: {e}")
            return []
    
    # ============================================================================
    # Connection management
    # ============================================================================
    
    async def close(self):
        """Close database connection"""
        if self._adapter:
            await self._adapter.close_pool()
            self.logger.debug("TemplateStore connection closed")

