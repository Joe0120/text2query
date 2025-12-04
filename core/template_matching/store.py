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
            llm_config=ModelConfig(...),  # For generating embeddings
        )
    """
    
    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        template_schema: str = "wisbi",
        embedding_dim: int = 768,
        llm_config: Optional[ModelConfig] = None,
    ):
        """Initialize TemplateStore
        
        Note: Recommended to use TemplateStore.initialize() which
        automatically checks and creates required tables.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            template_schema: Schema name for template tables (default: "wisbi")
            embedding_dim: Embedding vector dimension (default: 768)
            llm_config: LLM config for generating embeddings (optional)
        """
        self.postgres_config = postgres_config
        self.template_schema = template_schema
        self.embedding_dim = embedding_dim
        self.llm_config = llm_config
        self.logger = logging.getLogger(__name__)
        self._adapter: Optional[PostgreSQLAdapter] = None
    
    @classmethod
    async def initialize(
        cls,
        postgres_config: PostgreSQLConfig,
        template_schema: str = "wisbi",
        embedding_dim: int = 768,
        llm_config: Optional[ModelConfig] = None,
        auto_init_tables: bool = True,
    ) -> "TemplateStore":
        """Initialize TemplateStore instance and auto-setup tables
        
        This is the recommended initialization method, which automatically
        checks and creates required tables if they don't exist.
        
        Args:
            postgres_config: PostgreSQL connection configuration
            template_schema: Schema name for template tables (default: "wisbi")
            embedding_dim: Embedding vector dimension (default: 768)
            llm_config: LLM config for generating embeddings (optional)
            auto_init_tables: Whether to auto-check and create tables (default: True)
        
        Returns:
            TemplateStore: Initialized instance
        """
        store = cls(postgres_config, template_schema, embedding_dim, llm_config)
        
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
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            RuntimeError: If llm_config not provided
        """
        if self.llm_config is None:
            raise RuntimeError(
                "LLM config not provided. Please initialize TemplateStore with llm_config "
                "or use insert_template with pre-computed embeddings."
            )
        
        try:
            embedding = await aembed_text(self.llm_config, text)
            return embedding
        except Exception as e:
            self.logger.exception(f"Error generating embedding: {e}")
            raise
    
    async def _generate_sql_from_template(
        self,
        template: Template,
        db_schema: Optional[str] = None,
    ) -> Optional[str]:
        """Generate SQL command from template components
        
        Args:
            template: Template object
            db_schema: Database schema string (optional)
            
        Returns:
            Optional[str]: Generated SQL command or None if generation fails
        """
        if self.llm_config is None:
            self.logger.warning("LLM config not provided, cannot generate SQL")
            return None
        
        try:
            # Get database schema if not provided
            if db_schema is None:
                adapter = self._get_adapter()
                try:
                    db_schema = await adapter.get_schema_str()
                    if not db_schema:
                        self.logger.warning("Could not retrieve database schema")
                        db_schema = "Database schema information not available"
                except Exception as e:
                    self.logger.warning(f"Failed to get database schema: {e}")
                    db_schema = "Database schema information not available"
            
            # Build prompt for SQL generation
            metrics_list = []
            for metric in template.metrics:
                if isinstance(metric, dict):
                    metric_name = metric.get("metric") or metric.get("name")
                    if metric_name:
                        metrics_list.append(metric_name)
                elif isinstance(metric, str):
                    metrics_list.append(metric)
            
            dims = template.dimensions or {}
            filters = template.filters or {}
            
            prompt = f"""You are a PostgreSQL database expert. Generate a SQL query based on the following template requirements:

Template Description: {template.description}
Base Query Type: {template.base_query or 'N/A'}

Metrics to include:
{chr(10).join([f"  - {m}" for m in metrics_list]) if metrics_list else "  (none)"}

Dimensions:
{chr(10).join([f"  - {k}: {v}" for k, v in dims.items() if v]) if dims else "  (none)"}

Filters:
{json.dumps(filters, indent=2) if filters else "  (none)"}

Database Schema:
{db_schema}

Instructions:
1. Generate a SQL SELECT query based on the base_query type:
   - timeseries: Time-series query with GROUP BY on time dimension
   - snapshot: Point-in-time query
   - comparison: Comparison query (e.g., quarter-over-quarter)
   - ranking: Ranking query with ORDER BY
2. Include all specified metrics and dimensions
3. Apply filters as needed
4. Use proper PostgreSQL syntax with double quotes for identifiers
5. Only return the SQL statement, no explanations or markdown formatting

Generate the SQL query:"""

            messages = [{"role": "user", "content": prompt}]
            response = await agenerate_chat(self.llm_config, messages)
            
            # Clean up the response
            sql = self._extract_sql_from_response(response)
            
            return sql
            
        except Exception as e:
            self.logger.exception(f"Failed to generate SQL from template: {e}")
            return None
    
    def _clean_sql_response(self, sql: str) -> str:
        """Clean up LLM SQL response"""
        import re
        # Remove markdown code blocks
        if "```" in sql:
            pattern = r'```(?:\w+)?\n?(.*?)```'
            matches = re.findall(pattern, sql, re.DOTALL)
            if matches:
                sql = matches[0].strip()
        
        # Remove leading/trailing whitespace
        sql = sql.strip()
        
        return sql
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL statement from LLM response"""
        import re
        # Try to find CREATE/SELECT/INSERT/UPDATE/DELETE statement
        sql_match = re.search(r'(CREATE|SELECT|INSERT|UPDATE|DELETE).*?(?:;|$)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(0).strip()
        
        # If no SQL found, return cleaned response
        return self._clean_sql_response(response)
    
    # ============================================================================
    # Insert template
    # ============================================================================
    
    async def insert_template(
        self,
        template: Template,
        sql_command: Optional[str] = None,
        components: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_sql: bool = True,
    ) -> Optional[int]:
        """Insert a template into the store
        
        Args:
            template: Template object
            sql_command: SQL command associated with this template (if None and generate_sql=True, will be generated)
            components: Components dictionary (if None, will be derived from template)
            metadata: Additional metadata
            generate_sql: Whether to generate SQL if sql_command is None (default: True)
            
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
            
            # Generate SQL if not provided and generation is enabled
            if sql_command is None and generate_sql:
                sql_command = await self._generate_sql_from_template(template)
                if sql_command:
                    self.logger.info(f"Generated SQL for template {template.id}")
            
            # Convert components to text for embedding
            components_text = self._components_to_text(components)
            
            # Generate embedding
            embedding = await self._generate_embedding(components_text)
            
            # Prepare data
            dimensions_json = json.dumps(template.dimensions, ensure_ascii=False)
            filters_json = json.dumps(template.filters, ensure_ascii=False)
            metrics_json = json.dumps(template.metrics, ensure_ascii=False)
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            insert_sql = f"""
                INSERT INTO {self.template_schema}.templates (
                    template_id, description, base_query,
                    dimensions, filters, metrics,
                    sql_command, components_text, embedding, metadata
                ) VALUES (
                    $1, $2, $3,
                    $4::jsonb, $5::jsonb, $6::jsonb,
                    $7, $8, $9::vector, $10::jsonb
                )
                ON CONFLICT (template_id) 
                DO UPDATE SET
                    description = EXCLUDED.description,
                    base_query = EXCLUDED.base_query,
                    dimensions = EXCLUDED.dimensions,
                    filters = EXCLUDED.filters,
                    metrics = EXCLUDED.metrics,
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
                template.base_query,
                dimensions_json,
                filters_json,
                metrics_json,
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
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Tuple[float, Template]]:
        """Find top K similar templates using vector similarity
        
        Args:
            components: Components dictionary to search for
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
            query_embedding = await self._generate_embedding(components_text)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Use cosine similarity for vector search
            search_sql = f"""
                SELECT 
                    template_id, description, base_query,
                    dimensions, filters, metrics,
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
                    base_query=row_dict["base_query"],
                    dimensions=row_dict["dimensions"] or {},
                    filters=row_dict["filters"] or {},
                    metrics=row_dict["metrics"] or [],
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

