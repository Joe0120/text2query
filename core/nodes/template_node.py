from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import yaml
import logging

from .state import WisbiState
from ..utils.models import agenerate_chat, ModelConfig
from ..connections.postgresql import PostgreSQLConfig
from ..template_matching.store import TemplateStore
from ..template_matching.models import Template

logger = logging.getLogger(__name__)


class TemplateRepo:
    """Template repository with RAG-based retrieval
    
    This class manages templates using vector similarity search.
    Templates are loaded from YAML and stored in PostgreSQL with embeddings.
    """
    
    def __init__(
        self,
        templates_file: Optional[str] = None,
        template_store: Optional[TemplateStore] = None,
        db_config: Optional[PostgreSQLConfig] = None,
        llm_config: Optional[ModelConfig] = None,
        template_schema: str = "wisbi",
        embedding_dim: int = 768,
    ):
        """Initialize TemplateRepo
        
        Args:
            templates_file: Path to templates.yaml file (optional, will try to find it)
            template_store: Pre-initialized TemplateStore instance (optional)
            db_config: PostgreSQL config for TemplateStore (required if template_store not provided)
            llm_config: LLM config for embeddings (required if template_store not provided)
            template_schema: Schema name for template tables (default: "wisbi")
            embedding_dim: Embedding vector dimension (default: 768)
        """
        self.templates_file = templates_file or self._find_templates_file()
        self.template_store = template_store
        self.db_config = db_config
        self.llm_config = llm_config
        self.template_schema = template_schema
        self.embedding_dim = embedding_dim
        self.templates: Dict[str, Template] = {}

    def _find_templates_file(self) -> str:
        """Try to find templates.yaml file in common locations"""
        current_file = Path(__file__)
        possible_paths = [
            current_file.parent.parent / "utils" / "templates" / "templates.yaml",
            Path("core/utils/templates/templates.yaml"),
            Path("templates/templates.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError("Could not find templates.yaml file")

    def load_templates(self) -> Dict[str, Template]:
        """Load templates from YAML file
        
        Returns:
            Dict[str, Template]: Dictionary mapping template IDs to Template objects
        """
        try:
            with open(self.templates_file, 'r') as file:
                data = yaml.safe_load(file)
            templates = data.get('templates', {})
            for template_id, template_data in templates.items():
                template = Template(
                    id=template_id,
                    description=template_data.get('description', ''),
                    base_query=template_data.get('base_query', None),
                    dimensions=template_data.get('dimensions', {}),
                    filters=template_data.get('filters', {}),
                    metrics=template_data.get('metrics', []),
                    sql=template_data.get('sql', None))
                self.templates[template_id] = template
            logger.info(f"Loaded {len(self.templates)} templates from {self.templates_file}")
            return self.templates
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize TemplateRepo and auto-populate templates if store is empty"""
        if self.template_store is None:
            if self.db_config is None or self.llm_config is None:
                raise ValueError(
                    "Either template_store must be provided, or both db_config and llm_config "
                    "must be provided for TemplateRepo initialization"
                )
            self.template_store = await TemplateStore.initialize(
                postgres_config=self.db_config,
                template_schema=self.template_schema,
                embedding_dim=self.embedding_dim,
                llm_config=self.llm_config,
            )
        
        # Auto-populate templates from YAML if store is empty
        is_empty = await self.template_store.is_empty()
        if is_empty:
            logger.info("Template store is empty, auto-populating from templates.yaml")
            templates = self.load_templates()
            
            # Insert all templates in the store with SQL from YAML or auto-generation
            for template in templates.values():
                await self.template_store.insert_template(
                    template=template,
                    sql_command=template.sql,  # Use SQL from YAML if available
                    generate_sql=template.sql is None,  # Only generate if not provided
                )
            
            logger.info(f"Auto-populated {len(templates)} templates to store")
            # Clear templates dict after use to free memory
            self.templates.clear()
        else:
            logger.info("Template store already contains templates, skipping auto-population")
    
    async def find_top_templates(
        self,
        components: Dict[str, Any],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Tuple[float, Template]]:
        """
        Find top K matching templates using RAG-based vector similarity.
        
        Args:
            components: Dictionary with base_query, dimensions, filters, etc.
            top_k: Number of top templates to return (default: 3)
            min_similarity: Minimum similarity score threshold (default: 0.0)
            
        Returns:
            List[Tuple[float, Template]]: List of (similarity_score, template) tuples,
                sorted by similarity (highest first). The template.sql field
                contains the corresponding SQL command (if available).
        """
        await self.initialize()
        return await self.template_store.find_similar_templates(
            components=components,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    
    async def close(self):
        """Close the template store connection"""
        if self.template_store:
            await self.template_store.close()


class TemplateNode:
    _PROMPT = (
        "You are a planner. Produce ONLY YAML matching the schema below under a single 'components' root.\n"
        "If a field is not inferable, use null or omit it.\n\n"
        "Allowed values:\n"
        "- base_query: [timeseries, snapshot, comparison, ranking]\n"
        "- dimensions.time: [day, month, quarter, year]\n"
        "- dimensions.org: [single, multiple, all]\n"
        "- filters.standard: [date_range, org_filter]\n"
        "- filters.business: [doc_type, threshold]\n\n"
        "YAML output (no code fences):\n"
        "components:\n"
        "  base_query: <enum|null>\n"
        "  dimensions:\n"
        "    time: <enum|null>\n"
        "    org: <enum|null>\n"
        "    region: <string|null>\n"
        "    industry: <string|null>\n"
        "  filters:\n"
        "    standard: {date_range: <bool>, org_filter: <bool>}\n"
        "    business: {doc_type: <bool>, threshold: <bool>}\n\n"
        "Query: {query}\n"
    ).strip()
    
    def __init__(
        self,
        template_repo: TemplateRepo,
    ):
        """
        Initialize TemplateNode.
        
        Args:
            template_repo: Template repository instance
        """
        self.template_repo = template_repo

    async def plan(self, state: WisbiState, query_text: str) -> Dict[str, Any]:
        prompt = self._PROMPT.format(query=query_text)
        model_config = state.get("llm_config")
        messages = [{"role": "user", "content": prompt}]
        response = await agenerate_chat(model_config, messages)
        return yaml.safe_load(response)

    async def __call__(self, state: WisbiState) -> WisbiState:
        query_text = state.get("query_text")
        if query_text is None:
            return state
        
        components_raw = await self.plan(state, query_text)
        components = components_raw.get("components", {}) or {}
        
        # Find top 3 templates using RAG
        top_templates = await self.template_repo.find_top_templates(components, top_k=3)
        
        if not top_templates:
            logger.warning("No templates found matching components")
            return state
        
        # Store top templates in state
        state["top_templates"] = top_templates
        # Store the best match
        best_score, best_template = top_templates[0]
        state["template_score"] = best_score
        state["template"] = best_template
        state["template_sql"] = best_template.sql
        
        return state
