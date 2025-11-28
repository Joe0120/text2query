from __future__ import annotations
from typing import Any, Dict, Optional
from ..template_matching.template_repo import TemplateRepo
from .state import WisbiState
from ..utils.models import agenerate_chat
import yaml

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
    def __init__(self, template_repo: TemplateRepo):
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
        
        components = await self.plan(state, query_text)
        best_score, template = self.template_repo.find_best_match(components)
        state["template_score"] = best_score
        state["template"] = template
        return state