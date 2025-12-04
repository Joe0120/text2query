from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import date
import logging
import yaml
from ..utils.models import agenerate_chat
from ..utils.model_configs import ModelConfig

class SlotExtraction:
    """
    Request-based slot extraction using agenerate_chat.
    Extracts slots from queries without using llama_index.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    _PROMPT_TEMPLATE = (
        """
You are an extraction model that outputs ONLY strict YAML matching the schema below. No prose.

Task: Given a user query (Chinese or English), extract slots.
Current date: {context_date}

YAML Schema (required keys and value types):
entities:
  - org: [string]            # Name of the organization, may be [] if none
  - time:                    # object or null
      type: "absolute" | "relative"
      reference: string      # e.g., 去年, 近3月, 本季, or a label for absolute range (e.g., 2024年)
      context_date: string   # ISO date of current context
      start_date: string     # ISO date if determinable
      end_date: string       # ISO date if determinable
  - metric: string | null
  - granularity: "day" | "month" | "quarter" | "year" | null

intent: "timeseries" | "comparison" | "ranking" | "summary" | "fact"

modifiers:
  - comparison: true | false
  - aggregation: "sum" | "avg" | "max" | "min" | "end_of_period" | "weighted_avg" | "none"
  - groupby: [string]

Output must be valid YAML only. Do not include markdown fences or explanations.

Example for query: 「台積電去年各月份的應收帳款」
entities:
  - org: ["台積電"]
  - time:
      type: "relative"
      reference: "去年"
      context_date: "{context_date}"
      start_date: "{context_date}"
      end_date: "{context_date}"
  - metric: "應收帳款"
  - granularity: "month"

intent: "timeseries"

modifiers:
  - comparison: false
  - aggregation: "end_of_period"
  - groupby: []

Now extract the slots for the following query:
{query}
        """
    ).strip()
    
    @staticmethod
    def _normalize_entities(entities_obj: Any) -> Dict[str, Any]:
        """Convert YAML 'entities' (list of maps) to our dict schema."""
        result: Dict[str, Any] = {
            "org": None,
            "time": None,
            "metric": None,
            "granularity": None,
        }
        if isinstance(entities_obj, list):
            for item in entities_obj:
                if not isinstance(item, dict):
                    continue
                if "org" in item:
                    result["org"] = item.get("org")
                if "time" in item:
                    result["time"] = item.get("time")
                if "metric" in item:
                    result["metric"] = item.get("metric")
                if "granularity" in item:
                    result["granularity"] = item.get("granularity")
        elif isinstance(entities_obj, dict):
            # In case the model returns a dict instead of the specified list format
            for key in ("org", "time", "metric", "granularity"):
                if key in entities_obj:
                    result[key] = entities_obj.get(key)
        return result
    
    @staticmethod
    def _map_intent(intent_value: Any) -> str:
        s = str(intent_value or "").strip().lower()
        if s in {"timeseries", "time-series", "series"}:
            return "timeseries"
        if s in {"comparison", "compare"}:
            return "comparison"
        if s in {"ranking", "rank", "topn"}:
            return "ranking"
        if s in {"summary", "overview"}:
            return "summary"
        # other -> default to fact
        return "fact"
    
    @staticmethod
    def _normalize_modifiers(modifiers_obj: Any) -> Dict[str, Any]:
        result = {
            "comparison": False,
            "aggregation": None,
            "groupby": [],
        }
        if isinstance(modifiers_obj, list):
            merged: Dict[str, Any] = {}
            for item in modifiers_obj:
                if isinstance(item, dict):
                    merged.update(item)
            modifiers_obj = merged
        if isinstance(modifiers_obj, dict):
            comp = modifiers_obj.get("comparison")
            if isinstance(comp, bool):
                result["comparison"] = comp
            elif isinstance(comp, str):
                result["comparison"] = comp.strip().lower() == "true"
            agg = modifiers_obj.get("aggregation")
            if isinstance(agg, str) and agg.strip().lower() not in {"", "none", "null"}:
                result["aggregation"] = agg.strip().lower()
            groupby = modifiers_obj.get("groupby")
            if isinstance(groupby, list):
                result["groupby"] = groupby
        return result
    
    async def extract_slots(
        self,
        llm_config: ModelConfig,
        question: str,
        context_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Extract slots from query using request-based LLM call.
        
        Args:
            llm_config: ModelConfig instance for LLM API calls
            question: The query/question to extract slots from
            context_date: Optional date context (defaults to today)
        
        Returns:
            Dictionary containing extracted slots with entities, intent, modifiers, etc.
        """
        ctx = context_date or date.today()
        prompt = self._PROMPT_TEMPLATE.format(query=question, context_date=ctx.isoformat())
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Use request-based agenerate_chat instead of llama_index
            text = await agenerate_chat(llm_config, messages)
            text = text.strip()
            
            # Parse YAML response
            data: Dict[str, Any]
            try:
                # Remove markdown code fences if present
                if text.startswith("```"):
                    lines = text.split("\n")
                    # Remove first line (```yaml or ```)
                    if len(lines) > 1:
                        lines = lines[1:]
                    # Remove last line (```)
                    if len(lines) > 1 and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines)
                
                data = yaml.safe_load(text) or {}
            except Exception as e:
                self.logger.error(f"Failed to parse YAML response: {e}\nResponse: {text[:500]}")
                raise ValueError(f"Failed to parse YAML response: {e}\nResponse: {text[:500]}") from e
            
            # Normalize entities
            entities = self._normalize_entities(data.get("entities")) if isinstance(data, dict) else {}
            intent = self._map_intent(data.get("intent")) if isinstance(data, dict) else "fact"
            modifiers = self._normalize_modifiers(data.get("modifiers")) if isinstance(data, dict) else {
                "comparison": False,
                "aggregation": None,
                "groupby": [],
            }
            
            # Minimal validation
            if not isinstance(entities, dict):
                entities = {"org": None, "time": None, "metric": None, "granularity": None}
            # time sanity: ensure dict if present
            if entities.get("time") is not None and not isinstance(entities.get("time"), dict):
                entities["time"] = None
            
            result = {
                "entities": entities,
                "intent": intent,
                "modifiers": modifiers,
                "mappings": {
                    "org": {"source": "llm"} if entities.get("org") else None,
                    "metric": {"source": "llm"} if entities.get("metric") else None,
                },
                "provenance": {"method": "llm"},
            }
            
            return result
        except Exception as e:
            # If extraction fails, return empty slots structure
            # Could optionally fall back to rule-based extraction here
            self.logger.warning(
                f"Slot extraction failed for query '{question[:100]}': {e}",
                exc_info=True
            )
            return {
                "entities": {"org": None, "time": None, "metric": None, "granularity": None},
                "intent": "fact",
                "modifiers": {"comparison": False, "aggregation": None, "groupby": []},
                "mappings": {},
                "provenance": {"method": "llm", "error": str(e)},
            }
