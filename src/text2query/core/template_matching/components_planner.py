from __future__ import annotations

from typing import Any, Dict, Optional


class ComponentPlanner:
    """
    LLM-first component planner.
    Given a natural language query, produce a strict 'components' YAML structure:

        components:
          base_query: timeseries | snapshot | comparison | ranking | null
          dimensions:
            time: day | month | quarter | year | null
            org: single | multiple | all | null
            region: string | null
            industry: string | null
          filters:
            standard:
              date_range: true|false
              org_filter: true|false
            business:
              doc_type: true|false
              threshold: true|false

    The planner asks the LLM to emit ONLY YAML. This class then normalizes the YAML
    into a deterministic Python dict with the 'components' root.
    """

    ALLOWED_BASE = {"timeseries", "snapshot", "comparison", "ranking"}
    ALLOWED_TIME = {"day", "month", "quarter", "year"}
    ALLOWED_ORG = {"single", "multiple", "all"}
    ALLOWED_STD_FILTERS = {"date_range", "org_filter"}
    ALLOWED_BIZ_FILTERS = {"doc_type", "threshold"}

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
        "Context date: {context_date}\n"
    ).strip()

    def __init__(self, llm: Any):
        self.llm = llm

    async def plan(self, query: str, *, context_date: str) -> Dict[str, Any]:
        """
        Ask the LLM to fill the 'components' YAML and normalize the result.
        """
        from llama_index.core.llms import ChatMessage

        prompt = self._PROMPT.format(query=query, context_date=context_date)
        raw = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        text = str(raw.message.content).strip()

        # Parse YAML (tolerate missing PyYAML by falling back to empty dict)
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}

        comps_in = data.get("components") if isinstance(data, dict) else {}
        return {"components": self._normalize_components(comps_in)}

    def _normalize_components(self, comps_in: Any) -> Dict[str, Any]:
        # Defaults
        out: Dict[str, Any] = {
            "base_query": None,
            "dimensions": {"time": None, "org": None, "region": None, "industry": None},
            "filters": {"standard": {}, "business": {}},
        }
        if not isinstance(comps_in, dict):
            return out

        # base_query
        bq = (comps_in.get("base_query") or "").strip().lower() if comps_in.get("base_query") else None
        if bq in self.ALLOWED_BASE:
            out["base_query"] = bq

        # dimensions
        dims = comps_in.get("dimensions") or {}
        if isinstance(dims, dict):
            time_v = (dims.get("time") or "").strip().lower() if dims.get("time") else None
            org_v = (dims.get("org") or "").strip().lower() if dims.get("org") else None
            region_v = dims.get("region")
            industry_v = dims.get("industry")
            out["dimensions"]["time"] = time_v if time_v in self.ALLOWED_TIME else None
            out["dimensions"]["org"] = org_v if org_v in self.ALLOWED_ORG else None
            out["dimensions"]["region"] = region_v if isinstance(region_v, str) and region_v.strip() else None
            out["dimensions"]["industry"] = (
                industry_v if isinstance(industry_v, str) and industry_v.strip() else None
            )

        # filters
        filts = comps_in.get("filters") or {}
        std = filts.get("standard") or {}
        biz = filts.get("business") or {}
        out["filters"]["standard"] = {k: True for k, v in (std.items() if isinstance(std, dict) else []) if k in self.ALLOWED_STD_FILTERS and bool(v)}
        out["filters"]["business"] = {k: True for k, v in (biz.items() if isinstance(biz, dict) else []) if k in self.ALLOWED_BIZ_FILTERS and bool(v)}

        return out

