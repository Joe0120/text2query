from __future__ import annotations

import yaml
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

TIME_ORDER = ["day", "month", "quarter", "year"]
TIME_INDEX = {v: i for i, v in enumerate(TIME_ORDER)}

ORG_ORDER = ["single", "multiple", "all"]
ORG_INDEX = {v: i for i, v in enumerate(ORG_ORDER)}


def _time_similarity(template_time: Optional[str], requested_time: Optional[str]) -> float:
    if not template_time or not requested_time:
        return 0.0
    if template_time == requested_time:
        return 2.0

    if template_time in TIME_INDEX and requested_time in TIME_INDEX:
        dist = abs(TIME_INDEX[template_time] - TIME_INDEX[requested_time])
        if dist == 1:
            return 1.0
        elif dist == 2:
            return 0.5
    return 0.0


def _org_similarity(template_org: Optional[str], requested_org: Optional[str]) -> float:
    if not template_org or not requested_org:
        return 0.0
    if template_org == requested_org:
        return 2.0

    if template_org == "all" and requested_org in {"multiple", "single"}:
        return 1.0
    if template_org == "multiple" and requested_org == "single":
        return 1.0

    return 0.0

@dataclass
class Template:
    id: str
    description: str
    base_query: Optional[str]
    dimensions: Dict[str, Any]
    filters: Dict[str, Any]
    fact_view: Optional[str]
    metrics: List[Dict[str, Any]]

class TemplateRepo:
    def __init__(self, templates_file: str = "templates/templates.yaml"):
        self.templates_file = templates_file
        self.templates: Dict[str, Template] = self.load_templates()

    def load_templates(self) -> Dict[str, Any]:
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
                fact_view=template_data.get('fact_view', None),
                metrics=template_data.get('metrics', []))
            self.templates[template_id] = template
        return self.templates
    
    def find_best_match(self, components: Dict[str, Any]) -> Template:
        base = components.get("base_query")
        dims = components.get("dimensions") or {}
        time = dims.get("time")
        org = dims.get("org")
        candidates = []
        for t in self.templates.values():
            # base_query must match (if defined)
            if t.base_query and base and t.base_query != base:
                continue
            score = 0
            score += self.calculate_score(t, components)
            candidates.append((score, t))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]
    
    def calculate_score(self, template: Template, components: Dict[str, Any]) -> float:
        score = 0.0

        if template.base_query and template.base_query == components.get("base_query"):
            score += 2.0

        dims = components.get("dimensions") or {}
        t_dims = template.dimensions or {}

        # Time similarity
        score += _time_similarity(
            t_dims.get("time"),
            dims.get("time"),
        )

        # Org similarity
        score += _org_similarity(
            t_dims.get("org"),
            dims.get("org"),
        )

        # Add filter scoring?

        return score / 6.0