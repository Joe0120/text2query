from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Template:
    """Template definition used for RAG-based template matching."""

    id: str
    description: str
    base_query: Optional[str]
    dimensions: Dict[str, Any]
    filters: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    # Optional SQL associated with this template (from YAML or DB)
    sql: Optional[str] = None


