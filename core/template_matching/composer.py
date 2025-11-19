from __future__ import annotations

from typing import Dict




def choose_template(components: Dict) -> str:
    """
    Deterministically choose a template id based on the normalized components.

    Examples (initial rules, extend as needed):
      - timeseries + month + single_org + snapshot → template_001
      - comparison + quarter + multiple_org → template_002
      - ranking + year + all_org + threshold → template_003
    """
    comps = components or {}
    bq = comps.get("base_query")
    dims = comps.get("dimensions") or {}
    time_g = dims.get("time")
    org_d = dims.get("org")
    filt_biz = (comps.get("filters") or {}).get("business") or {}

    # Rule: timeseries + month + single
    if bq == "timeseries" and time_g == "month" and org_d == "single":
        return "template_001"

    # Rule: comparison + quarter + multiple
    if bq == "comparison" and time_g == "quarter" and org_d == "multiple":
        return "template_002"

    # Rule: ranking + year + all + threshold
    if bq == "ranking" and time_g == "year" and org_d == "all" and bool(filt_biz.get("threshold")):
        return "template_003"

    # Additional useful defaults (metric-agnostic)
    if bq == "timeseries" and time_g == "month" and org_d == "single":
        return "template_timeseries_month_single_v1"
    if bq == "timeseries" and time_g == "quarter" and org_d == "single":
        return "template_timeseries_quarter_single_v1"
    if bq == "snapshot" and org_d in {"single", "multiple"}:
        return "template_snapshot_point_in_time_v1"

    # Fallback
    return "template_generic_v1"


