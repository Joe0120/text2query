import re
from typing import Optional, List, Dict, Any, Tuple
import calendar
from datetime import date, timedelta

class RuleBasedSlotExtraction:
    # METRICS, may be chaged in the future in favor of a registry of metrics
    METRICS: Dict[str, Dict[str, Any]] = {
        "應收帳款": {
            "canonical": "應收帳款",
            "aliases": ["AR", "應收", "應收款項"],
            "field": "ar_balance",
            "value_type": "snapshot",
            "default_aggregation": "end_of_period",  # <- correct for AR
            "allowed_grains": {"month", "quarter", "year"},
        },
        "銷售金額": {
            "canonical": "銷售金額",
            "aliases": ["營收", "收入", "銷售額"],
            "field": "sales_amount",
            "value_type": "flow",
            "default_aggregation": "sum",
            "allowed_grains": {"day", "month", "quarter", "year"},
        },
        "平均單價": {
            "canonical": "平均單價",
            "aliases": ["單價", "均價", "平均價格"],
            "field": "unit_price",
            "value_type": "rate",
            "default_aggregation": "weighted_avg",   # enforce SUM(amount)/SUM(qty)
            "allowed_grains": {"day", "month", "quarter", "year"},
        },
    }

    QUICK_TIME = {
        "last year": ["去年", "上一年", "last year"],
        "this year": ["今年", "本年", "this year"],
        "last month": ['上個月', 'last month', "上月", "上月份"],
        "this month": ['本月', 'this month', "本月", "本月份"],
        "last quarter": ['上季度', 'last quarter', "上季度", "上季"],
        "this quarter": ['本季度', 'this quarter', "本季度", "本季"],
        "this week": ["本週", "this week", "這週", "本禮拜", "本周", "今週"],
        "last week": ["上週", "last week", "上禮拜", "上周", "過去一週"]
    }

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        return any(k in text for k in keywords)

    METRIC_ALIASES: Dict[str, str] = {}
    for k, v in METRICS.items():
        METRIC_ALIASES[k] = k
        for a in v.get("aliases", []):
            METRIC_ALIASES[a] = k

    def _month_start_end(self, d: date) -> Tuple[date, date]:
        start = date(d.year, d.month, 1)
        end_day = calendar.monthrange(d.year, d.month)[1]
        end = date(d.year, d.month, end_day)
        return start, end

    def _add_months(self, d: date, months: int) -> date:
        # Add months with day clamped to last day of target month
        year = d.year + (d.month - 1 + months) // 12
        month = (d.month - 1 + months) % 12 + 1
        end_day = calendar.monthrange(year, month)[1]
        day = min(d.day, end_day)
        return date(year, month, day)

    def _quarter_start_end(self, d: date) -> Tuple[date, date]:
        q = (d.month - 1) // 3 + 1
        start_month = 3 * (q - 1) + 1
        start = date(d.year, start_month, 1)
        # next quarter start then minus one day
        next_q_start_month = start_month + 3
        next_q_year = d.year + (1 if next_q_start_month > 12 else 0)
        next_q_start_month = ((next_q_start_month - 1) % 12) + 1
        next_q_start = date(next_q_year, next_q_start_month, 1)
        end = next_q_start - timedelta(days=1)
        return start, end

    def _week_start_end(self, d: date) -> Tuple[date, date]:
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        return start, end

    def quick_lookup(self, text: str, ctx: date) -> Optional[Dict[str, str]]:
        #quick lookups
        # last year
        if self._contains_any(text, self.QUICK_TIME["last year"]):
            start = date(ctx.year - 1, 1, 1)
            end = date(ctx.year - 1, 12, 31)
            return {"type": "relative", "reference": "去年", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # this year
        if self._contains_any(text, self.QUICK_TIME["this year"]):
            start = date(ctx.year, 1, 1)
            end = date(ctx.year, 12, 31)
            return {"type": "relative", "reference": "今年", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # last month
        if self._contains_any(text, self.QUICK_TIME["last month"]):
            first_this_month = date(ctx.year, ctx.month, 1)
            prev_last_day = first_this_month - timedelta(days=1)
            start = date(prev_last_day.year, prev_last_day.month, 1)
            end = prev_last_day
            return {"type": "relative", "reference": "上月", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # this month
        if self._contains_any(text, self.QUICK_TIME["this month"]):
            start, end = self._month_start_end(ctx)
            return {"type": "relative", "reference": "本月", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # last quarter
        if self._contains_any(text, self.QUICK_TIME["last quarter"]):
            # compute last quarter by taking this quarter start then step back one day
            this_q_start, _ = self._quarter_start_end(ctx)
            prev_last_day = this_q_start - timedelta(days=1)
            start, end = self._quarter_start_end(prev_last_day)
            return {"type": "relative", "reference": "上季", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # this quarter
        if self._contains_any(text, self.QUICK_TIME["this quarter"]):
            start, end = self._quarter_start_end(ctx)
            return {"type": "relative", "reference": "本季", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # last week
        if self._contains_any(text, self.QUICK_TIME["last week"]):
            prev = ctx - timedelta(days=7)
            start, end = self._week_start_end(prev)
            return {"type": "relative", "reference": "上周", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}
        # this week
        if self._contains_any(text, self.QUICK_TIME["this week"]):
            start, end = self._week_start_end(ctx)
            return {"type": "relative", "reference": "今週", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": end.isoformat()}    
        return None
    
    def _parse_absolute_time(self, text: str) -> Optional[Dict[str, str]]:
        # Patterns: YYYY-MM, YYYY/MM/DD, YYYY年Q[1-4], YYYY年M月至M月 (same year)
        t = text.strip()
        # YYYY年 (whole year)
        m = re.search(r"(\d{4})年(?!\s*[Qq]|\s*\d)", t)
        if m:
            y = int(m.group(1))
            try:
                start = date(y, 1, 1)
                end = date(y, 12, 31)
                return {
                    "type": "absolute",
                    "reference": f"{y}年",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                }
            except ValueError:
                pass
        # YYYY-MM
        m = re.search(r"(\d{4})[-/](\d{1,2})(?!\d)", t)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            try:
                start = date(y, mo, 1)
                end_day = calendar.monthrange(y, mo)[1]
                end = date(y, mo, end_day)
                return {"type": "absolute", "reference": f"{y}-{mo:02d}", "start_date": start.isoformat(), "end_date": end.isoformat()}
            except ValueError:
                pass
        # YYYY/MM/DD
        m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", t)
        if m:
            y, mo, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                d = date(y, mo, dd)
                return {"type": "absolute", "reference": f"{y}-{mo:02d}-{dd:02d}", "start_date": d.isoformat(), "end_date": d.isoformat()}
            except ValueError:
                pass
        # YYYY年Q[1-4]
        m = re.search(r"(\d{4})年\s*[Qq]([1-4])", t)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            start_month = 3 * (q - 1) + 1
            start = date(y, start_month, 1)
            next_q_start_month = start_month + 3
            next_q_year = y + (1 if next_q_start_month > 12 else 0)
            next_q_start_month = ((next_q_start_month - 1) % 12) + 1
            end = date(next_q_year, next_q_start_month, 1) - timedelta(days=1)
            return {"type": "absolute", "reference": f"{y}年Q{q}", "start_date": start.isoformat(), "end_date": end.isoformat()}
        # YYYY年M月至M月 (same year range)
        m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*至\s*(\d{1,2})月", t)
        if m:
            y, m1, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                start = date(y, m1, 1)
                end_day = calendar.monthrange(y, m2)[1]
                end = date(y, m2, end_day)
                return {"type": "absolute", "reference": f"{y}年{m1}月至{m2}月", "start_date": start.isoformat(), "end_date": end.isoformat()}
            except ValueError:
                pass
        return None

    def _parse_relative_time(self, text: str, ctx: date) -> Optional[Dict[str, str]]:
        text = text.strip()
        result = self.quick_lookup(text, ctx)
        if result:
            return result
        abs_res = self._parse_absolute_time(text)
        if abs_res:
            # Include context_date for consistency
            abs_res["context_date"] = ctx.isoformat()
            return abs_res
        m = re.search(r"(\d+)\s*(天|日|週|周|月|年)", text)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit in ("天", "日"):
                start = ctx - timedelta(days=n)
            elif unit in ("週","周"):
                start = ctx - timedelta(weeks=n)
            elif unit == "月":
                # go to first of current month, then step back n months
                first_this_month = date(ctx.year, ctx.month, 1)
                start_month_anchor = self._add_months(first_this_month, -n)
                start = start_month_anchor
            else:
                # 年: try replace, fallback to Feb-28 if invalid
                target_year = max(1, ctx.year - n)
                try:
                    start = date(target_year, ctx.month, ctx.day)
                except ValueError:
                    # handle Feb 29 -> last valid day
                    end_day = calendar.monthrange(target_year, ctx.month)[1]
                    start = date(target_year, ctx.month, min(ctx.day, end_day))
            return {"type": "relative", "reference": f"近{n}{unit}", "context_date": ctx.isoformat(),
                    "start_date": start.isoformat(), "end_date": ctx.isoformat()}
        return None

    def _detect_granularity(self, text: str) -> Optional[str]:
        if self._contains_any(text, ["各月份","各月","按月","每月","月度","月"]):
            return "month"
        if self._contains_any(text, ["每季","季","季度","Q1","Q2","Q3","Q4"]):
            return "quarter"
        if self._contains_any(text, ["每天","每日","按日","日度","天"]):
            return "day"
        if self._contains_any(text, ["每年","年度","年"]):
            return "year"
        return None
        
    def _detect_intent(self, text: str) -> str:
        if self._contains_any(text, ["比較","對比","同比","環比","YoY","MoM","QoQ","差異"]):
            return "comparison"
        if self._contains_any(text, ["排名","前","Top","top"]):
            return "ranking"
        return "fact"


    def _detect_metric(self, question: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        for alias, canon in self.METRIC_ALIASES.items():
            if alias in question:
                m = self.METRICS[canon]
                return m["canonical"], {"field": m["field"], "value_type": m["value_type"], "canonical": m["canonical"]}
        return None, None

    # Detect subject/organization names (e.g., 台積電、聯電、鴻海、ABC公司)
    def _detect_org(self, question: str) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        q = question.strip()

        # Remove metric aliases from consideration to avoid false positives
        cleaned = q
        for alias in self.METRIC_ALIASES.keys():
            cleaned = cleaned.replace(alias, "")

        # Remove quick time expressions
        for kws in self.QUICK_TIME.values():
            for w in kws:
                cleaned = cleaned.replace(w, "")

        # Remove common structural words
        cleaned = re.sub(r"(各月份|各月|每月|按月|月份|月度|年度|比較|對比|同比|環比|排行|排名|前\d+|Top\d+|top\d+|的|之)", " ", cleaned)

        # Split by common conjunctions/separators to allow multiple orgs
        parts = re.split(r"[與和跟及、,，/&]", cleaned)

        # Heuristic pattern for Chinese/English org tokens with optional corporate suffixes
        org_pattern = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:股份有限公司|有限公司|公司|集團|科技|電子|製造|工業|實業)?)")

        candidates: List[str] = []
        for part in parts:
            s = part.strip()
            if not s:
                continue
            # Prefer token before possessive marker if present
            s = s.split(" 的 ")[0].strip() if " 的 " in s else s
            m = org_pattern.search(s)
            if not m:
                continue
            name = m.group(1).strip()
            # Filter overly generic words
            if len(name) < 2:
                continue
            candidates.append(name)

        if not candidates:
            return None, None

        # Deduplicate while preserving order
        seen = set()
        ordered = [x for x in candidates if not (x in seen or seen.add(x))]
        return ordered, {"source": "rule", "matches": ordered}

    def extract_slots(self, question: str, context_date: Optional[date] = None) -> Dict[str, Any]:
        q = question.strip()
        ctx = context_date or date.today()

        org_values, org_info = self._detect_org(q)
        metric_value, metric_info = self._detect_metric(q)
        time_payload = self._parse_relative_time(q, ctx)
        granularity = self._detect_granularity(q)
        intent = self._detect_intent(q)

        aggregation = None
        if metric_info:
            vt = metric_info["value_type"]
            if vt == "flow":
                aggregation = "sum"
            elif vt == "snapshot":
                aggregation = "end_of_period"
            else:
                aggregation = "avg"

        return {
            "entities": {
                "org": org_values,
                "time": time_payload,
                "metric": metric_value,
                "granularity": granularity,
            },
            "intent": intent,
            "modifiers": {
                "comparison": False,
                "aggregation": aggregation,
                "groupby": [],
            },
            "mappings": {
                "org": org_info,
                "metric": metric_info,
            },
            "provenance": {"method": "rule"},
        }
        
    # return a enriched slots post-sql generation and rag retrieval for more robust template matching
    def enrich_slots(self, slots: Dict[str, Any], sql_text: str) -> Dict[str, Any]:
        # TODO: implement this
        return slots

class LLMSlotExtraction:
    """LLM-driven slot extraction using the centralized LLMService.

    Attempts to parse a YAML response and normalize it to the same schema
    as the rule-based extractor. Falls back to rule-based on failure.
    """

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

    @classmethod
    async def extract_slots(
        cls,
        llm: Any,
        question: str,
        *,
        context_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Call LLM and parse YAML into our slot schema; fallback to rule-based."""
        ctx = context_date or date.today()
        prompt = cls._PROMPT_TEMPLATE.format(query=question, context_date=ctx.isoformat())

        from llama_index.core.llms import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            raw = await llm.achat(messages)
            text = str(raw.message.content).strip()

            data: Dict[str, Any]
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(text) or {}
            except Exception:
                # Very naive fallback: try to coerce YAML-ish to JSON-ish when no YAML loader
                # If it fails, fall back to rule-based below
                data = {}

            entities = cls._normalize_entities(data.get("entities")) if isinstance(data, dict) else {}
            intent = cls._map_intent(data.get("intent")) if isinstance(data, dict) else "fact"
            modifiers = cls._normalize_modifiers(data.get("modifiers")) if isinstance(data, dict) else {
                "comparison": False,
                "aggregation": None,
                "groupby": [],
            }

            # Minimal validation; if critical parts missing, we will hybrid-backfill
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
                    "metric": {"source": "llm"} if entities .get("metric") else None,
                },
                "provenance": {"method": "llm"},
            }

            return result
        except Exception:
            # Fallback to rule-based extraction if anything goes wrong
            rule = RuleBasedSlotExtraction()
            return rule.extract_slots(question, context_date=ctx)

class SlotExtraction:
    def __init__(self):
        self.rule = RuleBasedSlotExtraction()
        self.llm = LLMSlotExtraction()
    

    def backfill_slots(self, slots_llm: Dict[str, Any], slots_rb: Dict[str, Any]) -> Dict[str, Any]:
        """Merge LLM slots with rule-based slots to fill gaps.

        This function replaces the previous T2SQLService._backfill_slots dependency so that
        slot extraction is independent of any SQL generation service.
        """
        # Start with LLM result and fill missing pieces from rule-based
        slots = slots_llm or {}
        slots.setdefault("entities", {})
        slots.setdefault("modifiers", {})
        slots.setdefault("mappings", {})

        # Entities backfill: org, time, metric, granularity
        for key in ("org", "time", "metric", "granularity"):
            v_llm = slots.get("entities", {}).get(key)
            v_rb = slots_rb.get("entities", {}).get(key)
            if v_llm in (None, "", []) and v_rb not in (None, "", []):
                slots["entities"][key] = v_rb

        # Modifiers backfill: aggregation (None), groupby (empty), comparison (only if missing)
        mod_llm = slots.get("modifiers", {})
        mod_rb = slots_rb.get("modifiers", {})
        if mod_llm.get("aggregation") in (None, "") and mod_rb.get("aggregation") not in (None, ""):
            mod_llm["aggregation"] = mod_rb.get("aggregation")
        if isinstance(mod_llm.get("groupby"), list) and not mod_llm.get("groupby") and mod_rb.get("groupby"):
            mod_llm["groupby"] = mod_rb.get("groupby")
        if "comparison" not in mod_llm and "comparison" in mod_rb:
            mod_llm["comparison"] = mod_rb.get("comparison")
        slots["modifiers"] = mod_llm

        # If intent missing/empty from LLM, backfill from rule-based
        if not slots.get("intent") and slots_rb.get("intent"):
            slots["intent"] = slots_rb.get("intent")
        return slots

    async def extract_slots(
        self,
        question: str,
        context_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        ctx = context_date or date.today()
        llm_slots = await self.llm.extract_slots(
            question,
            context_date=ctx,
        )
        # rb_slots = self.rule.extract_slots(question, context_date=ctx)
        # merged = self.backfill_slots(llm_slots, rb_slots)  # type: ignore[arg-type]
        # merged["provenance"] = {"method": "hybrid"}
        return llm_slots