"""Planning-officer assistant: a tool-using LLM agent with retrieval and output guardrails.

Flow
  1. The model sees only a compact plan summary and may call tools:
       get_plan_summary, list_species, search_knowledge(query)
  2. It must finish by calling submit_recommendation(...) with structured fields.
  3. Guardrails (code, not prompt) validate that structured output:
       - every species is in the vetted list
       - every citation is an id returned by search_knowledge in THIS run
       - a rationale and caveat are present
  4. The final text is assembled by code. All numbers (cost, cooling, uncertainty)
     come from the planner, never from the model.
  5. If there is no API key, the call fails, or validation fails, a deterministic
     template is returned instead and the response says mode='template'.

A human still has to approve through POST /approve.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .config import VETTED_SPECIES
from .planner import agent_explain as template_explain
from .rag import get_kb
from .tracing import traceable, wrap_anthropic, wrap_openai

MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")          # Anthropic
def _norm_groq_model(m: str) -> str:
    """Forgive typos like 'gpt oss 20b' -> 'openai/gpt-oss-20b'."""
    m = (m or "").strip()
    if "/" not in m:
        m = m.lower().replace(" ", "-").replace("_", "-")
        if m.startswith("gpt-oss"):
            m = "openai/" + m
    return m


GROQ_MODEL = _norm_groq_model(os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))  # Groq (OpenAI-compatible API)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_TURNS = 6
VETTED_NAMES = {s["name"] for s in VETTED_SPECIES}

SYSTEM = """You assist a municipal planning officer in Peshawar, Pakistan with urban tree planting.
A planner has already chosen the blocks; you do NOT change them or invent numbers.
Your job: recommend which species from the vetted list suit this plan and explain why.
Rules:
- Use tools to look things up. Only recommend species returned by list_species.
- Support every claim about species or maintenance with search_knowledge results, and cite their ids.
- If the knowledge base does not cover something, say so in the caveat instead of guessing.
- Cooling estimates are correlational and coarse; the caveat must say so.
- Finish by calling submit_recommendation exactly once."""

TOOLS = [
    {"name": "get_plan_summary", "description": "Get the planner's chosen sites, cost and cooling estimate.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_species", "description": "List the vetted species with water need and cooling potential.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_knowledge", "description": "Search the local knowledge base (species, maintenance, heat evidence). Returns chunks with ids to cite.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "submit_recommendation", "description": "Submit the final recommendation. Ends the run.",
     "input_schema": {"type": "object", "properties": {
         "species": {"type": "array", "items": {"type": "string"}, "description": "Exact names from list_species."},
         "rationale": {"type": "string", "description": "2-4 sentences on why these species fit this plan."},
         "citations": {"type": "array", "items": {"type": "string"}, "description": "Chunk ids from search_knowledge."},
         "caveat": {"type": "string"}},
         "required": ["species", "rationale", "citations", "caveat"]}},
]


def _plan_summary(plan: Dict) -> Dict:
    sites = plan.get("selected") or plan.get("selected_summary") or []
    return {
        "area": plan.get("area"), "n_sites": plan["n_sites"], "total_cost": plan["total_cost"],
        "aggregate_cooling_c": round(plan["aggregate_cooling_c"], 2),
        "aggregate_uncertainty_c": round(plan["aggregate_uncertainty_c"], 2),
        "mean_current_green": round(sum(s["current_green"] for s in sites) / len(sites), 3) if sites else None,
        "mean_surface_temp_c": round(sum(s["heat"] for s in sites) / len(sites), 1) if sites else None,
        "caveat": plan.get("caveat"),
    }


def validate(sub: Dict, retrieved_ids: set) -> List[str]:
    """Guardrail checks on the model's structured output. Returns a list of problems (empty = pass)."""
    issues = []
    species = sub.get("species") or []
    if not species:
        issues.append("no species given")
    issues += [f"species not in vetted list: {s!r}" for s in species if s not in VETTED_NAMES]
    cites = sub.get("citations") or []
    if not cites:
        issues.append("no citations")
    issues += [f"citation not retrieved in this run: {c!r}" for c in cites if c not in retrieved_ids]
    if len((sub.get("rationale") or "").strip()) < 20:
        issues.append("rationale missing")
    cav = (sub.get("caveat") or "").lower()
    if not cav or not any(w in cav for w in ("correlat", "estimate", "uncertain", "coarse")):
        issues.append("caveat must state estimates are correlational/uncertain")
    return issues


def _render(plan: Dict, sub: Dict, kb) -> str:
    by_id = {c.id: c for c in kb.chunks}
    spec = {s["name"]: s for s in VETTED_SPECIES}
    lines = [
        f"Recommendation for Planning Officer ({plan.get('area') or 'selected area'}):",
        f"- {plan['n_sites']} planting blocks within budget (cost ≈ {plan['total_cost']:.0f}).",
        f"- Estimated average cooling per planted block {plan['aggregate_cooling_c']:.1f} °C (± {plan['aggregate_uncertainty_c']:.1f} °C).",
        "", "Species (vetted list only):",
    ]
    for n in sub["species"]:
        s = spec[n]
        lines.append(f"  • **{n}**: water={s['water']}, cooling={s['cooling']}")
    lines += ["", sub["rationale"].strip(), "", "Caveat: " + sub["caveat"].strip(), "", "Sources:"]
    for cid in sub["citations"]:
        c = by_id.get(cid)
        lines.append(f"  [{cid}] {c.title} ({c.source})" if c else f"  [{cid}]")
    lines += ["", "Please review the chosen blocks on the map and approve or request changes."]
    return "\n".join(lines)


def _fallback(plan: Dict, reason: str, trace: List[dict]) -> Dict:
    return {"text": template_explain(plan), "mode": "template", "reason": reason,
            "species": [s["name"] for s in VETTED_SPECIES[:5]], "citations": [], "tool_calls": trace}


OAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                "parameters": t["input_schema"]}} for t in TOOLS]


def _key(name: str) -> str:
    """Read an env var, tolerating stray spaces and quotes pasted into .env."""
    return os.getenv(name, "").strip().strip("'\"").strip()


def key_diagnostics() -> dict:
    from .config import ENV_FOUND, ENV_PATH, ENV_TXT_MISTAKE
    g, a = _key("GROQ_API_KEY"), _key("ANTHROPIC_API_KEY")
    return {
        "env_file_expected_at": str(ENV_PATH), "env_file_found": ENV_FOUND,
        "found_env_txt_by_mistake": ENV_TXT_MISTAKE,
        "groq_key_set": bool(g), "groq_key_preview": (g[:4] + "…" + f"({len(g)} chars)") if g else None,
        "anthropic_key_set": bool(a), "provider_forced": os.getenv("AGENT_PROVIDER") or None,
        "groq_model": GROQ_MODEL,
        "hint": None if (g or a) else (
            "No key visible. Create a file named exactly .env (not .env.txt) at the path above with a line "
            "GROQ_API_KEY=your_key , then restart the server."),
    }


def _make_client():
    """Pick a backend: AGENT_PROVIDER=groq|anthropic, else whichever key is set (Groq first).
    Returns (kind, client) or (None, reason)."""
    provider = os.getenv("AGENT_PROVIDER", "").strip().lower()
    if provider == "groq" or (not provider and _key("GROQ_API_KEY")):
        if not _key("GROQ_API_KEY"):
            return None, "GROQ_API_KEY not set"
        from openai import OpenAI
        return "openai", wrap_openai(OpenAI(api_key=_key("GROQ_API_KEY"), base_url=GROQ_BASE_URL))
    if _key("ANTHROPIC_API_KEY"):
        import anthropic
        return "anthropic", wrap_anthropic(anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY")))
    from .config import ENV_FOUND, ENV_PATH
    return None, (f"no API key set. Looked for .env at {ENV_PATH} (found: {'yes' if ENV_FOUND else 'NO - file missing or named .env.txt'}). "
                  "See http://localhost:8000/agent/status")


class _Ctx:
    """Shared state for one run: runs tools and records what was retrieved."""
    def __init__(self, plan):
        self.plan, self.kb, self.retrieved, self.trace = plan, get_kb(), set(), []
        self.summary = _plan_summary(plan)

    def call(self, name, args):
        self.trace.append({"tool": name, "input": args})
        if name == "get_plan_summary":
            return self.summary
        if name == "list_species":
            return VETTED_SPECIES
        if name == "search_knowledge":
            out = self.kb.search(str(args.get("query", "")))
            self.retrieved.update(r["id"] for r in out)
            return out
        return {"error": f"unknown tool {name}"}

    def finish(self, args, model):
        issues = validate(args, self.retrieved)
        if issues:
            return _fallback(self.plan, "guardrail: " + "; ".join(issues), self.trace)
        return {"text": _render(self.plan, args, self.kb), "mode": "llm", "species": args["species"],
                "citations": args["citations"], "tool_calls": self.trace, "model": model}


def _loop_anthropic(ctx, client, max_turns):
    messages = [{"role": "user", "content": USER_MSG}]
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=1200, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if getattr(b, "type", None) != "tool_use":
                continue
            args = b.input or {}
            if b.name == "submit_recommendation":
                ctx.trace.append({"tool": b.name, "input": args})
                return ctx.finish(args, MODEL)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(ctx.call(b.name, args))})
        if not results:
            return _fallback(ctx.plan, "model stopped without submitting", ctx.trace)
        messages.append({"role": "user", "content": results})
    return _fallback(ctx.plan, "turn limit reached", ctx.trace)


def _loop_openai(ctx, client, max_turns):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_MSG}]
    for _ in range(max_turns):
        extra = {"reasoning_effort": "low"} if "gpt-oss" in GROQ_MODEL else {}  # keep reasoning models fast
        resp = client.chat.completions.create(model=GROQ_MODEL, max_tokens=4000, messages=messages,
                                              tools=OAI_TOOLS, tool_choice="auto", extra_body=extra)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return _fallback(ctx.plan, "model stopped without submitting", ctx.trace)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            args = json.loads(c.function.arguments or "{}")
            if c.function.name == "submit_recommendation":
                ctx.trace.append({"tool": c.function.name, "input": args})
                return ctx.finish(args, GROQ_MODEL)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(ctx.call(c.function.name, args))})
    return _fallback(ctx.plan, "turn limit reached", ctx.trace)


USER_MSG = "Recommend species and rationale for the current plan. Look things up before answering."


@traceable(name="plan_agent", run_type="chain")
def run_agent(plan: Dict, client=None, kind: str = "anthropic", max_turns: int = MAX_TURNS) -> Dict:
    """Run the agent. Tests can inject `client` (+ kind='anthropic'|'openai')."""
    ctx = _Ctx(plan)
    if client is None:
        try:
            kind, client = _make_client()
        except Exception as e:  # noqa: BLE001
            return _fallback(plan, f"client init failed: {e}", [])
        if kind is None:
            return _fallback(plan, client, [])
    try:
        loop = _loop_openai if kind == "openai" else _loop_anthropic
        return loop(ctx, client, max_turns)
    except Exception as e:  # noqa: BLE001
        return _fallback(plan, f"LLM call failed: {e}", ctx.trace)
