"""Human approval gate: decisions are appended to artifacts/approvals.jsonl with a real timestamp."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from .config import ARTIFACTS_DIR

PKT = timezone(timedelta(hours=5))
LOG = ARTIFACTS_DIR / "approvals.jsonl"


def record_approval(plan: Dict, approved: bool = True, comment: str = "") -> Dict:
    summary = {"area": plan.get("area"), "n_sites": plan["n_sites"], "total_cost": plan["total_cost"],
               "aggregate_cooling_c": plan["aggregate_cooling_c"]}
    entry = {
        "approved": approved,
        "comment": comment or ("Approved as recommended" if approved else "Needs revision"),
        "n_sites": plan["n_sites"],
        "plan_hash": hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "agent_mode": (plan.get("agent") or {}).get("mode"),
        "timestamp": datetime.now(PKT).isoformat(timespec="seconds"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def list_approvals() -> List[Dict]:
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
