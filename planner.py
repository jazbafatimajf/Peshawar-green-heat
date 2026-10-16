"""Planting-site optimiser under budget & water constraints + simple agent explanation."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import DEFAULT_BUDGET, VETTED_SPECIES
from .heat_link import predict_cooling


def score_blocks(
    green_frac: np.ndarray,
    heat: np.ndarray,
    water_access: Optional[np.ndarray] = None,
    block_size: int = 20,
) -> List[Dict]:
    """Score candidate planting blocks by heat priority and low existing green.

    Automatically shrinks block_size if the map is smaller than the requested block
    so we never return zero candidates just because the grid is small.
    """
    h, w = green_frac.shape
    if h < 1 or w < 1:
        return []

    # Fit block size to the map — need at least a few candidates
    bs = int(block_size)
    bs = max(3, min(bs, h, w))
    # If still only one cell possible, use half the smaller side (min 3)
    if h - bs + 1 <= 0 or w - bs + 1 <= 0:
        bs = max(3, min(h, w) // 2 or 1)

    step = max(1, bs // 2)  # overlap a bit so small maps still get several sites
    candidates = []
    if water_access is None:
        water_access = np.ones_like(green_frac) * 0.5

    for i in range(0, max(1, h - bs + 1), step):
        for j in range(0, max(1, w - bs + 1), step):
            i2 = min(i + bs, h)
            j2 = min(j + bs, w)
            g = float(green_frac[i:i2, j:j2].mean())
            t = float(heat[i:i2, j:j2].mean())
            wa = float(water_access[i:i2, j:j2].mean())
            score = (t - 30.0) * (1.0 - g) * (0.5 + 0.5 * wa)
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "block_size": bs,
                    "score": score,
                    "current_green": g,
                    "heat": t,
                    "water": wa,
                }
            )

    # Absolute fallback: whole map as one candidate
    if not candidates:
        candidates.append(
            {
                "i": 0,
                "j": 0,
                "block_size": min(h, w),
                "score": 1.0,
                "current_green": float(green_frac.mean()),
                "heat": float(heat.mean()),
                "water": 0.5,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def optimise_planting(
    candidates: List[Dict],
    budget: float = DEFAULT_BUDGET,
    cost_per_block: float = 50.0,
    target_green_increase: float = 0.25,
    cooling_model: Optional[Dict] = None,
) -> Dict:
    """Greedy select top-scoring blocks until budget exhausted."""
    if cost_per_block <= 0:
        cost_per_block = 50.0

    # If budget is large but we have few candidates, still use them all that fit
    selected = []
    remaining = float(budget)
    total_cooling = 0.0
    total_unc = 0.0

    for c in candidates:
        if remaining < cost_per_block:
            break
        new_green = min(1.0, c["current_green"] + target_green_increase)
        if cooling_model:
            pred = predict_cooling(c["current_green"], new_green, cooling_model)
            cool = float(pred["expected_cooling_c"])
            unc = float(pred["uncertainty_c"])
        else:
            cool = 1.5 * (new_green - c["current_green"])
            unc = 0.8

        selected.append(
            {
                **c,
                "new_green": new_green,
                "cost": cost_per_block,
                "expected_cooling_c": cool,
                "uncertainty_c": unc,
            }
        )
        remaining -= cost_per_block
        total_cooling += cool
        total_unc += unc ** 2

    return {
        "selected": selected,
        "n_sites": len(selected),
        "total_cost": float(budget - remaining),
        "remaining_budget": float(remaining),
        # Mean per planted block. Summing block cooling (old behaviour) gave nonsense like 22 °C
        # for a whole city; blocks overlap and cooling does not add across space.
        "aggregate_cooling_c": float(total_cooling / len(selected)) if selected else 0.0,
        "aggregate_uncertainty_c": float(np.sqrt(total_unc / len(selected))) if selected else 0.0,
        "cooling_basis": "average surface-temperature reduction per planted block",
        "caveat": "Estimates are coarse (~100 m LST) and correlational only",
    }


def agent_explain(
    plan: Dict,
    species_list: List[Dict] = VETTED_SPECIES,
    officer_name: str = "Planning Officer",
) -> str:
    n = plan["n_sites"]
    cool = plan["aggregate_cooling_c"]
    unc = plan["aggregate_uncertainty_c"]
    cost = plan["total_cost"]

    lines = [
        f"Recommendation for {officer_name}:",
        f"- Selected {n} planting blocks within the given budget (total cost ≈ {cost:.0f}).",
        f"- Estimated average cooling per planted block: {cool:.1f} °C (uncertainty ±{unc:.1f} °C).",
        f"- {plan.get('caveat', 'Estimates are coarse and correlational only')}.",
        "",
        "Species guidance (restricted to vetted local list):",
    ]
    for s in species_list[:5]:
        lines.append(
            f"  • **{s['name']}**: water need={s['water']}, cooling potential={s['cooling']}"
        )
    lines.append("")
    lines.append("Please review the map of chosen blocks and approve or request changes.")
    return "\n".join(lines)


def mock_approval(plan: Dict, approved: bool = True, comment: str = "") -> Dict:
    return {
        "approved": approved,
        "comment": comment or ("Approved as recommended" if approved else "Needs revision"),
        "n_sites": plan["n_sites"],
        "timestamp": "2026-10-14T21:00:00+05:00",
    }
