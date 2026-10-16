"""Simple multi-year green-cover change map (NDVI difference)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def green_change(
    ndvi_t0: np.ndarray,
    ndvi_t1: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Compute lost / gained vegetation between two dates.
    Returns binary masks and difference map.
    """
    # Align shapes if needed
    if ndvi_t0.shape != ndvi_t1.shape:
        # Simple center crop to min size
        h = min(ndvi_t0.shape[0], ndvi_t1.shape[0])
        w = min(ndvi_t0.shape[1], ndvi_t1.shape[1])
        ndvi_t0 = ndvi_t0[:h, :w]
        ndvi_t1 = ndvi_t1[:h, :w]

    veg0 = (ndvi_t0 >= threshold).astype(np.uint8)
    veg1 = (ndvi_t1 >= threshold).astype(np.uint8)
    lost = ((veg0 == 1) & (veg1 == 0)).astype(np.uint8)
    gained = ((veg0 == 0) & (veg1 == 1)).astype(np.uint8)
    diff = ndvi_t1 - ndvi_t0

    return {
        "lost": lost,
        "gained": gained,
        "diff": diff.astype(np.float32),
        "veg_frac_t0": float(veg0.mean()),
        "veg_frac_t1": float(veg1.mean()),
        "lost_frac": float(lost.mean()),
        "gained_frac": float(gained.mean()),
    }
