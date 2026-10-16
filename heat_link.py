"""Link green cover to land-surface temperature (coarse correlation)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def aggregate_to_blocks(
    green_frac: np.ndarray,
    lst: np.ndarray,
    block_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average green fraction and LST over non-overlapping blocks.
    green_frac and lst should be co-registered (or resampled).
    Returns (green_means, lst_means) 1-D arrays.
    """
    h, w = green_frac.shape
    g_list, t_list = [], []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            g = green_frac[i : i + block_size, j : j + block_size].mean()
            t = lst[i : i + block_size, j : j + block_size].mean()
            if np.isfinite(g) and np.isfinite(t):
                g_list.append(g)
                t_list.append(t)
    return np.array(g_list), np.array(t_list)


def fit_cooling_model(
    green_frac: np.ndarray,
    lst: np.ndarray,
    block_size: int = 10,
) -> Dict:
    """Simple linear model: LST ~ a + b * green_frac.
    Returns coefficients, R², and predicted delta-T per 0.1 increase in green.
    IMPORTANT: this is correlation only, not causal proof of planting cooling.
    """
    g, t = aggregate_to_blocks(green_frac, lst, block_size)
    if len(g) < 5:
        return {
            "slope": 0.0,
            "intercept": float(np.nanmean(lst)),
            "r2": 0.0,
            "delta_t_per_0_1_green": 0.0,
            "n_blocks": len(g),
            "note": "Insufficient blocks for regression",
        }

    X = g.reshape(-1, 1)
    reg = LinearRegression().fit(X, t)
    pred = reg.predict(X)
    r2 = r2_score(t, pred)
    slope = float(reg.coef_[0])
    # Negative slope expected if more green → cooler
    delta = slope * 0.1  # expected change for +10% green cover

    return {
        "slope": slope,
        "intercept": float(reg.intercept_),
        "r2": float(r2),
        "delta_t_per_0_1_green": float(delta),
        "n_blocks": int(len(g)),
        "green_mean": float(g.mean()),
        "lst_mean": float(t.mean()),
        "note": "Correlation only; hold-out neighbourhoods recommended for validation",
    }


def predict_cooling(
    current_green: float,
    planted_green: float,
    model: Dict,
) -> Dict:
    """Estimate cooling (and rough uncertainty) from a change in green fraction."""
    delta_green = planted_green - current_green
    expected_dt = model["slope"] * delta_green
    # Heuristic uncertainty: wider when R² low or |slope| small
    unc = abs(expected_dt) * (1.0 - model["r2"]) + 0.5  # °C
    return {
        "delta_green": float(delta_green),
        "expected_cooling_c": float(-expected_dt) if expected_dt < 0 else float(expected_dt * -1),  # positive = cooling
        "uncertainty_c": float(unc),
        "r2": model["r2"],
        "caveat": "Coarse Landsat ~100 m; correlation not causation",
    }


def holdout_neighbourhood_eval(
    green_frac: np.ndarray,
    lst: np.ndarray,
    n_folds: int = 4,
    block_size: int = 20,
) -> Dict:
    """Spatial hold-out: split into coarse tiles, train on some, test on others.
    Reports mean R² on held-out regions.
    """
    h, w = green_frac.shape
    tile_h = h // n_folds
    r2s = []
    for fold in range(n_folds):
        # Hold out a horizontal strip
        start = fold * tile_h
        end = (fold + 1) * tile_h if fold < n_folds - 1 else h
        train_g = np.concatenate([green_frac[:start], green_frac[end:]], axis=0)
        train_t = np.concatenate([lst[:start], lst[end:]], axis=0)
        test_g = green_frac[start:end]
        test_t = lst[start:end]

        model = fit_cooling_model(train_g, train_t, block_size=max(5, block_size // 2))
        g_test, t_test = aggregate_to_blocks(test_g, test_t, block_size=max(5, block_size // 2))
        if len(g_test) < 3:
            continue
        pred = model["intercept"] + model["slope"] * g_test
        r2s.append(r2_score(t_test, pred))

    return {
        "mean_holdout_r2": float(np.mean(r2s)) if r2s else 0.0,
        "std_holdout_r2": float(np.std(r2s)) if r2s else 0.0,
        "n_folds_used": len(r2s),
        "note": "Spatial hold-out (neighbourhood strips). Low R² → weak generalisation.",
    }


# Synthetic LST for demo when real Landsat window not yet downloaded
def synthetic_lst_from_green(green_frac: np.ndarray, base_temp: float = 38.0, slope: float = -8.0) -> np.ndarray:
    """Generate a plausible LST field correlated with green cover + noise."""
    noise = np.random.normal(0, 1.5, green_frac.shape)
    return (base_temp + slope * green_frac + noise).astype(np.float32)
