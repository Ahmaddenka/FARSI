
"""
fuzzy_to_prob.py
----------------

Utilities to convert expert-elicited Likert scores (e.g., 1–5) into
fuzzy (trapezoidal) possibility distributions and then into probability
distributions using an α-average (Dubois–Prade style) possibility→probability
transform. Finally, the resulting probability distribution can be mapped
to any target interval (default: [-1, 1]) for simulation models that assume
bounded node outputs.

Author: AHMAD
Date: 2025-10-13

Requirements
------------
- numpy
- pandas

Overview
--------
We have an Excel file where:
- Column 0 (first column) contains variable names (one per row).
- Columns 1..K contain expert scores for that variable (Likert scale, e.g., 1..5).
There are at least two experts per variable (K >= 2). Missing cells are ignored.

Pipeline (per variable)
-----------------------
1) Aggregate expert scores into a *trapezoidal fuzzy number* ˜X = Trap(a, b, c, d).
   Intuition:
     - [a, d] is the *support* (where possibility > 0).
     - [b, c] is the *core* or *plateau* (where possibility = 1).
   You can control how (a, b, c, d) are chosen from the expert samples via flags.

2) Interpret the membership function μ_˜X(x) as a *possibility distribution* π(x).

3) Convert possibility π(x) to a probability density p(x) by the *α-average* rule:
   For trapezoid Trap(a, b, c, d), the α-cut is
       A_α = [ a + α (b - a),  d - α (d - c) ],
   whose length is |A_α| = (d - a) - α[(d - c) + (b - a)] = L0 - α L1,
   where L0 = d - a,  L1 = (d - c) + (b - a).

   The α-average transform (continuous form) yields, for a given x with π(x) = μ(x):
       p(x)  ∝  ∫_{0}^{μ(x)} [ 1 / |A_α| ] dα
            =  (1 / L1) * ln( L0 / (L0 - μ(x) * L1) ),   (for L1 > 0)

   We evaluate this on a grid x ∈ [scale_min, scale_max], then normalize so that
   ∑ p(x) Δx = 1 (numerical normalization). This produces a proper discrete PDF.

4) Optional linear mapping of the probability distribution to a *target range*:
   Let x' = t_min + (t_max - t_min) * (x - s_min) / (s_max - s_min).
   Then the density transforms by change-of-variables:
       p'(x') = p(x) * (dx/dx') = p(x) * (s_max - s_min) / (t_max - t_min).
   In code we recompute and renormalize on the transformed grid for numerical stability.

5) Produce an output dictionary ready for simulation:
   {
     "Variable Name": {
         "trap": (a, b, c, d),                         # on the source scale
         "grid": np.ndarray (x'-grid on target range), # target-range grid
         "pdf":  np.ndarray (discrete pdf on target grid, sums to 1),
         "cdf":  np.ndarray (discrete cdf on target grid),
         "sampler": callable (rng, size) -> samples on target range,
         "stats": {"mean": ..., "std": ..., "mode_interval_source": [b, c],
                   "mean_target": ..., "std_target": ...}
     },
     ...
   }

Usage
-----
From Python:

    from fuzzy_to_prob import build_probability_inputs

    results = build_probability_inputs(
        excel_path="experts.xlsx",
        sheet_name=0,                 # or None for first sheet
        id_col=0,                     # variable names in first column
        scale_min=1.0, scale_max=5.0, # Likert bounds
        trapezoid_fit="percentile",   # 'percentile' or 'std'
        support_p=(0.05, 0.95),
        core_p=(0.35, 0.65),
        std_core_width=0.5,
        std_support_width=2.0,
        n_grid=1001,
        target_range=(-1.0, 1.0)      # final mapping for simulation
    )

    var = "Management Commitment"
    x = results[var]["grid"]     # in target range
    pdf = results[var]["pdf"]
    samples = results[var]["sampler"](size=10000)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable, Any


def trapezoid_membership(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Compute membership μ_Trap(a,b,c,d)(x) for a trapezoidal fuzzy number.

    Definition (piecewise linear):
        μ(x) = 0                          for x <= a or x >= d
        μ(x) = (x - a) / (b - a)          for a < x < b
        μ(x) = 1                          for b <= x <= c
        μ(x) = (d - x) / (d - c)          for c < x < d

    Parameters
    ----------
    x : np.ndarray
        Points where to evaluate membership.
    a, b, c, d : float
        Trapezoid parameters with a < b ≤ c < d.

    Returns
    -------
    np.ndarray
        Membership values in [0, 1].
    """
    mu = np.zeros_like(x, dtype=float)

    # left slope
    mask = (x > a) & (x < b) & (b > a)
    mu[mask] = (x[mask] - a) / (b - a)

    # plateau
    mask = (x >= b) & (x <= c)
    mu[mask] = 1.0

    # right slope
    mask = (x > c) & (x < d) & (d > c)
    mu[mask] = (d - x[mask]) / (d - c)

    return mu


def alpha_average_pdf_from_trap(x: np.ndarray, a: float, b: float, c: float, d: float, eps: float = 1e-12) -> np.ndarray:
    """
    Compute a discrete PDF p(x) proportional to the α-average transform of
    a trapezoidal possibility distribution μ(x) = μ_Trap(a,b,c,d)(x).

    Mathematics
    -----------
    Let L0 = d - a, L1 = (d - c) + (b - a). The α-cut length is |A_α| = L0 - α L1.
    For a given x, let μ = μ(x) ∈ [0,1]. The α-average transform is:

        p̃(x) = ∫_0^{μ} [ 1 / (L0 - α L1) ] dα
              = (1 / L1) * ln( L0 / (L0 - μ L1) ),  provided L1 > 0.

    We evaluate p̃(x) on a fine grid, then normalize to obtain a proper discrete PDF.

    Parameters
    ----------
    x : np.ndarray
        Monotone grid (ascending) covering [a, d] or the full Likert range.
    a, b, c, d : float
        Trapezoid parameters with a < b ≤ c < d.
    eps : float
        Small constant to guard against division / log issues.

    Returns
    -------
    np.ndarray
        Discrete PDF on x (nonnegative, sums to 1 numerically when integrated by trapezoid rule).
    """
    mu = trapezoid_membership(x, a, b, c, d)

    L0 = max(d - a, eps)
    L1 = max((d - c) + (b - a), eps)

    denom = np.maximum(L0 - mu * L1, eps)
    unnorm = (1.0 / L1) * np.log(L0 / denom)

    unnorm = np.maximum(unnorm, 0.0)

    area = np.trapz(unnorm, x)
    if area <= eps:
        pdf = np.ones_like(x) / (x[-1] - x[0] + eps)
        pdf = pdf / np.trapz(pdf, x)
        return pdf

    pdf = unnorm / area
    return pdf


def inverse_cdf_sampler(x: np.ndarray, pdf: np.ndarray) -> Callable[..., np.ndarray]:
    """
    Build an inverse-CDF sampler for the discrete (x, pdf) grid (x ascending).

    Returns a function sampler(size=..., rng=...) that draws samples by
    inverse transform sampling on the tabulated CDF.
    """
    cdf = np.zeros_like(pdf)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * (x[1:] - x[:-1]))
    if cdf[-1] <= 0:
        cdf = np.linspace(0.0, 1.0, len(x))
    else:
        cdf = cdf / cdf[-1]

    def sampler(size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random(size=size)
        return np.interp(u, cdf, x)

    return sampler


def fit_trapezoid_from_experts(
    vals: np.ndarray,
    scale_min: float,
    scale_max: float,
    method: str = "percentile",
    support_p: Tuple[float, float] = (0.05, 0.95),
    core_p: Tuple[float, float] = (0.35, 0.65),
    std_core_width: float = 0.5,
    std_support_width: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    Fit a trapezoidal fuzzy number Trap(a, b, c, d) from expert scores.

    Parameters
    ----------
    vals : np.ndarray
        Expert scores for one variable (1-D, with NaNs allowed). At least 2 non-NaNs recommended.
    scale_min, scale_max : float
        Bounds of the Likert scale (e.g., 1 and 5).
    method : {'percentile', 'std'}
        - 'percentile': robust nonparametric fit using quantiles.
        - 'std': parametric fit using mean ± k * std.
    support_p : (float, float)
        Lower and upper quantiles for support [a, d] when method='percentile'.
    core_p : (float, float)
        Lower and upper quantiles for core [b, c] when method='percentile'.
    std_core_width : float
        k such that b = m - k*s, c = m + k*s for method='std'.
    std_support_width : float
        K such that a = m - K*s, d = m + K*s for method='std'.

    Returns
    -------
    (a, b, c, d) : tuple of floats
        Trapezoid parameters with scale_min <= a < b ≤ c < d <= scale_max.
        If the data are degenerate (zero variance), we widen slightly to avoid collapse.
    """
    v = vals[~np.isnan(vals)].astype(float)
    if v.size == 0:
        a, b, c, d = float(scale_min), float(scale_min + 0.25*(scale_max-scale_min)), \
                     float(scale_min + 0.75*(scale_max-scale_min)), float(scale_max)
        return a, b, c, d

    v = np.clip(v, scale_min, scale_max)

    if method == "percentile":
        q = np.quantile(v, [support_p[0], core_p[0], core_p[1], support_p[1]])
        a, b, c, d = map(float, q)

    elif method == "std":
        m = float(np.mean(v))
        s = float(np.std(v, ddof=1)) if v.size > 1 else 0.0

        if s == 0.0:
            eps = 0.1 * (scale_max - scale_min)
            a, b, c, d = m - 2*eps, m - eps, m + eps, m + 2*eps
        else:
            b = m - std_core_width * s
            c = m + std_core_width * s
            a = m - std_support_width * s
            d = m + std_support_width * s

        a = max(scale_min, min(a, m))
        b = max(a + 1e-6, min(b, c))
        c = max(b, min(c, d - 1e-6))
        d = min(scale_max, max(c, d))

    else:
        raise ValueError("Unknown method: choose 'percentile' or 'std'.")

    eps = 1e-6
    a = float(max(scale_min, min(a, b - eps)))
    d = float(min(scale_max, max(d, c + eps)))
    if b < a + eps: b = a + eps
    if c < b: c = b
    if d <= c + eps: d = c + eps

    return a, b, c, d


def _linear_map(x: np.ndarray, s_min: float, s_max: float, t_min: float, t_max: float) -> np.ndarray:
    """Linear monotone mapping from [s_min, s_max] to [t_min, t_max]."""
    return t_min + (t_max - t_min) * (x - s_min) / (s_max - s_min)


def build_probability_inputs(
    excel_path: str,
    sheet_name: Any = 0,
    id_col: int = 0,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
    trapezoid_fit: str = "percentile",
    support_p: Tuple[float, float] = (0.05, 0.95),
    core_p: Tuple[float, float] = (0.35, 0.65),
    std_core_width: float = 0.5,
    std_support_width: float = 2.0,
    n_grid: int = 1001,
    target_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, Dict[str, Any]]:
    """
    Load an Excel file of expert scores and build per-variable probability inputs,
    mapped to a desired target range (default: [-1, 1]).

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    sheet_name : Any
        Sheet name or index (pandas-compatible). Default: 0 (first sheet).
    id_col : int
        Column index of variable names (0-based). Default: 0.
    scale_min, scale_max : float
        Likert bounds (inclusive). Default: [1, 5].
    trapezoid_fit : {'percentile','std'}
        How to fit (a,b,c,d) from expert scores.
    support_p, core_p : (float, float)
        Percentile settings for 'percentile' fitting.
    std_core_width, std_support_width : float
        Width multipliers for 'std' fitting.
    n_grid : int
        Number of grid points for tabulation (odd number recommended).
    target_range : (float, float)
        Final range for the output distributions (e.g., [-1, 1]).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping variable_name -> info dict with keys:
          - "trap": (a,b,c,d) on source scale
          - "grid": x' grid on target range
          - "pdf":  pdf on target grid (sums to 1 numerically)
          - "cdf":  cdf on target grid
          - "sampler": callable(size, rng) -> samples on target
          - "stats": {"mean": float, "std": float, "mode_interval_source": [b, c],
                      "mean_target": float, "std_target": float}
          - "meta":  fit/setup metadata
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)

    if id_col < 0 or id_col >= df.shape[1]:
        raise ValueError(f"id_col={id_col} is out of bounds for dataframe with {df.shape[1]} columns.")

    names = df.iloc[:, id_col].astype(str).tolist()

    expert_cols = [i for i in range(df.shape[1]) if i != id_col]
    expert_data = df.iloc[:, expert_cols].to_numpy(dtype=float)

    # Source grid on original scale for numerical stability
    x_src = np.linspace(scale_min, scale_max, int(n_grid))

    t_min, t_max = float(target_range[0]), float(target_range[1])
    s_min, s_max = float(scale_min), float(scale_max)

    # Precompute the mapped grid (monotone linear mapping)
    x_tgt = _linear_map(x_src, s_min, s_max, t_min, t_max)

    result: Dict[str, Dict[str, Any]] = {}

    for idx, name in enumerate(names):
        vals = expert_data[idx, :]

        a, b, c, d = fit_trapezoid_from_experts(
            vals=vals,
            scale_min=scale_min,
            scale_max=scale_max,
            method=trapezoid_fit,
            support_p=support_p,
            core_p=core_p,
            std_core_width=std_core_width,
            std_support_width=std_support_width,
        )

        # PDF on source grid
        pdf_src = alpha_average_pdf_from_trap(x_src, a, b, c, d)

        # Change of variables to target grid
        # Continuous rule: p'(x') = p(x) * (dx/dx') with x' = linear(x)
        # Since grid mapping is one-to-one, we can compute scale factor once:
        scale_factor = (s_max - s_min) / (t_max - t_min)
        pdf_tgt_unnorm = pdf_src * scale_factor

        # Normalize on the target grid to guard numerical drift
        area_tgt = np.trapz(pdf_tgt_unnorm, x_tgt)
        if area_tgt > 0:
            pdf_tgt = pdf_tgt_unnorm / area_tgt
        else:
            pdf_tgt = np.ones_like(x_tgt) / (x_tgt[-1] - x_tgt[0])
            pdf_tgt = pdf_tgt / np.trapz(pdf_tgt, x_tgt)

        # CDF on target grid
        cdf_tgt = np.zeros_like(pdf_tgt)
        cdf_tgt[1:] = np.cumsum(0.5 * (pdf_tgt[1:] + pdf_tgt[:-1]) * (x_tgt[1:] - x_tgt[:-1]))
        if cdf_tgt[-1] > 0:
            cdf_tgt = cdf_tgt / cdf_tgt[-1]
        else:
            cdf_tgt = np.linspace(0.0, 1.0, len(x_tgt))

        sampler = inverse_cdf_sampler(x_tgt, pdf_tgt)

        # Stats on source and target
        mean_src = np.trapz(x_src * pdf_src, x_src)
        var_src = np.trapz((x_src - mean_src) ** 2 * pdf_src, x_src)
        std_src = float(np.sqrt(max(var_src, 0.0)))

        mean_tgt = np.trapz(x_tgt * pdf_tgt, x_tgt)
        var_tgt = np.trapz((x_tgt - mean_tgt) ** 2 * pdf_tgt, x_tgt)
        std_tgt = float(np.sqrt(max(var_tgt, 0.0)))

        result[name] = {
            "trap": (float(a), float(b), float(c), float(d)),  # on source scale
            "grid": x_tgt.copy(),
            "pdf": pdf_tgt.copy(),
            "cdf": cdf_tgt.copy(),
            "sampler": sampler,
            "stats": {
                "mean": float(mean_src),
                "std": float(std_src),
                "mode_interval_source": [float(b), float(c)],
                "mean_target": float(mean_tgt),
                "std_target": float(std_tgt),
            },
            "meta": {
                "fit_method": trapezoid_fit,
                "scale_bounds": (float(scale_min), float(scale_max)),
                "target_range": (t_min, t_max),
                "n_experts_used": int(np.sum(~np.isnan(vals))),
                "n_grid": int(n_grid),
            },
        }

    return result


# Optional CLI
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Build probability inputs from expert Likert data (Excel), mapped to a target range.")
    parser.add_argument("excel_path", type=str, help="Path to the Excel file.")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first).")
    parser.add_argument("--id-col", type=int, default=0, help="Index of the variable-name column (default: 0).")
    parser.add_argument("--scale-min", type=float, default=1.0, help="Likert min (default: 1).")
    parser.add_argument("--scale-max", type=float, default=5.0, help="Likert max (default: 5).")
    parser.add_argument("--fit", type=str, choices=["percentile","std"], default="percentile", help="Trapezoid fit method.")
    parser.add_argument("--support-p", type=float, nargs=2, default=(0.05, 0.95), help="Support percentiles (low high).")
    parser.add_argument("--core-p", type=float, nargs=2, default=(0.35, 0.65), help="Core percentiles (low high).")
    parser.add_argument("--std-core-width", type=float, default=0.5, help="k for core width in 'std' fit.")
    parser.add_argument("--std-support-width", type=float, default=2.0, help="K for support width in 'std' fit.")
    parser.add_argument("--n-grid", type=int, default=1001, help="Number of grid points.")
    parser.add_argument("--target-min", type=float, default=-1.0, help="Target range min (default: -1).")
    parser.add_argument("--target-max", type=float, default=1.0, help="Target range max (default: 1).")
    parser.add_argument("--dump-json", type=str, default="", help="Optional path to dump a compact JSON (sans sampler).")
    args = parser.parse_args()

    res = build_probability_inputs(
        excel_path=args.excel_path,
        sheet_name=args.sheet,
        id_col=args.id_col,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        trapezoid_fit=args.fit,
        support_p=tuple(args.support_p),
        core_p=tuple(args.core_p),
        std_core_width=args.std_core_width,
        std_support_width=args.std_support_width,
        n_grid=args.n_grid,
        target_range=(args.target_min, args.target_max),
    )

    if args.dump_json:
        out = {}
        for k, v in res.items():
            out[k] = {
                "trap": v["trap"],
                "stats": v["stats"],
                "meta": v["meta"],
            }
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote summary JSON to: {args.dump_json}")
    else:
        print("Built probability inputs for", len(res), "variables.",
              f"Target range = ({args.target_min}, {args.target_max})")



