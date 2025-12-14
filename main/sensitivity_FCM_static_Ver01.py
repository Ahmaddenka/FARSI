
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd

# Optional SciPy (for Spearman). If missing, fall back to numpy-based rank corr.
try:
    from scipy.stats import spearmanr  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================
# Config dataclasses
# =========================

@dataclass
class OATConfig:
    delta: float = 0.10                  # ±delta around baseline value (on initial state)
    clip_min: float = -1.0
    clip_max: float = 1.0
    aggregate: str = "mean"              # "mean" | "median" | "l2" | "l1" | "sum"
    save_dir: Union[str, Path] = Path("sensitivity_outputs")
    save_csv_name: str = "oat_results.csv"
    seed: Optional[int] = 123


@dataclass
class RandomScreenConfig:
    n_scenarios: int = 64                # keep small due to runtime cost
    range_type: str = "abs"              # "abs" or "percent"
    low: float = -0.2                    # absolute lower (if "abs") OR -0.2 => -20% (if "percent")
    high: float = 0.2                    # absolute upper (if "abs") OR +0.2 => +20% (if "percent")
    clip_min: float = -1.0
    clip_max: float = 1.0
    aggregate: str = "mean"              # how to collapse multiple output nodes to one scalar per run
    save_dir: Union[str, Path] = Path("sensitivity_outputs")
    save_csv_name: str = "randomized_screening_results.csv"
    seed: Optional[int] = 123


# =========================
# Small helpers
# =========================

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aggregate_outputs(y, how: str) -> float:
    import numpy as _np
    y = _np.asarray(y, dtype=float)
    if how == "mean":
        return float(_np.mean(y))
    if how == "median":
        return float(_np.median(y))
    if how == "l2":
        return float(_np.sqrt(_np.sum(y**2)))
    if how == "l1":
        return float(_np.sum(_np.abs(y)))
    if how == "sum":
        return float(_np.sum(y))
    raise ValueError(f"Unknown aggregate: {how}")


def _spearman_corr(x, y) -> float:
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    if SCIPY_AVAILABLE:
        rho, _ = spearmanr(x, y)
        return float(rho)
    # Fallback: compute Spearman via ranking with numpy only
    x_rank = x.argsort().argsort().astype(float)
    y_rank = y.argsort().argsort().astype(float)
    xr = (x_rank - x_rank.mean()) / (x_rank.std(ddof=0) + 1e-12)
    yr = (y_rank - y_rank.mean()) / (y_rank.std(ddof=0) + 1e-12)
    return float(_np.mean(xr * yr))


def _modify_init_state(init_state: Dict[int, float],
                       updates: Dict[int, float],
                       clip_min: float,
                       clip_max: float) -> Dict[int, float]:
    new_state = dict(init_state)
    for idx, val in updates.items():
        new_state[idx] = float(np.clip(val, clip_min, clip_max))
    return new_state


# =========================
# Core analyses
# =========================

def local_oat_sensitivity(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    target_node_indices: Sequence[int],
    cfg: OATConfig = OATConfig(),
) -> pd.DataFrame:
    """
    For each target node i in `target_node_indices`, run two simulations:
      baseline (no change), plus (x_i + delta), minus (x_i - delta).
    Compute central-difference sensitivity on a scalar aggregation of outputs.

    Returns a DataFrame with per-node sensitivities, and writes CSV.
    """
    out_dir = _ensure_dir(cfg.save_dir)

    # Dry run to get baseline init_state and output nodes
    init_state0, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_fn(**init_kwargs)

    # Baseline equilibrium
    sim_out = simulate_fn(
        fcm_matrix=fcm_matrix,
        init_state=init_state0,
        random_root_nodes=random_nodes,
        bsp_root_nodes=bsp_nodes_idx,
        output_nodes=output_nodes,
        **sim_kwargs
    )
    if not (isinstance(sim_out, (list, tuple)) and len(sim_out) >= 3):
        raise RuntimeError("simulate_fn must return at least (state_trace, final_step, final_state).")
    _, _, y0 = sim_out[:3]
    y0_agg = _aggregate_outputs(y0, cfg.aggregate)

    rows = []

    for idx in target_node_indices:
        x_baseline = float(init_state0.get(int(idx), 0.0))

        # plus
        x_plus = x_baseline + cfg.delta
        init_plus = _modify_init_state(init_state0, {int(idx): x_plus}, cfg.clip_min, cfg.clip_max)
        sim_plus = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_plus,
            random_root_nodes=random_nodes,
            bsp_root_nodes=bsp_nodes_idx,
            output_nodes=output_nodes,
            **sim_kwargs
        )
        _, _, y_plus = sim_plus[:3]
        y_plus_agg = _aggregate_outputs(y_plus, cfg.aggregate)

        # minus
        x_minus = x_baseline - cfg.delta
        init_minus = _modify_init_state(init_state0, {int(idx): x_minus}, cfg.clip_min, cfg.clip_max)
        sim_minus = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_minus,
            random_root_nodes=random_nodes,
            bsp_root_nodes=bsp_nodes_idx,
            output_nodes=output_nodes,
            **sim_kwargs
        )
        _, _, y_minus = sim_minus[:3]
        y_minus_agg = _aggregate_outputs(y_minus, cfg.aggregate)

        # Central-difference sensitivity around baseline
        sens_cd = (y_plus_agg - y_minus_agg) / (2.0 * cfg.delta)
        dy_plus = y_plus_agg - y0_agg
        dy_minus = y_minus_agg - y0_agg
        dy_max_abs = max(abs(dy_plus), abs(dy_minus))

        node_names_list = list(graph.nodes())
        node_name = node_names_list[int(idx)] if 0 <= int(idx) < len(node_names_list) else f"node_{int(idx)}"

        rows.append({
            "node_index": int(idx),
            "node_name": node_name,
            "baseline_x": x_baseline,
            "delta": cfg.delta,
            "y0_agg": y0_agg,
            "y_plus_agg": y_plus_agg,
            "y_minus_agg": y_minus_agg,
            "sens_central_diff": sens_cd,
            "abs_change_max": dy_max_abs,
            "abs_change_plus": abs(dy_plus),
            "abs_change_minus": abs(dy_minus),
            "aggregate": cfg.aggregate,
        })

    df = pd.DataFrame(rows).sort_values(by="abs_change_max", ascending=False).reset_index(drop=True)
    csv_path = Path(out_dir) / cfg.save_csv_name
    df.to_csv(csv_path, index=False)

    meta = {"analysis": "local_oat", "config": asdict(cfg), "n_targets": len(target_node_indices)}
    with open(Path(out_dir) / "oat_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OAT] Done. Wrote {csv_path}")
    return df


def randomized_screening(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    factor_node_indices: Sequence[int],
    cfg: RandomScreenConfig = RandomScreenConfig(),
) -> pd.DataFrame:
    """
    Randomized global-lite screening:
      - Draw n_scenarios random initializations for selected factor nodes.
      - For each scenario, run simulation to equilibrium.
      - Aggregate outputs to a scalar (e.g., mean).
      - Rank factors by |Spearman rho| with the scalar outcome.
    Returns a DataFrame with correlations and summary stats, and writes CSV.
    """
    out_dir = _ensure_dir(cfg.save_dir)

    # Baseline init to get dimensions and baseline values
    init_state0, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_fn(**init_kwargs)

    node_names_list = list(graph.nodes())

    # Base values for factors
    base_vals = np.array([float(init_state0.get(int(i), 0.0)) for i in factor_node_indices], dtype=float)

    if cfg.range_type not in ("abs", "percent"):
        raise ValueError("range_type must be 'abs' or 'percent'")

    # Storage
    X = np.empty((cfg.n_scenarios, len(factor_node_indices)), dtype=float)
    Y = np.empty(cfg.n_scenarios, dtype=float)

    rng = np.random.default_rng(cfg.seed)

    # Scenarios loop (kept small by user request)
    for s in range(cfg.n_scenarios):
        if cfg.range_type == "abs":
            deltas = rng.uniform(cfg.low, cfg.high, size=len(factor_node_indices))
            vals = base_vals + deltas
        else:  # percent
            scales = rng.uniform(1.0 + cfg.low, 1.0 + cfg.high, size=len(factor_node_indices))
            vals = base_vals * scales

        # Clip to allowed range
        vals = np.clip(vals, cfg.clip_min, cfg.clip_max)

        # Apply to init state
        updates = {int(idx): float(vals[k]) for k, idx in enumerate(factor_node_indices)}
        init_s = _modify_init_state(init_state0, updates, cfg.clip_min, cfg.clip_max)

        # Run simulation
        sim_out = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_s,
            random_root_nodes=random_nodes,
            bsp_root_nodes=bsp_nodes_idx,
            output_nodes=output_nodes,
            **sim_kwargs
        )
        if not (isinstance(sim_out, (list, tuple)) and len(sim_out) >= 3):
            raise RuntimeError("simulate_fn must return at least (state_trace, final_step, final_state).")
        _, _, y_fin = sim_out[:3]
        y_agg = _aggregate_outputs(y_fin, cfg.aggregate)

        X[s, :] = vals
        Y[s] = y_agg

    # Rank importance by |Spearman| for each factor
    rows = []
    for j, idx in enumerate(factor_node_indices):
        rho = _spearman_corr(X[:, j], Y)
        node_name = node_names_list[int(idx)] if 0 <= int(idx) < len(node_names_list) else f"node_{int(idx)}"
        rows.append({
            "node_index": int(idx),
            "node_name": node_name,
            "spearman_rho": float(rho),
            "importance_abs": float(abs(rho)),
            "x_mean": float(np.mean(X[:, j])),
            "x_std": float(np.std(X[:, j], ddof=0)),
            "x_min": float(np.min(X[:, j])),
            "x_max": float(np.max(X[:, j])),
        })

    df = pd.DataFrame(rows).sort_values(by="importance_abs", ascending=False).reset_index(drop=True)
    csv_path = Path(out_dir) / cfg.save_csv_name
    df.to_csv(csv_path, index=False)

    meta = {
        "analysis": "randomized_screening",
        "config": asdict(cfg),
        "n_factors": len(factor_node_indices),
        "n_scenarios": int(cfg.n_scenarios),
    }
    with open(Path(out_dir) / "randomized_screening_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Screening] Done. Wrote {csv_path}")
    return df


def run_both_analyses(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    oat_targets: Sequence[int],
    screen_factors: Sequence[int],
    oat_cfg: Optional[OATConfig] = None,
    screen_cfg: Optional[RandomScreenConfig] = None,
) -> Dict[str, Any]:
    """
    Run Local OAT and Randomized Screening back-to-back with shared model functions.
    Returns paths to CSVs and DataFrames in a dict.
    """
    oat_cfg = oat_cfg or OATConfig()
    screen_cfg = screen_cfg or RandomScreenConfig()

    df_oat = local_oat_sensitivity(
        graph=graph,
        fcm_matrix=fcm_matrix,
        initialize_state_fn=initialize_state_fn,
        simulate_fn=simulate_fn,
        init_kwargs=init_kwargs,
        sim_kwargs=sim_kwargs,
        target_node_indices=oat_targets,
        cfg=oat_cfg,
    )

    df_screen = randomized_screening(
        graph=graph,
        fcm_matrix=fcm_matrix,
        initialize_state_fn=initialize_state_fn,
        simulate_fn=simulate_fn,
        init_kwargs=init_kwargs,
        sim_kwargs=sim_kwargs,
        factor_node_indices=screen_factors,
        cfg=screen_cfg,
    )

    out_dir = _ensure_dir(oat_cfg.save_dir)
    summary_path = Path(out_dir) / "sensitivity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "oat_csv": str(Path(oat_cfg.save_dir) / oat_cfg.save_csv_name),
            "screen_csv": str(Path(screen_cfg.save_dir) / screen_cfg.save_csv_name),
            "oat_config": asdict(oat_cfg),
            "screen_config": asdict(screen_cfg),
        }, f, ensure_ascii=False, indent=2)

    print(f"[All] Summary: {summary_path}")
    return {
        "oat_df": df_oat,
        "screen_df": df_screen,
        "oat_csv": str(Path(oat_cfg.save_dir) / oat_cfg.save_csv_name),
        "screen_csv": str(Path(screen_cfg.save_dir) / screen_cfg.save_csv_name),
        "summary_json": str(summary_path),
    }
