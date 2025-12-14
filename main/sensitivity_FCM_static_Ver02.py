# sensitivity_FCM_static_Ver02.py
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
except Exception:  # SciPy not essential
    SCIPY_AVAILABLE = False


# =========================
# Config dataclasses
# =========================

@dataclass
class OATConfig:
    """
    Configuration for local One-At-a-Time (OAT) sensitivity analysis.

    Ver02 changes:
      - n_mc: number of Monte Carlo repetitions per scenario.
      - seed: base seed used to generate per-repetition seeds.
      تحلیل حساسیت روی امیدریاضی خروجی انجام می‌شود.
    """
    delta: float = 0.10                  # ±delta around baseline value (on initial state)
    clip_min: float = -1.0
    clip_max: float = 1.0
    aggregate: str = "mean"              # "mean" | "median" | "l2" | "l1" | "sum"
    save_dir: Union[str, Path] = Path("sensitivity_outputs")
    save_csv_name: str = "oat_results.csv"
    seed: Optional[int] = 123
    n_mc: int = 16                       # number of Monte Carlo repetitions per node


@dataclass
class RandomScreenConfig:
    """
    Configuration for randomized global-lite screening.
    """
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
# Helpers
# =========================

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aggregate_outputs(y, how: str) -> float:
    arr = np.asarray(y, dtype=float)
    if how == "mean":
        return float(arr.mean())
    if how == "median":
        return float(np.median(arr))
    if how == "l2":
        return float(np.sqrt(np.sum(arr ** 2)))
    if how == "l1":
        return float(np.sum(np.abs(arr)))
    if how == "sum":
        return float(np.sum(arr))
    raise ValueError(f"Unknown aggregate: {how}")


def _modify_init_state(
    init_state: Dict[int, float],
    updates: Dict[int, float],
    clip_min: float,
    clip_max: float,
) -> Dict[int, float]:
    new_state = dict(init_state)
    for idx, val in updates.items():
        new_state[idx] = float(np.clip(val, clip_min, clip_max))
    return new_state


def _spearman_corr(x, y) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return float("nan")
    if SCIPY_AVAILABLE:
        rho, _ = spearmanr(x_arr, y_arr)
        return float(rho)
    # Fallback: compute Spearman via ranking with numpy only
    x_rank = x_arr.argsort().argsort().astype(float)
    y_rank = y_arr.argsort().argsort().astype(float)
    xr = (x_rank - x_rank.mean()) / (x_rank.std(ddof=0) + 1e-12)
    yr = (y_rank - y_rank.mean()) / (y_rank.std(ddof=0) + 1e-12)
    return float(np.mean(xr * yr))


# =========================
# Local OAT with Monte Carlo
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
    Local OAT sensitivity on the *expected* outcome.

    For each target node i in `target_node_indices`:
      - Perform `cfg.n_mc` Monte Carlo repetitions.
      - In each repetition r:
          * Draw an initialization with seed_r.
          * Run three simulations on the same realization:
                baseline, plus (x_i + delta), minus (x_i - delta).
      - Aggregate outputs (e.g. mean over output nodes) for each run.
      - Average across repetitions and compute central-difference sensitivity
        on these Monte Carlo means.

    این طراحی از Common Random Numbers استفاده می‌کند:
      در هر تکرار، baseline / plus / minus روی یک realization مشترک
      از نودهای تصادفی و نمونه‌گیری BSM اجرا می‌شوند؛ پس نویز کنترل می‌شود.
    """
    out_dir = _ensure_dir(cfg.save_dir)
    n_mc = max(1, int(cfg.n_mc))

    # RNG for Monte Carlo seeds (deterministic across runs if cfg.seed fixed)
    base_rng = np.random.default_rng(cfg.seed)
    mc_seeds = base_rng.integers(low=0, high=2**31 - 1, size=n_mc, dtype=np.int64)

    node_names_list = list(graph.nodes())

    # --- 1) Build baseline ensemble (for y0) and store per-repetition init states ---
    mc_init_states: List[Dict[int, float]] = []
    mc_root_nodes: List[List[int]] = []
    mc_output_nodes: List[List[int]] = []
    mc_random_nodes: List[List[int]] = []
    mc_bsp_nodes: List[List[int]] = []
    mc_y0_list: List[float] = []

    for r in range(n_mc):
        kw = dict(init_kwargs)
        # Respect existing key if present, but override None
        if "seed" in kw and kw["seed"] is not None:
            # user-provided seed wins; still deterministic if they fix it
            pass
        else:
            kw["seed"] = int(mc_seeds[r])

        out = initialize_state_fn(**kw)
        if not (isinstance(out, (list, tuple)) and len(out) == 5):
            raise RuntimeError(
                "initialize_state_fn must return a 5-tuple: "
                "(init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx)"
            )
        init_state_r, root_nodes_r, output_nodes_r, random_nodes_r, bsp_nodes_idx_r = out

        sim_out = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_state_r,
            random_root_nodes=random_nodes_r,
            bsp_root_nodes=bsp_nodes_idx_r,
            output_nodes=output_nodes_r,
            **sim_kwargs,
        )
        if not (isinstance(sim_out, (list, tuple)) and len(sim_out) >= 3):
            raise RuntimeError(
                "simulate_fn must return at least (state_trace, final_step, final_state, ...)."
            )
        _, _, y0 = sim_out[:3]
        y0_agg = _aggregate_outputs(y0, cfg.aggregate)

        mc_init_states.append(init_state_r)
        mc_root_nodes.append(root_nodes_r)
        mc_output_nodes.append(output_nodes_r)
        mc_random_nodes.append(random_nodes_r)
        mc_bsp_nodes.append(bsp_nodes_idx_r)
        mc_y0_list.append(y0_agg)

    y0_mean = float(np.mean(mc_y0_list))
    y0_std = float(np.std(mc_y0_list, ddof=1)) if n_mc > 1 else 0.0

    rows: List[Dict[str, Any]] = []

    # --- 2) OAT for each target node ---
    for idx in target_node_indices:
        idx_int = int(idx)

        x_baselines: List[float] = []
        y_plus_list: List[float] = []
        y_minus_list: List[float] = []

        for r in range(n_mc):
            init_state_r = mc_init_states[r]
            root_nodes_r = mc_root_nodes[r]
            output_nodes_r = mc_output_nodes[r]
            random_nodes_r = mc_random_nodes[r]
            bsp_nodes_idx_r = mc_bsp_nodes[r]

            x_baseline_r = float(init_state_r.get(idx_int, 0.0))
            x_baselines.append(x_baseline_r)

            # plus
            x_plus = x_baseline_r + cfg.delta
            init_plus = _modify_init_state(init_state_r, {idx_int: x_plus}, cfg.clip_min, cfg.clip_max)
            sim_plus = simulate_fn(
                fcm_matrix=fcm_matrix,
                init_state=init_plus,
                random_root_nodes=random_nodes_r,
                bsp_root_nodes=bsp_nodes_idx_r,
                output_nodes=output_nodes_r,
                **sim_kwargs,
            )
            _, _, y_plus = sim_plus[:3]
            y_plus_agg = _aggregate_outputs(y_plus, cfg.aggregate)
            y_plus_list.append(y_plus_agg)

            # minus
            x_minus = x_baseline_r - cfg.delta
            init_minus = _modify_init_state(init_state_r, {idx_int: x_minus}, cfg.clip_min, cfg.clip_max)
            sim_minus = simulate_fn(
                fcm_matrix=fcm_matrix,
                init_state=init_minus,
                random_root_nodes=random_nodes_r,
                bsp_root_nodes=bsp_nodes_idx_r,
                output_nodes=output_nodes_r,
                **sim_kwargs,
            )
            _, _, y_minus = sim_minus[:3]
            y_minus_agg = _aggregate_outputs(y_minus, cfg.aggregate)
            y_minus_list.append(y_minus_agg)

        # MC statistics for this node
        x_baseline_mean = float(np.mean(x_baselines))
        x_baseline_std = float(np.std(x_baselines, ddof=1)) if n_mc > 1 else 0.0

        y_plus_mean = float(np.mean(y_plus_list))
        y_plus_std = float(np.std(y_plus_list, ddof=1)) if n_mc > 1 else 0.0

        y_minus_mean = float(np.mean(y_minus_list))
        y_minus_std = float(np.std(y_minus_list, ddof=1)) if n_mc > 1 else 0.0

        # central difference sensitivity on MC means
        sens_cd = (y_plus_mean - y_minus_mean) / (2.0 * cfg.delta)

        dy_plus_mean = y_plus_mean - y0_mean
        dy_minus_mean = y_minus_mean - y0_mean
        dy_max_abs = max(abs(dy_plus_mean), abs(dy_minus_mean))

        node_names_list = list(graph.nodes())
        node_name = (
            node_names_list[idx_int]
            if 0 <= idx_int < len(node_names_list)
            else f"node_{idx_int}"
        )

        rows.append(
            {
                "node_index": idx_int,
                "node_name": node_name,
                # baseline input stats
                "baseline_x_mean": x_baseline_mean,
                "baseline_x_std": x_baseline_std,
                "baseline_x": x_baseline_mean,  # backward compatibility
                "delta": cfg.delta,
                # baseline output stats
                "y0_mean": y0_mean,
                "y0_std": y0_std,
                "y0_agg": y0_mean,  # backward compatibility
                # plus / minus stats
                "y_plus_mean": y_plus_mean,
                "y_plus_std": y_plus_std,
                "y_plus_agg": y_plus_mean,
                "y_minus_mean": y_minus_mean,
                "y_minus_std": y_minus_std,
                "y_minus_agg": y_minus_mean,
                # sensitivities
                "sens_central_diff": sens_cd,
                "dy_plus_mean": dy_plus_mean,
                "dy_minus_mean": dy_minus_mean,
                "dy_plus": dy_plus_mean,
                "dy_minus": dy_minus_mean,
                "abs_change_max": dy_max_abs,
                "abs_change_plus": abs(dy_plus_mean),
                "abs_change_minus": abs(dy_minus_mean),
                "aggregate": cfg.aggregate,
                # MC meta
                "n_mc": n_mc,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="abs_change_max", ascending=False).reset_index(drop=True)
    csv_path = Path(out_dir) / cfg.save_csv_name
    df.to_csv(csv_path, index=False)

    meta = {
        "analysis": "local_oat_mc",
        "config": asdict(cfg),
        "n_targets": len(target_node_indices),
        "n_mc": n_mc,
        "seed": cfg.seed,
    }
    with open(Path(out_dir) / "oat_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OAT-MC] Done. Wrote {csv_path}")
    return df


# =========================
# Randomized screening
# =========================

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

      - For each scenario s in {1..n_scenarios}:
          * Draw a fresh initialization via `initialize_state_fn` with its own seed.
          * Randomly perturb the chosen factor nodes around their current values.
          * Run a simulation and aggregate outputs to a scalar y_s.

      - After all scenarios:
          * For each factor j, compute Spearman correlation rho_j between
            its scenario-wise values x_{s,j} and the scalar outputs y_s.
          * Rank factors by |rho_j| as an importance metric.

    The random perturbations are controlled by:
      - range_type="abs":      new_x = x0 + U(low, high)
      - range_type="percent":  new_x = x0 * (1 + U(low, high))

    All values are clipped to [clip_min, clip_max].
    """
    out_dir = _ensure_dir(cfg.save_dir)
    n_scen = int(cfg.n_scenarios)
    if n_scen <= 0:
        raise ValueError("RandomScreenConfig.n_scenarios must be positive.")

    factor_node_indices = [int(i) for i in factor_node_indices]
    n_factors = len(factor_node_indices)
    node_names_list = list(graph.nodes())

    rng_global = np.random.default_rng(cfg.seed)

    # X: scenario-wise factor values; Y: scenario-wise aggregated outputs
    X = np.zeros((n_scen, n_factors), dtype=float)
    Y = np.zeros(n_scen, dtype=float)

    for s in range(n_scen):
        kw = dict(init_kwargs)
        # ensure a fresh seed for every scenario, unless user fixed a specific one
        if "seed" in kw and kw["seed"] is not None:
            # If user set a fixed seed, we still keep it to remain deterministic.
            pass
        else:
            kw["seed"] = int(rng_global.integers(0, 2**31 - 1))

        out = initialize_state_fn(**kw)
        if not (isinstance(out, (list, tuple)) and len(out) == 5):
            raise RuntimeError(
                "initialize_state_fn must return a 5-tuple: "
                "(init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx)"
            )
        init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = out

        # Perturb selected factors for this scenario
        for j, idx in enumerate(factor_node_indices):
            idx_int = int(idx)
            x0 = float(init_state.get(idx_int, 0.0))

            if cfg.range_type == "percent":
                scale = rng_global.uniform(1.0 + cfg.low, 1.0 + cfg.high)
                x_new = x0 * scale
            else:  # "abs" or anything else treated as abs
                delta = rng_global.uniform(cfg.low, cfg.high)
                x_new = x0 + delta

            x_new = float(np.clip(x_new, cfg.clip_min, cfg.clip_max))
            init_state[idx_int] = x_new
            X[s, j] = x_new

        sim_out = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_state,
            random_root_nodes=random_nodes,
            bsp_root_nodes=bsp_nodes_idx,
            output_nodes=output_nodes,
            **sim_kwargs,
        )
        if not (isinstance(sim_out, (list, tuple)) and len(sim_out) >= 3):
            raise RuntimeError(
                "simulate_fn must return at least (state_trace, final_step, final_state, ...)."
            )
        _, _, y_fin = sim_out[:3]
        Y[s] = _aggregate_outputs(y_fin, cfg.aggregate)

    # Compute per-factor Spearman correlation with outputs
    rows: List[Dict[str, Any]] = []
    for j, idx in enumerate(factor_node_indices):
        xi = X[:, j]
        rho = _spearman_corr(xi, Y)
        node_idx = int(idx)
        node_name = (
            node_names_list[node_idx]
            if 0 <= node_idx < len(node_names_list)
            else f"node_{node_idx}"
        )

        rows.append(
            {
                "node_index": node_idx,
                "node_name": node_name,
                "spearman_rho": rho,
                "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                "x_mean": float(np.mean(xi)),
                "x_std": float(np.std(xi, ddof=1)) if n_scen > 1 else 0.0,
                "y_mean": float(np.mean(Y)),
                "y_std": float(np.std(Y, ddof=1)) if n_scen > 1 else 0.0,
                "aggregate": cfg.aggregate,
                "n_scenarios": n_scen,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="abs_spearman_rho", ascending=False).reset_index(drop=True)
    csv_path = Path(out_dir) / cfg.save_csv_name
    df.to_csv(csv_path, index=False)

    meta = {
        "analysis": "randomized_screening",
        "config": asdict(cfg),
        "n_factors": len(factor_node_indices),
        "n_scenarios": n_scen,
    }
    with open(Path(out_dir) / "randomized_screening_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Screening] Done. Wrote {csv_path}")
    return df


# =========================
# Combined runner
# =========================

def run_both_analyses(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    oat_targets: Sequence[int],
    screen_factors: Sequence[int],
    oat_cfg: OATConfig = OATConfig(),
    screen_cfg: RandomScreenConfig = RandomScreenConfig(),
) -> Dict[str, Any]:
    """
    Convenience wrapper: run local OAT (with MC) and randomized screening
    using the same graph, FCM matrix, initialization and simulation functions.

    Returns a dict with:
      - "oat_df", "screen_df": pandas DataFrames
      - "oat_csv", "screen_csv": paths to CSV files
      - "summary_json": path to a small JSON summary with configs
    """
    # ensure output dirs exist
    _ensure_dir(oat_cfg.save_dir)
    _ensure_dir(screen_cfg.save_dir)

    # 1) Local OAT with Monte Carlo on expected outcome
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

    # 2) Randomized screening
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

    # Summary JSON in the OAT folder (arbitrary choice)
    summary_path = Path(oat_cfg.save_dir) / "sensitivity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "oat_csv": str(Path(oat_cfg.save_dir) / oat_cfg.save_csv_name),
                "screen_csv": str(Path(screen_cfg.save_dir) / screen_cfg.save_csv_name),
                "oat_config": asdict(oat_cfg),
                "screen_config": asdict(screen_cfg),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[All] Summary: {summary_path}")
    return {
        "oat_df": df_oat,
        "screen_df": df_screen,
        "oat_csv": str(Path(oat_cfg.save_dir) / oat_cfg.save_csv_name),
        "screen_csv": str(Path(screen_cfg.save_dir) / screen_cfg.save_csv_name),
        "summary_json": str(summary_path),
    }
