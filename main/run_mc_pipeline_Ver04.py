from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import json
import numpy as np
import pandas as pd

# --- Optional SciPy bits (safe if missing) ---
try:
    from scipy.stats import gaussian_kde, genpareto, norm  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# =============================================================================
# Config
# =============================================================================

@dataclass
class MCPipelineConfig:
    # --- core ---
    n_total: int = 25_000
    batch_size: int = 5_000
    test_mode: bool = False
    verbose: bool = True
    seed: Optional[int] = 42

    # --- risk stats ---
    alpha: float = 0.05                 # VaR/CVaR level
    q_list: Tuple[float, ...] = (0.05, 0.90, 0.95)
    tail: str = "right"                 # 'right' or 'left'
    evt_threshold_quantile: float = 0.90

    # --- outputs ---
    output_dir: Union[str, Path] = Path("mc_outputs")
    dataset_filename: str = "mc_dataset.csv"
    stats_filename: str = "mc_stats.csv"
    meta_filename: str = "mc_meta.json"

    # --- auto-n (OFF by default) ---
    auto_n: bool = False
    auto_n_params: Optional[Dict[str, Any]] = None

    # --- performance/UX toggles (default: OFF to match conservative behavior) ---
    n_workers: int = 1                  # 1 = serial
    use_shared_csr: bool = False        # if True and n_workers>1, use shared-memory CSR (requires scipy)
    mp_chunksize: int = 64
    stream_dataset: bool = False        # write dataset incrementally to disk
    stream_every: int = 20_000
    float32_results: bool = False
    stats_only: bool = False            # if True, don't write mc_dataset.csv

    progress_report: bool = False
    progress_interval: float = 0.05     # fraction (e.g., 0.05 = every 5%)

    # --- new: fast per-node matrix & optional post-run excel ---
    save_per_node_feather: bool = True  # write 'mc_per_node_final_states.feather' (or Parquet fallback)
    post_export_excel: bool = False     # after simulation finishes, export per-node matrix to Excel
    post_export_stats_excel: bool = False  # after simulation, export per-node stats to Excel
    excel_engine: str = "openpyxl"      # or "xlsxwriter"


# =============================================================================
# Small helpers
# =============================================================================

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _progress(done: int, total: int) -> float:
    return 100.0 * done / max(1, total)

def _jsonify(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    return o


# =============================================================================
# Sample size (auto-n)
# =============================================================================

def compute_mc_runs(
    n_outputs: int,
    alphas: Tuple[float, ...] = (0.05,),
    q_targets: Tuple[float, ...] = (0.05, 0.90, 0.95),
    eps_quantile: Optional[float] = None,
    tail_samples_target: int = 1000,
    f_min: Optional[float] = None,
    ci: float = 0.90,
    use_bonferroni: bool = False,
    min_n: int = 5000,
) -> int:
    # crude, conservative recipe; can be refined later
    n_tail = max(int(np.ceil(tail_samples_target / max(1e-12, a))) for a in alphas)
    n_quant = 0
    if (eps_quantile is not None) and (f_min is not None) and eps_quantile > 0 and f_min > 0:
        delta = 1.0 - ci
        metrics_count = len(q_targets) + len(alphas)
        if use_bonferroni and n_outputs > 0 and metrics_count > 0:
            delta = delta / (n_outputs * metrics_count)
        eps_cdf = f_min * eps_quantile
        n_quant = int(np.ceil(np.log(2.0 / max(1e-16, delta)) / (2.0 * max(1e-16, eps_cdf) ** 2)))
    return max(min_n, n_tail, n_quant)


# =============================================================================
# Statistics
# =============================================================================

def kde_mode(y: np.ndarray) -> Optional[float]:
    if not SCIPY_AVAILABLE:
        return None
    try:
        xs = np.linspace(np.min(y), np.max(y), 2048)
        kde = gaussian_kde(y)
        dens = kde(xs)
        return float(xs[np.argmax(dens)])
    except Exception:
        return None

def mean_ci90_bootstrap(y: np.ndarray, n_boot: int = 2000, ci: float = 0.90, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return float(np.mean(y)), np.nan, np.nan
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(y, size=n, replace=True)
        means[i] = np.mean(sample)
    lower = np.quantile(means, (1 - ci) / 2)
    upper = np.quantile(means, 1 - (1 - ci) / 2)
    return float(np.mean(y)), float(lower), float(upper)

def var_cvar(y: np.ndarray, alpha: float = 0.05, tail: str = "right") -> Tuple[float, float]:
    if tail not in ("right", "left"):
        raise ValueError("tail must be 'right' or 'left'")
    p = 1 - alpha if tail == "right" else alpha
    var_val = float(np.quantile(y, p))
    mask = (y >= var_val) if tail == "right" else (y <= var_val)
    tail_samples = y[mask]
    cvar_val = float(np.mean(tail_samples)) if tail_samples.size else np.nan
    return var_val, cvar_val

def evt_pot_fit(y: np.ndarray, q_thresh: float = 0.90, tail: str = "right") -> Dict[str, Any]:
    if not SCIPY_AVAILABLE:
        return {"error": "SciPy not available"}
    if tail not in ("right", "left"):
        raise ValueError("tail must be 'right' or 'left'")
    if tail == "right":
        u = float(np.quantile(y, q_thresh))
        excess = y[y > u] - u
    else:
        u = float(np.quantile(y, 1 - q_thresh))
        excess = -(y[y < u] - u)
    k, N = int(excess.size), int(y.size)
    if k == 0:
        return {"u": u, "k": 0, "N": N, "error": "No exceedances"}
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    return {"u": float(u), "k": k, "N": N, "p_u": k / N, "xi": float(c), "beta": float(scale)}

def evt_var_cvar(evt: Dict[str, Any], p: float, tail: str = "right") -> Dict[str, float]:
    # simple EVT quantile / CVar based on GPD tail (crude)
    if "error" in evt:
        return {"VaR_evt": float("nan"), "CVaR_evt": float("nan")}
    xi, beta, u, p_u = evt["xi"], evt["beta"], evt["u"], evt["p_u"]
    p = min(max(p, 1e-12), 1 - 1e-12)
    if tail == "right":
        s, s_u = (1 - p), p_u
        if abs(xi) < 1e-12:
            q = u + beta * np.log(s_u / s)
            cvar = q + beta
        else:
            q = u + (beta/xi) * ((s_u / s)**xi - 1)
            cvar = (q + beta - xi*u) / (1 - xi) if xi < 1 else np.inf
        return {"VaR_evt": float(q), "CVaR_evt": float(cvar)}
    elif tail == "left":
        up, s, s_u = -u, p, p_u
        if abs(xi) < 1e-12:
            zq = up + beta * np.log(s_u / s)
            zc = zq + beta
        else:
            zq = up + (beta/xi) * ((s_u / s)**xi - 1)
            zc = (zq + beta - xi*up) / (1 - xi) if xi < 1 else np.inf
        q = -zq
        cvar = -zc
        return {"VaR_evt": float(q), "CVaR_evt": float(cvar)}
    else:
        raise ValueError("tail must be 'right' or 'left'")


# =============================================================================
# Core: simulate a batch (serial)
# =============================================================================

def simulate_batch(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run `n` Monte Carlo simulations serially.
    Returns: ndarray of shape (n, n_outputs_per_run) with final states for output nodes.
    """
    results: List[List[float]] = []
    for run_idx in range(n):
        # Per-run seed perturbation (reproducible)
        if "seed" in init_kwargs and init_kwargs.get("seed") is not None:
            base = int(init_kwargs["seed"])
            init_kwargs = dict(init_kwargs)
            init_kwargs["seed"] = base + int(run_idx)

        init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_fn(**init_kwargs)

        sim_out = simulate_fn(
            fcm_matrix=fcm_matrix,
            init_state=init_state,
            random_root_nodes=random_nodes,
            bsp_root_nodes=bsp_nodes_idx,
            output_nodes=output_nodes,
            **sim_kwargs
        )
        if not (isinstance(sim_out, (list, tuple)) and len(sim_out) >= 3):
            raise RuntimeError("simulate_fn must return at least (state_trace, final_step, final_state).")
        _, _, final_state = sim_out[:3]
        if not final_state:
            raise RuntimeError("simulate_fn returned empty final_state (check output_nodes).")
        results.append(final_state)

    dtype = np.float32 if init_kwargs.get("dtype") == "float32" else float
    return np.array(results, dtype=dtype)


# =============================================================================
# Main pipeline (aggregated 1-D analysis + fast per-node save)
# =============================================================================

def run_mc_pipeline(
    graph: Any,
    fcm_matrix: Any,
    initialize_state_fn: Callable[..., Tuple[Dict[int, float], List[int], List[int], List[int], List[int]]],
    simulate_fn: Callable[..., Tuple[Any, Any, List[float], Any, Any]],
    init_kwargs: Dict[str, Any],
    sim_kwargs: Dict[str, Any],
    cfg: MCPipelineConfig
) -> Dict[str, Any]:

    out_dir = _ensure_dir(cfg.output_dir)
    dataset_path = out_dir / cfg.dataset_filename
    stats_path = out_dir / cfg.stats_filename
    meta_path = out_dir / cfg.meta_filename

    rng = np.random.default_rng(cfg.seed)

    # Dry init: to detect number of outputs and their names
    init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_fn(**init_kwargs)
    n_outputs_per_run = len(output_nodes)
    if n_outputs_per_run < 1:
        raise RuntimeError("No output nodes detected by initialize_state_fn().")

    node_names_list = list(graph.nodes())
    per_node_cols = [
        node_names_list[i] if 0 <= int(i) < len(node_names_list) else f"node_{i}"
        for i in output_nodes
    ]

    # Auto-n
    n_total = cfg.n_total
    if cfg.auto_n:
        params = cfg.auto_n_params or {}
        n_total = compute_mc_runs(
            n_outputs=1,
            alphas=(cfg.alpha,),
            q_targets=cfg.q_list,
            eps_quantile=params.get("eps_quantile"),
            tail_samples_target=int(params.get("tail_samples_target", 1000)),
            f_min=params.get("f_min"),
            ci=float(params.get("ci", 0.90)),
            use_bonferroni=bool(params.get("use_bonferroni", False)),
            min_n=int(params.get("min_n", 5000)),
        )
        if cfg.verbose:
            print(f"[MC] auto_n=True → computed n_total={n_total}")

    if cfg.test_mode:
        n_total = min(n_total, max(1000, cfg.batch_size))

    if cfg.verbose:
        print(f"[MC] Start simulation: n_total={n_total}, batch={cfg.batch_size}, outputs_per_run={n_outputs_per_run}")

    # --- Streaming simulation (serial batching) ---
    all_rows: List[np.ndarray] = []
    done = 0
    while done < n_total:
        m = int(min(cfg.batch_size, n_total - done))
        batch = simulate_batch(
            graph=graph,
            fcm_matrix=fcm_matrix,
            initialize_state_fn=initialize_state_fn,
            simulate_fn=simulate_fn,
            init_kwargs=init_kwargs,
            sim_kwargs=sim_kwargs,
            n=m,
            rng=rng,
        )
        all_rows.append(batch)
        done += m
        if cfg.verbose:
            print(f"[MC] Progress {done}/{n_total} ({_progress(done, n_total):.1f}%)")

    # --- Stack batches → samples (n_total, n_outputs)
    if len(all_rows) == 0:
        raise RuntimeError("No batches produced any data.")
    samples = np.vstack(all_rows)
    if samples.shape[1] == 0:
        raise RuntimeError("simulate_fn returned empty final_state (0 outputs per run).")

    if cfg.verbose:
        print(f"[MC] collected samples shape: {samples.shape}")

    # --- Fast per-node matrix (Feather or Parquet fallback)
    per_node_df = pd.DataFrame(samples, columns=per_node_cols)
    if cfg.save_per_node_feather:
        try:
            feather_path = out_dir / "mc_per_node_final_states.feather"
            per_node_df.to_feather(feather_path)
            if cfg.verbose:
                print(f"[MC] Saved per-node matrix (Feather): {feather_path.name}")
        except Exception as e:
            pq_path = out_dir / "mc_per_node_final_states.parquet"
            per_node_df.to_parquet(pq_path, compression="snappy")
            if cfg.verbose:
                print(f"[WARN] Feather failed ({e}); wrote Parquet: {pq_path.name}")

    # --- Flatten to 1-D (aggregate vector) for legacy single-column dataset & aggregate stats
    Y_all = samples.ravel(order="C")
    finite = np.isfinite(Y_all)
    if not finite.any():
        raise RuntimeError("All observations became NaN/inf; stats input empty.")
    Y_all = Y_all[finite]

    if cfg.verbose:
        print(f"[MC] stats input: Y_all.size={Y_all.size}")

    # --- Legacy single-column dataset (optional when stats_only=False)
    if not cfg.stats_only:
        pd.DataFrame({"y": Y_all}).to_csv(dataset_path, index=False)

    # --- Aggregate stats (single vector)
    mean, ciL, ciU = mean_ci90_bootstrap(Y_all)
    median = float(np.median(Y_all))
    mode_est = kde_mode(Y_all)
    quantiles = {f"q_{int(p*100)}": float(np.quantile(Y_all, p)) for p in cfg.q_list}
    var_val, cvar_val = var_cvar(Y_all, alpha=cfg.alpha, tail=cfg.tail)
    evt = evt_pot_fit(Y_all, q_thresh=cfg.evt_threshold_quantile, tail=cfg.tail)
    p_evt = 1 - cfg.alpha if cfg.tail == "right" else cfg.alpha
    evt_q = evt_var_cvar(evt, p=p_evt, tail=cfg.tail)

    stats_record = {
        "n_total_runs": int(n_total),
        "n_outputs_per_run": int(n_outputs_per_run),
        "n_effective_samples": int(Y_all.size),
        "mean": mean, "mean_CI90_L": ciL, "mean_CI90_U": ciU,
        "median": median,
        "mode_KDE": (float(mode_est) if mode_est is not None else np.nan),
        "alpha": float(cfg.alpha), "tail": cfg.tail,
        "VaR_alpha": var_val, "CVaR_alpha": cvar_val,
        "evt_threshold_quantile": float(cfg.evt_threshold_quantile),
        "evt_u": float(evt.get("u", np.nan)),
        "evt_k": int(evt.get("k", 0)) if isinstance(evt.get("k", 0), (int, np.integer)) else 0,
        "evt_pu": float(evt.get("p_u", np.nan)),
        "evt_xi": float(evt.get("xi", np.nan)),
        "evt_beta": float(evt.get("beta", np.nan)),
        "evt_VaR": float(evt_q.get("VaR_evt", np.nan)),
        "evt_CVaR": float(evt_q.get("CVaR_evt", np.nan)),
    }
    stats_record.update(quantiles)
    pd.DataFrame([stats_record]).to_csv(stats_path, index=False)

    # --- Per-node stats (small; CSV fast)
    records = []
    q_list_sorted = tuple(sorted(cfg.q_list))
    for j, col_name in enumerate(per_node_cols):
        y = samples[:, j]
        y = y[np.isfinite(y)]
        if y.size == 0:
            rec = {
                "node": col_name,
                "n_runs": int(samples.shape[0]),
                "effective_n": 0,
                "mean": float("nan"),
                "mean_CI90_L": float("nan"),
                "mean_CI90_U": float("nan"),
                "median": float("nan"),
            }
            for p in q_list_sorted:
                rec[f"q_{int(p*100)}"] = float("nan")
            rec.update({"VaR_alpha": float("nan"), "CVaR_alpha": float("nan")})
            rec.update({
                "evt_threshold_quantile": float(cfg.evt_threshold_quantile),
                "evt_u": float("nan"), "evt_k": 0, "evt_pu": float("nan"),
                "evt_xi": float("nan"), "evt_beta": float("nan"),
                "evt_VaR": float("nan"), "evt_CVaR": float("nan"),
            })
            records.append(rec)
            continue

        m, l, u = mean_ci90_bootstrap(y, ci=0.90)
        med = float(np.median(y))
        qs = {f"q_{int(p*100)}": float(np.quantile(y, p)) for p in q_list_sorted}
        vr, cvr = var_cvar(y, alpha=cfg.alpha, tail=cfg.tail)
        evt_local = evt_pot_fit(y, q_thresh=cfg.evt_threshold_quantile, tail=cfg.tail)
        p_evt_local = 1 - cfg.alpha if cfg.tail == "right" else cfg.alpha
        evt_q_local = evt_var_cvar(evt_local, p=p_evt_local, tail=cfg.tail)

        rec = {
            "node": col_name,
            "n_runs": int(samples.shape[0]),
            "effective_n": int(y.size),
            "mean": float(m), "mean_CI90_L": float(l), "mean_CI90_U": float(u),
            "median": med,
        }
        rec.update(qs)
        rec.update({"VaR_alpha": float(vr), "CVaR_alpha": float(cvr)})
        rec.update({
            "evt_threshold_quantile": float(cfg.evt_threshold_quantile),
            "evt_u": float(evt_local.get("u", float("nan"))),
            "evt_k": int(evt_local.get("k", 0)) if isinstance(evt_local.get("k", 0), (int, np.integer)) else 0,
            "evt_pu": float(evt_local.get("p_u", float("nan"))),
            "evt_xi": float(evt_local.get("xi", float("nan"))),
            "evt_beta": float(evt_local.get("beta", float("nan"))),
            "evt_VaR": float(evt_q_local.get("VaR_evt", float("nan"))),
            "evt_CVaR": float(evt_q_local.get("CVaR_evt", float("nan"))),
        })
        records.append(rec)

    per_node_stats_df = pd.DataFrame.from_records(records)
    per_node_stats_csv = out_dir / "mc_stats_per_node.csv"
    per_node_stats_df.to_csv(per_node_stats_csv, index=False)

    # --- Optional post-export to Excel (after simulation; slower but off by default)
    if cfg.post_export_excel:
        try:
            xlsx_path = out_dir / "mc_per_node_final_states.xlsx"
            # Prefer reading back from Feather to reduce RAM spikes if needed
            try:
                df_for_excel = pd.read_feather(out_dir / "mc_per_node_final_states.feather")
            except Exception:
                df_for_excel = per_node_df  # fallback
            max_rows = 1_048_576  # Excel row limit
            with pd.ExcelWriter(xlsx_path, engine=cfg.excel_engine) as w:
                if len(df_for_excel) <= max_rows:
                    df_for_excel.to_excel(w, index=False, sheet_name="final_states")
                else:
                    start = 0
                    sheet_idx = 1
                    while start < len(df_for_excel):
                        end = min(start + max_rows, len(df_for_excel))
                        df_for_excel.iloc[start:end].to_excel(
                            w, index=False, sheet_name=f"final_{sheet_idx}"
                        )
                        sheet_idx += 1
                        start = end
            if cfg.verbose:
                print(f"[MC] Post-export Excel created: {xlsx_path.name}")
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Post-export Excel failed: {e}")

    if cfg.post_export_stats_excel:
        try:
            xlsx_stats_path = out_dir / "mc_stats_per_node.xlsx"
            with pd.ExcelWriter(xlsx_stats_path, engine=cfg.excel_engine) as w:
                per_node_stats_df.to_excel(w, index=False, sheet_name="stats")
            if cfg.verbose:
                print(f"[MC] Post-export Excel (stats) created: {xlsx_stats_path.name}")
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Post-export Excel (stats) failed: {e}")

    # --- Meta
    cfg_dict = _jsonify(asdict(cfg))
    meta = {
        "config": cfg_dict,
        "n_total_runs": int(n_total),
        "n_outputs_per_run": int(n_outputs_per_run),
        "n_effective_samples": int(Y_all.size),
        "scipy_available": SCIPY_AVAILABLE,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if cfg.verbose:
        print(f"[MC] Completed. Saved stats={stats_path.name}, per-node stats={per_node_stats_csv.name}")

    return {
        "dataset_path": (str(dataset_path) if not cfg.stats_only else None),
        "stats_path": str(stats_path),
        "meta_path": str(meta_path),
        "per_node_feather": str(out_dir / "mc_per_node_final_states.feather") if cfg.save_per_node_feather else None,
        "per_node_stats_csv": str(per_node_stats_csv),
        "n_total_runs": int(n_total),
        "n_outputs_per_run": int(n_outputs_per_run),
        "n_effective_samples": int(Y_all.size),
    }
