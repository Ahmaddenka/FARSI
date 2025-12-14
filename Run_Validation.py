import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

# --- pipeline + EVT helpers (already implemented in run_mc_pipeline_Ver04.py) ---
from main.run_mc_pipeline_Ver04 import (
    MCPipelineConfig,
    run_mc_pipeline,
    var_cvar,
    evt_pot_fit,
    evt_var_cvar,
    SCIPY_AVAILABLE,
)

# ---  sim + initialization plan prep ---
from main.run_initialize_state import prepare_state_plan
from main.run_fcm_simulation_trace import fcm_simulation_trace

# ---  project-specific modules (must exist in  project folder) ---
from main.fuzzy_to_prob import build_probability_inputs
from main.agent_and_control_number import build_control_summary
from main.graph_to_fcm import run_graph_to_fcm
from pathlib import Path


INPUT_DIR = Path("field_data")
#OUT_DIR = Path("output")

# =============================================================================
# Helpers
# =============================================================================
def clean_node_name(name: str) -> str:
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name.strip("_")

def clean_node_dict(prob_inputs: dict) -> dict:
    return {clean_node_name(k): v for k, v in prob_inputs.items()}

def score_0_100(x: float | np.ndarray) -> float | np.ndarray:
    # Score = 50 * (A + 1)
    return 50.0 * (np.asarray(x) + 1.0)

def safe_read_stats_csv(stats_path: str) -> dict:
    df = pd.read_csv(stats_path)
    if len(df) != 1:
        raise RuntimeError(f"Expected single-row stats CSV; got {len(df)} rows: {stats_path}")
    return df.iloc[0].to_dict()

def pooled_samples_from_outputs(out_dir: Path("output")) -> np.ndarray:
    """
    Loads pooled samples Y from either:
      - mc_dataset.csv (single column 'y'), OR
      - mc_per_node_final_states.feather/parquet (matrix), then ravel.
    """
    ds_csv = out_dir / "mc_dataset.csv"
    if ds_csv.exists():
        df = pd.read_csv(ds_csv)
        if "y" not in df.columns:
            raise RuntimeError("mc_dataset.csv found but no 'y' column.")
        y = df["y"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        return y

    feather = out_dir / "mc_per_node_final_states.feather"
    parquet = out_dir / "mc_per_node_final_states.parquet"

    if feather.exists():
        df = pd.read_feather(feather)
        y = df.to_numpy(dtype=float).ravel(order="C")
        y = y[np.isfinite(y)]
        return y

    if parquet.exists():
        df = pd.read_parquet(parquet)
        y = df.to_numpy(dtype=float).ravel(order="C")
        y = y[np.isfinite(y)]
        return y

    raise RuntimeError(
        f"Could not find pooled data in {out_dir}. "
        f"Expected mc_dataset.csv or mc_per_node_final_states.feather/parquet."
    )

def compute_evt_threshold_table(
    y: np.ndarray,
    tail: str,
    p_tail: float,
    q_thresh_list=(0.90, 0.95, 0.97),
) -> pd.DataFrame:
    """
    Produces the exact POT sensitivity table:
      q_thresh, u, N, k, k/N, xi, sigma, EVT-VaR_p, EVT-CVaR_p
    using the SAME EVT functions as your pipeline (evt_pot_fit / evt_var_cvar).
    """
    rows = []
    N = int(np.isfinite(y).sum())
    y = y[np.isfinite(y)]

    for q_thresh in q_thresh_list:
        evt = evt_pot_fit(y, q_thresh=float(q_thresh), tail=tail)
        # evt_var_cvar expects p = p_tail (e.g., 0.05) and uses tail to interpret left/right
        q = evt_var_cvar(evt, p=float(p_tail), tail=tail)

        u = float(evt.get("u", np.nan))
        k = int(evt.get("k", 0)) if "k" in evt else 0
        rate = (k / N) if N > 0 else np.nan

        rows.append({
            "q_thresh": float(q_thresh),
            "u = Q_{1-q_thresh}(Y) (left-tail cutoff)" if tail == "left" else "u = Q_{q_thresh}(Y) (right-tail cutoff)": u,
            "N_total": N,
            "k_exceedances": k,
            "exceedance_rate (k/N)": float(rate),
            "GPD_shape_xi": float(evt.get("xi", np.nan)),
            "GPD_scale_sigma": float(evt.get("beta", np.nan)),  # beta in code = scale
            f"EVT-VaR_p={p_tail}": float(q.get("VaR_evt", np.nan)),
            f"EVT-CVaR_p={p_tail}": float(q.get("CVaR_evt", np.nan)),
            "evt_error": evt.get("error", ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Seed-respecting initializer wrapper
# =============================================================================
def initialize_state_from_plan_seeded(
    plan,
    bsp_range=(-1.0, 1.0),
    random_range=(-1.0, 1.0),
    intermediate_range=(-1.0, 1.0),
    output_range=(-1.0, 1.0),
    get_bsp_input=False,
    seed=None,
    dtype="float32",
):
    try:
        from scipy.stats import truncnorm
        TRUNC_AVAILABLE = True
    except Exception:
        TRUNC_AVAILABLE = False

    rng = np.random.default_rng(seed)

    node_names        = plan["node_names"]
    n_nodes           = plan["n_nodes"]
    bsp_idx           = plan["bsp_idx"]
    rnd_idx           = plan["rnd_idx"]
    out_idx           = plan["out_idx"]
    inter_idx         = plan["inter_idx"]
    bsp_samplers      = plan["bsp_samplers"]
    bsp_has_sampler   = plan["bsp_has_sampler"]
    bsp_mean_targets  = plan["bsp_mean_targets"]

    state_arr = np.empty(n_nodes, dtype=dtype)
    state_arr.fill(np.nan)

    # Random nodes
    if rnd_idx:
        if TRUNC_AVAILABLE:

            vals = truncnorm.rvs(
                random_range[0], random_range[1],
                loc=(random_range[0] + random_range[1]) / 2,
                scale=1,
                size=len(rnd_idx),
                random_state=rng
            )
        else:
            vals = rng.uniform(low=random_range[0], high=random_range[1], size=len(rnd_idx))
        state_arr[np.fromiter(rnd_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)

    # Intermediate nodes
    if inter_idx:
        vals = rng.uniform(low=intermediate_range[0], high=intermediate_range[1], size=len(inter_idx))
        state_arr[np.fromiter(inter_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)

    # Output nodes (initial values)
    if out_idx:
        vals = rng.uniform(low=output_range[0], high=output_range[1], size=len(out_idx))
        state_arr[np.fromiter(out_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)

    # BSP nodes
    if bsp_idx:
        bsp_idx_arr = np.fromiter(bsp_idx, dtype=np.int32)

        if get_bsp_input and any(bsp_has_sampler):
            # Sample where sampler exists, fallback to mean_target/center
            for i_local, has_samp in enumerate(bsp_has_sampler):
                i_global = bsp_idx[i_local]
                if has_samp:
                    sampler = bsp_samplers[i_local]
                    try:
                        sampled = sampler(size=1, rng=rng)
                        val = float(sampled[0]) if hasattr(sampled, "__len__") else float(sampled)
                    except Exception:
                        mt = float(bsp_mean_targets[i_local])
                        if np.isnan(mt):
                            mt = 0.5 * (bsp_range[0] + bsp_range[1])
                        val = mt
                    state_arr[i_global] = val
                else:
                    mt = float(bsp_mean_targets[i_local])
                    if np.isnan(mt):
                        mt = 0.5 * (bsp_range[0] + bsp_range[1])
                    state_arr[i_global] = mt
        else:
            # If not sampling BSP inputs, keep your older behavior: random BSP init
            vals = rng.uniform(low=bsp_range[0], high=bsp_range[1], size=len(bsp_idx))
            state_arr[bsp_idx_arr] = vals.astype(dtype, copy=False)

    # Clamp
    set_mask = ~np.isnan(state_arr)
    if np.any(set_mask):
        state_arr[set_mask] = np.minimum(1.0, np.maximum(-1.0, state_arr[set_mask]))

    init_state = {int(i): float(state_arr[i]) for i in np.where(set_mask)[0].tolist()}
    root_nodes = list(bsp_idx) + list(rnd_idx)
    output_nodes_idx = list(out_idx)
    random_nodes_idx = list(rnd_idx)
    bsp_nodes_idx = list(bsp_idx)

    return init_state, root_nodes, output_nodes_idx, random_nodes_idx, bsp_nodes_idx


# =============================================================================
# Validation configuration
# =============================================================================
@dataclass
class ValidationPlan:
    out_root: str = "validation_outputs"

    # Seed / initialization robustness
    seed_list: tuple[int, ...] = (11, 21, 31, 41, 51)

    # MC stability (sample-size check)
    mc_sizes: tuple[int, int] = (5000, 15000)

    # Sensitivity around calibrated density parameters (alpha_dens,beta_dens,gamma_dens)
    perturb_pct: float = 0.10   # ±10%

    # EVT POT threshold sensitivity
    evt_q_thresh_list: tuple[float, ...] = (0.90, 0.95, 0.97)

    # Reported band (computed may include more)
    report_q_lo: float = 0.05
    report_q_hi: float = 0.95


# =============================================================================
# Main
# =============================================================================
def main():
    V = ValidationPlan()
    out_root = Path(V.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Build probability inputs, graph->FCM, and control data
    # ------------------------------------------------------------
    prob_inputs = build_probability_inputs(
        excel_path=str(INPUT_DIR/"BSM_input_DATA.xlsx"),
        sheet_name=0,
        id_col=0,
        scale_min=1.0, scale_max=5.0,
        trapezoid_fit="std",
        std_core_width=0.2,
        std_support_width=1.0,
        support_p=(0.05, 0.95),
        core_p=(0.35, 0.65),
        n_grid=1001,
        target_range=(-1.0, 1.0)
    )

    fcm_matrix, graph = run_graph_to_fcm()
    control_leaders = build_control_summary(str(INPUT_DIR/"Control_gorup_structure.xlsx"))

    node_names = list(graph.nodes())
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    control_nodes = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "control" and n not in control_leaders
    )
    if len(control_nodes) == 0:
        control_nodes = sorted(
            n for n, d in graph.nodes(data=True)
            if str(d.get("level", "")).lower() == "control"
        )
        print("[WARN] non-leader controls not found -> using ALL control nodes as outputs:", len(control_nodes))

    # Build control leader index dict
    control_leaders_idx = {}
    for name, vals in control_leaders.items():
        i = name_to_idx[name]
        control_leaders_idx[i] = {
            "agent_count": float(vals.get("agent_count", 1.0)),
            "control_count": float(vals.get("control_count", 0.0)),
        }

    control_index_map = {
        name_to_idx[n]
        for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "control" and n in control_leaders
    }

    # Prepare state plan
    bsp_data = clean_node_dict(prob_inputs)
    plan = prepare_state_plan(
        graph,
        bsp_data=bsp_data,
        random_nodes=None,
        output_nodes=control_nodes,
        random_keyword="random",
    )

    # init_kwargs (seed will be overridden in seed-robustness runs)
    init_kwargs_base = dict(
        plan=plan,
        bsp_range=(-1.0, 1.0),
        random_range=(-1.0, 1.0),
        intermediate_range=(-1.0, 1.0),
        output_range=(-1.0, 1.0),
        get_bsp_input=True,
        seed=0,              # IMPORTANT: enables per-run deterministic seeding via run_mc_pipeline
        dtype="float32"
    )

    # sim_kwargs (density parameters live here)
    sim_kwargs_base = dict(
        steps=300,
        delt=1e-3,
        control_dict=control_leaders_idx,
        control_index_map=control_index_map,
        alpha=0.05153744308202851,  # alpha_dens
        beta=0.8030979544125557,    # beta_dens
        gamma=0.6721727251459532,   # gamma_dens
        warmup_steps=30,
        freeze_bsp_in_warmup=True,
        freeze_bsp_in_main=False,
        min_main_steps=10,
        patience=2,
        debug_first_main=False,
        debug_bsp_nodes=None,
        Control_Data=None
    )

    # cfg base
    cfg_base = MCPipelineConfig()
    cfg_base.auto_n = True
    cfg_base.auto_n_params = dict(
        tail_samples_target=1000,
        ci=0.90,
        min_n=5000,
        use_bonferroni=False
    )
    cfg_base.alpha = 0.05                 # p_tail (VaR/CVaR alpha)
    cfg_base.q_list = (0.05, 0.90, 0.95)  # compute extra quantiles for internal diagnostics
    cfg_base.tail = "left"
    cfg_base.evt_threshold_quantile = 0.90
    #cfg_base.n_total = 10
    cfg_base.batch_size = 100
    cfg_base.test_mode = False
    cfg_base.make_plots = False
    cfg_base.verbose = True
    cfg_base.save_per_node_feather = True
    cfg_base.post_export_excel = True
    cfg_base.post_export_stats_excel = True
    cfg_base.seed = None
    cfg_base.output_dir = str(out_root / "baseline")
    cfg_base.n_workers = 5
    cfg_base.use_shared_csr = True
    cfg_base.mp_chunksize = 64
    cfg_base.stream_dataset = False
    cfg_base.float32_results = True
    cfg_base.stats_only = False
    cfg_base.progress_report = True
    cfg_base.progress_interval = 0.01

    # ------------------------------------------------------------
    # Runner for one experiment
    # ------------------------------------------------------------
    def run_one(tag: str, cfg: MCPipelineConfig, init_kwargs: dict, sim_kwargs: dict) -> dict:
        cfg = deepcopy(cfg)
        init_kwargs = deepcopy(init_kwargs)
        sim_kwargs = deepcopy(sim_kwargs)

        cfg.output_dir = str(out_root / tag)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        out = run_mc_pipeline(
            graph=graph,
            fcm_matrix=fcm_matrix,
            initialize_state_fn=initialize_state_from_plan_seeded,
            simulate_fn=fcm_simulation_trace,
            init_kwargs=init_kwargs,
            sim_kwargs=sim_kwargs,
            cfg=cfg
        )

        stats = safe_read_stats_csv(out["stats_path"])

        # Reportable indicators
        median_y = float(stats.get("median", np.nan))
        q5 = float(stats.get("q_5", np.nan))
        q95 = float(stats.get("q_95", np.nan))
        evt_cvar = float(stats.get("evt_CVaR", np.nan))

        return {
            "tag": tag,
            "out_dir": cfg.output_dir,
            "n_total_runs": int(out["n_total_runs"]),
            "n_outputs_per_run": int(out["n_outputs_per_run"]),
            "OSI_0_100": float(score_0_100(median_y)),
            "median_y": median_y,
            "q05_y": q5,
            "q95_y": q95,
            "q05_0_100": float(score_0_100(q5)),
            "q95_0_100": float(score_0_100(q95)),
            "EVT_CVaR_y": evt_cvar,
            "stats_path": out["stats_path"],
        }

    # ------------------------------------------------------------
    # 2) Baseline run
    # ------------------------------------------------------------
    print("\n=== Baseline Monte Carlo ===")
    baseline = run_one("baseline", cfg_base, init_kwargs_base, sim_kwargs_base)

    # ------------------------------------------------------------
    # 3) Seed / initialization robustness
    # ------------------------------------------------------------
    print("\n=== Seed robustness ===")
    seed_rows = []
    for s in V.seed_list:
        init_kwargs = deepcopy(init_kwargs_base)
        init_kwargs["seed"] = int(s)
        seed_rows.append(run_one(f"seed_{s}", cfg_base, init_kwargs, sim_kwargs_base))

    # ------------------------------------------------------------
    # 4) Monte Carlo stability (two nMC sizes)
    # ------------------------------------------------------------
    print("\n=== Monte Carlo sample-size stability ===")
    mc_rows = []
    N1, N2 = V.mc_sizes
    for n in (N1, N2):
        cfg = deepcopy(cfg_base)
        cfg.auto_n = False
        cfg.n_total = int(n)
        mc_rows.append(run_one(f"mc_n_{n}", cfg, init_kwargs_base, sim_kwargs_base))

    # ------------------------------------------------------------
    # 5) Sensitivity to calibrated density parameters (alpha_dens,beta_dens,gamma_dens)
    # ------------------------------------------------------------
    print("\n=== Density-parameter sensitivity (±pct) ===")
    sens_rows = []
    p = float(V.perturb_pct)

    for param in ("alpha", "beta", "gamma"):
        base_val = float(sim_kwargs_base[param])
        for mult, sign in ((1.0 - p, "minus"), (1.0 + p, "plus")):
            sim_kwargs = deepcopy(sim_kwargs_base)
            sim_kwargs[param] = base_val * mult
            sens_rows.append(run_one(f"{param}_{sign}_{int(p*100)}pct", cfg_base, init_kwargs_base, sim_kwargs))

    # ------------------------------------------------------------
    # 6) EVT (POT) threshold sensitivity table (single baseline pooled Y)
    # ------------------------------------------------------------
    print("\n=== EVT POT threshold sensitivity table ===")
    y_pool = pooled_samples_from_outputs(Path(baseline["out_dir"]))
    evt_table = compute_evt_threshold_table(
        y=y_pool,
        tail=cfg_base.tail,
        p_tail=cfg_base.alpha,
        q_thresh_list=V.evt_q_thresh_list
    )

    evt_csv = out_root / "evt_pot_threshold_sensitivity.csv"
    evt_xlsx = out_root / "evt_pot_threshold_sensitivity.xlsx"
    evt_table.to_csv(evt_csv, index=False)

    try:
        with pd.ExcelWriter(evt_xlsx) as w:
            evt_table.to_excel(w, index=False, sheet_name="POT_sensitivity")
    except Exception as e:
        print(f"[WARN] Could not write EVT Excel ({e}). CSV was written: {evt_csv.name}")

    # ------------------------------------------------------------
    # 7) Consolidate summary + deltas
    # ------------------------------------------------------------
    all_rows = []
    all_rows.append({**baseline, "group": "baseline"})

    for r in seed_rows:
        rr = {**r, "group": "seed"}
        rr["ΔOSI_vs_baseline"] = rr["OSI_0_100"] - baseline["OSI_0_100"]
        rr["ΔEVT_CVaR_vs_baseline"] = rr["EVT_CVaR_y"] - baseline["EVT_CVaR_y"]
        all_rows.append(rr)

    # compare N1 vs N2 in a simple way
    for r in mc_rows:
        rr = {**r, "group": "mc_size"}
        rr["ΔOSI_vs_baseline"] = rr["OSI_0_100"] - baseline["OSI_0_100"]
        rr["ΔEVT_CVaR_vs_baseline"] = rr["EVT_CVaR_y"] - baseline["EVT_CVaR_y"]
        all_rows.append(rr)

    for r in sens_rows:
        rr = {**r, "group": "density_sensitivity"}
        rr["ΔOSI_vs_baseline"] = rr["OSI_0_100"] - baseline["OSI_0_100"]
        rr["ΔEVT_CVaR_vs_baseline"] = rr["EVT_CVaR_y"] - baseline["EVT_CVaR_y"]
        all_rows.append(rr)

    summary_df = pd.DataFrame(all_rows)
    summary_csv = out_root / "validation_summary.csv"
    summary_xlsx = out_root / "validation_summary.xlsx"
    summary_df.to_csv(summary_csv, index=False)

    try:
        with pd.ExcelWriter(summary_xlsx) as w:
            summary_df.to_excel(w, index=False, sheet_name="summary")
            evt_table.to_excel(w, index=False, sheet_name="EVT_POT_table")
    except Exception as e:
        print(f"[WARN] Could not write summary Excel ({e}). CSV was written: {summary_csv.name}")

    # Save a tiny JSON meta
    meta = {
        "scipy_available": bool(SCIPY_AVAILABLE),
        "baseline": baseline,
        "seed_list": list(V.seed_list),
        "mc_sizes": list(V.mc_sizes),
        "perturb_pct": V.perturb_pct,
        "evt_q_thresh_list": list(V.evt_q_thresh_list),
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_xlsx": str(summary_xlsx),
            "evt_csv": str(evt_csv),
            "evt_xlsx": str(evt_xlsx),
        }
    }
    with open(out_root / "validation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Outputs folder:", out_root.resolve())
    print(" -", summary_csv.name)
    print(" -", summary_xlsx.name)
    print(" -", evt_csv.name)
    print(" -", evt_xlsx.name)
    if not SCIPY_AVAILABLE:
        print("[WARN] SciPy not available -> EVT table may contain NaNs.")


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    main()
