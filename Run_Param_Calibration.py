"""
GA calibration for (alpha, beta, gamma) using TRAIN split of field (expert) data, and MEDIAN-based model summary.

Updates vs previous:
- Adds gamma to GA (range 0.1..2.0) and passes it to Simulation_Run (new signature).
- Keeps 10% Test split held out from calibration; reports metrics separately at the end.
- Uses median (not mode).
- Adds tail-risk penalties to avoid solutions that push the left tail towards -1 (which can break EVT-CVaR).

NOTE:
- Tail penalties are computed empirically (finite). This is intentional to avoid EVT-fit instability during optimization.
"""

import numpy as np
import pandas as pd
import unicodedata
import re

from main.run_initialize_state import prepare_state_plan
from main.Run_Simulation_Test import Simulation_Run  # <-- your updated Simulation_Run has gamma :contentReference[oaicite:1]{index=1}
from main.fuzzy_to_prob import build_probability_inputs
from main.agent_and_control_number import build_control_summary
from main.graph_to_fcm import run_graph_to_fcm
from pathlib import Path


INPUT_DIR = Path("field_data")
OUT_DIR = Path("output")
# =======================
# Global settings
# =======================

# Search range for alpha, beta, gamma
ALPHA_MIN, ALPHA_MAX = 0.0, 2.0
BETA_MIN,  BETA_MAX  = 0.0, 2.0
GAMMA_MIN, GAMMA_MAX = 0.1, 2.0

# Monte Carlo and GA settings
N_MC           = 40
POP_SIZE       = 18          # slightly larger due to 3D search
N_GENERATIONS  = 18          # slightly larger due to 3D search

ELITE_FRACTION  = 0.2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE  = 0.9
MUTATION_RATE   = 0.85
MUTATION_STD    = 0.18

# Simulation settings
STEPS_MAX = 300

# Median window
MEDIAN_BANDWIDTH = 0.1

# Train/Test split settings
TEST_FRACTION = 0.10
SPLIT_SEED    = 42

# ---- Tail risk control (IMPORTANT) ----
# We punish solutions that produce too-negative left-tail behavior (near -1).
# Tune these depending on your EVT setup and acceptable risk tolerance.
CVAR_ALPHA_LEFT = 0.01   # 1% left-tail CVaR
TAIL_CVAR_FLOOR = -0.98  # we want CVaR_left >= -0.98 (avoid hugging -1 too much)
LOWER_BOUND_EPS = 1e-3   # "near -1" means <= (-1 + eps)
MAX_FRAC_NEAR_LOWER = 0.02  # allow max 2% of samples near -1

# Loss weights (tunable)
W_RMSE            = 1.0
W_MAE             = 0.5
W_BIAS_ABS        = 0.3
W_SPEARMAN_PEN    = 0.3   # penalty term: (1 - clip(r_s,0,1))
W_COVERAGE_PEN    = 0.7   # penalty term: (1 - coverage)

# Tail penalties:
W_TAIL_CVAR_PEN   = 1.2   # penalize too-negative CVaR_left
W_TAIL_NEAR_PEN   = 1.2   # penalize too many samples near -1

W_FRAC_MAX_STEPS  = 1.0
W_FRAC_INVALID    = 1.0
LAMBDA_REG        = 0.01  # reg near (1,1,0.5)

FACE_EXCEL_PATH = str(INPUT_DIR/"Face_Validation_Data.xlsx")
FACE_SHEET_NAME = 0  # adjust if needed


# =======================
# Utility: clean node names
# =======================

def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name.strip("_")


def clean_node_dict(prob_inputs):
    cleaned = {}
    for key, value in prob_inputs.items():
        new_key = clean_node_name(key)
        cleaned[new_key] = value
    return cleaned


# =======================
# Convert full_state_trace → numpy array
# =======================

def full_state_to_array(full_state_trace, context):
    """
    Convert full_state_trace (DataFrame, ndarray, or list-of-dicts/list-of-lists)
    into a float numpy array (T x N). Does NOT raise if NaN/Inf is present.
    """
    import pandas as pd

    if isinstance(full_state_trace, pd.DataFrame):
        return full_state_trace.to_numpy(dtype=float)
    if isinstance(full_state_trace, np.ndarray):
        return full_state_trace.astype(float)

    seq = list(full_state_trace)
    if len(seq) == 0:
        return np.empty((0, 0), dtype=float)

    first = seq[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        if all(isinstance(k, (int, np.integer)) for k in keys):
            max_idx = max(keys)
            n_nodes = max_idx + 1
            arr = np.full((len(seq), n_nodes), np.nan, float)
            for t, step_state in enumerate(seq):
                for idx, val in step_state.items():
                    arr[t, int(idx)] = float(val)
            return arr
        else:
            node_names = context["node_names"]
            name_to_pos = {name: i for i, name in enumerate(node_names)}
            n_nodes = len(node_names)
            arr = np.full((len(seq), n_nodes), np.nan, float)
            for t, step_state in enumerate(seq):
                for key, val in step_state.items():
                    if key in name_to_pos:
                        j = name_to_pos[key]
                        arr[t, j] = float(val)
            return arr

    return np.array(seq, dtype=float)


# =======================
# Train/Test split
# =======================

def split_face_train_test(face_df, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    face_df = face_df.reset_index(drop=True).copy()
    n = len(face_df)
    n_test = max(1, int(round(test_fraction * n)))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    face_test = face_df.iloc[test_idx].reset_index(drop=True)
    face_train = face_df.iloc[train_idx].reset_index(drop=True)
    return face_train, face_test


# =======================
# Build simulation context
# =======================

def build_simulation_context():
    prob_inputs = build_probability_inputs(
        excel_path=str(INPUT_DIR/"BSM_input_DATA.xlsx"),
        sheet_name=0,
        id_col=0,
        scale_min=1.0, scale_max=5.0,
        trapezoid_fit="percentile",
        support_p=(0.499999, 0.50001),
        core_p=(0.499999, 0.50001),
        n_grid=1001,
        target_range=(-1.0, 1.0),
    )

    fcm_matrix, graph = run_graph_to_fcm()

    Control_Leader_Nodes = build_control_summary(str(INPUT_DIR/"Control_gorup_structure.xlsx"))
    control_nodes = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "control"
        and n not in Control_Leader_Nodes
    )

    bsp_data = clean_node_dict(prob_inputs)

    plan = prepare_state_plan(
        graph,
        bsp_data=bsp_data,
        random_nodes=None,
        output_nodes=control_nodes,
        random_keyword="random",
    )

    node_names = list(graph.nodes())
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    Control_Leader_Nodes_idx = {}
    for name, vals in Control_Leader_Nodes.items():
        i = name_to_idx[name]
        Control_Leader_Nodes_idx[i] = {
            "agent_count": float(vals.get("agent_count", 1.0)),
            "control_count": float(vals.get("control_count", 0.0)),
        }

    control_index_map = {
        name_to_idx[n]
        for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "control"
        and n in Control_Leader_Nodes
    }

    face = pd.read_excel(FACE_EXCEL_PATH, sheet_name=FACE_SHEET_NAME)
    if "Control Nodes" not in face.columns or "Expert Evaluation Results" not in face.columns:
        raise ValueError("Face_Validation_Data.xlsx must have: 'Control Nodes', 'Expert Evaluation Results'.")
    # --- PUBLIC-REPO GUARD: stop if confidential/non-numeric face-validation data exists ---
    col = "Expert Evaluation Results"
    s = face[col].astype(str).str.strip().str.lower()

    if (s == "confidential").any():
        print(
            "[INFO] Run_Param_Calibration.py requires confidential organizational evaluation data.\n"
            "[INFO] In the public repository, these values are removed/masked (e.g., 'confidential').\n"
            "[INFO] Please provide the real numeric dataset locally to run calibration."
        )
        raise SystemExit(0)

    numeric = pd.to_numeric(face[col], errors="coerce")
    if numeric.isna().any():
        print(
            "[INFO] Calibration requires numeric values in 'Expert Evaluation Results'.\n"
            "[INFO] Some values are non-numeric or missing in the public dataset.\n"
            "[INFO] Provide the real numeric dataset locally to run calibration."
        )
        raise SystemExit(0)

    face_train, face_test = split_face_train_test(face, test_fraction=TEST_FRACTION, seed=SPLIT_SEED)

    return {
        "graph": graph,
        "fcm_matrix": fcm_matrix,
        "plan": plan,
        "control_nodes": control_nodes,
        "node_names": node_names,
        "name_to_idx": name_to_idx,
        "control_leader_dict": Control_Leader_Nodes_idx,
        "control_index_map": control_index_map,
        "face_train": face_train,
        "face_test": face_test,
    }


# =======================
# Run one simulation
# =======================

def simulate_once(context, alpha, beta, gamma):
    full_state_trace, output_nodes = Simulation_Run(
        fcm_matrix=context["fcm_matrix"],
        plan=context["plan"],
        control_dict=context["control_leader_dict"],
        control_index_map=context["control_index_map"],
        bsp_val_band=(0.9999, 1.0),
        random_val_band=(-1.0, 1.0),
        raw_val_band=(-1.0, 1.0),
        intermediate_range_band=(-1.0, 1.0),
        get_bsp_input=True,
        steps=STEPS_MAX,
        delt=1e-3,
        alpha=alpha,
        beta=beta,
        gamma=gamma,   # <-- new parameter :contentReference[oaicite:2]{index=2}
        warmup_steps=50,
        freeze_roots_in_warmup=True,
        freeze_roots_in_main=False,
        min_main_steps=10,
        patience=3,
        debug_first_main=False,
        boundary_test=None,
    )

    arr = full_state_to_array(full_state_trace, context)
    n_steps = arr.shape[0] if (arr.ndim == 2 and arr.size > 0) else len(full_state_trace)
    return arr, output_nodes, n_steps


# =======================
# Median estimation
# =======================

def estimate_median_and_p(samples, bandwidth=MEDIAN_BANDWIDTH):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.nan, 0.0
    med = float(np.median(samples))
    p_med = float(np.mean(np.abs(samples - med) <= bandwidth))
    return med, p_med


# =======================
# Tail metrics (empirical, finite)
# =======================

def empirical_cvar_left(samples, alpha=CVAR_ALPHA_LEFT):
    """
    Empirical left-tail CVaR (Expected Shortfall):
      CVaR_alpha = mean of the worst alpha-fraction of samples (lowest values)
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return np.nan
    s_sorted = np.sort(s)
    k = int(np.ceil(alpha * n))
    k = max(1, min(k, n))
    return float(np.mean(s_sorted[:k]))


def frac_near_lower_bound(samples, eps=LOWER_BOUND_EPS):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    return float(np.mean(s <= (-1.0 + eps)))


# =======================
# Run MC for one (alpha, beta, gamma)
# =======================

def run_mc_for_params(context, alpha, beta, gamma, n_mc=N_MC):
    name_to_idx = context["name_to_idx"]
    control_nodes = context["control_nodes"]

    values_per_node = {n: [] for n in control_nodes}
    steps_list = []
    invalid_runs = 0

    for _ in range(n_mc):
        arr, _, n_steps = simulate_once(context, alpha, beta, gamma)
        steps_list.append(float(n_steps))

        if arr.size == 0:
            invalid_runs += 1
            continue
        if not np.isfinite(arr).all():
            invalid_runs += 1
            continue

        final_state = arr[-1, :]
        for n in control_nodes:
            idx = name_to_idx[n]
            val = final_state[idx]
            if np.isfinite(val):
                values_per_node[n].append(float(val))

    # per-node summaries
    rows = []
    cvar_list = []
    near_list = []

    for n in control_nodes:
        vals = np.array(values_per_node[n], dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            median = np.nan
            p_median = 0.0
            mean_val = np.nan
            q5 = np.nan
            q95 = np.nan
            cvar_l = np.nan
            frac_near = np.nan
        else:
            median, p_median = estimate_median_and_p(vals, bandwidth=MEDIAN_BANDWIDTH)
            mean_val = float(np.mean(vals))
            q5 = float(np.percentile(vals, 5))
            q95 = float(np.percentile(vals, 95))
            cvar_l = empirical_cvar_left(vals, alpha=CVAR_ALPHA_LEFT)
            frac_near = frac_near_lower_bound(vals, eps=LOWER_BOUND_EPS)

        if np.isfinite(cvar_l):
            cvar_list.append(cvar_l)
        if np.isfinite(frac_near):
            near_list.append(frac_near)

        rows.append({
            "node": n,
            "median": median,
            "p_median": p_median,
            "mean": mean_val,
            "q_5": q5,
            "q_95": q95,
            "cvar_left": cvar_l,
            "frac_near_lower": frac_near,
        })

    mc_df = pd.DataFrame(rows)

    steps_arr = np.array(steps_list, dtype=float)
    mean_steps = float(np.mean(steps_arr)) if steps_arr.size else np.nan
    frac_max_steps = float(np.mean(steps_arr >= STEPS_MAX)) if steps_arr.size else np.nan
    frac_invalid = float(invalid_runs / max(n_mc, 1))

    # "worst-case" tail behavior across nodes
    cvar_left_min = float(np.min(cvar_list)) if len(cvar_list) else np.nan
    frac_near_lower_max = float(np.max(near_list)) if len(near_list) else np.nan

    stability = {
        "mean_steps": mean_steps,
        "frac_max_steps": frac_max_steps,
        "frac_invalid": frac_invalid,
        "cvar_left_min": cvar_left_min,
        "frac_near_lower_max": frac_near_lower_max,
    }
    return mc_df, stability


# =======================
# Validation metrics (median-based)
# =======================

def _try_spearman_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        from scipy.stats import spearmanr, pearsonr
        sr, sp = spearmanr(x, y)
        pr, pp = pearsonr(x, y)
        return float(sr), float(sp), float(pr), float(pp)
    except Exception:
        def _pearson(a, b):
            a = np.asarray(a, float) - np.mean(a)
            b = np.asarray(b, float) - np.mean(b)
            denom = (np.sqrt(np.sum(a*a)) * np.sqrt(np.sum(b*b)))
            return np.nan if denom == 0 else float(np.sum(a*b) / denom)

        pr = _pearson(x, y)
        rx = pd.Series(x).rank(method="average").to_numpy(float)
        ry = pd.Series(y).rank(method="average").to_numpy(float)
        sr = _pearson(rx, ry)
        return float(sr), np.nan, float(pr), np.nan


def compute_validation_metrics(mc_df, face_df):
    df = face_df.merge(mc_df, left_on="Control Nodes", right_on="node", how="inner")
    n_total = int(len(face_df))

    if len(df) == 0:
        return {k: np.nan for k in [
            "spearman_r","spearman_p","pearson_r","pearson_p","ME","MAE","RMSE","coverage_q5_q95",
            "n_total","n_used"
        ]} | {"n_total": n_total, "n_used": 0}

    df["field_percent"] = df["Expert Evaluation Results"].astype(float)
    df["field_norm"] = (df["field_percent"] / 50.0) - 1.0

    mask = np.isfinite(df["median"].to_numpy(float)) & np.isfinite(df["field_norm"].to_numpy(float))
    dfv = df[mask].copy()
    n_used = int(len(dfv))

    if n_used == 0:
        return {k: np.nan for k in [
            "spearman_r","spearman_p","pearson_r","pearson_p","ME","MAE","RMSE","coverage_q5_q95"
        ]} | {"n_total": n_total, "n_used": 0}

    pred = dfv["median"].to_numpy(float)
    truth = dfv["field_norm"].to_numpy(float)
    diff = pred - truth

    ME = float(np.mean(diff))
    MAE = float(np.mean(np.abs(diff)))
    RMSE = float(np.sqrt(np.mean(diff * diff)))

    q5 = dfv["q_5"].to_numpy(float)
    q95 = dfv["q_95"].to_numpy(float)
    cov_mask = np.isfinite(q5) & np.isfinite(q95)
    coverage_q5_q95 = float(np.mean((truth[cov_mask] >= q5[cov_mask]) & (truth[cov_mask] <= q95[cov_mask]))) if np.any(cov_mask) else np.nan

    spearman_r, spearman_p, pearson_r, pearson_p = _try_spearman_pearson(pred, truth)

    return {
        "n_total": n_total,
        "n_used": n_used,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "ME": ME,
        "MAE": MAE,
        "RMSE": RMSE,
        "coverage_q5_q95": coverage_q5_q95,
    }


# =======================
# Loss function (TRAIN only)
# =======================

def compute_loss(alpha, beta, gamma, val_metrics, stability_metrics):
    rmse = val_metrics.get("RMSE", np.nan)
    mae = val_metrics.get("MAE", np.nan)
    me = val_metrics.get("ME", np.nan)
    spearman_r = val_metrics.get("spearman_r", np.nan)
    coverage = val_metrics.get("coverage_q5_q95", np.nan)

    frac_max = stability_metrics.get("frac_max_steps", np.nan)
    frac_invalid = stability_metrics.get("frac_invalid", np.nan)

    cvar_left_min = stability_metrics.get("cvar_left_min", np.nan)
    frac_near_lower_max = stability_metrics.get("frac_near_lower_max", np.nan)

    needed = [rmse, mae, me, spearman_r, coverage, frac_max, frac_invalid, cvar_left_min, frac_near_lower_max]
    if any(np.isnan(x) for x in needed):
        return 1e6

    corr_pen = 1.0 - float(np.clip(spearman_r, 0.0, 1.0))
    cov_pen = 1.0 - float(np.clip(coverage, 0.0, 1.0))

    # Tail penalties:
    # 1) CVaR_left should not be too negative (avoid mass at -1)
    tail_cvar_pen = max(0.0, (TAIL_CVAR_FLOOR - cvar_left_min)) ** 2

    # 2) Too many samples near -1 is bad
    tail_near_pen = max(0.0, (frac_near_lower_max - MAX_FRAC_NEAR_LOWER)) ** 2

    loss_validity = (
        W_RMSE * rmse +
        W_MAE * mae +
        W_BIAS_ABS * abs(me) +
        W_SPEARMAN_PEN * corr_pen +
        W_COVERAGE_PEN * cov_pen +
        W_TAIL_CVAR_PEN * tail_cvar_pen +
        W_TAIL_NEAR_PEN * tail_near_pen
    )

    loss_steps = W_FRAC_MAX_STEPS * (float(frac_max) ** 2)
    loss_invalid = W_FRAC_INVALID * (float(frac_invalid) ** 2)

    # Regularization near defaults (1,1,0.5)
    reg = (float(alpha) - 1.0) ** 2 + (float(beta) - 1.0) ** 2 + (float(gamma) - 0.5) ** 2

    return float(loss_validity + loss_steps + loss_invalid + LAMBDA_REG * reg)


# =======================
# Evaluation wrapper for GA
# =======================

def evaluate_params(alpha, beta, gamma, context, cache=None, n_mc=N_MC):
    key = (round(float(alpha), 4), round(float(beta), 4), round(float(gamma), 4))
    if cache is not None and key in cache:
        return cache[key]

    try:
        mc_df, stab = run_mc_for_params(context, alpha, beta, gamma, n_mc=n_mc)
        val_train = compute_validation_metrics(mc_df, context["face_train"])
        loss = compute_loss(alpha, beta, gamma, val_train, stab)
    except Exception as e:
        print(f"  [WARN] evaluate_params failed for a={alpha:.3f}, b={beta:.3f}, g={gamma:.3f}: {e}")
        loss = 1e6
        val_train = {k: np.nan for k in ["RMSE","MAE","ME","spearman_r","coverage_q5_q95"]}
        stab = {k: np.nan for k in ["frac_max_steps","frac_invalid","cvar_left_min","frac_near_lower_max"]}

    result = (loss, val_train, stab)
    if cache is not None:
        cache[key] = result
    return result


# =======================
# GA components
# =======================

def init_population():
    pop = []
    for _ in range(POP_SIZE):
        alpha = np.random.uniform(ALPHA_MIN, ALPHA_MAX)
        beta  = np.random.uniform(BETA_MIN,  BETA_MAX)
        gamma = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
        pop.append({"alpha": alpha, "beta": beta, "gamma": gamma})
    return pop


def tournament_select(pop):
    cand_idx = np.random.choice(len(pop), size=min(TOURNAMENT_SIZE, len(pop)), replace=False)
    candidates = [pop[i] for i in cand_idx]
    best = min(candidates, key=lambda ind: ind["loss"])
    return best


def crossover(p1, p2):
    if np.random.rand() > CROSSOVER_RATE:
        return (
            {"alpha": p1["alpha"], "beta": p1["beta"], "gamma": p1["gamma"]},
            {"alpha": p2["alpha"], "beta": p2["beta"], "gamma": p2["gamma"]},
        )

    w = np.random.rand()
    c1 = {
        "alpha": w * p1["alpha"] + (1 - w) * p2["alpha"],
        "beta":  w * p1["beta"]  + (1 - w) * p2["beta"],
        "gamma": w * p1["gamma"] + (1 - w) * p2["gamma"],
    }
    c2 = {
        "alpha": w * p2["alpha"] + (1 - w) * p1["alpha"],
        "beta":  w * p2["beta"]  + (1 - w) * p1["beta"],
        "gamma": w * p2["gamma"] + (1 - w) * p1["gamma"],
    }
    return c1, c2


def mutate(ind):
    if np.random.rand() < MUTATION_RATE:
        ind["alpha"] += np.random.normal(0.0, MUTATION_STD)
        ind["beta"]  += np.random.normal(0.0, MUTATION_STD)
        ind["gamma"] += np.random.normal(0.0, MUTATION_STD)

        ind["alpha"] = float(np.clip(ind["alpha"], ALPHA_MIN, ALPHA_MAX))
        ind["beta"]  = float(np.clip(ind["beta"],  BETA_MIN,  BETA_MAX))
        ind["gamma"] = float(np.clip(ind["gamma"], GAMMA_MIN, GAMMA_MAX))
    return ind


# =======================
# GA main loop
# =======================

def genetic_algorithm_calibration():
    context = build_simulation_context()
    cache = {}

    print("=== Face data split ===")
    print(f"Total rows: {len(context['face_train']) + len(context['face_test'])}")
    print(f"Train rows: {len(context['face_train'])}")
    print(f"Test  rows: {len(context['face_test'])}")
    print(f"Test fraction target: {TEST_FRACTION:.2f}, seed={SPLIT_SEED}")

    population = init_population()
    global_best = None
    history_rows = []
    eval_counter = 0

    for gen in range(N_GENERATIONS):
        print(f"\n===== Generation {gen+1}/{N_GENERATIONS} =====")

        for ind in population:
            loss, val_train, stab = evaluate_params(ind["alpha"], ind["beta"], ind["gamma"], context, cache=cache, n_mc=N_MC)
            ind["loss"] = loss
            ind["val_train"] = val_train
            ind["stab"] = stab

            if (global_best is None) or (loss < global_best["loss"]):
                global_best = {
                    "alpha": float(ind["alpha"]),
                    "beta": float(ind["beta"]),
                    "gamma": float(ind["gamma"]),
                    "loss": float(loss),
                    "generation": int(gen),
                    "RMSE_train": float(val_train.get("RMSE", np.nan)),
                    "MAE_train": float(val_train.get("MAE", np.nan)),
                    "ME_train": float(val_train.get("ME", np.nan)),
                    "spearman_r_train": float(val_train.get("spearman_r", np.nan)),
                    "coverage_q5_q95_train": float(val_train.get("coverage_q5_q95", np.nan)),
                    "frac_invalid": float(stab.get("frac_invalid", np.nan)),
                    "frac_max_steps": float(stab.get("frac_max_steps", np.nan)),
                    "cvar_left_min": float(stab.get("cvar_left_min", np.nan)),
                    "frac_near_lower_max": float(stab.get("frac_near_lower_max", np.nan)),
                }

            eval_counter += 1
            print(
                f"eval {eval_counter:4d}: a={ind['alpha']:.3f}, b={ind['beta']:.3f}, g={ind['gamma']:.3f} | "
                f"loss={loss:.4f}, best={global_best['loss']:.4f} | "
                f"RMSE={val_train.get('RMSE', np.nan):.3f}, "
                f"SpR={val_train.get('spearman_r', np.nan):.3f}, "
                f"Cov={val_train.get('coverage_q5_q95', np.nan):.3f}, "
                f"CVaRmin={stab.get('cvar_left_min', np.nan):.3f}, "
                f"NearMax={stab.get('frac_near_lower_max', np.nan):.3f}"
            )

        population.sort(key=lambda ind: ind["loss"])
        best_gen = population[0]
        vt = best_gen["val_train"]
        st = best_gen["stab"]
        print(
            f"--> Best gen {gen}: loss={best_gen['loss']:.4f}, "
            f"a={best_gen['alpha']:.3f}, b={best_gen['beta']:.3f}, g={best_gen['gamma']:.3f} | "
            f"RMSE={vt.get('RMSE', np.nan):.3f}, "
            f"MAE={vt.get('MAE', np.nan):.3f}, "
            f"ME={vt.get('ME', np.nan):.3f}, "
            f"SpR={vt.get('spearman_r', np.nan):.3f}, "
            f"Cov={vt.get('coverage_q5_q95', np.nan):.3f}, "
            f"CVaRmin={st.get('cvar_left_min', np.nan):.3f}, "
            f"NearMax={st.get('frac_near_lower_max', np.nan):.3f}"
        )

        history_rows.append(global_best.copy())

        n_elite = max(1, int(ELITE_FRACTION * POP_SIZE))
        elites = population[:n_elite]
        new_pop = [{"alpha": e["alpha"], "beta": e["beta"], "gamma": e["gamma"]} for e in elites]

        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        population = new_pop

    df_hist = pd.DataFrame(history_rows)
    df_hist.to_csv(str(OUT_DIR/"alpha_beta_gamma_calibration_GA_median_train_history.csv"), index=False)

    print("\n=== GA calibration finished (optimized on TRAIN) ===")
    print("Global best found (TRAIN loss):")
    print(global_best)

    # ===== Final evaluation on TRAIN and TEST using best params =====
    best_alpha = float(global_best["alpha"])
    best_beta  = float(global_best["beta"])
    best_gamma = float(global_best["gamma"])

    mc_df_best, stab_best = run_mc_for_params(context, best_alpha, best_beta, best_gamma, n_mc=N_MC)
    train_metrics = compute_validation_metrics(mc_df_best, context["face_train"])
    test_metrics  = compute_validation_metrics(mc_df_best, context["face_test"])

    print("\n=== Final validation (TRAIN) [median-based] ===")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")
    print("\n=== Tail/stability (TRAIN+TEST shared MC run) ===")
    for k, v in stab_best.items():
        print(f"{k}: {v}")

    print("\n=== Final validation (TEST) [median-based] ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    final_rows = []
    for split_name, m in [("train", train_metrics), ("test", test_metrics)]:
        row = {"split": split_name, "alpha": best_alpha, "beta": best_beta, "gamma": best_gamma, **m, **stab_best}
        final_rows.append(row)

    pd.DataFrame(final_rows).to_csv(str(OUT_DIR/"alpha_beta_gamma_calibration_final_train_test_metrics.csv"), index=False)

    return df_hist, global_best, train_metrics, test_metrics


if __name__ == "__main__":
    genetic_algorithm_calibration()
