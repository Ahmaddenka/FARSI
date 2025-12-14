from main.sensitivity_FCM_static_Ver01 import (
    OATConfig, RandomScreenConfig, randomized_screening
)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import unicodedata
import numpy as np
from pathlib import Path
from main.run_initialize_state import prepare_state_plan, initialize_state_from_plan
from main.run_fcm_simulation_trace import fcm_simulation_trace
from main.fuzzy_to_prob import build_probability_inputs
from main.agent_and_control_number import build_control_summary
from main.graph_to_fcm import run_graph_to_fcm
from pathlib import Path


INPUT_DIR = Path("field_data")
OUT_DIR = Path("output")

# === Helper ===
def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name.strip("_")


def clean_node_dict(prob_inputs: dict) -> dict:
    return {clean_node_name(k): v for k, v in prob_inputs.items()}


def main():
    # --- 1) ورودی‌های احتمالی BSM ---
    prob_inputs = build_probability_inputs(
        excel_path=str(INPUT_DIR/"BSM_input_DATA.xlsx"),
        sheet_name=0,
        id_col=0,
        scale_min=1.0, scale_max=5.0,
        trapezoid_fit="std",
        std_core_width=0.2,
        std_support_width=1.0,
        support_p=(0.499999, 0.50001),
        core_p=(0.499999, 0.50001),
        n_grid=1001,
        target_range=(-1.0, 1.0)
    )

    # --- 2) ساخت ماتریس FCM و گراف ---
    fcm_matrix, graph = run_graph_to_fcm()

    control_leaders = build_control_summary(str(INPUT_DIR/"Control_gorup_structure.xlsx"))

    node_names = list(graph.nodes())
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    # --- 3) انتخاب نودها (BSM و collective_behavior) ---
    control_nodes = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "control" and n not in control_leaders
    )

    BSM = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "bsm"
    )
    BSM_idx = [name_to_idx[n] for n in BSM if n in name_to_idx]

    # --- 4) اطلاعات کنترل ---
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

    # --- 5) برنامه حالت اولیه ---
    bsp_data = clean_node_dict(prob_inputs)
    plan = prepare_state_plan(
        graph,
        bsp_data=bsp_data,
        random_nodes=None,
        output_nodes=control_nodes,   # خروجی‌ها به صورت نامی
        random_keyword="random",
    )

    init_kwargs = dict(
        plan=plan,
        bsp_range=(-1.0, 1.0),
        random_range=(-0.000001,0.000001),
        intermediate_range=(-0.000001,0.000001),
        output_range=(-0.000001,0.000001),
        get_bsp_input=True,
        seed=None,          # workerها run_idx را اضافه می‌کنند
        dtype="float32"
    )

    sim_kwargs = dict(
        steps=300,
        delt=1e-3,
        control_dict=control_leaders_idx,
        control_index_map=control_index_map,
                                                 alpha =0.05153744308202851,
                                                 beta = 0.8030979544125557,
                                                 gamma = 0.6721727251459532, 
        warmup_steps=10,
        freeze_bsp_in_warmup=True,
        freeze_bsp_in_main=False,
        min_main_steps=10,
        patience=2,
        debug_first_main=False,
        debug_bsp_nodes=None,
        Control_Data=None
    )


    screen_cfg = RandomScreenConfig(
        n_scenarios=500,
        range_type="abs",
        low=0.0, high=0.4,
        aggregate="mean",
        save_dir=str(OUT_DIR/"Optimization_outputs"),
        seed=123
    )

    res = randomized_screening(
        graph=graph,
        fcm_matrix=fcm_matrix,
        initialize_state_fn=initialize_state_from_plan,
        simulate_fn=fcm_simulation_trace,
        init_kwargs=init_kwargs,
        sim_kwargs=sim_kwargs,
        factor_node_indices=BSM_idx,
        cfg=screen_cfg,
    )

if __name__ == "__main__":
    main()
#===============================================================================================


"""
Top-10 BSMs: positive-shift randomized screening
===============================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# 1) Path configuration
# -----------------------------
SCREENING_CSV = Path("output/Optimization_outputs/randomized_screening_results.csv")
#OUT_DIR = Path("Optimization_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP10_CSV = OUT_DIR / "Optimization_outputs" / "top10_BSM_positive_screening_positive_shift.csv"
TOP10_FIG = OUT_DIR / "Optimization_outputs" / "fig_top10_BSM_positive_screening_positive_shift.png"
TOP10_MD  = OUT_DIR / "Optimization_outputs" / "top10_BSM_positive_screening_table_screening_positive_shift.md"


# -----------------------------
# 2) Load screening results
# -----------------------------
df = pd.read_csv(SCREENING_CSV)


# -----------------------------
# 3) Filter to positive effects
# -----------------------------
# فقط BSMهایی که هم‌بستگی مثبت با خروجی دارند (بهبود BSM → بهبود رفتار جمعی)
df_pos = df[df["spearman_rho"] > 0].copy()

# اگر هیچ‌کدام مثبت نبودند، می‌توانی این شرط را عوض کنی یا روی |rho| کار کنی.
if df_pos.empty:
    raise RuntimeError(
        "No positive Spearman correlations found. "
        "Consider relaxing the filter or checking the screening setup."
    )

# -----------------------------
# 4) Select top-10 by rho
# -----------------------------
df_top10 = df_pos.sort_values("spearman_rho", ascending=False).head(10).reset_index(drop=True)

# ذخیره در CSV
df_top10.to_csv(TOP10_CSV, index=False)
print(f"[INFO] Saved top-10 BSMs CSV to: {TOP10_CSV}")

# -----------------------------
# 5) Make bar plot
# -----------------------------
labels = df_top10["node_name"].astype(str)
values = df_top10["spearman_rho"].astype(float)

plt.figure(figsize=(8, 5))
plt.barh(labels, values)
plt.gca().invert_yaxis()  # بالاترین مقدار در بالا
plt.xlabel("Spearman rank correlation (ρ)")
plt.title("Top-10 BSMs with positive impact on Control Nodes")
plt.tight_layout()
plt.savefig(TOP10_FIG, dpi=300)
plt.close()
print(f"[INFO] Saved bar plot to: {TOP10_FIG}")


# -----------------------------
# 6) Markdown table for paper/appendix
# -----------------------------
md_lines = []
md_lines.append("| Rank | BSM node | ρ (Spearman) | |ρ| | x_mean | x_sd |")
md_lines.append("|------|----------|--------------|------|--------|------|")

for i, row in df_top10.iterrows():
    rank = i + 1
    name = str(row["node_name"])
    rho = f"{row['spearman_rho']:.3f}"
    imp = f"{row['importance_abs']:.3f}"
    x_mean = f"{row['x_mean']:.3f}"
    x_std = f"{row['x_std']:.3f}"
    md_lines.append(f"| {rank} | {name} | {rho} | {imp} | {x_mean} | {x_std} |")

TOP10_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"[INFO] Saved markdown table to: {TOP10_MD}")

print("\n[INFO] Top-10 BSMs with positive impact on Control:")
print(df_top10[["node_index", "node_name", "spearman_rho", "importance_abs", "x_mean", "x_std"]])
