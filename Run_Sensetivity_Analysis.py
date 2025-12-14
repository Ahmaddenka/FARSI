from main.sensitivity_FCM_static_Ver02 import (
    OATConfig, RandomScreenConfig, run_both_analyses
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
    

    collective_behavior = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "group"
        and "collective_behavior" in str(n).lower()
        and "behavioral" not in str(n).lower()
    )
    collective_behavior_idx = [name_to_idx[n] for n in collective_behavior if n in name_to_idx]

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
        output_nodes=control_nodes,   
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
        warmup_steps=50,
        freeze_bsp_in_warmup=True,
        freeze_bsp_in_main=True,
        min_main_steps=10,
        patience=2,
        debug_first_main=False,
        debug_bsp_nodes=None,
        Control_Data=None
    )

    # --- 6) پیکربندی حساسیت‌ها ---
    oat_cfg = OATConfig(
        delta=0.20,
        aggregate="mean",
        save_dir=str(OUT_DIR/"Sensitivity_Outputs"),
        seed=123
    )

    screen_cfg = RandomScreenConfig(
        n_scenarios=500,
        range_type="abs",
        low=-0.2, high=0.2,
        aggregate="mean",
        save_dir=str(OUT_DIR/"Sensitivity_Outputs"),
        seed=123
    )

    res = run_both_analyses(
        graph=graph,
        fcm_matrix=fcm_matrix,
        initialize_state_fn=initialize_state_from_plan,
        simulate_fn=fcm_simulation_trace,
        init_kwargs=init_kwargs,
        sim_kwargs=sim_kwargs,
        oat_targets=BSM_idx,
        screen_factors=BSM_idx,
        oat_cfg=oat_cfg,
        screen_cfg=screen_cfg,
    )

    print("OAT  CSV  ->", res["oat_csv"])
    print("SCRN CSV  ->", res["screen_csv"])


if __name__ == "__main__":
    main()
#===============================================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt

from pathlib import Path
from pathlib import Path
OUT_DIR = Path("output")

def plot_oat_bar(
    csv_path=str(OUT_DIR/"sensitivity_outputs/oat_results.csv"),
    top_k=15,
    save_path=str(OUT_DIR/"sensitivity_outputs/fig_OAT_top_nodes.png"),
    figsize=(8, 5)
):
    """
    OAT: نودهایی که بیشترین تغییر مطلق در خروجی ایجاد کرده‌اند (abs_change_max).
    نمودار: bar افقی از top_k نود.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # مرتب‌سازی بر اساس بیشترین تغییر مطلق
    df_sorted = df.sort_values("abs_change_max", ascending=False).head(top_k)

    # اگر نام نود طولانی است، بهتره یک لیبل کوتاه هم داشته باشی؛
    # فعلاً از node_name استفاده می‌کنیم.
    labels = df_sorted["node_name"].astype(str)
    values = df_sorted["abs_change_max"]

    plt.figure(figsize=figsize)
    plt.barh(labels, values)
    plt.gca().invert_yaxis()  # بالا = نود مهم‌تر
    plt.xlabel("Absolute change in aggregated output (|Δy|)")
    plt.title(f"Local OAT sensitivity – top {top_k} nodes")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[OAT plot] saved to {save_path}")


def plot_screening_bar(
    csv_path=str(OUT_DIR/"sensitivity_outputs/randomized_screening_results.csv"),
    top_k=15,
    save_path=str(OUT_DIR/"sensitivity_outputs/fig_SCREENING_top_factors.png"),
    figsize=(8, 5)
):
    """
    Screening: اهمیت نسبی بر اساس |Spearman rho|.
    نمودار: bar افقی از top_k فاکتور.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    #df_sorted = df.sort_values("importance_abs", ascending=False).head(top_k)
    df_sorted = df.sort_values("abs_spearman_rho", ascending=False).head(top_k)

    labels = df_sorted["node_name"].astype(str)
    #values = df_sorted["importance_abs"]
    values = df_sorted["abs_spearman_rho"]

    plt.figure(figsize=figsize)
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.xlabel("|Spearman correlation with aggregated output|")
    plt.title(f"Randomized screening – top {top_k} factors")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[Screening plot] saved to {save_path}")


def main():
    plot_oat_bar(
        csv_path=str(OUT_DIR/"sensitivity_outputs/oat_results.csv"),
        top_k=15,
        save_path=str(OUT_DIR/"sensitivity_outputs/fig_OAT_top15.png"),
        figsize=(15, 10),
    )

    plot_screening_bar(
        csv_path=str(OUT_DIR/"sensitivity_outputs/randomized_screening_results.csv"),
        top_k=15,
        save_path=str(OUT_DIR/"sensitivity_outputs/fig_SCREENING_top15.png"),
        figsize=(15, 10),
    )


if __name__ == "__main__":
    main()

