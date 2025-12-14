import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import re
import unicodedata
import numpy as np
from pathlib import Path
from main.run_mc_pipeline_Ver04 import MCPipelineConfig, run_mc_pipeline
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
    
    prob_inputs = build_probability_inputs(
        excel_path=str(INPUT_DIR/"BSM_input_DATA.xlsx"),
        sheet_name=0,
        id_col=0,
        scale_min=1.0, scale_max=5.0,
        trapezoid_fit="std",   # 'std' OR "percentile" --> std is proper when there is just one evaluator
        std_core_width=0.2,        
        std_support_width=1.0,     
        support_p=(0.05, 0.95),#(0.499999, 0.50001)
        core_p=(0.35, 0.65),#((0.499999, 0.50001)
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

    Culture =sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "group"
                        and "culture" in str(n).lower()])

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

    bsp_data = clean_node_dict(prob_inputs)
    plan = prepare_state_plan(
        graph,
        bsp_data=bsp_data,
        random_nodes=None,
        output_nodes=Culture,
        random_keyword="random",
    )

    init_kwargs = dict(
        plan=plan,
        bsp_range=(-1.0, 1.0),
        random_range=(-1.0, 1.0),
        intermediate_range=(-1.0, 1.0),
        output_range=(-1.0, 1.0),
        get_bsp_input=True,
        seed=None,         
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
        warmup_steps=30,
        freeze_bsp_in_warmup=True,
        freeze_bsp_in_main=False,
        min_main_steps=10,
        patience=2,
        debug_first_main=False,
        debug_bsp_nodes=None,
        Control_Data=None
    )

    # --- 6) تنظیمات مونت‌کارلو ---
    cfg = MCPipelineConfig()

    cfg.auto_n = False
    cfg.auto_n_params = dict(tail_samples_target=10,#1000,
                             ci=0.90,
                             min_n=5,#5000,
                             use_bonferroni=False)
    cfg.alpha = 0.05
    cfg.q_list = (0.05, 0.90, 0.95)
    cfg.tail = "left"
    cfg.evt_threshold_quantile = 0.90
    cfg.n_total = 200
    cfg.batch_size = 10#5000
    cfg.test_mode = False
    cfg.make_plots = False
    cfg.verbose = True
    cfg.save_per_node_feather = True       # Feather سریع
    cfg.post_export_excel = True          # اکسل بعد از اتمام (اختیاری)
    cfg.post_export_stats_excel = True
    cfg.seed = None
    cfg.output_dir = str(OUT_DIR/"mc_outputs_culture_analysis")

    # موازی‌سازی امن در ویندوز با اشتراک CSR 
    cfg.n_workers = 5             # تعداد هسته‌ها 
    cfg.use_shared_csr = True
    cfg.mp_chunksize = 64

    # استریم دیتاست (کاهش RAM) – اختیاری:
    cfg.stream_dataset = False     # اگر RAM محدود است → True
    cfg.stream_every = 20000

    # نتایج float32 – اختیاری:
    cfg.float32_results = True

    # فقط آمار (ننوشتن mc_dataset.csv) – اختیاری:
    cfg.stats_only = False

    # ریپورت پیشرفت هر 5%:
    cfg.progress_report = True
    cfg.progress_interval = 0.01

    # --- 7) اجرای پایپلاین مونت‌کارلو ---
#=========Debug=============================================================================================================
    print("controls total:", sum(1 for _,d in graph.nodes(data=True) if str(d.get("level","")).lower()=="control"))
    print("leaders:", len(control_leaders))
    print("non-leader controls (outputs):", len(control_nodes))
    
    if len(control_nodes) == 0:
        control_nodes = sorted(n for n,d in graph.nodes(data=True)
                               if str(d.get("level","")).lower()=="control")
        print("[warn] non-leader controls not found → using ALL control nodes as outputs:", len(control_nodes))
    
    from main.run_initialize_state import initialize_state_from_plan
    init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_from_plan(**init_kwargs)
    print("init check → outputs:", len(output_nodes))
#=========================================================================================================================
    out = run_mc_pipeline(
        graph=graph,
        fcm_matrix=fcm_matrix,
        initialize_state_fn=initialize_state_from_plan,
        simulate_fn=fcm_simulation_trace,
        init_kwargs=init_kwargs,
        sim_kwargs=sim_kwargs,
        cfg=cfg
    )

    print("\nMonte Carlo finished.")
    for k, v in out.items():
        print(f"  {k}: {v}")

    print("\nFiles saved in:", Path(cfg.output_dir).resolve())


if __name__ == "__main__":
    # ویندوز: محافظ الزامی برای multiprocessing
    import sys
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    main()
##========================================================================================================================




import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

def plot_mc_results(
    dataset_path,
    stats_path=None,
    bins=60,
    save=True,
    show=True,
    xlim=None,                 # ← بازهٔ دلخواه محور X، مثل (-1, 1) یا (a, b)
    add_bg_gradient=True       # ← پس‌زمینهٔ قرمز→زرد→سبز
):
    """
    Plot histogram and statistical markers from saved Monte Carlo results.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the saved dataset CSV (must contain a single column 'y').
    stats_path : str or Path, optional
        Path to the stats CSV (if provided, markers like mean/median/VaR are shown).
    bins : int, default=60
        Number of histogram bins.
    save : bool, default=True
        Whether to save the figure as 'hist_result.png' next to the dataset.
    show : bool, default=True
        Whether to display the plot interactively.
    xlim : (float, float) or None
        If provided, sets the x-axis range and limits the histogram to this range.
    add_bg_gradient : bool
        If True, adds a horizontal background gradient from red (left) to green (right).
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # --- Load dataset ---
    df = pd.read_csv(dataset_path)
    if "y" not in df.columns:
        raise ValueError("Dataset must have a single column named 'y'.")
    y = df["y"].dropna().values.astype(float)

    if xlim is not None:
        x_min, x_max = float(xlim[0]), float(xlim[1])
        mask = (y >= x_min) & (y <= x_max)
        y = y[mask]

    # --- Load statistics if available ---
    stats = {}
    if stats_path and Path(stats_path).exists():
        s = pd.read_csv(stats_path)
        if len(s) > 0:
            stats = s.iloc[0].to_dict()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    # --- Histogram ---
    hist_kwargs = dict(bins=bins, density=False, alpha=0.6, color="#4c72b0", edgecolor="white")
    if xlim is not None:
        hist_kwargs["range"] = (x_min, x_max)
    ax.hist(y, **hist_kwargs)

    # --- Add background gradient (red→yellow→green) along X ---
    if add_bg_gradient:
        # محدودهٔ X و Y روی محور فعلی
        cur_xlim = ax.get_xlim() if xlim is None else xlim
        cur_ylim = ax.get_ylim()
        # گرادیان افقی: یک تصویر 1×W که در محور X از 0→1 تغییر می‌کند
        W = 600
        grad = np.linspace(0, 1, W).reshape(1, W)  # 0 = left, 1 = right
        cmap = LinearSegmentedColormap.from_list("ryg_soft", [(1, 0, 0), (1, 1, 0), (0, 0.6, 0)], N=256)
        ax.imshow(
            grad,
            extent=(-1, 1, cur_ylim[0], cur_ylim[1]),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            alpha=0.25,
            zorder=0
        )
        ax.set_xlim(cur_xlim)
        ax.set_ylim(cur_ylim)

    # --- Add statistical markers (if available) ---
    def _draw_line(val, label, color, lw=2, style="--"):
        if val is None:
            return
        try:
            v = float(val)
        except Exception:
            return
        if np.isfinite(v):
            ax.axvline(v, color=color, linestyle=style, linewidth=lw, label=label)

    _draw_line(stats.get("mean"), "Mean", "#ffb000")
    _draw_line(stats.get("q_95"), "q95", "#ee7733")
    _draw_line(stats.get("q_5"), "q5", "#0077bb")
    # CVaR و EVT-CVaR
    alpha_val = stats.get('alpha', 0.05)
    #_draw_line(stats.get("CVaR_alpha"), f"CVaR(α={alpha_val})", "#009988")
    _draw_line(stats.get("evt_CVaR"), "EVT CVaR", "#882255", style=":")

    # --- Aesthetics ---
    ax.set_title("Aggregated Monte Carlo Output Distribution", fontsize=13)
    ax.set_xlabel("Output Value (Safety_Culture)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()

    # --- Save & Show ---
    out_path = dataset_path.parent / "hist_result_density.png"
    if save:
        plt.savefig(out_path, dpi=300)
        print(f"[Saved] Histogram → {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


plot_mc_results(
     dataset_path="output/mc_outputs_culture_analysis/mc_dataset.csv",
     stats_path="output/mc_outputs_culture_analysis/mc_stats.csv",
     bins=100,
     xlim=(-1, 1),              # ← بازهٔ دلخواه
     add_bg_gradient=True,      # ← گرادیان پس‌زمینه
     save=True,
    show=True
 )
#=============================================================================================================
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
        trapezoid_fit="percentile",
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
    

    Culture = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "group"
        and "culture" in str(n).lower()
        #and "behavioral" not in str(n).lower()
    )


    Culture_idx = [name_to_idx[n] for n in Culture if n in name_to_idx]

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
        output_nodes=Culture,   # خروجی‌ها به صورت نامی
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
        warmup_steps=30,
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
        save_dir=str(OUT_DIR/"Culture_Sensitivity_Outputs"),
        seed=123
    )

    screen_cfg = RandomScreenConfig(
        n_scenarios=50,
        range_type="abs",
        low=-0.2, high=0.2,
        aggregate="mean",
        save_dir=str(OUT_DIR/"Culture_Sensitivity_Outputs"),
        seed=123
    )

    # --- 7) اجرای دو تحلیل (بدون مونت‌کارلو) ---
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

#========================================================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt

from pathlib import Path
OUT_DIR = Path("output")

def plot_oat_bar(
    csv_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/oat_results.csv"),
    top_k=15,
    save_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/fig_OAT_top_nodes.png"),
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
    csv_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/randomized_screening_results.csv"),
    top_k=15,
    save_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/fig_SCREENING_top_factors.png"),
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
        csv_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/oat_results.csv"),
        top_k=15,
        save_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/fig_OAT_top15.png"),
        figsize=(15, 10),
    )

    plot_screening_bar(
        csv_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/randomized_screening_results.csv"),
        top_k=15,
        save_path=str(OUT_DIR/"Culture_Sensitivity_Outputs/fig_SCREENING_top15.png"),
        figsize=(15, 10),
    )


if __name__ == "__main__":
    main()
#=====================================================================================================

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
        trapezoid_fit="percentile",
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
    Random_Root = sorted(
        n for n, d in graph.nodes(data=True)
        if (str(d.get("level", "")).lower() == "group" or  str(d.get("level", "")).lower() == "agent")
        and "random" in str(n).lower()      
    )
    Random_Root = [name_to_idx[n] for n in Random_Root if n in name_to_idx]
    Root_idx = Random_Root + BSM_idx
    Culture = sorted(
        n for n, d in graph.nodes(data=True)
        if str(d.get("level", "")).lower() == "group"
        and "culture" in str(n).lower()
        #and "behavioral" not in str(n).lower()
    )


    Culture_idx = [name_to_idx[n] for n in Culture if n in name_to_idx]

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
        output_nodes=Culture,   # خروجی‌ها به صورت نامی
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
        warmup_steps=30,
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
        save_dir=str(OUT_DIR/"Culture_Optimization_outputs"),
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
#=================================================================================

"""
Top-10 BSMs: positive-shift randomized screening
===============================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# 1) Path configuration
# -----------------------------
SCREENING_CSV = Path("output/Culture_Optimization_outputs/randomized_screening_results.csv")
#OUT_DIR = Path("Culture_Optimization_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP10_CSV = OUT_DIR / "Culture_Optimization_outputs" / "top10_BSM_positive_screening_positive_shift.csv"
TOP10_FIG = OUT_DIR / "Culture_Optimization_outputs" / "fig_top10_BSM_positive_screening_positive_shift.png"
TOP10_MD  = OUT_DIR / "Culture_Optimization_outputs" / "top10_BSM_positive_screening_table_screening_positive_shift.md"


# -----------------------------
# 2) Load screening results
# -----------------------------
df = pd.read_csv(SCREENING_CSV)


# -----------------------------
# 3) Filter to positive effects
# -----------------------------
df_pos = df[df["spearman_rho"] > 0].copy()

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
plt.xlabel("Spearman rank correlation (ρ) with Culture")
plt.title("Top-10 BSMs with positive impact on Culture")
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

print("\n[INFO] Top-10 BSMs with positive impact on Culture:")
print(df_top10[["node_index", "node_name", "spearman_rho", "importance_abs", "x_mean", "x_std"]])

