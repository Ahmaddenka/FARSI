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
        trapezoid_fit="std",   # 'std' OR "percentile" --> std is proper when there is just one Evaluation Value
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

    collective_behavior =sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "group"
                        and "collective_behavior" in str(n).lower()
                        and not "behavioral" in str(n).lower()])

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
        output_nodes=control_nodes,
        random_keyword="random",
    )

    init_kwargs = dict(
        plan=plan,
        bsp_range=(-1.0, 1.0),
        random_range=(-1.0, 1.0),
        intermediate_range=(-1.0, 1.0),
        output_range=(-1.0, 1.0),
        get_bsp_input=True,
        seed=None,          # برای تکرارپذیری؛ workerها run_idx را به آن اضافه می‌کنند
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

    cfg.auto_n = True
    cfg.auto_n_params = dict(tail_samples_target=1000,
                             ci=0.90,
                             min_n=5000,
                             use_bonferroni=False)
    cfg.alpha = 0.05
    cfg.q_list = (0.05, 0.90, 0.95)
    cfg.tail = "left"
    cfg.evt_threshold_quantile = 0.90
    #cfg.n_total = 10
    cfg.batch_size = 10#5000
    cfg.test_mode = False
    cfg.make_plots = False
    cfg.verbose = True
    cfg.save_per_node_feather = True       # Feather سریع
    cfg.post_export_excel = True          # اکسل بعد از اتمام (اختیاری)
    cfg.post_export_stats_excel = True
    cfg.seed = None
    cfg.output_dir = str(OUT_DIR/"mc_outputs")

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
##===================================================================================viz_mc_violins==


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------- Config --------------------
OUT_DIR = Path("output")
STATS_CSV = OUT_DIR /"mc_outputs"/ "mc_stats_per_node.csv"
FEATHER = OUT_DIR /"mc_outputs"/ "mc_per_node_final_states.feather"
PARQUET = OUT_DIR /"mc_outputs"/ "mc_per_node_final_states.parquet"
CSV_ALT = OUT_DIR /"mc_outputs"/ "mc_per_node_final_states.csv"

PNG_PATH = OUT_DIR /"mc_outputs"/ "mc_violins_per_node.png"
PDF_PATH = OUT_DIR /"mc_outputs"/ "mc_violins_per_node.pdf"

Y_MIN, Y_MAX = -1.0, 1.0
SYNTH_SAMPLES_PER_NODE = 1000  # used only if raw per-node matrix is unavailable
MARKER_SIZE = 20

def try_read_samples():
    try:
        if FEATHER.exists():
            return pd.read_feather(FEATHER)
    except Exception:
        pass
    try:
        if PARQUET.exists():
            return pd.read_parquet(PARQUET)
    except Exception:
        pass
    try:
        if CSV_ALT.exists():
            return pd.read_csv(CSV_ALT)
    except Exception:
        pass
    return None

def truncated_normal_from_stats(mean, q5, q95, n=SYNTH_SAMPLES_PER_NODE, low=-1.0, high=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(1234)
    z = 1.6448536269514722  # N(0,1) quantile at 95%
    sigma = max(1e-6, (q95 - q5) / (2.0 * z))
    x = rng.normal(loc=mean, scale=sigma, size=n)
    return np.clip(x, low, high)

def main():
    assert STATS_CSV.exists(), f"Stats file not found: {STATS_CSV}"
    df_stats = pd.read_csv(STATS_CSV)

    # Expected columns (we handle missing gracefully)
    df_stats = df_stats.copy()
    df_stats["node"] = df_stats["node"].astype(str)
    df_stats.sort_values("node", inplace=True)
    nodes = df_stats["node"].tolist()

    df_samples = try_read_samples()
    if df_samples is not None:
        sample_cols = [c for c in df_samples.columns if c in nodes]
        df_stats = df_stats[df_stats["node"].isin(sample_cols)].copy().sort_values("node")
        nodes = df_stats["node"].tolist()
        df_samples = df_samples[nodes]  # align columns order with nodes

    n_nodes = len(nodes)
    if n_nodes == 0:
        raise RuntimeError("No nodes to plot. Check your stats file.")

    # Prepare violin data
    data_per_node = []
    if df_samples is not None:
        for name in nodes:
            vals = df_samples[name].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            data_per_node.append(np.clip(vals, Y_MIN, Y_MAX))
    else:
        rng = np.random.default_rng(2025)
        for _, row in df_stats.iterrows():
            mu = float(row.get("mean", 0.0))
            q5 = float(row.get("q_5", mu))
            q95 = float(row.get("q_95", mu))
            data_per_node.append(truncated_normal_from_stats(mu, q5, q95, n=SYNTH_SAMPLES_PER_NODE, low=Y_MIN, high=Y_MAX, rng=rng))

    # Figure size
    fig_w = max(12.0, min(0.35 * n_nodes, 60.0))
    fig_h = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Background gradient (red→yellow→green)
    grad_res = 400
    grad = np.linspace(0, 1, grad_res).reshape(grad_res, 1)
    cmap = LinearSegmentedColormap.from_list("ryg_soft", [(1, 0, 0), (1, 1, 0), (0, 0.6, 0)], N=256)
    ax.imshow(grad, extent=(-0.5, n_nodes - 0.5, Y_MIN, Y_MAX), origin="lower", aspect="auto", cmap=cmap, alpha=0.35, zorder=0)

    # Violin plot
    parts = ax.violinplot(
        data_per_node,
        positions=np.arange(n_nodes),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8
    )
    for body in parts["bodies"]:
        body.set_facecolor("#CCCCCC")
        body.set_edgecolor("#444444")
        body.set_alpha(0.7)
        body.set_linewidth(0.6)

    # Overlay stats
    x = np.arange(n_nodes)
    means = df_stats.get("mean", pd.Series(np.full(n_nodes, np.nan))).to_numpy(dtype=float)
    q5s = df_stats.get("q_5", pd.Series(np.full(n_nodes, np.nan))).to_numpy(dtype=float)
    q95s = df_stats.get("q_95", pd.Series(np.full(n_nodes, np.nan))).to_numpy(dtype=float)
    cvars_evt = df_stats.get("evt_CVaR", df_stats.get("CVaR_alpha", pd.Series(np.full(n_nodes, np.nan)))).to_numpy(dtype=float)

    w = 0.35
    for i in range(n_nodes):
        if np.isfinite(means[i]):
            ax.plot([i - w, i + w], [means[i], means[i]], lw=2.0, color="k", zorder=3)
        if np.isfinite(q5s[i]):
            ax.plot([i - w/1.5, i + w/1.5], [q5s[i], q5s[i]], lw=1.8, color="#1f77b4", zorder=3)
        if np.isfinite(q95s[i]):
            ax.plot([i - w/1.5, i + w/1.5], [q95s[i], q95s[i]], lw=1.8, color="#1f77b4", zorder=3)
        if np.isfinite(cvars_evt[i]):
            ax.scatter([i], [cvars_evt[i]], s=MARKER_SIZE, marker="D", color="#d62728", zorder=4)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="k", lw=2, label="Mean"),
        Line2D([0], [0], color="#1f77b4", lw=2, label="q5 & q95"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728", markersize=8, label="EVT-CVaR")
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=True)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.5, n_nodes - 0.5)
    ax.set_ylabel("Final state")
    ax.set_title("Per-Node Final-State Distributions (Violin)\nBackground: -1 (red) → 0 (yellow) → +1 (green)")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=90, ha="center", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG_PATH, dpi=200)
    plt.savefig(PDF_PATH)
    plt.show()
    print("Saved:", PNG_PATH, "and", PDF_PATH)

if __name__ == "__main__":
    main()
#============================================================================================================
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

    #_draw_line(stats.get("mean"), "Mean", "#ffb000")
    _draw_line(stats.get("median"), "Median", "#009988")
    _draw_line(stats.get("q_95"), "q95", "#ee7733")
    _draw_line(stats.get("q_5"), "q5", "#0077bb")
    # CVaR و EVT-CVaR
    alpha_val = stats.get('alpha', 0.05)
    #_draw_line(stats.get("CVaR_alpha"), f"CVaR(α={alpha_val})", "#009988")
    _draw_line(stats.get("evt_CVaR"), "EVT CVaR", "#882255", style=":")

    # --- Aesthetics ---
    ax.set_title("Aggregated Monte Carlo Output Distribution", fontsize=13)
    ax.set_xlabel("Output Value (Blunt-End Safety Control Roles)", fontsize=12)
    #ax.set_xlabel("Output Value (Collective Safety Behavior)", fontsize=12)
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
     dataset_path="output/mc_outputs/mc_dataset.csv",
     stats_path="output/mc_outputs/mc_stats.csv",
     bins=100,
     xlim=(-1, 1),              # ← بازهٔ دلخواه
     add_bg_gradient=True,      # ← گرادیان پس‌زمینه
     save=True,
    show=True
 )
