from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Plot Function ---
def plot_node_traces(state_trace, output_nodes, all_nodes=False,
                     save=False, out_dir="simulation_test", filename=None):
    num_steps = len(state_trace)
    num_nodes = len(state_trace[0])
    traces = {i: [] for i in range(num_nodes)}

    for t in range(num_steps):
        for i in range(num_nodes):
            traces[i].append(state_trace[t][i])

    fig = plt.figure(figsize=(16, 8))
    for i in range(num_nodes):
        if not all_nodes and (i not in output_nodes):
            continue
        label = f"Node {i}"
        plt.plot(traces[i], "-", label=label)

    plt.title("FCM Node Evolution", fontsize=20)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Activation Level",  fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()

    if save:
        # 1) ساخت پوشه اگر نبود
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # 2) نام فایل
        if filename is None:
            filename = f"fcm_plot_{datetime.now():%Y%m%d_%H%M%S}.png"
        # 3) ذخیره
        fig.savefig(out_path / filename, dpi=600, bbox_inches="tight")

    plt.show()
