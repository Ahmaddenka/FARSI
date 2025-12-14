from pathlib import Path
import runpy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import statistics
import unicodedata
import pandas as pd  
from scipy.stats import truncnorm


INPUT_DIR = Path("graph_file")

def ensure_graph_file(filename="all_network.graphml", build_script = Path("main") /"Graph_Builder.py"):
    """
    Check if a .graphml file exists in current directory.
    If not, run the provided Python script (build_script).
    """
    file_path = INPUT_DIR/filename
    if file_path.exists():
        print(f"[OK] Found existing file: {filename}")
    else:
        print(f"[INFO] {filename} not found — running {build_script} ...")
        try:
            runpy.run_path(str(build_script))
            print(f"[DONE] {build_script} executed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to run {build_script}: {e}")


def make_or_load_graph():
    ensure_graph_file("all_network.graphml", "Graph_Builder.py")
    
    # Load the graph from the file
    all_network = nx.read_graphml(str(INPUT_DIR/'all_network.graphml'))
    print("Graph has been loaded from all_network.graphml")
    return all_network


# --- Utility Functions ---
def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    return name

def graph_to_fcm(g, weight_property="weight", normalize=True):
    """
    Convert a NetworkX graph to an adjacency matrix (row j -> col i means j -> i).
    """
   
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, d in g.edges(data=True):
        w = float(d.get(weight_property, 1))
        W[idx[u], idx[v]] = (w / 5.0) if normalize else w
    return W

def run_graph_to_fcm():
    all_network = make_or_load_graph()
    W = graph_to_fcm(all_network, weight_property="weight", normalize=True)
    return W , all_network
    
    