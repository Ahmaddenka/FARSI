# -*- coding: utf-8 -*-
# Final integrated builder for multi-level directed graph from Excel matrices
# Global rule: any numeric entry => add a directed edge from COLUMN (source) to ROW (target)

import pandas as pd
import networkx as nx
import re
from collections import defaultdict
from pathlib import Path
# =========================
# Config
# =========================

BASE_DIR = Path("graph_data")
OUT_DIR = Path("graph_file")
PATHS = {
    "agent": BASE_DIR / "agent_graph.xlsx",
    "group": BASE_DIR / "group_graph.xlsx",
    "soe": BASE_DIR / "SOE_graph.xlsx",
    "control": BASE_DIR / "Control_graph.xlsx",
    "bsm": BASE_DIR / "BSM_graph.xlsx",
    "agent_group": BASE_DIR / "Agent_Group_graph.xlsx",
    "agent_soe": BASE_DIR / "Agent_SOE_graph.xlsx",
    "soe_group": BASE_DIR / "SOE_Group_graph.xlsx",
    "control_group": BASE_DIR / "Control_gorup_graph.xlsx",  
    "bsm_control": BASE_DIR / "BSM_Control_graph.xlsx",
    "bsm_linkage": BASE_DIR / "BSM_linkage_graph.xlsx",
    "structure": BASE_DIR / "SOE_and_Group_name.xlsx",
}

OUTPUT_NODES_ONLY = str(OUT_DIR / "nodes_only.graphml")
OUTPUT_ALL = str(OUT_DIR / "all_network.graphml")


TREAT_BIDIRECTIONAL_AS_ERROR = False
INCLUDE_BSM_LINKAGE_TO_CONTROL = False

# =========================
# Helpers
# =========================
def clean_name(name) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s.strip("_")

def excel_read_indexed(path):
    df = pd.read_excel(path, index_col=0)
    df.index = [clean_name(i) for i in df.index]
    df.columns = [clean_name(c) for c in df.columns]
    return df

def extract_nodes(path):
    df = pd.read_excel(path, index_col=0)
    clean_cols = [clean_name(c) for c in df.columns]
    return sorted(set(c for c in clean_cols if c))

def load_base_names_from_index(path):
    df = pd.read_excel(path, index_col=0)
    return set(clean_name(i) for i in df.index)

def load_base_names_from_columns(path):
    df = pd.read_excel(path, index_col=0)
    return set(clean_name(c) for c in df.columns)

def coerce_numeric(x):
    return pd.to_numeric(x, errors="coerce")

# =========================
# Load structure & unique maps
# =========================
structure_df = pd.read_excel(PATHS["structure"])
structure_df.columns = [c.strip().lower() for c in structure_df.columns]
if "soe" not in structure_df.columns or "department (group)" not in structure_df.columns:
    raise ValueError("Structure sheet must contain columns 'SOE' and 'Department (Group)'.")
if "number of agents" not in structure_df.columns:
    poss = [c for c in structure_df.columns if "agent" in c]
    if not poss:
        raise ValueError("Structure sheet needs 'Number of Agents' column.")
    structure_df.rename(columns={poss[0]: "number of agents"}, inplace=True)

unique_depts = sorted({clean_name(x) for x in structure_df["soe"]})
unique_groups = sorted({
    (clean_name(r["soe"]), clean_name(r["department (group)"]))
    for _, r in structure_df.iterrows()
})
agents_per_group = {}
for _, r in structure_df.iterrows():
    dept = clean_name(r["soe"])
    grp  = clean_name(r["department (group)"])
    n_ag = int(r["number of agents"])
    agents_per_group[(dept, grp)] = max(n_ag, agents_per_group.get((dept, grp), 0))

# Base name sets
agent_base   = load_base_names_from_index(PATHS["agent"])
group_base   = load_base_names_from_index(PATHS["group"])
soe_base     = load_base_names_from_index(PATHS["soe"])
control_base = load_base_names_from_columns(PATHS["control"])  # بدون پیشوند
bsm_base     = load_base_names_from_columns(PATHS["bsm"])      # بدون پیشوند

def fq_list(level, base, dept, group, n_agents):
    # فقط برای این سطوح fully-qualified می‌سازیم
    if level == "agent":
        return [f"{dept}_{group}_{a}_{base}" for a in range(1, n_agents+1)]
    if level == "group":
        return [f"{dept}_{group}_{base}"]
    if level == "soe":
        return [f"{dept}_{base}"]
    # BSM و Control یکتا هستند
    if level == "control":
        return [base]
    if level == "bsm":
        return [base]
    return []

def resolve_level(base_name, allowed_levels):
    def in_set(lvl):
        return ((lvl=="agent"   and base_name in agent_base)   or
                (lvl=="group"   and base_name in group_base)   or
                (lvl=="soe"     and base_name in soe_base)     or
                (lvl=="control" and base_name in control_base) or
                (lvl=="bsm"     and base_name in bsm_base))
    cands = [lvl for lvl in allowed_levels if in_set(lvl)]
    if len(cands) == 1:
        return cands[0]
    return None

# =========================
# Build nodes
# =========================
G = nx.DiGraph()

# BSM: بدون پیشوند
bsm_nodes = extract_nodes(PATHS["bsm"])
for n in bsm_nodes:
    if n:
        G.add_node(n, level="bsm")

# CONTROL: بدون پیشوند
control_nodes = extract_nodes(PATHS["control"])
for n in control_nodes:
    if n:
        G.add_node(n, level="control")

# SOE
soe_internal_nodes = extract_nodes(PATHS["soe"])
for dept in unique_depts:
    for n in soe_internal_nodes:
        if n:
            G.add_node(f"{dept}_{n}", level="soe", department=dept)

# GROUP
group_internal_nodes = [n for n in set(extract_nodes(PATHS["group"])) if n]
created_groups = set()
for (dept, group) in unique_groups:
    if (dept, group) in created_groups:
        continue
    created_groups.add((dept, group))
    for n in group_internal_nodes:
        G.add_node(f"{dept}_{group}_{n}", level="group", department=dept)

# AGENT
agent_internal_nodes = extract_nodes(PATHS["agent"])
for (dept, group), n_agents in agents_per_group.items():
    for a in range(1, n_agents + 1):
        for n in agent_internal_nodes:
            G.add_node(f"{dept}_{group}_{a}_{n}", level="agent", department=dept)

nx.write_graphml(G, OUTPUT_NODES_ONLY)
print(f"✅ Nodes created and saved as {OUTPUT_NODES_ONLY}")
print("Node counts:",
      "BSM:",     sum(1 for _,d in G.nodes(data=True) if d.get("level")=="bsm"),
      "CONTROL:", sum(1 for _,d in G.nodes(data=True) if d.get("level")=="control"),
      "SOE:",     sum(1 for _,d in G.nodes(data=True) if d.get("level")=="soe"),
      "GROUP:",   sum(1 for _,d in G.nodes(data=True) if d.get("level")=="group"),
      "AGENT:",   sum(1 for _,d in G.nodes(data=True) if d.get("level")=="agent"),
      "TOTAL:",   len(G.nodes()))

# =========================
# Edge helpers
# =========================
edge_attempts = 0
duplicate_attempts = 0
seen_edges = set()

def add_edge_safe(u, v, weight, src_level=None, dst_level=None):
    global edge_attempts, duplicate_attempts
    if u not in G or v not in G:
        return False
    w = coerce_numeric(weight)
    if pd.isna(w) or float(w) == 0.0:
        return False
    edge_attempts += 1
    if (u, v) in seen_edges:
        duplicate_attempts += 1
    seen_edges.add((u, v))
    G.add_edge(u, v, weight=float(w), src_level=src_level, dst_level=dst_level)
    return True

def add_edges_from_df(df, src_prefix, dst_prefix, src_level, dst_level):
    added = 0
    for src in df.columns:

        for dst in df.index:
            w = df.loc[dst, src]
            u = f"{src_prefix}{clean_name(src)}"
            v = f"{dst_prefix}{clean_name(dst)}"
            if add_edge_safe(u, v, w, src_level, dst_level):
                added += 1
    return added

# =========================
# internal edges
# =========================
def add_internal_edges(file, level):
    df = excel_read_indexed(file)
    total = 0
    if level == "agent":
        for (dept, group), n_agents in agents_per_group.items():
            for a in range(1, n_agents + 1):
                total += add_edges_from_df(df,
                    f"{dept}_{group}_{a}_", f"{dept}_{group}_{a}_", "agent", "agent")
    elif level == "group":
        for (dept, group) in unique_groups:
            total += add_edges_from_df(df,
                f"{dept}_{group}_", f"{dept}_{group}_", "group", "group")
    elif level == "soe":
        for dept in unique_depts:
            total += add_edges_from_df(df, f"{dept}_", f"{dept}_", "soe", "soe")
    elif level == "control":
        # بدون پیشوند
        total += add_edges_from_df(df, "", "", "control", "control")
    elif level == "bsm":
        # بدون پیشوند
        total += add_edges_from_df(df, "", "", "bsm", "bsm")
    print(f"  + Internal {level:<7} edges added: {total}")

# =========================
# Multilevel cross (both ways, column→row)
# =========================
def add_cross_edges_multilevel(df_path, allowed_levels):
    df = excel_read_indexed(df_path)
    added = 0
    for (dept, group) in unique_groups:
        n_agents = agents_per_group[(dept, group)]
        for src_raw in df.columns:
            src = clean_name(src_raw)
            src_level = resolve_level(src, allowed_levels)
            if not src_level:
                continue
            #for dst_raw in df.columns:  
            for dst_raw in df.index:
                dst = clean_name(dst_raw)
                dst_level = resolve_level(dst, allowed_levels)
                if not dst_level:
                    continue
                w = df.loc[dst, src]
                if pd.isna(w) or float(pd.to_numeric(w, errors="coerce") or 0.0) == 0.0:
                    continue
                src_nodes = fq_list(src_level, src, dept, group, n_agents)
                dst_nodes = fq_list(dst_level, dst, dept, group, n_agents)
                for u in src_nodes:
                    for v in dst_nodes:
                        if add_edge_safe(u, v, w, src_level, dst_level):
                            added += 1
    print(f"  + Cross (multilevel) {sorted(allowed_levels)} edges added: {added}")
    return added

# =========================
# Special: Control <- Group
# =========================
def add_control_to_group_edges_special(file_path, control_ref_file):
    # از فایل کنترل لیست واقعی کنترل‌ها را بدون پیشوند بگیر
    df_control_ref = pd.read_excel(control_ref_file, index_col=0)
    valid_controls = set(clean_name(c) for c in df_control_ref.columns)

    df_raw = pd.read_excel(file_path, header=None)
    header_groups = [clean_name(x) for x in df_raw.iloc[0, 1:]]
    header_nodes  = [clean_name(x) for x in df_raw.iloc[1, 1:]]
    control_nodes = [clean_name(x) for x in df_raw.iloc[2:, 0]]
    weights = df_raw.iloc[2:, 1:]
    weights.columns = header_groups
    weights.index   = control_nodes

    added = 0
    for (dept, g_act) in unique_groups:
        idxs = [i for i, g in enumerate(header_groups) if g == g_act]
        if not idxs:
            continue
        for col_idx in sorted(set(idxs)):
            base_node = header_nodes[col_idx]
            target    = f"{dept}_{g_act}_{base_node}"
            if target not in G:
                continue
            for ctrl in control_nodes:
                w = weights.loc[ctrl, g_act] if (ctrl in weights.index and g_act in weights.columns) else float("nan")
                # این‌بار بدون پیشوند
                src_node = ctrl if ctrl in valid_controls else None
                if not src_node or src_node not in G:
                    continue
                if add_edge_safe(target, src_node, w, "control", "group"):
                    added += 1
    print("  + Group -> Control (special) edges added:", added)


# =========================
# BSM -> Control
# =========================
def add_bsm_to_control_edges():
    df = pd.read_excel(PATHS["bsm_control"], index_col=0)
    cols_clean = [clean_name(c) for c in df.columns]
    idx_clean  = [clean_name(i) for i in df.index]
    df.index, df.columns = idx_clean, cols_clean

    col_is_control = sum(1 for c in cols_clean if c in control_base)
    col_is_bsm     = sum(1 for c in cols_clean if c in bsm_base)

    # بدون پیشوند
    added = add_edges_from_df(df, "", "", "bsm", "control")
    print("  + Control -> BSM edges added:", added)

    
# =========================
# BSM linkage
# =========================
def add_bsm_linkage_edges():
    df_bl = pd.read_excel(PATHS["bsm_linkage"], index_col=0)
    df_bl.index   = [clean_name(i) for i in df_bl.index]
    df_bl.columns = [clean_name(c) for c in df_bl.columns]

    added = 0
    for bsm_src in df_bl.columns:
        src_node = bsm_src  # بدون پیشوند
        if src_node not in G:
            continue
        for dst_base in df_bl.index:
            w = pd.to_numeric(df_bl.loc[dst_base, bsm_src], errors="coerce")
            if pd.isna(w) or float(w) == 0.0:
                continue

            # Agent
            if dst_base in agent_base:
                for (dept, group), n_agents in agents_per_group.items():
                    for a in range(1, n_agents + 1):
                        v = f"{dept}_{group}_{a}_{dst_base}"
                        if add_edge_safe(src_node, v, w, "bsm", "agent"):
                            added += 1
                continue

            # Group
            if dst_base in group_base:
                for (dept, group) in unique_groups:
                    v = f"{dept}_{group}_{dst_base}"
                    if add_edge_safe(src_node, v, w, "bsm", "group"):
                        added += 1
                continue

            # SOE
            if dst_base in soe_base:
                for dept in unique_depts:
                    v = f"{dept}_{dst_base}"
                    if add_edge_safe(src_node, v, w, "bsm", "soe"):
                        added += 1
                continue

            # Control (اختیاری)
            if INCLUDE_BSM_LINKAGE_TO_CONTROL and dst_base in control_base:
                v = dst_base  # بدون پیشوند
                if add_edge_safe(src_node, v, w, "bsm", "control"):
                    added += 1
                continue

    print("  + BSM -> (agent/group/soe{ctrl}) linkage edges added:", added)

# =========================
# Build edges
# =========================
print("\n--- Adding edges ---")
add_internal_edges(PATHS["agent"], "agent")
add_internal_edges(PATHS["group"], "group")
add_internal_edges(PATHS["soe"], "soe")
add_internal_edges(PATHS["control"], "control")
add_internal_edges(PATHS["bsm"], "bsm")

# ماتریس‌های بین‌سطحی (هر دو جهت، ستون→سطر)
add_cross_edges_multilevel(PATHS["agent_group"], {"agent","group"})
add_cross_edges_multilevel(PATHS["agent_soe"],   {"agent","soe"})
add_cross_edges_multilevel(PATHS["soe_group"],   {"group","soe"})

# ویژه‌ها
add_control_to_group_edges_special(PATHS["control_group"], PATHS["control"])
add_bsm_to_control_edges()
add_bsm_linkage_edges()

print(f"\nEdge attempts: {edge_attempts} | duplicate attempts (same u,v): {duplicate_attempts}")
print(f"Graph now has: {len(G.nodes())} nodes, {len(G.edges())} edges")

# =========================
# Validation
# =========================
print("\n--- Validating Graph ---")
isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]
bidirectional_edges = [(u, v) for (u, v) in G.edges if G.has_edge(v, u)]
zero_weight_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if pd.isna(d.get("weight")) or float(d.get("weight", 0.0)) == 0.0]
components = list(nx.weakly_connected_components(G))

print(f"Isolated nodes: {len(isolated_nodes)}")
print(f"Bidirectional edges: {len(bidirectional_edges)}")
print(f"Duplicate edge attempts (overwrites): {duplicate_attempts}")
print(f"Zero/None/NaN-weight edges present: {len(zero_weight_edges)}")
print(f"Weakly connected components: {len(components)}")
if len(components) > 1:
    sizes = sorted([len(c) for c in components], reverse=True)
    print("  Component sizes (top 10):", sizes[:10])

ok = (len(isolated_nodes) == 0 and duplicate_attempts == 0 and len(zero_weight_edges) == 0)
if TREAT_BIDIRECTIONAL_AS_ERROR:
    ok = ok and (len(bidirectional_edges) == 0)

print("Validation PASSED ✅" if ok else "Validation FAILED ❌")

# =========================
# Save
# =========================
nx.write_graphml(G, OUTPUT_ALL)
print(f"\nGraph saved as '{OUTPUT_ALL}'")
