import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import statistics
import unicodedata
import pandas as pd  
from scipy.stats import truncnorm
from typing import Optional, Dict, Set, Tuple, Literal, Any

# --- Utility Functions ---
import re
import unicodedata

def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name.strip("_")

def clean_node_dict(prob_inputs):
    """
    Apply the same cleaning rules as clean_node_name to all keys in a dictionary.
    Values remain unchanged.
    Returns a new dictionary with cleaned keys.
    """
    cleaned = {}
    for key, value in prob_inputs.items():
        new_key = clean_node_name(key)
        cleaned[new_key] = value
    return cleaned

    
# --- Initialization Function ---
def prepare_state_plan(
    graph,
    bsp_data=None,                 # dict[name] -> {'sampler', 'stats': {'mean_target', ...}, ...}
    random_nodes=None,             # list/set of names (optional)
    output_nodes=None,             # list/set of names (optional)
    random_keyword="random",
):
    """
    One-time preparation: identify indices and precompute lookups.
    Returns a plan reused in Monte Carlo loops (no graph scans inside the loop).
    """
    # Clean name inputs
    output_name_set = set(clean_node_name(x) for x in (output_nodes or []))
    random_name_set = set(clean_node_name(x) for x in (random_nodes or []))
    bsp_data_clean  = clean_node_dict(bsp_data or {})

    # Index-aligned names
    node_names = [clean_node_name(data.get("name", str(n))) for n, data in graph.nodes(data=True)]
    n_nodes = len(node_names)

    # Identify indices (vector-friendly lists)
    bsp_idx = []
    rnd_idx = []
    out_idx = []
    inter_idx = []

    for i, name in enumerate(node_names):
        is_out = (name in output_name_set)
        is_rnd = (name in random_name_set) or (bool(random_keyword) and (random_keyword in name))

        if name in bsp_data_clean:
            bsp_idx.append(i)
            #out_idx.append(i)
        elif is_rnd:
            rnd_idx.append(i)
        elif is_out:
            out_idx.append(i)
        else:
            inter_idx.append(i)

    # For BSP: pre-extract mean_target and sampler availability to avoid dict lookups in the loop
    bsp_names = [node_names[i] for i in bsp_idx]
    bsp_entries = [bsp_data_clean.get(nm, {}) for nm in bsp_names]
    bsp_samplers = [e.get("sampler", None) for e in bsp_entries]
    bsp_has_sampler = [callable(s) for s in bsp_samplers]
    bsp_mean_targets = []
    for e in bsp_entries:
        stats = e.get("stats", {})
        if "mean_target" in stats:
            bsp_mean_targets.append(float(stats["mean_target"]))
        else:
            bsp_mean_targets.append(float("nan"))  # filled later by center if needed
            

    plan = {
        "node_names": node_names,
        "n_nodes": n_nodes,
        "bsp_idx": bsp_idx,
        "rnd_idx": rnd_idx,
        "out_idx": out_idx,
        "inter_idx": inter_idx,
        "bsp_names": bsp_names,
        "bsp_samplers": bsp_samplers,
        "bsp_has_sampler": bsp_has_sampler,
        "bsp_mean_targets": bsp_mean_targets,
        "bsp_data_clean": bsp_data_clean,  # kept for completeness/fallbacks
    }
    return plan


def initialize_state_from_plan(
    plan,
    bsp_range: Tuple[float, float] = (-1.0, 1.0),
    random_range: Tuple[float, float] = (-1.0, 1.0),
    intermediate_range: Tuple[float, float] = (-1.0, 1.0),
    output_range: Tuple[float, float] = (-1.0, 1.0),
    get_bsp_input: bool = False,            # if True and sampler exists, sample; else use mean_target or center
    seed=None,
    dtype="float32"                # internal array dtype; output types remain identical to before
):
    """
    Fast initialization using a precomputed plan (no graph traversal).
    Returns exactly the same 4-tuple as your current function:
        init_state (dict[int,float]), root_nodes (list[int]),
        output_nodes_idx (list[int]), random_nodes_idx (list[int]).
    """
    import numpy as np

    rng = np.random.default_rng(seed=None)

    node_names        = plan["node_names"]
    n_nodes           = plan["n_nodes"]
    bsp_idx           = plan["bsp_idx"]
    rnd_idx           = plan["rnd_idx"]
    out_idx           = plan["out_idx"]
    inter_idx         = plan["inter_idx"]
    bsp_names         = plan["bsp_names"]
    bsp_samplers      = plan["bsp_samplers"]
    bsp_has_sampler   = plan["bsp_has_sampler"]
    bsp_mean_targets  = plan["bsp_mean_targets"]


    # Internal contiguous array for fast assignment
    state_arr = np.empty(n_nodes, dtype=dtype)
    state_arr.fill(np.nan)

    # Vector RNG for Random and Intermediate
    if rnd_idx:
        #vals = rng.uniform(low=random_range[0], high=random_range[1], size=len(rnd_idx))
        

        # محاسبه پارامترهای توزیع قطع شده
        #  (-1 - 0) / 0.1 = -10  کران پایین نرمال شده
        #  (1 - 0) / 0.1 = 10   کران بالا نرمال شده

        vals = truncnorm.rvs(random_range[0], random_range[1], loc=(random_range[0]+random_range[1])/2, scale=1, size=len(rnd_idx), random_state=rng)
        state_arr[np.fromiter(rnd_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)

    if inter_idx:
        vals = rng.uniform(low=intermediate_range[0], high=intermediate_range[1], size=len(inter_idx))
        state_arr[np.fromiter(inter_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)

    if out_idx:
        vals = rng.uniform(low=output_range[0], high=output_range[1], size=len(out_idx))
        state_arr[np.fromiter(out_idx, dtype=np.int32)] = vals.astype(dtype, copy=False)
        
    # BSP handling: split into two groups to minimize Python-level loops
    if bsp_idx:
        bsp_idx_arr = np.fromiter(bsp_idx, dtype=np.int32)

        if get_bsp_input and any(bsp_has_sampler):
            # 1) For nodes WITH sampler: loop only those (cannot vectorize Python callables)
            for i_local, has_samp in enumerate(bsp_has_sampler):
                if not has_samp:
                    continue
                i_global = bsp_idx[i_local]
                sampler = bsp_samplers[i_local]
                try:
                    sampled = sampler(size=1, rng=rng)
                    val = float(sampled[0]) if hasattr(sampled, "__len__") else float(sampled)
                except Exception:
                    mt = bsp_mean_targets[i_local]
                    if np.isnan(mt):
                        mt = 0.5 * (bsp_range[0] + bsp_range[1])
                    val = float(mt)
                state_arr[i_global] = val

            # 2) For nodes WITHOUT sampler: fill from mean_target (vectorized), fallback to center
            no_samp_mask = np.logical_not(np.fromiter(bsp_has_sampler, dtype=bool))
            if np.any(no_samp_mask):
                no_samp_idx_arr = bsp_idx_arr[no_samp_mask]
                mt_arr = np.array([bsp_mean_targets[i] for i in range(len(bsp_idx))], dtype=dtype)[no_samp_mask]
                # Replace NaNs with center of BSP range
                nan_mask = np.isnan(mt_arr)
                if np.any(nan_mask):
                    mt_arr[nan_mask] = dtype(type(0.5 * (bsp_range[0] + bsp_range[1])))
                state_arr[no_samp_idx_arr] = mt_arr

        else:

            # get_bsp_input=False OR no samplers at all: fill from mean_target vectorized, fallback to center where needed
            mt_arr = np.array(rng.uniform(low=bsp_range[0], high=bsp_range[1], size=len(bsp_idx)), dtype=dtype)
            nan_mask = np.isnan(mt_arr)
            if np.any(nan_mask):
                vals = rng.uniform(low=bsp_range[0], high=bsp_range[1], size=len(bsp_idx))
                mt_arr[nan_mask] = vals.astype(dtype, copy=False)
            state_arr[bsp_idx_arr] = mt_arr

    # Clamp to [-1, 1] in one shot (protect against tiny numeric drift)
    # Only clamp the entries we actually set; build a mask of ~isnan
    set_mask = ~np.isnan(state_arr)
    if np.any(set_mask):
        state_arr[set_mask] = np.minimum(1.0, np.maximum(-1.0, state_arr[set_mask]))

    # Build outputs (same types as before)
    init_state = {}
    # Only BSP + Random + Intermediate are initialized (outputs are intentionally skipped)
    for i in np.where(set_mask)[0].tolist():
        init_state[int(i)] = float(state_arr[i])
    root_nodes = list(bsp_idx) + list(rnd_idx)
    output_nodes_idx = list(out_idx)
    random_nodes_idx = list(rnd_idx)
    bsp_nodes_idx = list(bsp_idx)

    return init_state, root_nodes, output_nodes_idx, random_nodes_idx , bsp_nodes_idx
    
#===========================================================

