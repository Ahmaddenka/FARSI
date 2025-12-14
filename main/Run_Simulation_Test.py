import numpy as np
import runpy
import matplotlib.pyplot as plt
import networkx as nx
import re
import statistics
import unicodedata
import pandas as pd  
from scipy.stats import truncnorm
from main.run_fcm_simulation_trace import fcm_simulation_trace
from main.run_initialize_state import initialize_state_from_plan
from typing import Optional, Dict, Set, Tuple, Literal, Any


# --- Run Test Function ---
def Simulation_Run( fcm_matrix: Optional[np.ndarray] = None,
                    plan: Optional[Dict[str, Dict[str, Any]]] = None,
                    control_dict: Optional[Dict[str, Dict[str, Any]]] = None,
                    control_index_map: Optional[Set[int]] = None,
                    bsp_val_band: Tuple[float, float] = (-1.0, 1.0),
                    random_val_band: Tuple[float, float] = (-1.0, 1.0),
                    raw_val_band: Tuple[float, float] = (-1.0, 1.0),
                    intermediate_range_band: Tuple[float, float] = (-1.0, 1.0),
                    get_bsp_input: bool = False,
                    steps: int = 1000,
                    delt: float = 1e-3,
                    alpha: float = 1.5,
                    beta: float = 1.1,
                    gamma: float = 0.5,
                    warmup_steps: int = 10,
                    freeze_roots_in_warmup: bool = True,
                    freeze_roots_in_main: bool = False,
                            # --- warm-up ---
            
                    freeze_random_in_warmup: bool = True,
                    freeze_random_in_main: bool = True,   # randoms stay frozen in MAIN

                    freeze_bsp_in_warmup = True,
                    freeze_bsp_in_main = False,
                    # ---- convergence control ----
                    min_main_steps: int = 10,
                    patience: int = 3,
                    # ---- debug ----
                    debug_first_main: bool = True,
                    boundary_test: Optional[Literal["positive", "negative"]] = None,
                   ):

    init_state, root_nodes, output_nodes, random_nodes, bsp_nodes_idx = initialize_state_from_plan(
    plan,
    bsp_range = bsp_val_band,
    random_range = random_val_band,
    intermediate_range = intermediate_range_band,
    output_range = raw_val_band,   
    get_bsp_input = get_bsp_input,            # if True and sampler exists, sample; otherwise use stats.mean_target or center
    seed=None
)

    #  boundary_test logic -----------------------------------------------------
    if isinstance(boundary_test, str):
        mode = boundary_test.strip().lower()
    else:
        mode = None

    if mode in ("positive", "negative"):
        num_nodes = fcm_matrix.shape[0]
        all_nodes_idx = list(range(num_nodes))
        random_with_neg_out = set()
        for i in random_nodes:
            if np.any(fcm_matrix[i, :] < 0):
                random_with_neg_out.add(i)
        #print(random_with_neg_out)
        if mode == "positive":
            for i in random_nodes:
                init_state[i] = -1.0 if i in random_with_neg_out else 1.0
        elif mode == "negative":
            for i in random_nodes:
                init_state[i] = 1.0 if i in random_with_neg_out else -1.0
    # --- END boundary_test logic ---------------------------------------------
    
    trace, Simulation_Final_Step, final_state, warmup_final_state, full_state_trace = fcm_simulation_trace(
        fcm_matrix=fcm_matrix,
        init_state=init_state,
        # --- roots (new API) ---
        random_root_nodes=random_nodes,
        bsp_root_nodes=bsp_nodes_idx,
        # root_nodes=root_nodes,
        output_nodes=output_nodes,
        # --- steps & tol ---
        steps=steps, delt=delt,
        # --- control ---
        control_dict=control_dict,
        control_index_map=control_index_map,
        alpha=alpha, beta=beta, gamma = gamma,
        # --- warm-up ---
        warmup_steps=warmup_steps,
        freeze_roots_in_warmup=freeze_roots_in_warmup,
        freeze_roots_in_main=freeze_roots_in_main,
        # --- category freeze policy  ---
        freeze_random_in_warmup=freeze_random_in_warmup,
        freeze_random_in_main=freeze_random_in_main,   # randoms stay frozen in MAIN
        freeze_bsp_in_warmup=freeze_bsp_in_warmup,
        freeze_bsp_in_main=freeze_bsp_in_main,     # BSPs update in MAIN
        # --- convergence ---
        min_main_steps=min_main_steps,
        patience=patience,
        # --- debug ---
        debug_first_main = debug_first_main,
        debug_bsp_nodes=list(bsp_nodes_idx)[:10],
    )






    print(f"Simulation Final Step = {Simulation_Final_Step}")
    if len(final_state) > 0:
        print("Target Nodes Final Values:")
        print(f"Average = {statistics.mean(final_state):.6f}")
        print(f"Median  = {statistics.median(final_state):.6f}")
        print(f"Min     = {min(final_state):.6f}")
        print(f"Max     = {max(final_state):.6f}")
        print(f"Range   = {max(final_state) - min(final_state):.6f}")
    else:
        print("[WARN] No output nodes detected; statistics skipped.")

    #plot_node_traces(full_state_trace, output_nodes, all_nodes=all_nodes_show)
    return full_state_trace , output_nodes