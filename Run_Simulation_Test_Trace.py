from main.run_initialize_state import prepare_state_plan , initialize_state_from_plan
from main.Run_Simulation_Test import Simulation_Run
from main.fuzzy_to_prob import build_probability_inputs
from main.agent_and_control_number import build_control_summary
from main.graph_to_fcm import run_graph_to_fcm
from main.run_plot_node_traces import plot_node_traces
import pandas as pd  
import unicodedata
import re
import networkx as nx
from pathlib import Path


INPUT_DIR = Path("field_data")
OUT_DIR = Path("output")

# --- Utility Functions ---
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

#-- get input bsp data and convert to probability function--------------------------
prob_inputs = build_probability_inputs(
    excel_path=str(INPUT_DIR/"BSM_input_DATA.xlsx"),
    sheet_name=0,
    id_col=0,
    scale_min=1.0, scale_max=5.0,   
    trapezoid_fit="std",     # OR 'std'
    std_core_width=0.2,        
    std_support_width=1.0,    
    support_p= (0.499999, 0.50001),
    core_p=(0.499999, 0.50001),
    n_grid=1001,
    target_range=(-1.0, 1.0)        
)

fcm_matrix, graph = run_graph_to_fcm()

Control_Leader_Nodes = build_control_summary(str(INPUT_DIR/"Control_gorup_structure.xlsx"))
control_nodes = sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "control"
                        and n not in Control_Leader_Nodes])
Control_Leader=list(Control_Leader_Nodes) 
bsp_data = clean_node_dict(prob_inputs)  

collective_behavior =sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "group"
                        and "collective_behavior" in str(n).lower()
                        and not "behavioral" in str(n).lower()])

Group =sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "group"
                        and not "collective_behavior" in str(n).lower()])
                        
behavior_nodes = sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "agent"
                        and "behavior" in str(n).lower()
                        and not "behavioral" in str(n).lower()])
agent = sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "agent"
                        and not "behavior" in str(n).lower()
                        and not "random" in str(n).lower()
                        and not "reason_against"in str(n).lower()])
                        

SOE = sorted([n for n, d in graph.nodes(data=True)
                        if str(d.get("level", "")).lower() == "soe"
                        ])

plan= prepare_state_plan(
    graph,
    bsp_data=bsp_data,                 # dict[name] -> { 'sampler', 'stats', ... }
    random_nodes=None,                 # list/set of names (optional)
    output_nodes=control_nodes,        # list/set of names (optional)
    random_keyword="random",
)

node_names = list(graph.nodes()) 
name_to_idx = {name: i for i, name in enumerate(node_names)}
name_to_index = {n: i for i, n in enumerate(node_names)}


Control_Leader_Nodes_idx = {}
for name, vals in Control_Leader_Nodes.items():
    i = name_to_idx[name]   
    Control_Leader_Nodes_idx[i] = {
        "agent_count": float(vals.get("agent_count", 1.0)),
        "control_count": float(vals.get("control_count", 0.0)),
    }
control_index_map = {
    name_to_index[n]
    for n, d in graph.nodes(data=True)
    if str(d.get("level", "")).lower() == "control" and n in Control_Leader_Nodes
}


full_state_trace, output_nodes = Simulation_Run(  fcm_matrix = fcm_matrix ,
                                                  plan = plan,
                                                 control_dict = Control_Leader_Nodes_idx,
                                                 control_index_map = control_index_map,
                                                 bsp_val_band=(-1.0, -0.999999),   
                                                 random_val_band= (-1.0, 1.0),
                                                 raw_val_band = (-1.0, 1.0), 
                                                 intermediate_range_band=(-1.0, 1.0), 
                                                 get_bsp_input = True,
                                                 steps = 300,
                                                 delt = 1e-3,
                                                 alpha =0.05153744308202851,
                                                 beta = 0.8030979544125557,
                                                 gamma = 0.6721727251459532,
                                                 warmup_steps = 30,
                                                 freeze_bsp_in_warmup = True,
                                                 freeze_bsp_in_main = False,
                                                 min_main_steps = 10,
                                                 patience = 3,
                                                 # ---- debug ----
                                                 debug_first_main = False,
                                                 boundary_test =None#"negative"# or "positive"
                                                 )

plot_node_traces(full_state_trace, output_nodes, all_nodes=False, save=True, out_dir=str(OUT_DIR/"simulation_test"), filename="Trace_Control_Nodes_LowerBand")   



