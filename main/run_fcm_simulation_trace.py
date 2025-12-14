
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix, issparse





ArrayLike = Union[np.ndarray, Sequence[float]]

def _normalize_control_dict(ctrl: Optional[Dict], N: int) -> Optional[Dict[int, dict]]:
    if ctrl is None:
        return None
    norm: Dict[int, dict] = {}
    for k, v in ctrl.items():
        try:
            i = int(k)
        except Exception as e:
            raise TypeError(f"Control_Data/control_dict has non-integer key: {k!r}") from e
        if not (0 <= i < N):
            raise IndexError(f"Control_Data/control_dict index {i} out of bounds for N={N}")
        if isinstance(v, dict):
            na = float(v.get("agent_count", 1.0))
            nc = float(v.get("control_count", 0.0))
        else:
            raise TypeError(f"Control_Data/control_dict[{i}] must be a dict with keys 'agent_count' and 'control_count'; got {type(v)}")
        norm[i] = {"agent_count": na, "control_count": nc}
    return norm


def _prepare_control_arrays(
    N: int,
    control_dict: Optional[Dict[int, dict]],
    control_index_map: Optional[Iterable[int]],
    alpha: float,
    beta: float
):
    use_ctl = np.zeros(N, dtype=bool)
    na = np.ones(N, dtype=np.float64)
    nc = np.zeros(N, dtype=np.float64)

    if control_dict is None and control_index_map is None:
        return use_ctl, np.zeros(N, dtype=np.float64)

    ctl_keys = set(control_dict.keys()) if control_dict is not None else set()

    if control_index_map is not None:
        ctl_index_set = {int(i) for i in control_index_map}
        candidates = ctl_keys.intersection(ctl_index_set) if control_dict is not None else ctl_index_set
    else:
        candidates = ctl_keys

    for i in candidates:
        if 0 <= i < N:
            if control_dict is not None and i in control_dict:
                info = control_dict[i]
                if isinstance(info, dict):
                    use_ctl[i] = True
                    na[i] = float(info.get("agent_count", 1.0))
                    nc[i] = float(info.get("control_count", 0.0))
                else:
                    use_ctl[i] = True
                    na[i] = 1.0
                    nc[i] = 1.0
            else:
                use_ctl[i] = True
                na[i] = 1.0
                nc[i] = 1.0

    ratio = (beta ** na) / (alpha ** np.maximum(nc, 1e-12))
    return use_ctl, ratio


def _coerce_state(init_state, N: int) -> np.ndarray:
    if isinstance(init_state, dict):
        x = np.zeros(N, dtype=np.float64)
        for k, v in init_state.items():
            try:
                i = int(k)
            except Exception as e:
                raise TypeError(f"init_state dict has non-integer key: {k!r}") from e
            if not (0 <= i < N):
                raise IndexError(f"init_state index {i} out of bounds for N={N}")
            try:
                x[i] = float(v)
            except Exception as e:
                raise TypeError(f"init_state[{i}] value is not numeric: {v!r}") from e
        return x

    arr = np.asarray(init_state, dtype=np.float64)
    if arr.ndim == 1 and arr.size == N:
        return arr.copy()
    elif arr.ndim == 2 and 1 in arr.shape and arr.size == N:
        return arr.reshape(N).copy()

    raise TypeError(
        f"init_state must be array-like of length {N} or dict[int->float]; got shape={getattr(arr,'shape',None)} type={type(init_state)}"
    )


def _as_set(iterable_or_none) -> set[int]:
    if iterable_or_none is None:
        return set()
    return {int(i) for i in iterable_or_none}


def fcm_simulation_trace(
    fcm_matrix,
    init_state,
    # ---- root nodes (new, preferred API) ----
    random_root_nodes: Iterable[int] | None = None,  # no inputs; should stay frozen in MAIN
    bsp_root_nodes: Iterable[int] | None = None,     # receive inputs; frozen in warm-up only
    # ---- legacy combined roots (kept for backward compatibility) ----
    root_nodes: Iterable[int] | None = None,
    # ---- outputs ----
    output_nodes: Iterable[int] | None = None,
    # ---- steps & thresholds ----
    steps: int = 1000,
    delt: float = 0.001,
    # ---- control info ----
    control_dict: dict | None = None,
    control_index_map: set | None = None,
    alpha: float = 1.5,
    beta: float = 1.1,
    gamma: float = 0.5,
    # ---- warm-up options ----
    warmup_steps: int = 0,
    # old coarse switches kept, but overridden by category-specific defaults below
    freeze_roots_in_warmup = True,
    freeze_roots_in_main = False,
    # ---- category-specific freeze policy (defaults implement your desired behavior) ----
    freeze_random_in_warmup = True,
    freeze_random_in_main = True,
    freeze_bsp_in_warmup: bool = True,
    freeze_bsp_in_main: bool = False,
    # ---- convergence control ----
    min_main_steps: int = 10,
    patience: int = 3,
    # ---- debug ----
    debug_first_main: bool = False,
    debug_bsp_nodes: Sequence[int] | None = None,
    # ---- alias ----
    Control_Data: dict | None = None
):
    """
    Vectorized FCM simulation with category-separated roots:

      * random_root_nodes: roots with ZERO in-degree (random). Frozen in MAIN by default.
      * bsp_root_nodes:    roots that receive inputs (e.g., from control). Frozen only in warm-up by default.

    Defaults implement:
      - Warm-up: freeze random + BSP roots.
      - MAIN:    freeze random, unfreeze BSP.

    Backward compatibility:
      - If both random_root_nodes and bsp_root_nodes are None, we fall back to legacy `root_nodes`
        and the old freeze flags `freeze_roots_in_warmup/freeze_roots_in_main`.
    """
    # ----- matrix prep -----
    

    if issparse(fcm_matrix):
        W = fcm_matrix.tocsr()
    else:
        W = csr_matrix(np.asarray(fcm_matrix, dtype=np.float64))
    N = W.shape[0]
    WT = W.T.tocsr()

    col_l1 = np.asarray(np.abs(W).sum(axis=0)).ravel().astype(np.float64)


    safe_den = np.maximum(col_l1, 1.0)


    # ----- roots & outputs -----
    random_set = _as_set(random_root_nodes)
    bsp_set    = _as_set(bsp_root_nodes)
    legacy_roots = _as_set(root_nodes)

    # If new API not used, fall back to legacy
    using_new_api = (len(random_set) > 0) or (len(bsp_set) > 0)
    if not using_new_api:
        root_nodes_set = legacy_roots
    else:
        root_nodes_set = random_set.union(bsp_set)

    output_nodes = list(output_nodes or [])
    output_nodes_set = set(int(i) for i in output_nodes)

    # ----- control -----
    raw_ctrl = control_dict if control_dict is not None else Control_Data
    norm_ctrl = _normalize_control_dict(raw_ctrl, N)
    use_ctl, ratio = _prepare_control_arrays(N, norm_ctrl, control_index_map, alpha, beta)

    # ----- state -----
    current_state = _coerce_state(init_state, N)
    z = np.empty(N, dtype=np.float64)
    next_state = np.empty(N, dtype=np.float64)

    state_trace: List[Dict[int, float]] = []
    full_state_trace: List[Dict[int, float]] = []
    full_state_trace.append({i: float(current_state[i]) for i in range(N)})

    def _compute_next(x: np.ndarray) -> np.ndarray:

        z[:] = WT.dot(x)

        ws_norm = z / safe_den
        non_ctl = ~use_ctl
        next_state[non_ctl] = np.clip(ws_norm[non_ctl], -1.0, 1.0)

        if np.any(use_ctl):

            def g(y):
                out = np.empty_like(y, dtype=np.float64)
                pos = (y >= 0)
                out[pos] = np.exp(-gamma * y[pos])          
                out[~pos] = np.expm1(gamma * y[~pos])       
                return out                
            y= z[use_ctl] * ratio[use_ctl]   
            next_state[use_ctl] = g(y)

        return next_state

    # Helper: apply freeze mask
    def _apply_freeze(x_prev: np.ndarray, x_next: np.ndarray, *, which: set[int]) -> None:
        if which:
            idx = np.fromiter(which, dtype=int)
            idx = idx[(idx >= 0) & (idx < N)]
            if idx.size > 0:
                x_next[idx] = x_prev[idx]

    # -------------------- Warm-up --------------------
    if warmup_steps and warmup_steps > 0:
        for _ in range(warmup_steps):
            ns = _compute_next(current_state)
            if using_new_api:
                # freeze categories according to new flags
                if freeze_random_in_warmup:
                    _apply_freeze(current_state, ns, which=random_set)
                if freeze_bsp_in_warmup:
                    _apply_freeze(current_state, ns, which=bsp_set)
            else:
                # legacy behavior
                if freeze_roots_in_warmup and len(root_nodes_set) > 0:
                    _apply_freeze(current_state, ns, which=root_nodes_set)

            current_state = ns.copy()
            full_state_trace.append({i: float(current_state[i]) for i in range(N)})
    warmup_final_state = current_state.copy()

    # -------------------- MAIN --------------------
    delta = [np.inf] * len(output_nodes)
    Simulation_Final_Step = steps
    final_state = None
    stable_runs = 0

    for step in range(1, steps + 1):
        # snapshot BEFORE update
        state_trace.append({i: float(current_state[i]) for i in range(N)})

        # ---- optional debug at step 1 (before update) ----
        if debug_first_main and step == 1:
            z_dbg = WT.dot(current_state)
            ctrl_total = int(np.sum(use_ctl))
            ctrl_zero_ratio = int(np.sum((use_ctl) & (ratio == 0.0)))
            roots_list = sorted(list(root_nodes_set))
            roots_sample = roots_list[:10]
            roots_vals = {i: float(current_state[i]) for i in roots_sample}
            print("[FCM DEBUG] MAIN step #1 diagnostics (before update):")
            print(f"  N={N}, nnz={W.nnz}, warmup_done={warmup_steps>0}")
            print(f"  state: max|x|={np.max(np.abs(current_state)):.6g}, mean|x|={np.mean(np.abs(current_state)):.6g}")
            print(f"  pre-act (raw): max|z|={np.max(np.abs(z_dbg)):.6g}, mean|z|={np.mean(np.abs(z_dbg)):.6g}")
            print(f"  control: total={ctrl_total}, ratio_zero={ctrl_zero_ratio}")
            print(f"  zero_in_degree_cols={int(np.sum(col_l1==0))}")
            if using_new_api:
                print(f"  roots (random={len(random_set)}, bsp={len(bsp_set)}), "
                      f"frozen_in_main: random={freeze_random_in_main}, bsp={freeze_bsp_in_main}")
            else:
                print(f"  roots: count={len(root_nodes_set)}, frozen_in_main={freeze_roots_in_main}, sample_values={roots_vals}")

        # compute next
        ns = _compute_next(current_state)

        # apply MAIN-phase freezing policy
        if using_new_api:
            if freeze_random_in_main:
                _apply_freeze(current_state, ns, which=random_set)   # random stay fixed in MAIN
            if freeze_bsp_in_main:
                _apply_freeze(current_state, ns, which=bsp_set)      # usually False (BSP update in MAIN)
        else:
            if freeze_roots_in_main and len(root_nodes_set) > 0:
                _apply_freeze(current_state, ns, which=root_nodes_set)

        # ---- optional BSP per-node debug (AFTER computing ns) ----
        if debug_first_main and step == 1 and debug_bsp_nodes:
            print("[FCM DEBUG] MAIN step #1 BSP updates (after update, before commit):")
            ws_norm_dbg = z / safe_den
            for i in debug_bsp_nodes:
                if not (0 <= i < N):
                    continue
                print(f"  BSP[{i}]: x={current_state[i]:+.6f}  z={z[i]:+.6f}  z_norm={ws_norm_dbg[i]:+.6f}"
                      f"  control={bool(use_ctl[i])}  ratio={ratio[i]:.6f}  next={ns[i]:+.6f}")

        # compute deltas for outputs vs previous current_state
        if len(output_nodes) > 0:
            out_idx = -1
            frozen_mask_main = set()
            if using_new_api:
                if freeze_random_in_main:
                    frozen_mask_main |= random_set
                if freeze_bsp_in_main:
                    frozen_mask_main |= bsp_set
            else:
                if freeze_roots_in_main:
                    frozen_mask_main |= root_nodes_set

            for i in range(N):
                if i in output_nodes_set:
                    out_idx += 1
                    if i in frozen_mask_main:
                        delta[out_idx] = 0.0
                    else:
                        delta[out_idx] = abs(ns[i] - current_state[i])

        # advance
        current_state = ns.copy()
        full_state_trace.append({i: float(current_state[i]) for i in range(N)})

        # convergence check
        if len(delta) > 0 and step >= min_main_steps:
            if max(delta) < delt:
                stable_runs += 1
            else:
                stable_runs = 0

            if stable_runs >= patience:
                Simulation_Final_Step = step
                final_state = [float(current_state[i]) for i in output_nodes]
                break

    if final_state is None:
        final_state = [float(current_state[i]) for i in output_nodes]

    return state_trace, Simulation_Final_Step, final_state, warmup_final_state, full_state_trace
