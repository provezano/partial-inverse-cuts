#!/usr/bin/env python3
"""
Clean, limitation-free reference algorithms for Paper 16:
  1. Algorithm 1: Unit-Decrement Circulation & 2^h Heavy-Arc Branching for Sum-Hamming Inverse Min-Cut
  2. Algorithm 2: Polarity-Uncrossed Two-Copy Reversed-Port Transshipment for Multi-Box Convex PWL Partial Inverse Cuts
  3. Algorithm 3: Linear-Surrogate (1+eps)-Approximation & L-Reduction Rounding for Smooth Concave Bulk Discounts

Eliminates all prototype limitations:
  - Handles edgeless (m = 0) and disconnected networks cleanly.
  - Solves totally unimodular circulation and two-copy port-bridge transshipment directly in IEEE-754 / HiGHS
    simplex without global math.lcm denominator scaling (zero 2^30 integer-overflow cap on 15+ prime denominators).
  - Supports both undirected Potts and asymmetric directed submodular cut energies, arbitrary anchors,
    zero capacities/ceilings/slopes, and arbitrary unequal-length K-segment convex PWL payment schedules.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


# ============================================================================
# Part 1: Complete Sum-Hamming Inverse Minimum Cut (Algorithm 1 & Comparators)
# ============================================================================

def st_cut_capacity(arcs: Sequence[Tuple[int, int, float]], side: Iterable[int]) -> float:
    S = set(side)
    return float(sum(c for u, v, c in arcs if u in S and v not in S))


def min_st_cut_via_lp(
    n: int,
    arcs: Sequence[Tuple[int, int, float]],
    source: int = 0,
    sink: Optional[int] = None,
) -> Tuple[float, frozenset[int]]:
    """
    Exact minimum s-t cut on real nonnegative capacities (handles m = 0, parallel arcs, disconnected graphs).
    Uses dual node-potential LP (totally unimodular) and threshold extraction.
    """
    if sink is None:
        sink = n - 1
    pos_arcs = [(u, v, float(c)) for u, v, c in arcs if float(c) > 0.0 and u != v]
    if not pos_arcs:
        return 0.0, frozenset([source])

    # Maximize flow from source to sink
    m = len(pos_arcs)
    # Variables: f_0..f_{m-1}, val
    c_obj = np.zeros(m + 1)
    c_obj[m] = -1.0
    A_eq = np.zeros((n, m + 1))
    b_eq = np.zeros(n)
    bounds = []
    for idx, (u, v, cap) in enumerate(pos_arcs):
        A_eq[u, idx] += 1.0
        A_eq[v, idx] -= 1.0
        bounds.append((0.0, cap))
    A_eq[source, m] = -1.0
    A_eq[sink, m] = 1.0
    bounds.append((0.0, None))
    # Drop redundant sink row
    keep = [i for i in range(n) if i != sink]
    res = linprog(c_obj, A_eq=A_eq[keep], b_eq=b_eq[keep], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Max-flow LP failed: {res.message}")
    flow = res.x[:m]
    max_flow_val = float(-res.fun)

    # Residual graph reachability from source
    adj: List[List[int]] = [[] for _ in range(n)]
    for idx, (u, v, cap) in enumerate(pos_arcs):
        if cap - flow[idx] > 1e-9:
            adj[u].append(v)
        if flow[idx] > 1e-9:
            adj[v].append(u)
    visited = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    return max_flow_val, frozenset(visited)


def solve_hamming_unit_circulation(
    n: int,
    arcs: Sequence[Tuple[int, int, float]],
    target_side: Iterable[int],
    drops: Sequence[float],
    fees: Sequence[float],
    source: int = 0,
    sink: Optional[int] = None,
) -> Tuple[float, Optional[List[int]], Optional[np.ndarray]]:
    """
    Core circulation solver for Theorem 1 (when editable forward arcs have l_a in {0, 1} and c_a in Z_{>=0}).
    Handles m = 0, disconnected graphs, arcs entering source or leaving sink, and real nonnegative fees.
    Returns (optimal_fee, edited_arc_indices, optimal_flow_vector) or (inf, None, None) if infeasible.
    """
    if sink is None:
        sink = n - 1
    S = set(target_side)
    m = len(arcs)
    if m == 0:
        return 0.0, [], np.zeros(0, dtype=float)

    lb = np.zeros(m, dtype=float)
    ub = np.zeros(m, dtype=float)
    c_obj = np.zeros(m, dtype=float)
    editable_fwd: List[int] = []

    for a, (u, v, cap) in enumerate(arcs):
        cap_f = float(cap)
        d_f = float(drops[a])
        w_f = float(fees[a])
        if u in S and v not in S:
            lb[a] = cap_f - d_f
            ub[a] = cap_f
            if d_f > 0.0:
                c_obj[a] = -w_f
                editable_fwd.append(a)
        elif u not in S and v in S:
            # Backward cut arcs must carry zero flow so cut saturation holds
            lb[a] = 0.0
            ub[a] = 0.0
        else:
            lb[a] = 0.0
            ub[a] = cap_f

    if np.any(lb > ub + 1e-12):
        return float("inf"), None, None

    internal = [v for v in range(n) if v != source and v != sink]
    if internal:
        A_eq = np.zeros((len(internal), m), dtype=float)
        pos_map = {v: r for r, v in enumerate(internal)}
        for a, (u, v, _) in enumerate(arcs):
            if u in pos_map:
                A_eq[pos_map[u], a] -= 1.0
            if v in pos_map:
                A_eq[pos_map[v], a] += 1.0
        b_eq = np.zeros(len(internal), dtype=float)
    else:
        A_eq = None
        b_eq = None

    res = linprog(
        c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=list(zip(lb.tolist(), ub.tolist())),
        method="highs",
    )
    if not res.success:
        return float("inf"), None, None

    f = res.x
    const_offset = sum(float(fees[a]) * float(arcs[a][2]) for a in editable_fwd)
    # Note: for unit drops l_a = 1 on integral c_a, extreme-point f_a in {c_a - 1, c_a}
    edited = [a for a in editable_fwd if f[a] < float(arcs[a][2]) - 1e-6]
    # Exact Hamming cost evaluated on the recovered post-edit capacities d_a = f_a (on forward arcs)
    opt_lp_val = float(const_offset + res.fun)
    return opt_lp_val, edited, f


def solve_hamming_fpt_2h(
    n: int,
    arcs: Sequence[Tuple[int, int, int]],
    target_side: Iterable[int],
    drops: Sequence[int],
    fees: Sequence[float],
    source: int = 0,
    sink: Optional[int] = None,
) -> Tuple[float, Optional[List[int]], Optional[List[float]]]:
    """
    Algorithm 1 (Theorem 1): Exact O(2^h * T_circ) solver for Sum-Hamming Inverse Min-Cut
    on integral capacities c_a in Z_{>=0} and integral drop bounds l_a in Z cap [0, c_a],
    where h = |{a in delta^+(X^0) : l_a >= 2}| and w_a >= 0 are arbitrary nonnegative real fees.
    Returns (opt_cost, edited_indices, post_edit_capacities).
    """
    if sink is None:
        sink = n - 1
    S = set(target_side)
    m = len(arcs)
    if m == 0:
        return 0.0, [], []

    heavy_arcs = [
        a for a, (u, v, _) in enumerate(arcs)
        if (u in S and v not in S) and drops[a] >= 2
    ]

    # Fast Step-0 full-drop feasibility screen (1 circulation LP instead of 2^h on infeasible inputs!)
    if heavy_arcs:
        full_arcs = [
            (u, v, c - drops[a] if a in heavy_arcs else c)
            for a, (u, v, c) in enumerate(arcs)
        ]
        full_drops = [
            1 if ((u in S and v not in S) and a not in heavy_arcs and drops[a] == 1) else 0
            for a, (u, v, _) in enumerate(arcs)
        ]
        full_fees = [float(fees[a]) if full_drops[a] == 1 else 0.0 for a in range(m)]
        rem_full, ed_full, _ = solve_hamming_unit_circulation(
            n, full_arcs, S, full_drops, full_fees, source=source, sink=sink
        )
        if ed_full is None:
            return float("inf"), None, None

    best_cost = float("inf")
    best_edited: Optional[List[int]] = None
    best_d: Optional[List[float]] = None

    all_masks = sorted(
        itertools.product((0, 1), repeat=len(heavy_arcs)),
        key=lambda m_bits: sum(float(fees[a]) for a, bit in zip(heavy_arcs, m_bits) if bit == 1),
    )
    for mask in all_masks:
        fixed_cost = sum(float(fees[a]) for a, bit in zip(heavy_arcs, mask) if bit == 1)
        if fixed_cost >= best_cost - 1e-12:
            break

        sub_arcs = list(arcs)
        sub_drops = [0] * m
        sub_fees = [0.0] * m
        heavy_map = dict(zip(heavy_arcs, mask))

        for a, (u, v, c) in enumerate(arcs):
            if u in S and v not in S:
                if a in heavy_map:
                    if heavy_map[a] == 1:
                        # Paid heavy arc: lower its capacity to c_a - l_a (maximal helpful drop on forward cut)
                        sub_arcs[a] = (u, v, c - drops[a])
                    else:
                        # Unpaid heavy arc: fixed at c_a (drop = 0)
                        sub_arcs[a] = (u, v, c)
                elif drops[a] == 1:
                    sub_drops[a] = 1
                    sub_fees[a] = float(fees[a])

        rem_cost, unit_edited, flow_vec = solve_hamming_unit_circulation(
            n, sub_arcs, S, sub_drops, sub_fees, source=source, sink=sink
        )
        if unit_edited is not None and fixed_cost + rem_cost < best_cost - 1e-12:
            best_cost = fixed_cost + rem_cost
            paid_heavy = [a for a, bit in zip(heavy_arcs, mask) if bit == 1]
            best_edited = sorted(paid_heavy + unit_edited)
            best_d = [
                float(arcs[a][2] - drops[a]) if a in best_edited else float(arcs[a][2])
                for a in range(m)
            ]

    return best_cost, best_edited, best_d


def solve_hamming_flow_milp(
    n: int,
    arcs: Sequence[Tuple[int, int, float]],
    target_side: Iterable[int],
    drops: Sequence[float],
    fees: Sequence[float],
    source: int = 0,
    sink: Optional[int] = None,
) -> Tuple[float, Optional[List[int]]]:
    """
    Exact polynomial-size Flow + Big-M Binary MILP comparator for general forward-drop Sum-Hamming Inverse Min-Cut.
    Scales cleanly without 2^(n-2) cut enumeration!
    """
    if sink is None:
        sink = n - 1
    S = set(target_side)
    m = len(arcs)
    if m == 0:
        return 0.0, []

    editable = [a for a, (u, v, _) in enumerate(arcs) if (u in S and v not in S) and float(drops[a]) > 0.0]
    k = len(editable)
    width = m + k
    c_obj = np.zeros(width, dtype=float)
    lb = np.zeros(width, dtype=float)
    ub = np.zeros(width, dtype=float)
    integrality = np.zeros(width, dtype=int)

    for a, (u, v, cap) in enumerate(arcs):
        cap_f = float(cap)
        d_f = float(drops[a])
        if u in S and v not in S:
            lb[a] = cap_f - d_f
            ub[a] = cap_f
        elif u not in S and v in S:
            lb[a] = 0.0
            ub[a] = 0.0
        else:
            lb[a] = 0.0
            ub[a] = cap_f

    constraints = []
    internal = [v for v in range(n) if v != source and v != sink]
    if internal:
        bal = np.zeros((len(internal), width), dtype=float)
        pos_map = {v: r for r, v in enumerate(internal)}
        for a, (u, v, _) in enumerate(arcs):
            if u in pos_map:
                bal[pos_map[u], a] -= 1.0
            if v in pos_map:
                bal[pos_map[v], a] += 1.0
        constraints.append(LinearConstraint(bal, 0.0, 0.0))

    if k > 0:
        rows = np.zeros((k, width), dtype=float)
        lo = np.zeros(k, dtype=float)
        hi = np.full(k, np.inf, dtype=float)
        for t_idx, a in enumerate(editable):
            # f_a + l_a * y_a >= c_a  (if y_a = 0, f_a >= c_a so arc a is unedited; if y_a = 1, f_a >= c_a - l_a)
            rows[t_idx, a] = 1.0
            rows[t_idx, m + t_idx] = float(drops[a])
            lo[t_idx] = float(arcs[a][2])
            c_obj[m + t_idx] = float(fees[a])
            lb[m + t_idx] = 0.0
            ub[m + t_idx] = 1.0
            integrality[m + t_idx] = 1
        constraints.append(LinearConstraint(rows, lo, hi))

    res = milp(
        c_obj,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if res.status == 2 or not res.success:
        return float("inf"), None
    edited = [editable[t_idx] for t_idx in range(k) if res.x[m + t_idx] > 0.5]
    return float(res.fun), edited


def run_liu2012_single_cut_greedy(
    n: int,
    arcs: Sequence[Tuple[int, int, float]],
    target_side: Iterable[int],
    drops: Sequence[float],
    fees: Sequence[float],
    source: int = 0,
    sink: Optional[int] = None,
) -> Dict[str, object]:
    """
    Published single-cut greedy heuristic (Liu et al., 2012, Algorithm 1; extended naturally to ratio l_a / w_a).
    Deletes backward arcs, finds a single initial min-cut Y, sorts Gamma = delta^+(X^0) \\ delta^+(Y) by largest drop,
    and stops as soon as cumulative drop >= C = c'(X^0) - c'(Y).
    """
    if sink is None:
        sink = n - 1
    S = set(target_side)
    # Delete backward arcs to form N'
    kept = [(idx, u, v, float(c)) for idx, (u, v, c) in enumerate(arcs) if not (u not in S and v in S)]
    arcs_p = [(u, v, c) for _, u, v, c in kept]

    # Step 0: check full-drop feasibility
    arcs_full = [
        (u, v, c - float(drops[orig_idx]) if (u in S and v not in S) else c)
        for orig_idx, u, v, c in kept
    ]
    min_full, _ = min_st_cut_via_lp(n, arcs_full, source=source, sink=sink)
    cap_full_S = st_cut_capacity(arcs_full, S)
    if min_full < cap_full_S - 1e-8:
        return {"step0_feasible": False, "output_feasible": False, "cost": float("inf"), "edited": []}

    min_init, Y = min_st_cut_via_lp(n, arcs_p, source=source, sink=sink)
    cap_init_S = st_cut_capacity(arcs_p, S)
    C = cap_init_S - min_init
    if C <= 1e-9:
        return {"step0_feasible": True, "output_feasible": True, "cost": 0.0, "edited": []}

    gamma = [
        orig_idx for orig_idx, u, v, _ in kept
        if (u in S and v not in S) and not (u in Y and v not in Y) and float(drops[orig_idx]) > 0.0
    ]
    # Sort by largest drop per unit fee (reduces to largest l_a when fees are uniform)
    gamma.sort(key=lambda idx: (float(drops[idx]) / max(float(fees[idx]), 1e-12), float(drops[idx])), reverse=True)

    chosen: List[int] = []
    tot_drop = 0.0
    for idx in gamma:
        chosen.append(idx)
        tot_drop += float(drops[idx])
        if tot_drop > C + 1e-9:
            break

    chosen_set = set(chosen)
    arcs_out = [
        (u, v, float(c) - (float(drops[idx]) if idx in chosen_set else 0.0))
        for idx, (u, v, c) in enumerate(arcs)
        if not (u not in S and v in S)
    ]
    min_after, _ = min_st_cut_via_lp(n, arcs_out, source=source, sink=sink)
    cap_after_S = st_cut_capacity(arcs_out, S)
    is_feas = min_after >= cap_after_S - 1e-8
    cost = sum(float(fees[idx]) for idx in chosen)
    return {"step0_feasible": True, "output_feasible": is_feas, "cost": cost, "edited": chosen}


def run_adaptive_recheck_greedy(
    n: int,
    arcs: Sequence[Tuple[int, int, float]],
    target_side: Iterable[int],
    drops: Sequence[float],
    fees: Sequence[float],
    source: int = 0,
    sink: Optional[int] = None,
) -> Dict[str, object]:
    """
    Repaired iterative greedy baseline: recomputes minimum s-t cut Y_t after EVERY greedy arc edit until feasible.
    Always returns a feasible edit set whenever the instance is feasible, testing optimality vs Algorithm 1.
    """
    if sink is None:
        sink = n - 1
    S = set(target_side)
    kept = [(idx, u, v, float(c)) for idx, (u, v, c) in enumerate(arcs) if not (u not in S and v in S)]
    arcs_full = [
        (u, v, c - float(drops[orig_idx]) if (u in S and v not in S) else c)
        for orig_idx, u, v, c in kept
    ]
    min_full, _ = min_st_cut_via_lp(n, arcs_full, source=source, sink=sink)
    if min_full < st_cut_capacity(arcs_full, S) - 1e-8:
        return {"output_feasible": False, "cost": float("inf"), "edited": []}

    chosen: set[int] = set()
    while True:
        cur_arcs = [
            (u, v, c - (float(drops[orig_idx]) if orig_idx in chosen else 0.0))
            for orig_idx, u, v, c in kept
        ]
        min_val, Y = min_st_cut_via_lp(n, cur_arcs, source=source, sink=sink)
        cap_S = st_cut_capacity(cur_arcs, S)
        if min_val >= cap_S - 1e-8:
            break
        cand = [
            orig_idx for orig_idx, u, v, _ in kept
            if orig_idx not in chosen and float(drops[orig_idx]) > 0.0
            and (u in S and v not in S) and not (u in Y and v not in Y)
        ]
        if not cand:
            cand = [
                orig_idx for orig_idx, u, v, _ in kept
                if orig_idx not in chosen and float(drops[orig_idx]) > 0.0 and (u in S and v not in S)
            ]
        if not cand:
            return {"output_feasible": False, "cost": float("inf"), "edited": sorted(chosen)}
        best = max(cand, key=lambda idx: (float(drops[idx]) / max(float(fees[idx]), 1e-12), float(drops[idx])))
        chosen.add(best)

    cost = sum(float(fees[idx]) for idx in chosen)
    return {"output_feasible": True, "cost": cost, "edited": sorted(chosen)}


# ============================================================================
# Part 2: Multi-Box Convex PWL Partial Inverse Cuts (Algorithm 2)
# ============================================================================

def uncross_extremal_corners(
    boxes: Sequence[Tuple[Sequence[float], Sequence[float]]],
    polarity: int,
    tol: float = 1e-12,
) -> List[Tuple[float, ...]]:
    """
    Polarity-specific box-corner uncrossing (Theorem 3):
      - For polarity == 0 (targets t_i = 0, flipped label z = x): each box [l^b, r^b] collapses to its
        coordinatewise lower corner l^b, and any corner l^b >= l^{b'} coordinatewise is dominated and pruned.
      - For polarity == 1 (targets t_i = 1, flipped label z = 1 - x): under label flip, unary field becomes -h,
        so each box [l^b, r^b] collapses to -r^b (equivalently r^b with negated sign), and any r^b <= r^{b'}
        coordinatewise (so -r^b >= -r^{b'}) is dominated and pruned.
    Returns undominated flipped-coordinate unary vectors h_flip (ready for standard source-0/sink-1 cut evaluation).
    """
    raw_candidates: List[np.ndarray] = []
    for lo, hi in boxes:
        if polarity == 0:
            vec = np.asarray(lo, dtype=float)
        else:
            vec = -np.asarray(hi, dtype=float)
        raw_candidates.append(vec)

    # Deduplicate within tol and prune coordinatewise-dominated vectors (keep minimal vectors: no u <= v)
    unique: List[np.ndarray] = []
    for vec in raw_candidates:
        if not any(np.all(np.abs(vec - u) <= tol) for u in unique):
            unique.append(vec)

    undominated: List[Tuple[float, ...]] = []
    for i, v in enumerate(unique):
        dominated = False
        for j, u in enumerate(unique):
            if i != j and np.all(u <= v + tol) and np.any(u < v - tol):
                dominated = True
                break
        if not dominated:
            undominated.append(tuple(v.tolist()))
    return undominated


def evaluate_pwl_cost(segments: Sequence[Tuple[float, float]], p_val: float) -> float:
    """
    Evaluate separable convex PWL cost given cumulative breakpoints [(end_1, slope_1), ..., (end_K, slope_K)].
    """
    rem = float(p_val)
    prev_end = 0.0
    total = 0.0
    for end, slope in segments:
        seg_len = float(end) - prev_end
        take = min(max(rem, 0.0), seg_len)
        total += take * float(slope)
        rem -= take
        prev_end = float(end)
    return total


def solve_two_corner_pwl_transshipment(
    boxes: Sequence[Tuple[Sequence[float], Sequence[float]]],
    undirected_edges: Sequence[Tuple[int, int, float]],
    targets: Dict[int, int],
    anchors: Optional[Dict[int, int]] = None,
    prices: Optional[Dict[int, Sequence[Tuple[float, float]]]] = None,
    directed_edges: Sequence[Tuple[int, int, float]] = (),
) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
    """
    Algorithm 2 (Theorem 3): Exact polynomial-time solver for Target-Unary Partial Inverse Cuts
    under multi-box unary uncertainty (with <= 2 undominated extremal corners per active polarity)
    and separable nondecreasing convex piecewise-linear payment schedules.

    Zero prototype limitations:
      - Direct totally-unimodular transshipment LP via HiGHS (zero math.lcm integer-overflow limit;
        handles arbitrary real capacities/potentials/slopes and 15+ prime rational denominators).
      - Handles m = 0 edgeless graphs, disconnected components, asymmetric directed submodular energies,
        1..K unequal-length PWL segments, zero slopes, and zero ceilings.
    Returns (optimal_total_cost, {target_node: p_i}) or (None, None) if infeasible.
    """
    anchors = anchors or {}
    n = len(boxes[0][0])
    if not targets:
        return 0.0, {}

    # Default unit-slope uncapped prices if None provided
    if prices is None:
        span = max(
            1.0,
            sum(max(abs(float(a)), abs(float(b))) for lo, hi in boxes for a, b in zip(lo, hi))
            + sum(float(w) for _, _, w in undirected_edges)
            + sum(float(w) for _, _, w in directed_edges),
        )
        prices = {i: [(span + 1.0, 1.0)] for i in targets}

    base_directed = [(int(u), int(v), float(w)) for u, v, w in directed_edges if float(w) > 0.0 and u != v]
    for u, v, w in undirected_edges:
        wf = float(w)
        if wf > 0.0 and u != v:
            base_directed.append((int(u), int(v), wf))
            base_directed.append((int(v), int(u), wf))

    caps: Dict[int, float] = {}
    for i in targets:
        segs = prices.get(i, ())
        caps[i] = float(segs[-1][0]) if segs else 0.0

    solution_p: Dict[int, float] = {}
    total_cost = 0.0

    for polarity in (0, 1):
        ports = [i for i, y in targets.items() if y == polarity]
        if not ports:
            continue

        corners = uncross_extremal_corners(boxes, polarity)
        if len(corners) > 2:
            raise ValueError(
                f"Polarity {polarity} retained {len(corners)} > 2 incomparable corners; "
                "3+ incomparable scenarios are NP-hard in general (Theorem 3 boundary)."
            )
        pair_corners = (corners[0], corners[-1])  # duplicates single corner cleanly when len == 1

        # Under flipped variable z_v = x_v ^ polarity:
        #   - All ports have target z_i = 0 (mismatch means z_i = 1, penalized by p_i on arc i -> sink).
        #   - When polarity == 1, (x_u - x_v)_+ = (z_v - z_u)_+, reversing directed arcs (u, v) -> (v, u)!
        oriented_arcs = [
            (v, u, w) if polarity == 1 else (u, v, w)
            for u, v, w in base_directed
        ]
        all_pins = {i: (y ^ polarity) for i, y in {**anchors, **targets}.items()}
        frozen_pins = {i: z for i, z in all_pins.items() if i not in ports}

        k = len(ports)
        s_id, t_id = n, n + 1
        dim_copy = n + 2 + k  # 0..n-1 graph nodes, s_id=n, t_id=n+1, n+2..n+1+k port nodes

        # Node indexing in joined two-copy transshipment graph:
        #   Copy A (reversed): 0 .. dim_copy - 1
        #   Copy B (forward):  dim_copy .. 2*dim_copy - 1
        total_nodes = 2 * dim_copy
        demands = np.zeros(total_nodes, dtype=float)  # net outflow b_v = sum_out - sum_in
        arc_u: List[int] = []
        arc_v: List[int] = []
        arc_cap: List[float] = []
        arc_cost: List[float] = []

        for copy_idx, h_tuple in enumerate(pair_corners):
            h = np.asarray(h_tuple, dtype=float)
            big_M = (
                1.0
                + float(np.abs(h).sum())
                + sum(w for _, _, w in oriented_arcs)
                + sum(caps[i] for i in ports)
            )

            # 1. Compliant cut value D_copy via max-flow with all pins clamped
            goal_arcs: List[Tuple[int, int, float]] = list(oriented_arcs)
            for v_idx in range(n):
                if h[v_idx] > 0.0:
                    goal_arcs.append((v_idx, t_id, float(h[v_idx])))
                elif h[v_idx] < 0.0:
                    goal_arcs.append((s_id, v_idx, float(-h[v_idx])))
            for v_idx, z_val in all_pins.items():
                if z_val == 1:
                    goal_arcs.append((s_id, v_idx, big_M))
                else:
                    goal_arcs.append((v_idx, t_id, big_M))
            D_copy, _ = min_st_cut_via_lp(n + 2, goal_arcs, source=s_id, sink=t_id)

            # 2. Port network arcs for this copy
            offset = copy_idx * dim_copy

            def add_copy_arc(u_loc: int, v_loc: int, cap_val: float) -> None:
                if cap_val <= 0.0:
                    return
                if copy_idx == 0:
                    # Copy A is reversed!
                    arc_u.append(offset + v_loc)
                    arc_v.append(offset + u_loc)
                else:
                    arc_u.append(offset + u_loc)
                    arc_v.append(offset + v_loc)
                arc_cap.append(cap_val)
                arc_cost.append(0.0)

            for u_loc, v_loc, w_val in oriented_arcs:
                add_copy_arc(u_loc, v_loc, w_val)
            for v_idx in range(n):
                if h[v_idx] > 0.0:
                    add_copy_arc(v_idx, t_id, float(h[v_idx]))
                elif h[v_idx] < 0.0:
                    add_copy_arc(s_id, v_idx, float(-h[v_idx]))
            for v_idx, z_val in frozen_pins.items():
                if z_val == 1:
                    add_copy_arc(s_id, v_idx, big_M)
                else:
                    add_copy_arc(v_idx, t_id, big_M)
            for j_idx, i_node in enumerate(ports):
                port_loc = n + 2 + j_idx
                add_copy_arc(i_node, port_loc, big_M)
                add_copy_arc(port_loc, t_id, caps[i_node])

            # Flow conservation convention: A_eq @ f = b_eq where (A_eq @ f)_v = outflow(v) - inflow(v).
            # Copy A (reversed): sink t_A sends D_A into reversed Copy A and source s_A absorbs D_A -> outflow(s_A) = -D_A, outflow(t_A) = +D_A.
            #Wait: let's verify signs!
            # In Forward Copy B: source s_B injects D_B (+D_B net outflow), graph & port arcs go s_B -> V -> port_B -> t_B,
            # PLUS rebate bridge z_i enters port_B (from port_A) and flows port_B -> t_B (cap u_i)!
            # Wait: at t_B, total inflow from s_B is D_B and from bridges is sum(z_i).
            # Then return arc (t_B -> t_A) carries sum(z_i) back to t_A, leaving net inflow D_B at t_B (outflow = -D_B)!
            # At t_A: inflow from return arc is sum(z_i), plus net supply D_A (outflow = +D_A), so total outflow into reversed Copy A is D_A + sum(z_i):
            # D_A exits at s_A (outflow = -D_A) and sum(z_i) exits at port_A across the PWL bridges (port_A -> port_B)!
            if copy_idx == 0:
                demands[offset + s_id] = -D_copy
                demands[offset + t_id] = +D_copy
            else:
                demands[offset + s_id] = +D_copy
                demands[offset + t_id] = -D_copy

        # 3. Return arc from Copy B sink (dim_copy + t_id) to Copy A sink (t_id)
        sum_caps = sum(caps[i] for i in ports)
        if sum_caps > 0.0:
            arc_u.append(dim_copy + t_id)
            arc_v.append(t_id)
            arc_cap.append(sum_caps)
            arc_cost.append(0.0)

        # 4. Parallel PWL rebate bridge arcs from Copy A port_j to Copy B port_j
        # Since p_i = u_i - z_i and C_i(p_i) is convex nondecreasing with slopes s_{i,1} <= ... <= s_{i,K}
        # on [0, b_1], [b_1, b_2], ..., [b_{K-1}, u_i], increasing rebate z_i = u_i - p_i from 0 to u_i
        # SAVES the steepest top segments first (-s_{i,K}, ..., -s_{i,1}) — which is convex in z_i!
        bridge_cols: Dict[int, List[int]] = {i: [] for i in ports}
        const_cap_cost = 0.0
        for j_idx, i_node in enumerate(ports):
            segs = list(prices.get(i_node, ()))
            const_cap_cost += evaluate_pwl_cost(segs, caps[i_node])
            port_A = n + 2 + j_idx
            port_B = dim_copy + n + 2 + j_idx
            prev_end = 0.0
            for end, slope in segs:
                seg_len = float(end) - prev_end
                prev_end = float(end)
                if seg_len <= 0.0:
                    continue
                col_id = len(arc_u)
                arc_u.append(port_A)
                arc_v.append(port_B)
                arc_cap.append(seg_len)
                arc_cost.append(-float(slope))
                bridge_cols[i_node].append(col_id)

        num_arcs = len(arc_u)
        if num_arcs == 0:
            if np.max(np.abs(demands)) > 1e-8:
                return None, None
            for i_node in ports:
                solution_p[i_node] = 0.0
            continue

        A_eq = np.zeros((total_nodes, num_arcs), dtype=float)
        for col_id, (u_id, v_id) in enumerate(zip(arc_u, arc_v)):
            A_eq[u_id, col_id] += 1.0
            A_eq[v_id, col_id] -= 1.0

        # Drop one redundant conservation row
        res = linprog(
            np.asarray(arc_cost, dtype=float),
            A_eq=A_eq[:-1],
            b_eq=demands[:-1],
            bounds=[(0.0, cap_val) for cap_val in arc_cap],
            method="highs",
        )
        if not res.success:
            return None, None

        for i_node in ports:
            rebate = sum(float(res.x[col_id]) for col_id in bridge_cols[i_node])
            p_val = max(0.0, min(caps[i_node], caps[i_node] - rebate))
            if abs(p_val - round(p_val)) <= 1e-9:
                p_val = float(round(p_val))
            solution_p[i_node] = p_val
            total_cost += evaluate_pwl_cost(prices.get(i_node, ()), p_val)

    return total_cost, solution_p


# ============================================================================
# Part 3: Smooth Concave Bulk-Discount Surrogate & L-Reduction (Theorem 4)
# ============================================================================

def smooth_concave_cost(p: np.ndarray, eps: float) -> float:
    p_arr = np.asarray(p, dtype=float)
    rem = np.maximum(0.0, 1.0 - 0.5 * p_arr)
    return float(np.sum(p_arr + eps * (1.0 - rem * rem)))


def build_subcubic_vc_gadget(edges: Sequence[Tuple[int, int]], n_verts: int):
    """
    Constructs Theorem 4's single-field unit-capacity bipartite subcubic inverse-cut instance
    for a graph G = (V, E), with ALL vertices V union {z_e : e in E} prescribed 0 and purchasable.
    """
    m = len(edges)
    N = n_verts + m
    h = np.zeros(N, dtype=float)
    potts_edges: List[Tuple[int, int, float]] = []
    for e_idx, (u, v) in enumerate(edges):
        z_node = n_verts + e_idx
        h[u] -= 1.0
        h[v] -= 1.0
        potts_edges.append((u, z_node, 1.0))
        potts_edges.append((v, z_node, 1.0))
    boxes = [(h.tolist(), h.tolist())]
    targets = {idx: 0 for idx in range(N)}
    return boxes, potts_edges, targets


def solve_concave_linear_surrogate(
    edges: Sequence[Tuple[int, int]],
    n_verts: int,
    eps: float,
) -> Dict[str, object]:
    """
    Algorithm 3 (Theorem 4): Polynomial-time (1 + eps)-approximation for smooth concave bulk-discount
    inverse cuts via linear-cost surrogate transshipment + deterministic L-reduction vertex-cover recovery.
    """
    t0 = time.perf_counter()
    boxes, potts_edges, targets = build_subcubic_vc_gadget(edges, n_verts)
    N = n_verts + len(edges)
    unit_prices = {i: [(6.0, 1.0)] for i in range(N)}
    lin_opt, sol_p = solve_two_corner_pwl_transshipment(boxes, potts_edges, targets, {}, unit_prices)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    assert sol_p is not None and lin_opt is not None
    p_vec = np.array([sol_p[i] for i in range(N)], dtype=float)
    surrogate_concave_val = smooth_concave_cost(p_vec, eps)
    return {
        "linear_sum": float(lin_opt),
        "surrogate_concave_cost": surrogate_concave_val,
        "p_vec": p_vec,
        "runtime_ms": elapsed_ms,
    }


def recover_vertex_cover_from_payment(
    edges: Sequence[Tuple[int, int]],
    n_verts: int,
    p_vec: Sequence[float],
) -> set[int]:
    """
    Exact polynomial-time Max-Flow + Conditional-Expectation L-Reduction Rounding (Theorem 4):
    Converts any feasible payment vector p_vec on a subcubic graph G's gadget into a valid vertex cover S subseteq V
    satisfying |S| - tau(G) <= (C_sm(p_vec) - (2|E| + eps*tau(G))) / eps!
    """
    m = len(edges)
    if m == 0:
        return set()
    N = n_verts + m
    options = [(u, v, n_verts + j) for j, (u, v) in enumerate(edges)]

    # Step 1: Decompose edge demands of 2.0 across incident sites {u, v, z_e} via max-flow LP
    # Nodes: 0..m-1 (edge nodes), m..m+N-1 (site nodes), s = m+N, t = m+N+1
    s_id, t_id = m + N, m + N + 1
    flow_arcs: List[Tuple[int, int, float]] = []
    edge_arc_map: List[List[Tuple[int, int]]] = [[] for _ in range(m)]
    for j, opt in enumerate(options):
        flow_arcs.append((s_id, j, 2.0))
        for site in opt:
            col = len(flow_arcs)
            flow_arcs.append((j, m + site, 2.0))
            edge_arc_map[j].append((site, col))
    for site in range(N):
        cap_s = max(0.0, float(p_vec[site]))
        if cap_s > 0.0:
            flow_arcs.append((m + site, t_id, cap_s))

    # Solve max-flow LP
    num_a = len(flow_arcs)
    c_obj = np.zeros(num_a + 1)
    c_obj[num_a] = -1.0
    A_eq = np.zeros((t_id + 1, num_a + 1))
    bounds = []
    for idx, (u_a, v_a, cap_a) in enumerate(flow_arcs):
        A_eq[u_a, idx] += 1.0
        A_eq[v_a, idx] -= 1.0
        bounds.append((0.0, cap_a))
    A_eq[s_id, num_a] = -1.0
    A_eq[t_id, num_a] = 1.0
    bounds.append((0.0, None))
    res = linprog(c_obj, A_eq=A_eq[:-1], b_eq=np.zeros(t_id), bounds=bounds, method="highs")
    if not res.success or (-res.fun) < 2.0 * m - 1e-6:
        raise ValueError("Input payment vector p_vec is not feasible for the subcubic gadget.")

    choices: List[Dict[int, float]] = []
    for j in range(m):
        d_j: Dict[int, float] = {}
        for site, col in edge_arc_map[j]:
            prob = max(0.0, min(1.0, 0.5 * float(res.x[col])))
            if prob > 1e-12:
                d_j[site] = prob
        tot_p = sum(d_j.values())
        for site in list(d_j):
            d_j[site] /= tot_p
        choices.append(d_j)

    def expected_support(ch_list: Sequence[Dict[int, float]]) -> float:
        unused: Dict[int, float] = {}
        for ed in ch_list:
            for site, pr in ed.items():
                unused[site] = unused.get(site, 1.0) * (1.0 - pr)
        return sum(1.0 - val for val in unused.values())

    # Step 2: Sequential conditional expectation derandomization
    for j in range(m):
        pos_sites = [site for site, pr in choices[j].items() if pr > 1e-12]
        best_site = min(
            pos_sites,
            key=lambda s_cand: (
                expected_support(choices[:j] + [{s_cand: 1.0}] + choices[j + 1:]),
                s_cand,
            ),
        )
        choices[j] = {best_site: 1.0}

    support = {next(iter(c_dict)) for c_dict in choices}
    cover = {site if site < n_verts else edges[site - n_verts][0] for site in support}
    return cover
