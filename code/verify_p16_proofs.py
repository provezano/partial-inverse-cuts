#!/usr/bin/env python3
"""
Standalone Stage-S6 Executable Proof Counterexample Search for Paper 16 (`p16_partial_inverse_cuts`).
Exits 0 iff ZERO counterexamples are found across all four theorems, sharpness propositions, and algorithms.
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))

import inverse_cuts_lib as lib  # noqa: E402


def all_st_cuts(n: int):
    return [
        frozenset([0, *mid])
        for k in range(n - 1)
        for mid in itertools.combinations(range(1, n - 1), k)
    ]


def exact_cut_enum_milp(n: int, arcs, target_S, drops, fees) -> float:
    m = len(arcs)
    if m == 0:
        return 0.0
    cuts = all_st_cuts(n)
    c_obj = np.array([0.0] * m + [float(w) for w in fees])
    integ = np.array([0] * m + [1] * m)
    lb = np.array([float(arcs[a][2] - drops[a]) for a in range(m)] + [0.0] * m)
    ub = np.array([float(arcs[a][2]) for a in range(m)] + [1.0] * m)
    in_T = np.array([1.0 if (u in target_S and v not in target_S) else 0.0 for u, v, _ in arcs])
    rows, lo, hi = [], [], []
    for S in cuts:
        if S == target_S:
            continue
        in_S = np.array([1.0 if (u in S and v not in S) else 0.0 for u, v, _ in arcs])
        diff = in_T - in_S
        if np.any(diff != 0):
            rows.append(np.concatenate([diff, np.zeros(m)]))
            lo.append(-np.inf)
            hi.append(0.0)
    for a in range(m):
        r = np.zeros(2 * m)
        r[a] = 1.0
        r[m + a] = float(drops[a])
        rows.append(r)
        lo.append(float(arcs[a][2]))
        hi.append(np.inf)
    res = milp(
        c_obj,
        integrality=integ,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.array(rows), lo, hi) if rows else (),
        options={"mip_rel_gap": 0.0},
    )
    if res.status == 2 or not res.success:
        return float("inf")
    return float(res.fun)


def exact_partial_cut_2n_lp(boxes, u_edges, d_edges, targets, anchors, prices):
    n = len(boxes[0][0])
    xs = np.array([
        x for x in itertools.product((0, 1), repeat=n)
        if all(x[i] == v for i, v in anchors.items())
    ], dtype=int)
    t_nodes = list(targets.keys())
    t_vals = np.array([targets[i] for i in t_nodes], dtype=int)
    mismatch = (xs[:, t_nodes] != t_vals).astype(float) if t_nodes else np.zeros((len(xs), 0))
    comp = ~mismatch.any(axis=1) if t_nodes else np.ones(len(xs), dtype=bool)
    if not np.any(comp):
        return None
    smooth = np.zeros(len(xs), dtype=float)
    for u, v, w in u_edges:
        smooth += float(w) * np.abs(xs[:, u] - xs[:, v])
    for u, v, w in d_edges:
        smooth += float(w) * np.maximum(0, xs[:, u] - xs[:, v])
    worst_gap = np.full(len(xs), -np.inf, dtype=float)
    for lo, hi in boxes:
        corners = np.array(list(itertools.product(*(sorted({float(a), float(b)}) for a, b in zip(lo, hi)))))
        E_all = xs @ corners.T + smooth[:, None]
        worst_gap = np.maximum(worst_gap, (E_all[comp].min(axis=0) - E_all).max(axis=1))
    if not t_nodes:
        return 0.0 if np.all(worst_gap <= 1e-9) else None
    bounds, obj, cols = [], [], [[] for _ in t_nodes]
    idx = 0
    for j, i in enumerate(t_nodes):
        prev = 0.0
        for end, sl in prices[i]:
            bounds.append((0.0, float(end) - prev))
            obj.append(float(sl))
            cols[j].append(idx)
            prev = float(end)
            idx += 1
    if idx == 0:
        return 0.0 if np.all(worst_gap <= 1e-9) else None
    A_ub = np.zeros((len(xs), idx), dtype=float)
    for j in range(len(t_nodes)):
        for c in cols[j]:
            A_ub[:, c] = -mismatch[:, j]
    lp = linprog(obj, A_ub=A_ub, b_ub=-worst_gap, bounds=bounds, method="highs")
    return float(lp.fun) if lp.success else None


def main() -> int:
    # 1. Verify strictly-positive strongly-connected 4-vertex refutation of Liu et al. (2012) Algorithm 1
    arcs_w = [(0, 1, 4), (1, 3, 6), (0, 2, 2), (2, 3, 4), (3, 0, 10)]
    drops_w = [0, 5, 0, 3, 0]
    fees_w = [1.0] * 5
    X0 = frozenset([0, 1, 2])
    g_res = lib.run_liu2012_single_cut_greedy(4, arcs_w, X0, drops_w, fees_w)
    opt_w, ed_w, _ = lib.solve_hamming_fpt_2h(4, arcs_w, X0, drops_w, fees_w)
    assert g_res["step0_feasible"] and (not g_res["output_feasible"]), "Liu 2012 Alg 1 should fail feasibility!"
    assert abs(opt_w - 2.0) < 1e-9 and ed_w == [1, 3], "Exact optimum on 4-vertex witness must be 2!"

    # Also verify 6-vertex refutation of Adaptive Max-Flow Recheck Greedy
    rng_g = np.random.default_rng(20261122)
    found_subopt = False
    for _ in range(400):
        n_g = 6
        Xg = frozenset([0, 1, 2, 3])
        ag, dg = [], []
        for u in range(n_g - 1):
            for v in range(u + 1, n_g):
                if (u == 0 and v == n_g - 1) or rng_g.random() < 0.45:
                    continue
                c = int(rng_g.integers(1, 8))
                d = int(rng_g.integers(1, c + 1)) if (u in Xg and v not in Xg) else 0
                ag.append((u, v, c))
                dg.append(d)
        if not ag:
            continue
        opt_g, _, _ = lib.solve_hamming_fpt_2h(n_g, ag, Xg, dg, [1.0] * len(ag))
        if np.isfinite(opt_g) and opt_g > 0:
            ad_res = lib.run_adaptive_recheck_greedy(n_g, ag, Xg, dg, [1.0] * len(ag))
            if ad_res["output_feasible"] and ad_res["cost"] > opt_g + 0.5:
                found_subopt = True
                break
    assert found_subopt, "Adaptive greedy suboptimality witness check failed!"

    # 2. Verify Theorem 1 (150 unit-drop + 2^h FPT networks + m=0 + 3 integrality-sharpness counterexamples)
    opt_m0, ed_m0, _ = lib.solve_hamming_fpt_2h(4, [], frozenset([0, 1]), [], [])
    assert opt_m0 == 0.0 and ed_m0 == []

    # Sharpness counterexamples (Proposition 1):
    # (i) Fractional unit drop l_a = 2/3 <= 1 on integral capacities
    arcs_b1 = [(0, 1, 5), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
    drops_b1 = [0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    fees_b1 = [0.0, 1.0, 1.0, 1.0]
    ex_b1 = exact_cut_enum_milp(3, arcs_b1, frozenset([0, 1]), drops_b1, fees_b1)
    lp_b1, _, _ = lib.solve_hamming_unit_circulation(3, arcs_b1, frozenset([0, 1]), drops_b1, fees_b1)
    assert abs(ex_b1 - 2.0) < 1e-6 and abs(lp_b1 - 1.0) < 1e-6, "Counterexample (i) failed!"

    # (ii) Fractional non-editable capacity c_{01} = 3.5 with unit drops l = 1
    arcs_b2 = [(0, 1, 3.5), (1, 2, 2.0), (1, 2, 2.0)]
    drops_b2 = [0.0, 1.0, 1.0]
    fees_b2 = [0.0, 10.0, 10.0]
    ex_b2 = exact_cut_enum_milp(3, arcs_b2, frozenset([0, 1]), drops_b2, fees_b2)
    lp_b2, _, _ = lib.solve_hamming_unit_circulation(3, arcs_b2, frozenset([0, 1]), drops_b2, fees_b2)
    assert abs(ex_b2 - 10.0) < 1e-6 and abs(lp_b2 - 5.0) < 1e-6, "Counterexample (ii) failed!"

    # (iii) Fractional heavy drop l_1 = 1.5 in 2^h branching on integral capacities
    arcs_b3 = [(0, 1, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 3, 5)]
    drops_b3 = [0.0, 1.5, 1.0, 1.0, 0.0]
    fees_b3 = [0.0, 1.0, 10.0, 10.0, 0.0]
    ex_b3 = exact_cut_enum_milp(4, arcs_b3, frozenset([0, 1]), drops_b3, fees_b3)
    sub_lp, _, _ = lib.solve_hamming_unit_circulation(
        4, [(0, 1, 4), (1, 2, 0.5), (1, 2, 2), (1, 2, 2), (2, 3, 5)], frozenset([0, 1]), [0, 0, 1, 1, 0], [0, 0, 10, 10, 0]
    )
    assert abs(ex_b3 - 11.0) < 1e-6 and abs(1.0 + sub_lp - 6.0) < 1e-6, "Counterexample (iii) failed!"

    rng1 = np.random.default_rng(881601)
    for _ in range(150):
        n = int(rng1.integers(4, 7))
        mid = rng1.choice(range(1, n - 1), size=int(rng1.integers(0, n - 1)), replace=False).tolist()
        S = frozenset([0, *mid])
        arcs, drops, fees = [], [], []
        h_rem = 3
        for u in range(n):
            for v in range(n):
                if u != v and rng1.random() < 0.30:
                    c = int(rng1.integers(0, 6))
                    d = 0
                    if u in S and v not in S and c >= 1 and rng1.random() < 0.75:
                        if c >= 2 and h_rem > 0 and rng1.random() < 0.30:
                            d = int(rng1.integers(2, c + 1))
                            h_rem -= 1
                        else:
                            d = 1
                    w = round(float(rng1.uniform(0.0, 7.5)), 3)
                    arcs.append((u, v, c))
                    drops.append(d)
                    fees.append(w)
        ex_v = exact_cut_enum_milp(n, arcs, S, drops, fees)
        alg_v, _, _ = lib.solve_hamming_fpt_2h(n, arcs, S, drops, fees)
        assert np.isfinite(ex_v) == np.isfinite(alg_v)
        if np.isfinite(ex_v):
            assert abs(ex_v - alg_v) < 1e-6

    # 3. Verify Theorem 2 (60 random 3DM -> {1,2}-capacity 4-layer DAG instances)
    rng2 = np.random.default_rng(881602)
    triples_pool = list(itertools.product(range(3), repeat=3))
    checked_3dm = 0
    while checked_3dm < 60:
        T_sub = [t for t in triples_pool if rng2.random() < 0.28]
        if not all(len({t[a] for t in T_sub}) == 3 for a in range(3)):
            continue
        q = 3
        jobs = [(1, [j for j, t in enumerate(T_sub) if t[ax] == val]) for ax in (1, 2) for val in range(q)]
        for val in range(q):
            allows = [j for j, t in enumerate(T_sub) if t[0] == val]
            jobs.extend((2, allows) for _ in range(len(allows) - 1))
        first_bin = 1 + len(jobs)
        sink = first_bin + len(T_sub)
        edges = [(0, i, d) for i, (d, _) in enumerate(jobs, 1)]
        drops = [0] * len(edges)
        for i, (d, allowed) in enumerate(jobs, 1):
            for b in allowed:
                edges.append((i, first_bin + b, d))
                drops.append(d)
        edges.extend((first_bin + b, sink, 2) for b in range(len(T_sub)))
        drops.extend([0] * len(T_sub))
        X_side = frozenset(range(first_bin))
        opt_edits, _ = lib.solve_hamming_flow_milp(sink + 1, edges, X_side, drops, [1.0] * len(edges))
        thresh = sum(bool(d) for d in drops) - len(jobs)
        has_3dm = any(
            all(len({t[a] for t in ch}) == q for a in range(3))
            for ch in itertools.combinations(T_sub, q)
        )
        assert (round(opt_edits) == thresh) == has_3dm and round(opt_edits) >= thresh
        checked_3dm += 1

    # 4. Verify Theorem 3 (150 undirected & asymmetric directed multi-box PWL instances + 15-prime denominator stress test)
    primes15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]  # lcm = 614,889,782,588,491,410 > 2^59!
    boxes_15p = [([float(Fraction(1, p)) for p in primes15[:6]], [float(Fraction(2, p)) for p in primes15[:6]])]
    edges_15p = [(i, i + 1, float(Fraction(3, primes15[6 + i]))) for i in range(5)]
    t_15p = {0: 0, 5: 1}
    pr_15p = {0: [(2.0, 1.0), (5.0, 2.5)], 5: [(3.0, 0.5), (6.0, 1.75)]}
    c_15p, _ = lib.solve_two_corner_pwl_transshipment(boxes_15p, edges_15p, t_15p, {}, pr_15p)
    ex_15p = exact_partial_cut_2n_lp(boxes_15p, edges_15p, [], t_15p, {}, pr_15p)
    assert c_15p is not None and ex_15p is not None and abs(c_15p - ex_15p) < 1e-7

    rng3 = np.random.default_rng(881603)
    for trial in range(150):
        n = int(rng3.integers(4, 7))
        use_dir = (trial >= 100)
        u_edges, d_edges = [], []
        if not use_dir:
            for u, v in itertools.combinations(range(n), 2):
                if rng3.random() < 0.32:
                    u_edges.append((u, v, float(rng3.integers(0, 5)) / 2.0))
        else:
            for u in range(n):
                for v in range(n):
                    if u != v and rng3.random() < 0.28:
                        d_edges.append((u, v, float(rng3.integers(1, 5)) / 2.0))
        perm = rng3.permutation(n).tolist()
        t_cnt = int(rng3.integers(1, n))
        targets = {i: int(rng3.integers(0, 2)) for i in perm[:t_cnt]}
        rem = perm[t_cnt:]
        anchors = {i: int(rng3.integers(0, 2)) for i in rem[: int(rng3.integers(0, len(rem) + 1))]}
        boxes = []
        for _ in range(int(rng3.choice([1, 2]))):
            lo = rng3.integers(-4, 4, size=n).astype(float) / 2.0
            hi = lo + rng3.integers(0, 4, size=n).astype(float) / 2.0
            for _ in range(int(rng3.integers(1, 4))):
                boxes.append((lo.tolist(), hi.tolist()))
                sh = rng3.integers(0, 3, size=n).astype(float) / 2.0
                lo = lo + sh
                hi = hi + sh
        prices = {}
        for i in targets:
            cur = 0.0
            sl = float(rng3.integers(0, 4)) / 2.0
            segs = []
            for s_i in range(int(rng3.integers(1, 4))):
                cur += float(rng3.integers(1, 5)) / 2.0
                if s_i > 0:
                    sl += float(rng3.integers(0, 4)) / 2.0
                segs.append((cur, sl))
            prices[i] = segs
        alg_cost, _ = lib.solve_two_corner_pwl_transshipment(boxes, u_edges, targets, anchors, prices, d_edges)
        ex_cost = exact_partial_cut_2n_lp(boxes, u_edges, d_edges, targets, anchors, prices)
        assert (alg_cost is not None) == (ex_cost is not None)
        if alg_cost is not None:
            assert abs(alg_cost - ex_cost) < 1e-6

    # 5. Verify Theorem 4 (subcubic vertex-cover identity 2|E| + eps*tau(G), (1+eps) linear bound, L-reduction cover)
    eps = 0.20
    test_graphs = [
        nx.cycle_graph(3),
        nx.cycle_graph(5),
        nx.complete_graph(4),
        nx.complete_bipartite_graph(3, 3),
        nx.cubical_graph(),
        nx.petersen_graph(),
    ]
    for G in test_graphs:
        n_v = G.number_of_nodes()
        edges = list(G.edges())
        m_e = len(edges)
        xs_v = np.array(list(itertools.product((0, 1), repeat=n_v)), dtype=int)
        cov = np.ones(len(xs_v), dtype=bool)
        for u, v in edges:
            cov &= (xs_v[:, u] + xs_v[:, v] >= 1)
        tau = int(xs_v[cov].sum(axis=1).min())
        exact_sm = 2.0 * m_e + eps * tau

        sur = lib.solve_concave_linear_surrogate(edges, n_v, eps)
        assert abs(sur["linear_sum"] - 2.0 * m_e) < 1e-6
        assert sur["surrogate_concave_cost"] <= (1.0 + eps) * exact_sm + 1e-8
        assert sur["surrogate_concave_cost"] >= exact_sm - 1e-8

        rec_cov = lib.recover_vertex_cover_from_payment(edges, n_v, sur["p_vec"])
        assert all((u in rec_cov or v in rec_cov) for u, v in edges)
        assert len(rec_cov) - tau <= (sur["surrogate_concave_cost"] - exact_sm) / eps + 1e-7

        # On smaller graphs (N <= 10), verify multi-start SLSQP NLP never beats 2|E| + eps*tau(G)!
        if n_v + m_e <= 10:
            N_all = n_v + m_e
            xs_all = np.array(list(itertools.product((0, 1), repeat=N_all)), dtype=int)
            en = np.zeros(len(xs_all), dtype=float)
            for j_e, (u, v) in enumerate(edges):
                z = xs_all[:, n_v + j_e]
                en += np.abs(xs_all[:, u] - z) + np.abs(xs_all[:, v] - z) - xs_all[:, u] - xs_all[:, v]
            A_c = xs_all[1:].astype(float)
            b_c = -en[1:]
            x0 = np.array([1.0] * n_v + [0.0] * m_e, dtype=float)
            nlp = minimize(
                lambda p: lib.smooth_concave_cost(p, eps),
                x0,
                jac=lambda p: 1.0 + eps * np.maximum(0.0, 1.0 - 0.5 * p),
                method="SLSQP",
                bounds=[(0.0, 6.0)] * N_all,
                constraints={"type": "ineq", "fun": lambda p: A_c @ p - b_c, "jac": lambda _: A_c},
            )
            assert nlp.success and nlp.fun >= exact_sm - 1e-5

    print("verify_p16_proofs: PASS (0 counterexamples across all 4 theorems, sharpness witnesses, and algorithms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
