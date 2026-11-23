#!/usr/bin/env python3
"""
Build the three publication figures for Paper 16 (`p16_partial_inverse_cuts`):
  1. figures/fig1_hamming_dichotomy.pdf       (2 rows: equal-aspect 3-stage 4-node network/cut diagrams + 4-bar failure & log runtime)
  2. figures/fig2_multibox_transshipment.pdf  (2 rows: 2D box uncrossing + wide equal-aspect 2-copy transshipment + B & n scaling)
  3. figures/fig3_curvature_and_prices.pdf    (2x2 grid: concave split lens + zoomed % excess on 21 subcubic graphs + 2D polytope + price sweep)

All schematic panels use `set_aspect("equal")`, explicit wide container geometry, Bezier arcs routed outside boxes/nodes,
and strict artist + curve/border clearance checks alongside `tools/paperfig.py` and `tools/fig_preflight.py`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperfig as pf  # noqa: E402

FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"


def _draw_network_stage(
    ax: plt.Axes,
    panel_title: str,
    caps: dict[tuple[str, str], int],
    edited_edges: set[tuple[str, str]],
    highlight_mode: str,
    footer_text: str,
    footer_color: str,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.56, 2.68)
    ax.set_ylim(-1.72, 1.38)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_title(panel_title, fontsize=8.0, fontweight="bold", pad=3)

    pos = {
        "s": (0.00, 0.16),
        "u": (1.02, 0.70),
        "v": (1.02, -0.38),
        "t": (2.16, 0.16),
    }
    node_r = 0.185

    # 1. Spacious target cut box X^* = {s, u, v}
    hull = mpatches.FancyBboxPatch(
        (-0.34, -0.66),
        1.70,
        1.68,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#eef5ff",
        edgecolor=pf.PAL["ours"],
        linestyle=(0, (4, 2.5)),
        linewidth=1.1,
        alpha=0.65,
        zorder=1,
    )
    ax.add_patch(hull)
    ax.text(
        0.51,
        1.12,
        r"Target cut $X^*=\{s,u,v\}$",
        fontsize=6.8,
        color=pf.PAL["ours"],
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=5,
    )

    # 2. Competing cut {s, u} capsule in (b) and (c) — generous clearance around s, u, shaft, and label "4"
    if highlight_mode in ("greedy_fail", "opt_ok"):
        comp_col = pf.PAL["invalid"] if highlight_mode == "greedy_fail" else pf.PAL["comp2"]
        comp_poly = mpatches.Polygon(
            [
                (-0.27, -0.08),
                (-0.27, 0.41),
                (0.80, 0.96),
                (1.29, 0.96),
                (1.29, 0.42),
                (0.28, -0.11),
            ],
            closed=True,
            facecolor="#fdecec" if highlight_mode == "greedy_fail" else "none",
            edgecolor=comp_col,
            linestyle=":",
            linewidth=1.25,
            alpha=0.88 if highlight_mode == "greedy_fail" else 0.72,
            zorder=2,
        )
        ax.add_patch(comp_poly)

    # 3. Forward directed arcs with labels placed cleanly away from shafts & borders
    orig_caps = {("s", "u"): 4, ("u", "t"): 6, ("s", "v"): 2, ("v", "t"): 4}
    fwd_specs = [
        ("s", "u", (0.41, 0.60)),
        ("s", "v", (0.41, -0.29)),
        ("u", "t", (1.76, 0.64)),   # strictly outside X^* right border (x=1.40)
        ("v", "t", (1.76, -0.32)),  # strictly outside X^* right border
    ]
    for u, v, (lx, ly) in fwd_specs:
        c_now = caps[(u, v)]
        c_old = orig_caps[(u, v)]
        is_ed = (u, v) in edited_edges
        ec = (pf.PAL["comp1"] if highlight_mode == "greedy_fail" else pf.PAL["ours"]) if is_ed else "#334155"
        lw = 2.0 if is_ed else (1.05 + 0.14 * c_now)

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dist = float(np.hypot(dx, dy))
        ux, uy = dx / dist, dy / dist
        start = (x1 + ux * (node_r + 0.03), y1 + uy * (node_r + 0.03))
        end = (x2 - ux * (node_r + 0.03), y2 - uy * (node_r + 0.03))

        arrow = mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>,head_length=4.6,head_width=2.9",
            shrinkA=0.0,
            shrinkB=0.0,
            linewidth=lw,
            color=ec,
            zorder=3,
        )
        ax.add_patch(arrow)

        lbl = f"{c_old}→{c_now}" if is_ed else f"{c_now}"
        ax.text(
            lx,
            ly,
            lbl,
            fontsize=6.9,
            fontweight="bold" if is_ed else "normal",
            color=ec if is_ed else "#1f2937",
            ha="center",
            va="center",
            zorder=5,
        )

    # 4. Backward arc t -> s routed vertically down/up and cleanly UNDERNEATH X^* via cubic Bezier
    verts_back = [
        (pos["t"][0], pos["t"][1] - node_r - 0.03),
        (pos["t"][0], -1.10),
        (pos["s"][0], -1.10),
        (pos["s"][0], pos["s"][1] - node_r - 0.03),
    ]
    codes_back = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    back_arrow = mpatches.FancyArrowPatch(
        path=MplPath(verts_back, codes_back),
        arrowstyle="-|>,head_length=4.4,head_width=2.7",
        linewidth=1.15,
        color=pf.PAL["comp3"],
        zorder=3,
    )
    ax.add_patch(back_arrow)
    ax.text(
        1.08,
        -1.06,
        "back 10",
        fontsize=6.4,
        color="#475569",
        ha="center",
        va="top",
        zorder=4,
    )

    # 5. Circular vertices (true circles via equal aspect!)
    for name, (px, py) in pos.items():
        fill = "#ffffff" if name != "t" else "#f3f4f6"
        edge_c = pf.PAL["ours"] if name in ("s", "u", "v") else "#1f2937"
        circ = mpatches.Circle((px, py), radius=node_r, facecolor=fill, edgecolor=edge_c, linewidth=1.4, zorder=6)
        ax.add_patch(circ)
        ax.text(px, py, f"${name}$", fontsize=8.2, fontweight="bold", ha="center", va="center", color="#111827", zorder=7)

    # 6. Auto-sized rounded status pill (expands dynamically to text glyph bounds — impossible to overflow!)
    ax.text(
        1.06,
        -1.48,
        footer_text,
        fontsize=6.35,
        fontweight="bold",
        color=footer_color,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.34,rounding_size=0.20",
            facecolor="#f8fafc",
            edgecolor=footer_color,
            linewidth=1.05,
        ),
        zorder=7,
    )


def build_fig1(summary: dict) -> dict:
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.65))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.24, 1.10], hspace=0.42)
    gs_top = gs_outer[0].subgridspec(1, 3, wspace=0.12)
    gs_bot = gs_outer[1].subgridspec(1, 2, wspace=0.34)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])

    _draw_network_stage(
        ax_a,
        "(a) Initial Network",
        caps={("s", "u"): 4, ("u", "t"): 6, ("s", "v"): 2, ("v", "t"): 4, ("t", "s"): 10},
        edited_edges=set(),
        highlight_mode="init",
        footer_text="d(X*) = 10 > min {s} = 6",
        footer_color=pf.PAL["comp2"],
    )
    _draw_network_stage(
        ax_b,
        "(b) Single-Cut Greedy (1 edit)",
        caps={("s", "u"): 4, ("u", "t"): 1, ("s", "v"): 2, ("v", "t"): 4, ("t", "s"): 10},
        edited_edges={("u", "t")},
        highlight_mode="greedy_fail",
        footer_text="d({s,u}) = 3 < d(X*) = 5 [Fail]",
        footer_color=pf.PAL["invalid"],
    )
    _draw_network_stage(
        ax_c,
        "(c) Algorithm 1 (2 edits)",
        caps={("s", "u"): 4, ("u", "t"): 1, ("s", "v"): 2, ("v", "t"): 1, ("t", "s"): 10},
        edited_edges={("u", "t"), ("v", "t")},
        highlight_mode="opt_ok",
        footer_text="d(X*) = 2 ≤ cuts (3,5,6) [Opt]",
        footer_color=pf.PAL["ours"],
    )

    # Bottom-left (d): 4 side-by-side grouped bars
    ax_d = fig.add_subplot(gs_bot[0, 0])
    hs = [0, 2, 4, 6, 8]
    by_h = summary["exp1_hamming_dichotomy"]["by_h"]
    liu_infeas = [100.0 * by_h[str(h)]["liu2012_infeasible_rate"] for h in hs]
    liu_exact = [100.0 * by_h[str(h)]["liu2012_exact_opt_rate"] for h in hs]
    adap_exact = [100.0 * (1.0 - by_h[str(h)]["adaptive_greedy_suboptimal_rate"]) for h in hs]
    alg1_exact = [100.0 * by_h[str(h)]["alg1_exact_opt_rate"] for h in hs]

    x = np.arange(len(hs))
    w = 0.19
    ax_d.bar(x - 1.5 * w, liu_infeas, width=w, color=pf.PAL["invalid"], hatch="xx", edgecolor="white", label="Single-cut infeas.")
    ax_d.bar(x - 0.5 * w, liu_exact, width=w, color=pf.PAL["comp1"], label="Single-cut exact")
    ax_d.bar(x + 0.5 * w, adap_exact, width=w, color=pf.PAL["comp2"], hatch="..", edgecolor="white", label="Adaptive exact")
    ax_d.bar(x + 1.5 * w, alg1_exact, width=w, color=pf.PAL["ours"], label="Alg. 1 exact")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([str(h) for h in hs])
    ax_d.set_yticks([0, 25, 50, 75, 100])
    ax_d.set_ylim(0, 154)
    ax_d.set_xlabel(r"Heavy forward arcs $h = |\{a \in \delta^+(X^*): l_a \geq 2\}|$")
    ax_d.set_ylabel("Instances (%)")
    ax_d.set_title("(d) Greedy Failure vs. Exactness", fontsize=8.0, fontweight="bold")
    ax_d.legend(
        loc="upper center",
        ncol=2,
        fontsize=6.3,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    # Bottom-right (e): Log-scale runtime crossover with clean open-space callout
    ax_e = fig.add_subplot(gs_bot[0, 1])
    alg1_ms = [by_h[str(h)]["alg1_median_ms"] for h in hs]
    milp_ms = [by_h[str(h)]["milp_median_ms"] for h in hs]
    with (DATA_DIR / "exp1_hamming_greedy_vs_exact.csv").open(encoding="utf-8") as f:
        r_exp1 = list(csv.DictReader(f))
    adap_ms = [
        float(np.median([float(r["adaptive_greedy_ms"]) for r in r_exp1 if int(r["h_heavy_arcs"]) == h]))
        for h in hs
    ]

    ax_e.plot(hs, alg1_ms, color=pf.PAL["ours"], marker="o", linewidth=1.8, markersize=4.6, label=r"Alg. 1 (Circ. / $2^h$)")
    ax_e.plot(hs, milp_ms, color=pf.PAL["comp1"], marker="s", linewidth=1.6, linestyle="--", markersize=4.2, label="General Flow-MILP")
    ax_e.plot(hs, adap_ms, color=pf.PAL["comp2"], marker="^", linewidth=1.4, linestyle=":", markersize=4.2, label="Adaptive greedy")
    ax_e.set_yscale("log")
    ax_e.set_xticks(hs)
    ax_e.set_ylim(0.24, 420.0)
    ax_e.set_xlabel(r"Heavy forward arcs $h$")
    ax_e.set_ylabel("Median solve time (ms)")
    ax_e.set_title(r"(e) Runtime Crossover ($h \leq 2$ vs. $h \geq 4$)", fontsize=8.0, fontweight="bold")
    ax_e.annotate(
        "6.0× faster at h = 0",
        xy=(0.12, alg1_ms[0]),
        xytext=(2.35, 0.42),
        fontsize=6.6,
        color=pf.PAL["ours"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.95),
    )
    ax_e.legend(loc="upper left", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.95)

    pdf_out = FIG_DIR / "fig1_hamming_dichotomy.pdf"
    png_out = FIG_DIR / "fig1_hamming_dichotomy.png"
    return pf.finalize_figure(
        fig,
        pdf_out,
        script_path=Path(__file__),
        input_paths=[DATA_DIR / "P16_RESULTS.json", DATA_DIR / "exp1_hamming_greedy_vs_exact.csv"],
        preview_png=png_out,
        strict_overlap=True,
    )


def build_fig2(summary: dict) -> dict:
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.70))
    gs = GridSpec(2, 20, figure=fig, height_ratios=[1.22, 1.08], hspace=0.58, wspace=0.45)

    # Panel (a) [36% width]: 2D Unary Box-Corner Uncrossing
    ax_a = fig.add_subplot(gs[0, 0:7])
    boxes_2d = [
        (-2.2, 0.2, 1.3, 1.4),   # surviving corner 1: (-2.2, 0.2)
        (-1.6, 0.7, 1.4, 1.3),
        (-0.9, 1.1, 1.2, 1.2),
        (-0.3, -1.7, 1.4, 1.3),  # surviving corner 2: (-0.3, -1.7)
        (0.4, -1.1, 1.3, 1.4),
        (1.0, -0.5, 1.2, 1.3),
    ]
    cone1 = mpatches.Rectangle((-2.2, 0.2), 4.8, 2.5, facecolor="#dbeafe", alpha=0.30, edgecolor="none", zorder=1)
    cone2 = mpatches.Rectangle((-0.3, -1.7), 2.9, 4.4, facecolor="#ffedd5", alpha=0.30, edgecolor="none", zorder=1)
    ax_a.add_patch(cone1)
    ax_a.add_patch(cone2)

    for idx_b, (x0, y0, w_b, h_b) in enumerate(boxes_2d):
        is_surv = idx_b in (0, 3)
        ec = pf.PAL["ours"] if idx_b == 0 else (pf.PAL["comp1"] if idx_b == 3 else "#64748b")
        rect = mpatches.Rectangle(
            (x0, y0),
            w_b,
            h_b,
            facecolor="#cbd5e1" if not is_surv else "none",
            alpha=0.28 if not is_surv else 0.95,
            edgecolor=ec,
            linewidth=1.4 if is_surv else 0.85,
            linestyle="-" if is_surv else "--",
            zorder=2,
        )
        ax_a.add_patch(rect)
        if not is_surv:
            ax_a.plot(x0, y0, marker="x", color="#475569", markersize=4.8, markeredgewidth=1.2, zorder=4)

    ax_a.scatter([-2.2], [0.2], color=pf.PAL["ours"], s=40, zorder=6, label=r"Corner $\ell^{(1)}$")
    ax_a.scatter([-0.3], [-1.7], color=pf.PAL["comp1"], marker="s", s=38, zorder=6, label=r"Corner $\ell^{(2)}$")
    ax_a.plot([], [], marker="x", linestyle="None", color="#475569", markersize=4.8, label="Pruned")
    ax_a.set_xlim(-3.75, 2.55)
    ax_a.set_ylim(-2.90, 2.75)
    ax_a.set_xlabel(r"Potential $h_1$ ($t_i=0$)")
    ax_a.set_ylabel(r"Potential $h_2$")
    ax_a.set_title("(a) Box-Corner Uncrossing", fontsize=8.0, fontweight="bold")
    ax_a.legend(loc="lower left", fontsize=6.2, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (b) [60% width]: Spacious Equal-Aspect Two-Copy Reversed-Port PWL Transshipment Schematic
    ax_b = fig.add_subplot(gs[0, 8:20])
    ax_b.set_aspect("equal")
    ax_b.set_xlim(-0.32, 5.72)
    ax_b.set_ylim(-1.48, 1.28)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.axis("off")
    ax_b.set_title("(b) Two-Copy Reversed-Port PWL Transshipment", fontsize=8.0, fontweight="bold", pad=3)

    # Spacious Copy A & Copy B containers with an extra-wide 2.40-unit central bridge corridor!
    boxA = mpatches.FancyBboxPatch(
        (-0.14, -0.66),
        1.62,
        1.50,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor="#eff6ff",
        edgecolor=pf.PAL["ours"],
        linewidth=1.05,
        linestyle="--",
        zorder=1,
    )
    boxB = mpatches.FancyBboxPatch(
        (3.92, -0.66),
        1.62,
        1.50,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor="#fff7ed",
        edgecolor=pf.PAL["comp1"],
        linewidth=1.05,
        linestyle="--",
        zorder=1,
    )
    ax_b.add_patch(boxA)
    ax_b.add_patch(boxB)
    ax_b.text(0.67, 0.95, "Copy A (Reversed)", fontsize=6.7, color=pf.PAL["ours"], ha="center", va="bottom", fontweight="bold")
    ax_b.text(4.73, 0.95, "Copy B (Forward)", fontsize=6.7, color=pf.PAL["comp1"], ha="center", va="bottom", fontweight="bold")

    npos = {
        "sA": (0.18, 0.46),
        "tA": (0.18, -0.22),
        "pA1": (1.16, 0.46),
        "pA2": (1.16, -0.22),
        "pB1": (4.24, 0.46),
        "pB2": (4.24, -0.22),
        "sB": (5.22, 0.46),
        "tB": (5.22, -0.22),
    }
    nlabels = {
        "sA": r"$s_A$", "tA": r"$t_A$", "pA1": r"$q_{1}^A$", "pA2": r"$q_{2}^A$",
        "pB1": r"$q_{1}^B$", "pB2": r"$q_{2}^B$", "sB": r"$s_B$", "tB": r"$t_B$",
    }
    nr = 0.21

    internal_arrows = [
        ("tA", "pA1", pf.PAL["ours"], 0.0),
        ("tA", "pA2", pf.PAL["ours"], 0.0),
        ("pA1", "sA", pf.PAL["ours"], 0.0),
        ("tA", "sA", pf.PAL["ours"], 0.0),
        ("sB", "pB1", pf.PAL["comp1"], 0.0),
        ("sB", "tB", pf.PAL["comp1"], 0.0),
        ("pB1", "tB", pf.PAL["comp1"], 0.0),
        ("pB2", "tB", pf.PAL["comp1"], 0.0),
    ]
    for u, v, col, rad in internal_arrows:
        x1, y1 = npos[u]
        x2, y2 = npos[v]
        dx, dy = x2 - x1, y2 - y1
        d_len = float(np.hypot(dx, dy))
        ux, uy = dx / d_len, dy / d_len
        ax_b.add_patch(mpatches.FancyArrowPatch(
            (x1 + ux * (nr + 0.03), y1 + uy * (nr + 0.03)),
            (x2 - ux * (nr + 0.03), y2 - uy * (nr + 0.03)),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=4.0,head_width=2.5",
            shrinkA=0.0,
            shrinkB=0.0,
            linewidth=1.15,
            color=col,
            zorder=3,
        ))

    # Parallel PWL rebate bridges across the open 2.34-unit center corridor (1.53 -> 3.87)
    for pA, pB in [("pA1", "pB1"), ("pA2", "pB2")]:
        x1, y1 = npos[pA]
        x2, y2 = npos[pB]
        for dy_off, sl_style in [(0.065, "-"), (-0.065, "--")]:
            ax_b.add_patch(mpatches.FancyArrowPatch(
                (x1 + nr + 0.03, y1 + dy_off),
                (x2 - nr - 0.03, y2 + dy_off),
                arrowstyle="-|>,head_length=4.2,head_width=2.6",
                shrinkA=0.0,
                shrinkB=0.0,
                linewidth=1.35,
                linestyle=sl_style,
                color=pf.PAL["ours_alt"],
                zorder=4,
            ))

    ax_b.text(
        2.70,
        0.69,
        r"PWL bridges $z_i = u_i - p_i$",
        fontsize=6.1,
        color=pf.PAL["ours_alt"],
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax_b.text(
        2.70,
        0.12,
        r"slopes $-s_{i,1}, \dots, -s_{i,K}$",
        fontsize=5.95,
        color=pf.PAL["ours_alt"],
        ha="center",
        va="center",
    )

    # True t_B -> t_A return arc routed vertically down/up and cleanly UNDERNEATH both Copy A and Copy B!
    ret_verts = [
        (npos["tB"][0], npos["tB"][1] - nr - 0.03),
        (npos["tB"][0], -1.16),
        (npos["tA"][0], -1.16),
        (npos["tA"][0], npos["tA"][1] - nr - 0.03),
    ]
    ret_codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax_b.add_patch(mpatches.FancyArrowPatch(
        path=MplPath(ret_verts, ret_codes),
        arrowstyle="-|>,head_length=4.4,head_width=2.7",
        linewidth=1.25,
        linestyle=":",
        color="#334155",
        zorder=4,
    ))
    ax_b.text(
        2.70,
        -1.12,
        r"Return flow $\sum_i z_i$ ($t_B \to t_A$)",
        fontsize=6.6,
        color="#334155",
        ha="center",
        va="top",
    )

    for k_n, (nx_c, ny_c) in npos.items():
        ec = pf.PAL["ours"] if "A" in k_n else pf.PAL["comp1"]
        ax_b.add_patch(mpatches.Circle((nx_c, ny_c), radius=nr, facecolor="white", edgecolor=ec, linewidth=1.35, zorder=5))
        ax_b.text(nx_c, ny_c, nlabels[k_n], fontsize=7.0, ha="center", va="center", zorder=6)

    # Panel (c): Box-count scaling B in {1..32} — annotation sits in clean empty wedge between curves
    ax_c = fig.add_subplot(gs[1, 0:9])
    Bs = [1, 2, 4, 8, 16, 32]
    by_B = summary["exp2_multibox_transshipment"]["by_B_at_n10"]
    alg2_B = [by_B[str(b)]["alg2_median_ms"] for b in Bs]
    multi_B = [by_B[str(b)]["ununcrossed_multicopy_median_ms"] for b in Bs]
    enum_B = [by_B[str(b)]["explicit_2n_median_ms"] for b in Bs]

    ax_c.plot(Bs, alg2_B, color=pf.PAL["ours"], marker="o", linewidth=1.8, markersize=4.6, label="Alg. 2 (Uncrossed 2-Copy)")
    ax_c.plot(Bs, multi_B, color=pf.PAL["comp1"], marker="s", linewidth=1.6, linestyle="--", markersize=4.3, label="Multi-copy LP (no uncross.)")
    ax_c.plot(Bs, enum_B, color=pf.PAL["comp2"], marker="^", linewidth=1.4, linestyle=":", markersize=4.3, label=r"Explicit $2^n$ cut LP")
    ax_c.annotate(
        "",
        xy=(32, alg2_B[-1] + 1.5),
        xytext=(32, multi_B[-1] - 1.5),
        arrowprops=dict(arrowstyle="<->", color=pf.PAL["ours"], lw=1.1),
    )
    ax_c.text(
        29.8,
        13.2,
        "8.3× faster\nat B = 32",
        fontsize=6.4,
        color=pf.PAL["ours"],
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax_c.set_xscale("log", base=2)
    ax_c.set_xlim(0.82, 38.0)
    ax_c.set_xticks(Bs)
    ax_c.set_xticklabels([str(b) for b in Bs])
    ax_c.set_ylim(0, 66)
    ax_c.set_xlabel(r"Uncertainty boxes $B$ ($n=10$, 4-seg PWL)")
    ax_c.set_ylabel("Median solve time (ms)")
    ax_c.set_title("(c) Scaling with Box Count $B$", fontsize=8.0, fontweight="bold")
    ax_c.legend(loc="upper left", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (d): Graph-size scaling n in {10..160} at B = 16
    ax_d = fig.add_subplot(gs[1, 11:20])
    ns = [10, 20, 40, 80, 160]
    by_n = summary["exp2_multibox_transshipment"]["by_n_at_B16"]
    alg2_n = [by_n[str(n_v)]["alg2_median_ms"] for n_v in ns]
    multi_n = [by_n[str(n_v)]["ununcrossed_multicopy_median_ms"] for n_v in ns]

    ax_d.plot(ns, alg2_n, color=pf.PAL["ours"], marker="o", linewidth=1.8, markersize=4.6, label="Alg. 2 (Uncrossed 2-Copy)")
    ax_d.plot(ns, multi_n, color=pf.PAL["comp1"], marker="s", linewidth=1.6, linestyle="--", markersize=4.3, label="Multi-copy LP (B = 16)")
    ax_d.annotate(
        "",
        xy=(160, alg2_n[-1] * 1.12),
        xytext=(160, multi_n[-1] * 0.88),
        arrowprops=dict(arrowstyle="<->", color=pf.PAL["ours"], lw=1.1),
    )
    ax_d.text(
        148.0,
        72.0,
        "8.7× faster\nat n = 160",
        fontsize=6.4,
        color=pf.PAL["ours"],
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax_d.set_xscale("log", base=2)
    ax_d.set_yscale("log")
    ax_d.set_xlim(8.5, 190.0)
    ax_d.set_xticks(ns)
    ax_d.set_xticklabels([str(n_v) for n_v in ns])
    ax_d.set_ylim(2.8, 620.0)
    ax_d.set_xlabel(r"Graph vertices $n$ ($B=16$ boxes, up to 455 arcs)")
    ax_d.set_ylabel("Median solve time (ms, log)")
    ax_d.set_title("(d) Scaling with Graph Size $n$ (0.0 obj. error)", fontsize=8.0, fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.95)

    pdf_out = FIG_DIR / "fig2_multibox_transshipment.pdf"
    png_out = FIG_DIR / "fig2_multibox_transshipment.png"
    return pf.finalize_figure(
        fig,
        pdf_out,
        script_path=Path(__file__),
        input_paths=[DATA_DIR / "P16_RESULTS.json", DATA_DIR / "exp2_multibox_scaling.csv"],
        preview_png=png_out,
        strict_overlap=True,
    )


def build_fig3(summary: dict) -> dict:
    """
    Upgraded spacious 2x2 visualization for Figure 3 (`6.8 x 4.45 in`):
      (a) Nonlinear Curvature Deviation C(p) - p (dramatically reveals the concave split-penalty lens vs convex PWL!)
      (b) Zoomed Subcubic Benchmark (% excess over 2|E| + eps*tau(G), making every single one of the 21 graphs readable!)
      (c) Geometric 2D Feasible-Payment Polytopes (visually proving why floor dominance freezes p* while coupled facets switch!)
      (d) Empirical Price-Ratio Switching & Cost Reduction across R in {1x..16x}
    """
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.50))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.05], hspace=0.52, wspace=0.34)

    # Panel (a): Nonlinear Surcharge / Curvature Profile C(p) - 0.75*p vs Payment p_w
    # Plotting C(p) with eps_vis = 1.10 and shaded concave lens between C_sm(p) and secant chord L(p) on [0, 2]
    ax_a = fig.add_subplot(gs[0, 0])
    p_grid = np.linspace(0.0, 2.5, 240)
    eps_vis = 1.05
    c_lin = p_grid
    c_sm = p_grid + eps_vis * (1.0 - np.maximum(0.0, 1.0 - 0.5 * p_grid) ** 2)
    c_sec = np.minimum((1.0 + 0.5 * eps_vis) * p_grid, p_grid + eps_vis)
    c_pwl = np.where(p_grid <= 1.1, 0.65 * p_grid, 0.715 + 1.55 * (p_grid - 1.1))

    mask_lens = p_grid <= 2.0
    ax_a.fill_between(
        p_grid[mask_lens],
        c_sec[mask_lens],
        c_sm[mask_lens],
        color="#fecdd3",
        alpha=0.55,
        zorder=1,
    )
    ax_a.plot(p_grid, c_sm, color=pf.PAL["invalid"], linewidth=1.85, label=r"Concave $C_{\mathrm{sm}}$ (APX-hard)", zorder=4)
    ax_a.plot(p_grid, c_sec, color=pf.PAL["invalid"], linewidth=1.0, linestyle="--", alpha=0.80, label=r"Secant $L(p)$", zorder=3)
    ax_a.plot(p_grid, c_pwl, color=pf.PAL["ours"], linewidth=1.75, linestyle="-.", label="Convex PWL (Alg. 2)", zorder=4)
    ax_a.plot(p_grid, c_lin, color=pf.PAL["comp2"], linewidth=1.2, linestyle=":", label=r"Linear surr. $p$", zorder=2)

    ax_a.vlines(1.0, 1.0 + 0.5 * eps_vis, 1.0 + 0.75 * eps_vis, color=pf.PAL["invalid"], linewidth=2.1, zorder=5)
    ax_a.annotate(
        r"Split penalty $+\frac{\epsilon}{4}\bar{p}(2-\bar{p})$",
        xy=(1.0, 1.0 + 0.75 * eps_vis),
        xytext=(0.06, 2.92),
        fontsize=6.3,
        color=pf.PAL["invalid"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=pf.PAL["invalid"], lw=0.95),
    )
    ax_a.set_xlim(0.0, 2.5)
    ax_a.set_yticks([0, 1, 2, 3, 4])
    ax_a.set_ylim(0.0, 5.25)
    ax_a.set_xlabel(r"Site payment $p_w$")
    ax_a.set_ylabel(r"Payment cost $C(p_w)$")
    ax_a.set_title("(a) Cost Curvature & Concave Split Lens", fontsize=8.0, fontweight="bold")
    ax_a.legend(
        loc="upper center",
        ncol=2,
        fontsize=5.65,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        columnspacing=0.65,
        handlelength=1.45,
        handletextpad=0.32,
    )

    # Panel (b): Zoomed Excess (%) Over Exact Optimum 2|E| + eps*tau(G) across all 21 Subcubic Graphs!
    ax_b = fig.add_subplot(gs[0, 1])
    with (DATA_DIR / "exp3_curvature_and_prices.csv").open(encoding="utf-8") as f:
        r_exp3 = list(csv.DictReader(f))
    vc_rows = [r for r in r_exp3 if r["part"] == "concave_vc_gadget"]
    vc_rows.sort(key=lambda r: (int(r["m_edges"]), int(r["n_verts"])))

    g_idx = np.arange(1, len(vc_rows) + 1)
    alg3_excess_pct = [100.0 * (float(r["alg3_approx_ratio"]) - 1.0) for r in vc_rows]
    nlp_x = [i + 1 for i, r in enumerate(vc_rows) if r["multistart_nlp_min"] != ""]
    nlp_excess_pct = [
        100.0 * (float(r["multistart_nlp_min"]) / float(r["exact_concave_opt"]) - 1.0)
        for r in vc_rows if r["multistart_nlp_min"] != ""
    ]
    mean_excess = float(np.mean(alg3_excess_pct))
    peak_i = int(np.argmax(alg3_excess_pct))

    ax_b.axhline(0.0, color=pf.PAL["comp3"], linestyle=":", linewidth=1.0)
    ax_b.axhline(mean_excess, color=pf.PAL["ours_alt"], linestyle="--", linewidth=1.15, label=f"Mean (+{mean_excess:.2f}%)")
    ax_b.scatter(nlp_x, nlp_excess_pct, facecolors="none", edgecolors=pf.PAL["comp1"], s=30, linewidth=1.2, label="SLSQP", zorder=4)
    ax_b.plot(g_idx, alg3_excess_pct, color=pf.PAL["ours"], marker="^", markersize=4.4, linewidth=1.5, label="Lin. surr. (Thm. 4c)", zorder=5)

    ax_b.annotate(
        f"Max +{alg3_excess_pct[peak_i]:.2f}% (ratio 1.0465)",
        xy=(peak_i + 1, alg3_excess_pct[peak_i]),
        xytext=(5.5, 6.05),
        fontsize=6.3,
        color=pf.PAL["ours"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9),
    )
    ax_b.set_xlim(0.2, 21.8)
    ax_b.set_xticks([1, 5, 10, 15, 20])
    ax_b.set_ylim(-0.60, 11.20)
    ax_b.set_xlabel("21 connected subcubic graphs (ordered by $|E|$)")
    ax_b.set_ylabel("Excess over exact opt. (%)")
    ax_b.set_title(r"(b) Subcubic Excess ($\epsilon=0.25$, cap $+25\%$)", fontsize=8.0, fontweight="bold")
    ax_b.legend(loc="upper center", ncol=3, fontsize=5.8, frameon=True, facecolor="white", framealpha=0.95, columnspacing=0.65, handletextpad=0.3)

    # Panel (c): Geometric 2D Feasible-Payment Polytopes Explaining Floor Invariance vs Coupled Switching
    ax_c = fig.add_subplot(gs[1, 0])
    # Coupled polytope: p1 + p2 >= 2.50, p1 >= 0.45, p2 >= 0.45 -> two extreme corners (2.05, 0.45) and (0.45, 2.05)
    x_c = np.linspace(0.45, 3.05, 120)
    y_c_low = np.maximum(0.45, 2.50 - x_c)
    ax_c.fill_between(x_c, y_c_low, 3.00, facecolor="#ffedd5", alpha=0.55, zorder=1)
    ax_c.plot([0.45, 0.45, 2.05, 3.05], [3.00, 2.05, 0.45, 0.45], color=pf.PAL["comp1"], linestyle="--", linewidth=1.3, zorder=2)
    # Floor-dominated cone: p1 >= 1.48, p2 >= 1.48 -> single dominant corner (1.48, 1.48)
    x_f = np.linspace(1.48, 3.05, 60)
    ax_c.fill_between(x_f, 1.48, 3.00, facecolor="#dbeafe", alpha=0.72, zorder=2)
    ax_c.plot([1.48, 1.48, 3.05], [3.00, 1.48, 1.48], color=pf.PAL["ours"], linestyle="-", linewidth=1.4, zorder=3)
    ax_c.scatter([1.48], [1.48], color=pf.PAL["ours"], s=44, zorder=5, label=r"Floor $\underline{p}$ (0% switch)")
    ax_c.scatter([2.05, 0.45], [0.45, 2.05], color=pf.PAL["comp1"], marker="s", s=38, zorder=5, label="Coupled (switches)")
    ax_c.annotate(
        r"All $c > 0$ hit $\underline{p}$",
        xy=(1.43, 1.43),
        xytext=(0.14, 0.85),
        fontsize=6.3,
        color=pf.PAL["ours"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9),
    )
    ax_c.set_xlim(0.0, 3.05)
    ax_c.set_yticks([0, 1, 2, 3])
    ax_c.set_ylim(0.0, 3.72)
    ax_c.set_xlabel(r"Target site payment $p_1$")
    ax_c.set_ylabel(r"Target site payment $p_2$")
    ax_c.set_title("(c) Floor Cone vs. Coupled Polytope", fontsize=8.0, fontweight="bold")
    ax_c.legend(
        loc="upper center",
        ncol=2,
        fontsize=5.75,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        columnspacing=0.65,
        handletextpad=0.28,
    )

    # Panel (d): Empirical Price-Ratio Sweep (R in {1x..16x})
    ax_d = fig.add_subplot(gs[1, 1])
    ratios_R = [1, 2, 4, 8, 16]
    p_sweep = summary["exp3_curvature_and_prices"]["price_ratio_sweep_20_pairs"]
    sw_floor = [100.0 * p_sweep["coordinatewise_floor_switch_rate_by_ratio"][str(r)] for r in ratios_R]
    sw_coupled = [100.0 * p_sweep["coupled_two_corner_switch_rate_by_ratio"][str(r)] for r in ratios_R]
    sav_coupled = [100.0 * p_sweep["coupled_two_corner_mean_cost_saving_by_ratio"][str(r)] for r in ratios_R]

    ax_d.plot(ratios_R, sw_coupled, color=pf.PAL["ours"], marker="o", linewidth=1.8, markersize=4.5, label="Coupled: switched % (85%)")
    ax_d.plot(ratios_R, sav_coupled, color=pf.PAL["ours_alt"], marker="s", linestyle="--", linewidth=1.65, markersize=4.3, label="Coupled: cost saved %")
    ax_d.plot(ratios_R, sw_floor, color=pf.PAL["comp1"], marker="D", linestyle=":", linewidth=1.55, markersize=4.3, label="Floor: switched % (0%)")
    ax_d.annotate(
        "30.9% saved\nat 16×",
        xy=(16, sav_coupled[-1]),
        xytext=(5.2, 46.0),
        fontsize=6.4,
        color=pf.PAL["ours_alt"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=pf.PAL["ours_alt"], lw=0.9),
    )
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(ratios_R)
    ax_d.set_xticklabels([f"{r}×" for r in ratios_R])
    ax_d.set_yticks([0, 25, 50, 75, 100])
    ax_d.set_ylim(-7, 138)
    ax_d.set_xlabel(r"Regional price ratio $c_{\mathrm{high}} / c_{\mathrm{low}}$")
    ax_d.set_ylabel("Rate / Cost Saving (%)")
    ax_d.set_title("(d) Empirical Price Switching (20 pairs)", fontsize=8.0, fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white", framealpha=0.95)

    pdf_out = FIG_DIR / "fig3_curvature_and_prices.pdf"
    png_out = FIG_DIR / "fig3_curvature_and_prices.png"
    return pf.finalize_figure(
        fig,
        pdf_out,
        script_path=Path(__file__),
        input_paths=[DATA_DIR / "P16_RESULTS.json", DATA_DIR / "exp3_curvature_and_prices.csv"],
        preview_png=png_out,
        strict_overlap=True,
    )


def main() -> int:
    summary = json.loads((DATA_DIR / "P16_RESULTS.json").read_text(encoding="utf-8"))
    p1 = build_fig1(summary)
    p2 = build_fig2(summary)
    p3 = build_fig3(summary)
    print("Built 3 redesigned vector figures cleanly:")
    for pr in (p1, p2, p3):
        print(f"  - {pr['figure']} (fonttype={pr['pdf_fonttype']}, overlaps={len(pr['overlap_issues'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
