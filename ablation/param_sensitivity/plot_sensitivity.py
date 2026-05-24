"""
DeltaMem 参数敏感性配图 v5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.titlepad":     4,
    "axes.labelsize":    13,
    "axes.labelpad":     3,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "ytick.major.pad":   4,
    "xtick.major.pad":   3,
    "legend.fontsize":   12,
    "legend.framealpha": 0.88,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
})

# 小清新绿色调：teal → 草绿 → 浅绿 → 淡绿 → 暖蜜
DEPTH_COLORS = {
    1: "#3A9E90",
    2: "#65B882",
    3: "#96CC80",
    4: "#C2E09A",
    5: "#E8CE78",
}

CSV_DEFAULT = os.path.join(os.path.dirname(__file__), "results", "sensitivity_all.csv")

GRID_VALUES = {
    "alfworld": {"tb": [0.70, 0.75, 0.80, 0.85], "eb": [0.80, 0.85, 0.88, 0.91]},
    "sciworld": {"tb": [0.80, 0.82, 0.84, 0.86], "eb": [0.82, 0.84, 0.86, 0.88]},
}
BM_LABEL   = {"alfworld": "ALFWorld", "sciworld": "SciWorld"}
METRIC_COL = {"alfworld": "sr",       "sciworld": "avg_reward"}
METRIC_FN  = {"alfworld": lambda v: v,      "sciworld": lambda v: v}
METRIC_LBL = {"alfworld": "Avg Reward",      "sciworld": "Avg Reward"}
METRIC_FMT = {"alfworld": "{:.3f}",          "sciworld": "{:.3f}"}

K_ORDER  = [3, 5, 8, 0]
K_LABELS = {3: "3", 5: "5", 8: "8", 0: "∞"}

K_COMP_TB_EB = {"alfworld": (0.70, 0.88), "sciworld": (0.82, 0.82)}
VARY_TB_CFG  = {
    "alfworld": {"eb": 0.85, "k": 8, "tb_vals": [0.70, 0.75, 0.80, 0.85]},
    "sciworld": {"eb": 0.88, "k": 5, "tb_vals": [0.80, 0.82, 0.84, 0.86]},
}
VARY_EB_CFG  = {
    "alfworld": {"tb": 0.70, "k": 8, "eb_vals": [0.80, 0.85, 0.88, 0.91]},
    "sciworld": {"tb": 0.80, "k": 8, "eb_vals": [0.82, 0.84, 0.86, 0.88]},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["avg_reward", "sr", "consolidation_count",
                "task_tree_total_nodes", "env_tree_total_nodes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_lc(s):
    if pd.isna(s) or not s:
        return {}
    try:
        return {int(k): int(v) for k, v in json.loads(s).items()}
    except Exception:
        return {}


def _get_row(df, benchmark, tb, eb, k):
    mask = (
        (df["benchmark"] == benchmark) &
        (df["tb"].round(4) == round(tb, 4)) &
        (df["eb"].round(4) == round(eb, 4)) &
        (df["k"] == k)
    )
    rows = df[mask]
    return rows.iloc[0] if len(rows) else None


def _bar_text_color(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    lum = 0.299*r + 0.587*g + 0.114*b
    return "#1a1a1a" if lum > 0.50 else "white"


def _save(fig, output_path, dpi):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── Figure 1: Heatmaps ────────────────────────────────────────────────────────

def plot_heatmaps(csv_path, benchmark, output_path, dpi=200):
    df  = _load(csv_path)
    sub = df[
        (df["benchmark"] == benchmark) &
        (df["sweep_type"].isin(["grid_sweep", "single"]))
    ].copy()
    if sub.empty:
        print(f"[heatmaps] No data for {benchmark}.")
        return

    grid    = GRID_VALUES[benchmark]
    tb_vals = grid["tb"]
    eb_vals = grid["eb"]
    mc, mfn = METRIC_COL[benchmark], METRIC_FN[benchmark]
    mfmt    = METRIC_FMT[benchmark]
    mlbl    = METRIC_LBL[benchmark]

    matrices = {}
    for k in K_ORDER:
        sk  = sub[sub["k"] == k]
        mat = np.full((len(tb_vals), len(eb_vals)), np.nan)
        for _, row in sk.iterrows():
            ti = tb_vals.index(round(float(row["tb"]), 4)) \
                 if round(float(row["tb"]), 4) in tb_vals else -1
            ei = eb_vals.index(round(float(row["eb"]), 4)) \
                 if round(float(row["eb"]), 4) in eb_vals else -1
            if ti >= 0 and ei >= 0:
                mat[ti, ei] = mfn(float(row[mc]))
        matrices[k] = mat

    all_v = np.concatenate([m.flatten() for m in matrices.values()])
    all_v = all_v[~np.isnan(all_v)]
    pad   = 0.001
    vmin, vmax = all_v.min() - pad, all_v.max() + pad

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)

    k_annot = {3: r"$K_\mathrm{cons}=3$", 5: r"$K_\mathrm{cons}=5$",
               8: r"$K_\mathrm{cons}=8$", 0: r"$K_\mathrm{cons}=\infty$"}
    letters  = ["(a)", "(b)", "(c)", "(d)"]

    ims = []
    for i, (ax, k, lbl) in enumerate(zip(axes, K_ORDER, letters)):
        mat = matrices[k]
        im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn", origin="lower",
                        interpolation="nearest", vmin=vmin, vmax=vmax)
        ims.append(im)

        ax.set_xticks(range(len(eb_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in eb_vals], rotation=30, ha="right")
        if i == 0:
            ax.set_yticks(range(len(tb_vals)))
            ax.set_yticklabels([f"{v:.2f}" for v in tb_vals])
            ax.set_ylabel(r"$\tau^\mathrm{task}_\mathrm{base}$")
        ax.text(0.5, 1.03, f"{lbl} {k_annot[k]}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=13)

        best_pos = np.unravel_index(np.nanargmax(mat), mat.shape) \
                   if not np.all(np.isnan(mat)) else None
        for ii in range(len(tb_vals)):
            for jj in range(len(eb_vals)):
                v = mat[ii, jj]
                if np.isnan(v):
                    continue
                rel = (v - vmin) / (vmax - vmin + 1e-9)
                fc  = "white" if rel > 0.62 else "black"
                ax.text(jj, ii, mfmt.format(v), ha="center", va="center",
                        fontsize=11, color=fc)
        if best_pos is not None:
            bi, bj = best_pos
            ax.add_patch(plt.Rectangle(
                (bj - 0.5, bi - 0.5), 1, 1,
                fill=False, edgecolor="#FFD700", linewidth=2.2, zorder=5))

    # x 轴标签先放在 axes[1]，让 tight_layout 确定正确的 y 距离
    axes[1].set_xlabel(r"$\tau^\mathrm{env}_\mathrm{base}$", fontsize=12)

    # 先 tight_layout，再读取真实坐标轴位置
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])

    # colorbar 与坐标轴完全对齐
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([0.924, pos.y0, 0.013, pos.height])

    # 将 xlabel 的 x 移到 axes[1] 右边缘与 axes[2] 左边缘的正中间
    pos1, pos2 = axes[1].get_position(), axes[2].get_position()
    x_center_fig = (pos1.x1 + pos2.x0) / 2
    # 转换为 axes[1] 的归一化 x 坐标
    new_x = (x_center_fig - pos1.x0) / (pos1.x1 - pos1.x0)
    axes[1].xaxis.label.set_x(new_x)
    cb = fig.colorbar(ims[0], cax=cbar_ax)
    cb.set_label(mlbl, fontsize=12)
    cb.ax.tick_params(labelsize=11)

    _save(fig, output_path, dpi)


# ── shared: horizontal stacked bar ───────────────────────────────────────────

def _draw_barh(ax, y_labels, lc_list):
    all_depths = sorted({d for lc in lc_list for d in lc if d > 0})
    bar_h = 0.54
    for yi, lc in enumerate(lc_list):
        left = 0
        for d in all_depths:
            cnt = lc.get(d, 0)
            color = DEPTH_COLORS.get(d, "#cccccc")
            ax.barh(yi, cnt, height=bar_h, left=left, color=color,
                    label=f"Depth {d}" if yi == 0 else "_nolegend_",
                    edgecolor="white", linewidth=0.4)
            left += cnt
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Node Count", fontsize=12)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _depth_legend(fig, fontsize=12):
    handles = [plt.Rectangle((0, 0), 1, 1, color=DEPTH_COLORS[d])
               for d in range(1, 6)]
    labels  = [f"Depth {d}" for d in range(1, 6)]
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.0), fontsize=fontsize,
               framealpha=0.88, columnspacing=0.9, handlelength=1.2,
               borderpad=0.4)


# ── Figure 2: K comparison ────────────────────────────────────────────────────

def plot_k_comparison(csv_path, output_path, dpi=200):
    df = _load(csv_path)
    trees   = [("task_tree_level_counts", "Task Tree"),
               ("env_tree_level_counts",  "Env Tree")]
    letters = [["(a)", "(b)"], ["(c)", "(d)"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    for ri, bm in enumerate(["alfworld", "sciworld"]):
        tb, eb = K_COMP_TB_EB[bm]
        for ci, (lc_col, tree_label) in enumerate(trees):
            lc_list = [
                _parse_lc(_get_row(df, bm, tb, eb, k)[lc_col])
                if _get_row(df, bm, tb, eb, k) is not None else {}
                for k in K_ORDER
            ]
            ax = axes[ri, ci]
            _draw_barh(ax, [K_LABELS[k] for k in K_ORDER], lc_list)
            ax.set_title(f"{letters[ri][ci]} {BM_LABEL[bm]} — {tree_label}",
                         fontsize=13, pad=4)
            if ci == 0:
                ax.set_ylabel(r"$K_\mathrm{cons}$", fontsize=12)

    _depth_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.36, wspace=0.22)
    _save(fig, output_path, dpi)


# ── Figure 3: vary_tb ────────────────────────────────────────────────────────

def plot_vary_tb(csv_path, output_path, dpi=200):
    df = _load(csv_path)
    trees   = [("task_tree_level_counts", "Task Tree"),
               ("env_tree_level_counts",  "Env Tree")]
    letters = [["(a)", "(b)"], ["(c)", "(d)"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    for ri, bm in enumerate(["alfworld", "sciworld"]):
        cfg = VARY_TB_CFG[bm]
        eb, k, tb_vals = cfg["eb"], cfg["k"], cfg["tb_vals"]
        for ci, (lc_col, tree_label) in enumerate(trees):
            lc_list = [
                _parse_lc(_get_row(df, bm, tb, eb, k)[lc_col])
                if _get_row(df, bm, tb, eb, k) is not None else {}
                for tb in tb_vals
            ]
            ax = axes[ri, ci]
            _draw_barh(ax, [f"{tb:.2f}" for tb in tb_vals], lc_list)
            ax.set_title(f"{letters[ri][ci]} {BM_LABEL[bm]} — {tree_label}",
                         fontsize=13, pad=4)
            if ci == 0:
                ax.set_ylabel(r"$\tau^\mathrm{task}_\mathrm{base}$", fontsize=12)

    _depth_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.36, wspace=0.22)
    _save(fig, output_path, dpi)


# ── Figure 4: vary_eb ────────────────────────────────────────────────────────

def plot_vary_eb(csv_path, output_path, dpi=200):
    df = _load(csv_path)
    trees   = [("task_tree_level_counts", "Task Tree"),
               ("env_tree_level_counts",  "Env Tree")]
    letters = [["(a)", "(b)"], ["(c)", "(d)"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    for ri, bm in enumerate(["alfworld", "sciworld"]):
        cfg = VARY_EB_CFG[bm]
        tb, k, eb_vals = cfg["tb"], cfg["k"], cfg["eb_vals"]
        for ci, (lc_col, tree_label) in enumerate(trees):
            lc_list = [
                _parse_lc(_get_row(df, bm, tb, eb, k)[lc_col])
                if _get_row(df, bm, tb, eb, k) is not None else {}
                for eb in eb_vals
            ]
            ax = axes[ri, ci]
            _draw_barh(ax, [f"{eb:.2f}" for eb in eb_vals], lc_list)
            ax.set_title(f"{letters[ri][ci]} {BM_LABEL[bm]} — {tree_label}",
                         fontsize=13, pad=4)
            if ci == 0:
                ax.set_ylabel(r"$\tau^\mathrm{env}_\mathrm{base}$", fontsize=12)

    _depth_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.36, wspace=0.22)
    _save(fig, output_path, dpi)


# ── Figure 5: combined trees (k_comparison | vary_tb | vary_eb) ──────────────

def plot_combined_trees(csv_path, output_path, dpi=200):
    df = _load(csv_path)
    trees = [("task_tree_level_counts", "Task Tree"),
             ("env_tree_level_counts",  "Env Tree")]

    # 8 GridSpec 列：panel0[0,1] | gap[2] | panel1[3,4] | gap[5] | panel2[6,7]
    fig = plt.figure(figsize=(21, 7))
    gs = fig.add_gridspec(
        2, 8,
        width_ratios=[1, 1, 0.04, 1, 1, 0.04, 1, 1],
        hspace=0.30, wspace=0.22,
    )

    COL_OFFSETS   = [0, 3, 6]
    PANEL_YLABELS = [
        r"$K_\mathrm{cons}$",
        r"$\tau^\mathrm{task}_\mathrm{base}$",
        r"$\tau^\mathrm{env}_\mathrm{base}$",
    ]
    TOP_LTRS    = list("abcdef")
    BOT_LTRS    = list("ghijkl")

    for ri, bm in enumerate(["alfworld", "sciworld"]):
        for pi, col_off in enumerate(COL_OFFSETS):
            for ci, (lc_col, tree_label) in enumerate(trees):
                ax = fig.add_subplot(gs[ri, col_off + ci])

                if pi == 0:
                    tb, eb = K_COMP_TB_EB[bm]
                    lc_list = [
                        _parse_lc(_get_row(df, bm, tb, eb, k)[lc_col])
                        if _get_row(df, bm, tb, eb, k) is not None else {}
                        for k in K_ORDER
                    ]
                    y_labels = [K_LABELS[k] for k in K_ORDER]
                elif pi == 1:
                    cfg = VARY_TB_CFG[bm]
                    eb_v, k_v, tb_vals = cfg["eb"], cfg["k"], cfg["tb_vals"]
                    lc_list = [
                        _parse_lc(_get_row(df, bm, tb, eb_v, k_v)[lc_col])
                        if _get_row(df, bm, tb, eb_v, k_v) is not None else {}
                        for tb in tb_vals
                    ]
                    y_labels = [f"{tb:.2f}" for tb in tb_vals]
                else:
                    cfg = VARY_EB_CFG[bm]
                    tb_v, k_v, eb_vals = cfg["tb"], cfg["k"], cfg["eb_vals"]
                    lc_list = [
                        _parse_lc(_get_row(df, bm, tb_v, eb, k_v)[lc_col])
                        if _get_row(df, bm, tb_v, eb, k_v) is not None else {}
                        for eb in eb_vals
                    ]
                    y_labels = [f"{eb:.2f}" for eb in eb_vals]

                _draw_barh(ax, y_labels, lc_list)
                ax.tick_params(labelsize=17)
                ax.xaxis.label.set_size(17)

                if ri == 0:
                    ax.set_xlabel("")   # xlabel 只在底行显示

                ltr = f"({TOP_LTRS[pi*2+ci]})" if ri == 0 else f"({BOT_LTRS[pi*2+ci]})"
                ax.set_title(f"{ltr} {BM_LABEL[bm]} — {tree_label}", fontsize=18, pad=4)

                if ci == 0:
                    ax.set_ylabel(PANEL_YLABELS[pi], fontsize=17)
                else:
                    ax.set_ylabel("")

    _depth_legend(fig, fontsize=18)
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.04, right=0.99,
                        hspace=0.30, wspace=0.22)
    _save(fig, output_path, dpi)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=CSV_DEFAULT)
    parser.add_argument("--benchmark",   choices=["alfworld", "sciworld"])
    parser.add_argument("--figure-type", required=True,
                        choices=["heatmaps", "k_comparison", "vary_tb", "vary_eb",
                                 "combined_trees"])
    parser.add_argument("--output",      default=None)
    parser.add_argument("--dpi",         type=int, default=200)
    args = parser.parse_args()

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")

    if args.figure_type == "heatmaps":
        bms = [args.benchmark] if args.benchmark else ["alfworld", "sciworld"]
        for bm in bms:
            out = args.output or os.path.join(fig_dir, f"{bm}_heatmaps.pdf")
            plot_heatmaps(args.input, bm, out, args.dpi)
    elif args.figure_type == "k_comparison":
        out = args.output or os.path.join(fig_dir, "tree_k_comparison.pdf")
        plot_k_comparison(args.input, out, args.dpi)
    elif args.figure_type == "vary_tb":
        out = args.output or os.path.join(fig_dir, "tree_vary_tb.pdf")
        plot_vary_tb(args.input, out, args.dpi)
    elif args.figure_type == "vary_eb":
        out = args.output or os.path.join(fig_dir, "tree_vary_eb.pdf")
        plot_vary_eb(args.input, out, args.dpi)
    elif args.figure_type == "combined_trees":
        out = args.output or os.path.join(fig_dir, "tree_combined.pdf")
        plot_combined_trees(args.input, out, args.dpi)


if __name__ == "__main__":
    main()
