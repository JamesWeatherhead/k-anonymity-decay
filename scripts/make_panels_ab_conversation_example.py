#!/usr/bin/env python3
"""
Split-workflow figure generation (Panels A & B only).

This script intentionally does NOT attempt to typeset the conversation trace
(Panel C). Panel C is provided as LaTeX (tcolorbox + TikZ) so the text matches
the paper font exactly.

Outputs (vector PDFs):
  - results/figures/panel_a_conversation_k_decay.pdf
  - results/figures/panel_b_privacy_risk_key.pdf
  - results/figures/panel_ab_conversation_example.pdf (optional combined)

Notes (publication specs):
  - Serif + Computer Modern-like mathtext (no external LaTeX dependency)
  - Vector PDF export
  - Muted/colorblind-friendly palette
  - Log-scale with clean 10^x tick labels
  - Line weights >= 0.5 pt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------------------------------------------------------
# Matplotlib "publication" rcParams (Panel A & B)
# -----------------------------------------------------------------------------
mpl.rcParams.update(
    {
        # Academic look (Computer Modern). If system LaTeX exists, we can use it.
        "font.family": "serif",
        "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        # Vector export
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        # Minimum line weights: 0.5 pt ~= 0.7 px at 100 dpi; use >= 0.8 for safety
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 5.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        # Clean look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "0.6",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# -----------------------------------------------------------------------------
# Example data for the qualitative figure (Panels A & B)
# -----------------------------------------------------------------------------
TRACE = [
    {"turn": 0, "k": 133262},
    {"turn": 1, "k": 8943},
    {"turn": 2, "k": 892},
    {"turn": 3, "k": 156},
    {"turn": 4, "k": 23},
    {"turn": 5, "k": 8},
    {"turn": 6, "k": 3},
    {"turn": 7, "k": 1},
]


def _muted_palette():
    # Paper-friendly, desaturated palette (user-provided hexes).
    # NOTE: These are used in both Python (Panels A/B) and LaTeX (Panel C / UI)
    # to ensure perfect cross-panel color matching.
    return {
        "safe": "#4A79A7",    # muted blue
        "caution": "#D58C32", # muted orange
        "small": "#AE5A27",   # muted brown-red
        "unique": "#B07AA1",  # muted purple
        "ink": (0.15, 0.18, 0.22),
        "grid": (0.88, 0.88, 0.88),
    }


def _risk_bucket(k: int) -> str:
    # Convention (small-cell suppression style):
    # - Safe: k > 10
    # - Caution: 5 < k <= 10
    # - Small cell: 1 < k <= 5
    # - Unique: k = 1
    if k > 10:
        return "safe"
    if k > 5:
        return "caution"
    if k > 1:
        return "small"
    return "unique"


def _enable_usetex_if_available() -> None:
    """
    Use system LaTeX for text rendering if available, otherwise keep mathtext.
    This avoids hard failures on systems without TeX while still supporting
    the 'text.usetex: True' publication setting when possible.
    """
    has_tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    if has_tex:
        mpl.rcParams.update({"text.usetex": True})


def make_panel_a(output_pdf: Path) -> None:
    colors = _muted_palette()

    turns = [r["turn"] for r in TRACE]
    ks = [r["k"] for r in TRACE]
    buckets = [_risk_bucket(k) for k in ks]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(turns, ks, color=colors["ink"], marker="o", zorder=2)
    marker_by_bucket = {"safe": "o", "caution": "s", "small": "^", "unique": "X"}
    for t, k, b in zip(turns, ks, buckets):
        ax.scatter(
            t,
            k,
            s=60,
            marker=marker_by_bucket[b],
            color=colors[b],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    # Risk-zone background bands (reduce cognitive load in print/grayscale)
    y_top = max(ks) * 1.4
    ax.set_ylim(bottom=1, top=y_top)
    ax.axhspan(10, y_top, facecolor=colors["safe"], alpha=0.07, zorder=0)
    ax.axhspan(5, 10, facecolor=colors["caution"], alpha=0.08, zorder=0)
    ax.axhspan(1, 5, facecolor=colors["small"], alpha=0.08, zorder=0)

    # Threshold lines (match legend inequalities exactly)
    ax.axhline(10, color=colors["caution"], linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axhline(5, color=colors["small"], linestyle="--", linewidth=0.9, alpha=0.9)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("Turn (0 = before any disclosure)")
    ax.set_ylabel(r"Anonymity set size ($k$, log$_{10}$ scale)")
    ax.text(
        0.0,
        1.02,
        r"A. Illustrative $k$-anonymity contraction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(turns)
    ax.grid(True, axis="y", color=colors["grid"], linestyle="-", alpha=0.8)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


def make_panel_b(output_pdf: Path) -> None:
    colors = _muted_palette()

    fig, axb = plt.subplots(figsize=(5, 3))
    axb.axis("off")
    axb.text(
        0.0,
        1.02,
        "B. Privacy risk levels (key)",
        transform=axb.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    items = [
        ("Safe zone", r"$k > 10$", colors["safe"], "Low re-identification risk"),
        ("Caution zone", r"$5 < k \leq 10$", colors["caution"], "Review / generalize as needed"),
        ("Small cell", r"$1 < k \leq 5$", colors["small"], "High risk (suppress/generalize)"),
        ("Unique", r"$k = 1$", colors["unique"], "Uniquely identifiable"),
    ]
    y0 = 0.83
    dy = 0.2
    for i, (label, rng, col, desc) in enumerate(items):
        y = y0 - i * dy
        axb.add_patch(
            plt.Rectangle(
                (0.02, y - 0.055),
                0.11,
                0.11,
                transform=axb.transAxes,
                facecolor=col,
                edgecolor="white",
                linewidth=0.9,
            )
        )
        axb.text(0.17, y + 0.03, label, transform=axb.transAxes, fontsize=10, fontweight="bold")
        axb.text(0.17, y - 0.03, rng, transform=axb.transAxes, fontsize=10)
        axb.text(0.17, y - 0.085, desc, transform=axb.transAxes, fontsize=9, color="0.35")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


def make_panels_ab_combined(output_pdf: Path) -> None:
    """
    Optional combined A+B (still vector PDF). Keep this for convenience when
    assembling without a vector editor.
    """
    colors = _muted_palette()
    turns = [r["turn"] for r in TRACE]
    ks = [r["k"] for r in TRACE]
    buckets = [_risk_bucket(k) for k in ks]

    fig = plt.figure(figsize=(7.2, 3.4))
    # More spacing so panel headers never collide in tight-bbox export.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(turns, ks, color=colors["ink"], marker="o", zorder=2)
    marker_by_bucket = {"safe": "o", "caution": "s", "small": "^", "unique": "X"}
    for t, k, b in zip(turns, ks, buckets):
        ax.scatter(
            t,
            k,
            s=60,
            marker=marker_by_bucket[b],
            color=colors[b],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    y_top = max(ks) * 1.4
    ax.set_ylim(bottom=1, top=y_top)
    ax.axhspan(10, y_top, facecolor=colors["safe"], alpha=0.07, zorder=0)
    ax.axhspan(5, 10, facecolor=colors["caution"], alpha=0.08, zorder=0)
    ax.axhspan(1, 5, facecolor=colors["small"], alpha=0.08, zorder=0)

    ax.axhline(10, color=colors["caution"], linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axhline(5, color=colors["small"], linestyle="--", linewidth=0.9, alpha=0.9)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Turn (0 = before any disclosure)")
    ax.set_ylabel(r"Anonymity set size ($k$, log$_{10}$ scale)")
    ax.text(
        0.0,
        1.02,
        r"A. Illustrative $k$-anonymity contraction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(turns)
    ax.grid(True, axis="y", color=colors["grid"], linestyle="-", alpha=0.8)

    axb = fig.add_subplot(gs[0, 1])
    axb.axis("off")
    axb.text(
        0.0,
        1.02,
        "B. Privacy risk levels (key)",
        transform=axb.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    items = [
        ("Safe zone", r"$k > 10$", colors["safe"], "Low re-identification risk"),
        ("Caution zone", r"$5 < k \leq 10$", colors["caution"], "Review / generalize as needed"),
        ("Small cell", r"$1 < k \leq 5$", colors["small"], "High risk (suppress/generalize)"),
        ("Unique", r"$k = 1$", colors["unique"], "Uniquely identifiable"),
    ]
    y0 = 0.83
    dy = 0.2
    for i, (label, rng, col, desc) in enumerate(items):
        y = y0 - i * dy
        axb.add_patch(
            plt.Rectangle(
                (0.02, y - 0.055),
                0.11,
                0.11,
                transform=axb.transAxes,
                facecolor=col,
                edgecolor="white",
                linewidth=0.9,
            )
        )
        axb.text(0.17, y + 0.03, label, transform=axb.transAxes, fontsize=10, fontweight="bold")
        axb.text(0.17, y - 0.03, rng, transform=axb.transAxes, fontsize=10)
        axb.text(0.17, y - 0.085, desc, transform=axb.transAxes, fontsize=9, color="0.35")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    _enable_usetex_if_available()

    base = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
    )
    out_a = base / "panel_a_conversation_k_decay.pdf"
    out_b = base / "panel_b_privacy_risk_key.pdf"
    out_ab = base / "panel_ab_conversation_example.pdf"

    make_panel_a(out_a)
    print(f"Saved: {out_a}")

    make_panel_b(out_b)
    print(f"Saved: {out_b}")

    make_panels_ab_combined(out_ab)
    print(f"Saved: {out_ab}")


if __name__ == "__main__":
    main()


