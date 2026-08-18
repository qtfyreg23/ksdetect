"""
plots.py — reusable, paper-ready plotting helpers built on top of style.py.

Every function here:
  - calls assert_ascii_only() on all text arguments before drawing anything,
  - returns (fig, ax) so the caller can do minor further tweaks if truly
    necessary, but should not need to for standard cases,
  - never plots a point estimate without its uncertainty when std/CI data is
    provided (project plan §4.5: "凡是带不确定性的结果,图上必须体现std或
    置信区间").
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import apply_style, assert_ascii_only, COLORS


def save_figure(fig, path: str) -> None:
    """
    Saves `fig` to `path`. If path ends in .pdf or .svg, saved as vector
    (preferred, per §4.5). If .png, uses the savefig.dpi set in apply_style
    (300, meets the >=300 DPI floor). Creates parent directories as needed.
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)


def line_with_ci(
    x: list[float],
    series: dict[str, dict],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (4.5, 3.2),
):
    """
    Line plot with a shaded confidence-interval band per series. Intended
    use case: layer-scan or sparsity-curve plots (AUC vs. layer depth, or
    AUC vs. k), one line per signal source / method.

    series: {
        "FFN": {"mean": [...], "ci_lower": [...], "ci_upper": [...]},
        "Residual": {"mean": [...], "ci_lower": [...], "ci_upper": [...]},
        ...
    }
    All three lists per series must be the same length as `x`.

    Series names ARE used as legend labels, so they must already be English
    (e.g. "FFN", "Attention", "Residual", not "前馈网络").
    """
    assert_ascii_only(title, xlabel, ylabel, *series.keys())
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, data) in enumerate(series.items()):
        color = COLORS[i % len(COLORS)]
        mean = np.asarray(data["mean"])
        ax.plot(x, mean, label=name, color=color, marker="o", markersize=3)
        if "ci_lower" in data and "ci_upper" in data:
            ax.fill_between(
                x, data["ci_lower"], data["ci_upper"],
                color=color, alpha=0.15, linewidth=0,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def bar_with_errorbars(
    labels: list[str],
    means: list[float],
    errors: list[float],
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (4.0, 3.0),
):
    """
    Bar chart with symmetric error bars (e.g. mean +/- (ci_upper-ci_lower)/2,
    or mean +/- std — pass whichever `errors` you intend, but be consistent
    across a figure and say which one it is in the caption when this is used
    in the paper).
    """
    assert_ascii_only(title, ylabel, *labels)
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(labels))
    ax.bar(
        x_pos, means, yerr=errors, capsize=3,
        color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cbar_label: str,
    figsize: tuple[float, float] = (6.0, 4.0),
    cmap: str = "viridis",
):
    """
    Heatmap for the wide-scan result grid (project plan §5): e.g. rows =
    signal source, columns = layer depth, cell value = mean AUC. Does NOT
    encode per-cell uncertainty visually (a heatmap has no natural room for
    error bars) — pair this with a companion table or line_with_ci plot of
    the specific row/column being discussed in the text when uncertainty
    matters for that claim.
    """
    assert_ascii_only(title, cbar_label, *row_labels, *col_labels)
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax
