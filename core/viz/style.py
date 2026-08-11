"""
style.py — one place that defines what every paper-bound figure looks like.

Enforces project plan §4.5:
  - all text in figures is English only (assert_ascii_only raises loudly if
    a non-ASCII character — i.e. almost certainly Chinese — is passed as a
    label/title/legend string, so mistakes are caught at generation time,
    not discovered after the figure is already in a draft).
  - a single, colorblind-friendlier qualitative palette (COLORS) used by all
    plotting helpers instead of matplotlib's default cycle.
  - font sizes, line widths, and export settings suitable for print
    (readable when scaled down to a two-column figure).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Colorblind-friendly qualitative palette (Okabe-Ito), used consistently
# across all plots. Index 0 is reserved as the "primary/method" color in
# comparison plots by convention — keep this ordering stable so a reader
# comparing two figures in the paper sees the same series in the same color.
COLORS = [
    "#0072B2",  # blue      - conventionally: our method / primary series
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (avoid for lines on white bg; ok for bars/fill)
    "#000000",  # black    - conventionally: baseline / reference line
]


def assert_ascii_only(*strings: str) -> None:
    """
    Raises ValueError if any given string contains non-ASCII characters.
    Call this on every title/xlabel/ylabel/legend-label string before
    plotting, so a stray Chinese character in a figure destined for the
    paper is caught immediately rather than discovered during a draft
    review.
    """
    for s in strings:
        if s is None:
            continue
        try:
            s.encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError(
                f"Non-ASCII text found in a figure string: {s!r}. "
                f"All paper-bound figures must use English-only text "
                f"(project plan §4.5)."
            ) from e


def apply_style() -> None:
    """
    Call once at the top of any plotting script/function before creating
    figures. Sets matplotlib rcParams for a consistent, print-suitable
    style. Idempotent — safe to call multiple times.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",   # always available, no missing-glyph
                                          # boxes; do NOT switch to a
                                          # CJK-capable font here — the point
                                          # is that figures never need CJK
                                          # glyphs in the first place.
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,               # on-screen preview
        "savefig.dpi": 300,               # export DPI floor per §4.5
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "pdf.fonttype": 42,               # embed fonts as TrueType, not
        "ps.fonttype": 42,                # Type3, so text stays editable/
                                            # selectable in the exported PDF
    })
