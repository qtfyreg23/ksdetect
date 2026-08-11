"""
core.viz — publication-grade plotting, English-only, unified style.

Rule (project plan §4.5): all figures destined for the paper must go through
apply_style() and the helper functions here, so that every figure in the
paper shares fonts, colors, and formatting without hand-tuning each one.
Do not call matplotlib directly from an experiments/ script for any figure
that might end up in the paper — build a new helper here instead if the
existing ones (line_with_ci, heatmap, bar_with_errorbars) don't fit.
"""

from .style import apply_style, COLORS, assert_ascii_only
from .plots import line_with_ci, heatmap, bar_with_errorbars, save_figure

__all__ = [
    "apply_style",
    "COLORS",
    "assert_ascii_only",
    "line_with_ci",
    "heatmap",
    "bar_with_errorbars",
    "save_figure",
]
