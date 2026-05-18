"""Optional plotting helpers for cityImage.

This subpackage contains the old matplotlib-based plotting capability, kept out
of the lightweight core. Importing ``cityImage`` alone does not import
matplotlib; plotting dependencies are imported only when this subpackage or a
plotting symbol is accessed.
"""

from __future__ import annotations

from .colors import kindlmann, lighten_color, normalize, rand_cmap, random_colors_list
from .static import (
    MultiPlot,
    Plot,
    plot_gdf,
    plot_grid_gdf_columns,
    plot_grid_gdfs_column,
)

__all__ = [
    "MultiPlot",
    "Plot",
    "kindlmann",
    "lighten_color",
    "normalize",
    "plot_gdf",
    "plot_grid_gdf_columns",
    "plot_grid_gdfs_column",
    "rand_cmap",
    "random_colors_list",
]
