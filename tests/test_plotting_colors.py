"""Offline tests for colour helpers in ``cityImage.plotting.colors``."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cityImage.plotting.colors import (
    kindlmann,
    lighten_color,
    normalize,
    rand_cmap,
    random_colors_list,
)


def test_random_colors_list_returns_hsv_triples():
    np.random.seed(0)
    colors = random_colors_list(5)
    assert len(colors) == 5
    assert all(len(c) == 3 for c in colors)

    # hsv=True exercises the RGB-conversion branch (return value stays HSV list).
    assert len(random_colors_list(3, hsv=True)) == 3


def test_rand_cmap_bright_soft_and_invalid_type():
    np.random.seed(1)
    bright = rand_cmap(4, type_color="bright")
    assert isinstance(bright, LinearSegmentedColormap) and bright.N == 4  # one colour per label
    assert rand_cmap(7, type_color="soft").N == 7
    # An unknown type falls back to "bright" (still builds a valid N-colour map).
    assert rand_cmap(4, type_color="unknown").N == 4


def test_kindlmann_returns_colormap():
    cmap = kindlmann()
    assert isinstance(cmap, LinearSegmentedColormap)
    # Kindlmann runs dark->bright: the top of the map is markedly lighter than the bottom.
    assert sum(cmap(1.0)[:3]) > sum(cmap(0.0)[:3])


def test_normalize_maps_between_ranges():
    assert normalize(5, [0, 10], [0, 100]) == 50
    assert normalize(0, [0, 10], [-1, 1]) == -1


def test_lighten_color_named_and_shorthand():
    # Named colour hits the cnames lookup; single-letter hits the KeyError fallback.
    lightened_named = lighten_color("green", amount=0.5)
    lightened_short = lighten_color("g", amount=0.5)
    assert len(lightened_named) == 3
    assert len(lightened_short) == 3
    # Smaller amount lightens more (toward white); larger amount stays closer to the base colour.
    assert sum(lighten_color("green", amount=0.1)) > sum(lighten_color("green", amount=0.8))
