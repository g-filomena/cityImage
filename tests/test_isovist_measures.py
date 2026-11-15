"""Isovist measures on ray fans whose answer is known."""

import numpy as np
import pytest

from cityImage.visibility2d import isovist_measures


def test_an_open_plane_is_a_circle_with_no_corners():
    k, r = 64, 100.0
    area, compactness, occlusions, occlusivity = isovist_measures(np.full(k, r), np.full(k, -1))
    assert area == pytest.approx(0.5 * k * r * r * np.sin(2 * np.pi / k))
    assert compactness == pytest.approx(1.0, abs=0.01)
    assert occlusions == 0
    assert occlusivity == 0.0


def test_a_facade_split_into_footprints_is_not_a_corner():
    radii = np.array([10.0, 10.5, 11.0, 10.5])
    hits = np.array([1, 2, 3, 4])
    assert isovist_measures(radii, hits)[2] == 0


def test_a_view_passing_a_corner_is_one_occluding_edge():
    radii = np.array([10.0, 10.0, 80.0, 80.0])
    hits = np.array([1, 1, -1, -1])
    area, compactness, occlusions, occlusivity = isovist_measures(radii, hits)
    # two jumps around the circle: 10 -> 80 and, wrapping round, 80 -> 10
    assert occlusions == 2
    assert occlusivity == pytest.approx(140.0)
