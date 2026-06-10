"""Tests for the hard angle core cleanup."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString

import cityImage as ci


def test_get_coord_angle_uses_historical_y_axis_convention():
    assert ci.get_coord_angle((0, 0), 10, 0) == pytest.approx((0, 10))
    assert ci.get_coord_angle((0, 0), 10, 90) == pytest.approx((10, 0))


def test_deflection_preserves_turn_semantics_for_dual_graphs():
    straight_a = LineString([(0, 0), (10, 0)])
    straight_b = LineString([(10, 0), (20, 0)])
    turn_b = LineString([(10, 0), (10, 10)])

    assert ci.angle_line_geometries(
        straight_a, straight_b, degree=True, calculation_type="deflection"
    ) == pytest.approx(0.0)
    assert ci.angle_line_geometries(
        straight_a, turn_b, degree=True, calculation_type="deflection"
    ) == pytest.approx(90.0)


def test_vectors_mode_preserves_historical_vector_orientation():
    line_a = LineString([(0, 0), (10, 0)])
    line_b = LineString([(10, 0), (20, 0)])

    assert ci.angle_line_geometries(
        line_a, line_b, degree=True, calculation_type="vectors"
    ) == pytest.approx(180.0)


def test_angular_change_uses_local_segments_for_multivertex_lines():
    line_a = LineString([(0, 0), (5, 0), (10, 0)])
    line_b = LineString([(10, 0), (10, 5), (10, 10)])

    assert ci.angle_line_geometries(
        line_a, line_b, degree=True, calculation_type="angular_change"
    ) == pytest.approx(90.0)


def test_angle_returns_radians_by_default_and_rejects_invalid_inputs():
    line_a = LineString([(0, 0), (10, 0)])
    line_b = LineString([(10, 0), (10, 10)])

    assert ci.angle_line_geometries(line_a, line_b, calculation_type="deflection") == pytest.approx(
        math.pi / 2
    )

    with pytest.raises(ci.AngleError):
        ci.angle_line_geometries(
            LineString([(0, 0), (1, 0)]),
            LineString([(2, 0), (3, 0)]),
            calculation_type="deflection",
        )

    with pytest.raises(ValueError):
        ci.angle_line_geometries(line_a, line_b, calculation_type="bad-mode")
