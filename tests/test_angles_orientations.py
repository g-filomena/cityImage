"""Offline tests for the angle helpers (``cityImage.angles``).

These exercise every endpoint-connection orientation for each calculation type,
the backwards-compatibility ``Settings`` shim, and the error paths.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

import cityImage as ci
from cityImage.angles import AngleError, Settings, get_coord_angle

# Line A runs from (0, 1) down to the shared vertex (0, 0).
LINE_A = LineString([(0, 1), (0, 0)])

# Each B shares exactly one endpoint with A, in one of the four orientations.
ORIENTATIONS = {
    "end_a_eq_end_b": LineString([(1, 0), (0, 0)]),  # end_a == end_b == (0, 0)
    "end_a_eq_start_b": LineString([(0, 0), (1, 0)]),  # end_a == start_b == (0, 0)
    "start_a_eq_start_b": LineString([(0, 1), (1, 1)]),  # start_a == start_b == (0, 1)
    "start_a_eq_end_b": LineString([(1, 1), (0, 1)]),  # start_a == end_b == (0, 1)
}


@pytest.mark.parametrize("calc", ["vectors", "angular_change", "deflection"])
@pytest.mark.parametrize("name", list(ORIENTATIONS))
def test_angle_line_geometries_handles_every_endpoint_orientation(name, calc):
    # LINE_A is vertical and every B geometry is horizontal, so the two lines are
    # perpendicular in every orientation and calculation type: the exact answer is 90 degrees.
    angle = ci.angle_line_geometries(LINE_A, ORIENTATIONS[name], degree=True, calculation_type=calc)
    assert angle == pytest.approx(90.0)


def test_angle_line_geometries_collinear_lines_are_180_degrees():
    # Two straight, co-linear segments meeting end-to-start read as a 180-degree (straight) turn,
    # proving the function is not simply returning 90 for everything.
    up_a = LineString([(0, 0), (0, 10)])
    up_b = LineString([(0, 10), (0, 20)])
    angle = ci.angle_line_geometries(up_a, up_b, degree=True, calculation_type="vectors")
    assert angle == pytest.approx(180.0)


def test_angle_line_geometries_rejects_non_linestrings():
    with pytest.raises(TypeError, match="LineString"):
        ci.angle_line_geometries("not a line", LINE_A)


def test_angle_line_geometries_rejects_unknown_calculation_type():
    with pytest.raises(ValueError, match="Invalid calculation type"):
        ci.angle_line_geometries(LINE_A, ORIENTATIONS["end_a_eq_start_b"], calculation_type="bogus")


def test_angle_line_geometries_raises_when_lines_do_not_share_an_endpoint():
    with pytest.raises(AngleError):
        ci.angle_line_geometries(
            LineString([(0, 0), (0, 1)]), LineString([(5, 5), (6, 6)]), calculation_type="vectors"
        )


def test_angle_line_geometries_degenerate_zero_length_line_returns_zero():
    degenerate = LineString([(0, 0), (0, 0)])  # zero magnitude vector
    angle = ci.angle_line_geometries(LINE_A, degenerate, degree=True, calculation_type="vectors")
    assert angle == 0.0


def test_get_coord_angle_uses_y_axis_convention():
    # 0 degrees -> straight up (+y); 90 degrees -> +x. Historical cityImage convention.
    assert get_coord_angle((0.0, 0.0), 10.0, 0.0) == pytest.approx((0.0, 10.0))
    assert get_coord_angle((0.0, 0.0), 10.0, 90.0) == pytest.approx((10.0, 0.0))


def test_settings_backwards_compat_exposes_oriented_lines():
    settings = Settings(
        list(LINE_A.coords), list(ORIENTATIONS["end_a_eq_start_b"].coords), "vectors"
    )
    assert settings.x_originA == 0.0 and settings.y_originA == 1.0
    assert isinstance(settings.lineA, tuple) and len(settings.lineA) == 2
    assert isinstance(settings.lineB, tuple) and len(settings.lineB) == 2
