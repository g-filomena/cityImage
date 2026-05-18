"""Angle helpers for cityImage graph semantics.

This module keeps the custom line-angle semantics used by ``dual_gdf`` and
legacy imageability workflows. There is nothing useful to delegate here:
generic trigonometry is tiny, while the important part is how two street
segments are oriented around their shared endpoint.

Three modes are preserved:

* ``vectors``: angle between the two oriented vectors at the shared endpoint;
* ``deflection``: turn/deflection angle along the full segments;
* ``angular_change``: turn/deflection using the local segment adjacent to the
  shared endpoint, useful for multi-vertex LineStrings.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString

VALID_CALCULATION_TYPES = {"vectors", "angular_change", "deflection"}


class Error(Exception):
    """Base class for angle exceptions."""


class AngleError(Error):
    """Raised when line geometries do not share an endpoint."""


def _dot(v_a: tuple[float, float], v_b: tuple[float, float]) -> float:
    """Return the two-dimensional dot product."""
    return v_a[0] * v_b[0] + v_a[1] * v_b[1]


def _round_coord(coord: Any, ndigits: int = 10) -> tuple[float, float]:
    """Return a 2D coordinate rounded for robust endpoint matching."""
    return (round(float(coord[0]), ndigits), round(float(coord[1]), ndigits))


def _coord_tuple(coord: Any) -> tuple[float, float]:
    """Return a 2D coordinate tuple without rounding."""
    return (float(coord[0]), float(coord[1]))


def get_coord_angle(origin: tuple[float, float], distance: float, angle: float) -> tuple[float, float]:
    """Return coordinates at ``distance`` and bearing-like angle from an origin.

    The angle convention is the historical cityImage convention: degrees from
    the positive y-axis, not the mathematical positive x-axis.
    """
    disp_x = distance * math.sin(math.radians(angle))
    disp_y = distance * math.cos(math.radians(angle))
    return (origin[0] + disp_x, origin[1] + disp_y)


def _full_segment(coords: list[Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the full first-to-last segment."""
    return (_coord_tuple(coords[0]), _coord_tuple(coords[-1]))


def _local_start_segment(coords: list[Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the segment adjacent to the first vertex."""
    return (_coord_tuple(coords[0]), _coord_tuple(coords[1]))


def _local_end_segment(coords: list[Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the segment adjacent to the last vertex."""
    return (_coord_tuple(coords[-2]), _coord_tuple(coords[-1]))


def _oriented_lines(
    coords_a: list[Any],
    coords_b: list[Any],
    calculation_type: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
    """Return the two oriented line segments used for angle calculation."""
    start_a, end_a = _round_coord(coords_a[0]), _round_coord(coords_a[-1])
    start_b, end_b = _round_coord(coords_b[0]), _round_coord(coords_b[-1])

    if calculation_type == "angular_change":
        if end_a == end_b:
            return _local_end_segment(coords_a), tuple(reversed(_local_end_segment(coords_b)))
        if end_a == start_b:
            return _local_end_segment(coords_a), _local_start_segment(coords_b)
        if start_a == start_b:
            return tuple(reversed(_local_start_segment(coords_a))), _local_start_segment(coords_b)
        if start_a == end_b:
            return tuple(reversed(_local_start_segment(coords_a))), tuple(reversed(_local_end_segment(coords_b)))

    elif calculation_type == "deflection":
        full_a = _full_segment(coords_a)
        full_b = _full_segment(coords_b)

        if end_a == end_b:
            return full_a, tuple(reversed(full_b))
        if end_a == start_b:
            return full_a, full_b
        if start_a == start_b:
            return tuple(reversed(full_a)), full_b
        if start_a == end_b:
            return tuple(reversed(full_a)), tuple(reversed(full_b))

    else:  # vectors
        full_a = _full_segment(coords_a)
        full_b = _full_segment(coords_b)

        if end_a == end_b:
            return tuple(reversed(full_a)), tuple(reversed(full_b))
        if end_a == start_b:
            return tuple(reversed(full_a)), full_b
        if start_a == start_b:
            return full_a, full_b
        if start_a == end_b:
            return full_a, tuple(reversed(full_b))

    raise AngleError("The lines do not intersect; provide lines that share an endpoint.")


class Settings:
    """Store two LineString coordinate sets and their oriented comparison lines.

    This class is retained for backwards compatibility with earlier cityImage
    code that accessed ``Settings.lineA`` and ``Settings.lineB`` directly.
    """

    def set_coordinates(self, coords: list[Any], prefix: str) -> None:
        """Store rounded start/end and neighbouring coordinates for a line."""
        setattr(self, "x_origin" + prefix, float(f"{coords[0][0]:.10f}"))
        setattr(self, "y_origin" + prefix, float(f"{coords[0][1]:.10f}"))
        setattr(self, "x_second" + prefix, float(f"{coords[1][0]:.10f}"))
        setattr(self, "y_second" + prefix, float(f"{coords[1][1]:.10f}"))
        setattr(self, "x_destination" + prefix, float(f"{coords[-1][0]:.10f}"))
        setattr(self, "y_destination" + prefix, float(f"{coords[-1][1]:.10f}"))
        setattr(self, "x_second_last" + prefix, float(f"{coords[-2][0]:.10f}"))
        setattr(self, "y_second_last" + prefix, float(f"{coords[-2][1]:.10f}"))

    def set_conditions(self, calculation_type: str) -> None:
        """Set ``lineA`` and ``lineB`` for the requested angle semantics."""
        self.lineA, self.lineB = _oriented_lines(self.coordsA, self.coordsB, calculation_type)

    def __init__(self, coordsA: list[Any], coordsB: list[Any], calculation_type: str):
        """Initialise coordinates and oriented comparison lines."""
        self.coordsA = coordsA
        self.coordsB = coordsB
        self.set_coordinates(coordsA, "A")
        self.set_coordinates(coordsB, "B")
        self.set_conditions(calculation_type)


def angle_line_geometries(
    line_geometryA: LineString,
    line_geometryB: LineString,
    degree: bool = False,
    calculation_type: str = "vectors",
) -> float:
    """Compute the angle between two endpoint-connected LineStrings.

    Parameters
    ----------
    line_geometryA, line_geometryB
        Shapely LineStrings sharing one endpoint.
    degree
        If True, return degrees. Otherwise return radians.
    calculation_type
        One of ``"vectors"``, ``"angular_change"``, or ``"deflection"``.
    """
    if not isinstance(line_geometryA, LineString) or not isinstance(line_geometryB, LineString):
        raise TypeError("Both inputs must be shapely.geometry.LineString objects.")

    if calculation_type not in VALID_CALCULATION_TYPES:
        raise ValueError(
            f"Invalid calculation type. Choose one of: {sorted(VALID_CALCULATION_TYPES)}."
        )

    coords_a = list(line_geometryA.coords)
    coords_b = list(line_geometryB.coords)

    if len(coords_a) < 2 or len(coords_b) < 2:
        raise ValueError("Both LineStrings must have at least two coordinates.")

    line_a, line_b = _oriented_lines(coords_a, coords_b, calculation_type)

    v_a = (line_a[0][0] - line_a[1][0], line_a[0][1] - line_a[1][1])
    v_b = (line_b[0][0] - line_b[1][0], line_b[0][1] - line_b[1][1])

    mag_a = _dot(v_a, v_a) ** 0.5
    mag_b = _dot(v_b, v_b) ** 0.5

    if mag_a == 0 or mag_b == 0:
        angle_rad = 0.0
    else:
        cosine = _dot(v_a, v_b) / mag_a / mag_b
        cosine = max(-1.0, min(1.0, cosine))
        angle_rad = math.acos(cosine)

    if degree:
        return math.degrees(angle_rad) % 360

    return angle_rad
