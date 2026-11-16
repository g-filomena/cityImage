"""Tests for the freehand district outlines of ``explore()``."""

from __future__ import annotations

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box

import cityImage.interactive as interactive


def test_freehand_district_survives_an_outline_that_crosses_itself(monkeypatch):
    # The wavering outline of a part of a district can cross itself, as this bow tie, and overlap
    # another part: their union made GEOS fail (TopologyException: side location conflict).
    outlines = iter(
        [
            np.array([(0, 0), (100, 100), (100, 0), (0, 100), (0, 0)], dtype=float),
            np.array([(20, 20), (80, 20), (80, 80), (20, 80), (20, 20)], dtype=float),
        ]
    )
    monkeypatch.setattr(interactive, "_waver", lambda ring, rng, scale=1.0: next(outlines))
    two_parts = MultiPolygon([box(0, 0, 400, 400), box(1000, 0, 1400, 400)])

    shape = interactive._freehand(two_parts, seed=0)

    assert shape.is_valid
    assert shape.area > 0


def test_freehand_district_is_valid_on_a_plain_square():
    shape = interactive._freehand(box(0, 0, 500, 500), seed=1)

    assert shape.is_valid
    assert shape.area > 0


def test_polygons_takes_every_polygon_of_nested_collections():
    a, b, c = box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)
    nested = GeometryCollection([MultiPolygon([a, b]), LineString([(0, 0), (1, 1)]), c])

    assert [p.bounds for p in interactive._polygons(nested)] == [a.bounds, b.bounds, c.bounds]
    assert interactive._polygons(Polygon()) == []
