"""End-to-end ``image_of_the_city`` on live OpenStreetMap data, a small area of Susa."""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.network

ox = pytest.importorskip("osmnx")

import cityImage as ci  # noqa: E402


def test_image_of_the_city_on_susa():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        img = ci.image_of_the_city(
            "Susa, Corso Francia",
            download_method="distance_from_address",
            distance=600,
            crs="EPSG:32632",
        )

    assert img.metadata["district_network"] == "OSM (drive)"
    assert img.metadata["crs"] == "EPSG:32632"
    for name in ("paths", "nodes", "landmarks"):
        layer = getattr(img, name)
        assert layer is not None and not layer.empty, name
        assert layer.crs == "EPSG:32632"
    for layer in (img.paths, img.nodes, img.landmarks):
        assert (layer["lynch_class"] == 1).any()
