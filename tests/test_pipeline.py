"""Offline tests for ``image_of_the_city``: the real York street network, synthetic buildings and
barriers, and a mocked OSM download for the place-based path."""

from __future__ import annotations

import random
import warnings

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import LineString, box

import cityImage as ci
import cityImage.osm as osm
from cityImage.pipeline import LYNCH_BINS, _district_area, _district_network, _rank
from tests.fixtures.cityimage_minimal import york_raw_network

CONFIG = ci.CityImageConfig(min_district_size=3)


def _layers():
    nodes, edges = york_raw_network()
    minx, miny, maxx, maxy = edges.total_bounds
    cx = (minx + maxx) / 2
    barriers = gpd.GeoDataFrame(
        {"barrier_type": ["water"]},
        geometry=[LineString([(cx, miny - 50), (cx + 20, maxy + 50)])],
        crs=edges.crs,
    )
    rng = random.Random(0)
    footprints = []
    for _ in range(40):
        x, y, size = rng.uniform(minx, maxx), rng.uniform(miny, maxy), rng.uniform(8, 30)
        footprints.append(box(x, y, x + size, y + size))
    buildings = gpd.GeoDataFrame(
        {
            "buildingID": range(40),
            "land_uses": [["residential"]] * 35 + [["retail"]] * 5,
            "land_uses_overlap": [[1.0]] * 40,
        },
        geometry=footprints,
        crs=edges.crs,
    )
    return nodes, edges, buildings, barriers


@pytest.fixture(scope="module")
def york_image():
    nodes, edges, buildings, barriers = _layers()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ci.image_of_the_city(
            nodes=nodes, edges=edges, buildings=buildings, barriers=barriers, config=CONFIG
        )


def test_all_five_elements_are_computed(york_image):
    assert set(york_image.layers()) == {"paths", "nodes", "edges", "districts", "landmarks"}
    assert {"Ab", "Ab_sc", "lynch_class", "sep_barr"} <= set(york_image.paths.columns)
    assert {"Bc", "Bc_sc", "lynch_class", "district", "gateway"} <= set(york_image.nodes.columns)
    assert {"gScore", "gScore_sc", "lynch_class"} <= set(york_image.landmarks.columns)
    assert "district" in york_image.districts.columns
    for layer in (york_image.paths, york_image.nodes, york_image.landmarks):
        assert (layer["lynch_class"] == 1).any()
        assert not layer.columns.str.startswith("is_").any()


def test_every_district_polygon_has_nodes(york_image):
    assert set(york_image.districts["district"]) == set(york_image.nodes["district"])


def test_streets_crossing_the_river_are_marked(york_image):
    assert york_image.paths["sep_barr"].any()
    assert not york_image.paths["sep_barr"].all()


def test_paths_are_angular_betweenness_of_the_dual_graph(york_image):
    paths, nodes = york_image.paths, york_image.nodes
    nodes_dual, edges_dual = ci.dual_gdf(nodes, paths, paths.crs)
    dual = ci.dual_graph_fromGDF(nodes_dual, edges_dual)
    # igraph, as calculate_centrality: unlike networkx it counts as tied two routes whose angle
    # sums differ only by float rounding, so the two disagree where such ties occur.
    expected = ci.calculate_centrality(dual, "betweenness", weight="rad")
    got = paths.set_index("edgeID")["Ab"]
    for edge_id, value in expected.items():
        assert got[edge_id] == pytest.approx(value)


def test_nodes_are_betweenness_of_the_primal_graph(york_image):
    nodes = york_image.nodes
    graph = ci.graph_fromGDF(nodes, york_image.paths)
    expected = nx.betweenness_centrality(graph, weight="length", normalized=False)
    got = nodes.set_index("nodeID")["Bc"]
    for node_id, value in expected.items():
        assert got[node_id] == pytest.approx(value)


@pytest.mark.parametrize(
    ("layer", "score"), [("paths", "Ab"), ("nodes", "Bc"), ("landmarks", "gScore")]
)
def test_elements_are_ranked_in_lynch_ranges(york_image, layer, score):
    gdf = getattr(york_image, layer)
    scaled = gdf[f"{score}_sc"]
    assert scaled.max() == pytest.approx(1.0)
    assert (gdf.loc[scaled >= 0.75, "lynch_class"] == 1).all()
    assert (gdf.loc[scaled < 0.125, "lynch_class"] == 0).all()


def test_metadata_records_how_the_result_was_made(york_image):
    meta = york_image.metadata
    assert meta["cityImage_version"] == ci.__version__
    assert meta["network"] == "given"
    assert meta["district_network"] == "paths network"
    assert meta["crs"] == "EPSG:2019"
    assert york_image.config is CONFIG


def test_missing_heights_warn_that_the_visual_component_is_left_out():
    nodes, edges, buildings, _ = _layers()
    with pytest.warns(UserWarning, match="visual component"):
        img = ci.image_of_the_city(
            nodes=nodes, edges=edges, buildings=buildings, elements=["landmarks"], config=CONFIG
        )
    assert img.paths is None and img.districts is None
    assert img.landmarks is not None


def test_element_without_data_warns_and_is_none():
    nodes, edges, _, _ = _layers()
    with pytest.warns(UserWarning, match="edges not computed"):
        img = ci.image_of_the_city(nodes=nodes, edges=edges, elements=["paths", "edges"])
    assert img.edges is None
    assert img.paths is not None and img.nodes is None


def test_to_file_writes_one_layer_per_element(york_image, tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    path = tmp_path / "york.gpkg"
    york_image.to_file(str(path))
    layers = {name for name, _ in pyogrio.list_layers(path)}
    assert layers == {"paths", "nodes", "edges", "districts", "landmarks"}
    landmarks = gpd.read_file(path, layer="landmarks")
    assert landmarks["land_uses"].iloc[0] in {"residential", "retail"}


def test_plot_draws_one_figure(york_image):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fig = york_image.plot(title="York")
    assert fig.axes
    matplotlib.pyplot.close(fig)


def test_explore_maps_one_layer_per_element(york_image, tmp_path):
    pytest.importorskip("folium")
    m = york_image.explore()
    html = tmp_path / "york.html"
    m.save(str(html))
    text = html.read_text(encoding="utf-8")
    for name in ("Districts", "Edges", "Paths", "Landmarks", "Nodes", "class 1"):
        assert name in text


def _clean_york():
    nodes, edges, _, _ = _layers()
    return ci.clean_network(nodes, edges, **dict(CONFIG.clean_kwargs))


def test_districts_leave_out_the_excluded_highways():
    nodes, edges = _clean_york()
    config = ci.CityImageConfig(district_excluded_highways=("footway", "service"))

    kept, dual = _district_network(nodes, edges, edges.crs, config)

    assert not kept["highway"].astype(str).isin(["footway", "service"]).any()
    assert len(kept) < len(edges)
    left_out = set(edges["edgeID"]) - set(kept["edgeID"])
    assert set(dual.nodes) <= set(kept["edgeID"]) and not set(dual.nodes) & left_out


def test_district_network_reuses_the_dual_graph_when_no_street_is_left_out():
    nodes, edges = _clean_york()
    dual = object()
    config = ci.CityImageConfig(district_excluded_highways=("motorway",))

    kept, reused = _district_network(nodes, edges, edges.crs, config, dual_graph=dual)

    assert reused is dual and kept is edges


def test_place_downloads_the_drive_network_for_districts(monkeypatch):
    nodes, edges, buildings, barriers = _layers()
    calls = []

    def fake_network(place, *, network_type, **kwargs):
        calls.append(("network", network_type, kwargs["download_method"]))
        return nodes.copy(), edges.copy()

    def fake_geocode(place):
        calls.append(("geocode", place))
        york = gpd.GeoSeries([edges.union_all().convex_hull], crs=edges.crs).to_crs(4326)
        return gpd.GeoDataFrame(geometry=york)

    def fake_barriers(place, **kwargs):
        calls.append(("barriers", None))
        return barriers.copy()

    def fake_buildings(place, **kwargs):
        calls.append(("buildings", None))
        return buildings.copy()

    monkeypatch.setattr(osm, "network_from_osm", fake_network)
    monkeypatch.setattr(osm, "barriers_from_osm", fake_barriers)
    monkeypatch.setattr(osm, "buildings_from_osm", fake_buildings)
    monkeypatch.setattr(ox, "geocode_to_gdf", fake_geocode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        img = ci.image_of_the_city("York, Ontario", config=CONFIG)

    # the paths network of the place; the drive network of the place's polygon and a margin
    assert ("network", "walk", "OSMplace") in calls
    assert ("geocode", "York, Ontario") in calls and ("network", "drive", "polygon") in calls
    assert ("barriers", None) in calls and ("buildings", None) in calls
    assert img.metadata["place"] == "York, Ontario"
    assert img.metadata["district_network"] == "OSM (drive), study area + 1000 m"
    assert set(img.layers()) == {"paths", "nodes", "edges", "districts", "landmarks"}


def test_district_area_is_the_study_area_and_a_margin():
    point = (43.77, -79.50)
    config = ci.CityImageConfig()
    assert _district_area(point, "distance_from_point", 1500, config) == (
        point,
        "distance_from_point",
        2500,
    )
    no_margin = ci.CityImageConfig(district_margin=0)
    assert _district_area(point, "distance_from_point", 1500, no_margin) == (
        point,
        "distance_from_point",
        1500,
    )

    square = gpd.GeoSeries([box(0, 0, 1000, 1000)], crs="EPSG:32617").to_crs(4326).iloc[0]
    wider, method, distance = _district_area(square, "polygon", None, config)

    metric = gpd.GeoSeries([square, wider], crs=4326).to_crs("EPSG:32617")
    assert method == "polygon" and distance is None
    assert metric.iloc[1].contains(metric.iloc[0])
    # a 1 km square with a 1 km margin: 3 km wide, the corners rounded
    assert 8.0e6 < metric.iloc[1].area < 9.0e6


def test_district_place_is_the_area_the_districts_partition():
    config = ci.CityImageConfig(district_place="Greater London", district_margin=500)

    assert _district_area("City of London", "OSMplace", None, config) == (
        "Greater London",
        "OSMplace",
        None,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "give a place"),
        ({"place": "x", "elements": ["streets"]}, "unknown elements"),
    ],
)
def test_invalid_calls_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ci.image_of_the_city(**kwargs)


def test_nodes_without_edges_raise():
    nodes, _, _, _ = _layers()
    with pytest.raises(ValueError, match="together"):
        ci.image_of_the_city(nodes=nodes)


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="lynch_bins"):
        ci.CityImageConfig(lynch_bins=(0.5, 0.25))
    with pytest.raises(ValueError, match="lynch_bins"):
        ci.CityImageConfig(lynch_bins=(0.0, 0.5))
    with pytest.raises(ValueError, match="district_network"):
        ci.CityImageConfig(district_network="walk")
    with pytest.raises(ValueError, match="district_margin"):
        ci.CityImageConfig(district_margin=-1)
    with pytest.raises(ValueError, match="district_resolution"):
        ci.CityImageConfig(district_resolution=0)


def test_rank_classes_follow_lynch_ranges():
    gdf = pd.DataFrame({"s": [0.0, 10.0, 12.5, 20.0, 25.0, 50.0, 74.0, 75.0, 100.0, float("nan")]})
    ranked = _rank(gdf, "s", LYNCH_BINS)
    assert ranked["lynch_class"].tolist() == [0, 0, 4, 4, 3, 2, 2, 1, 1, 0]


@pytest.mark.parametrize(
    "scoring", [ci.CityImageConfig().landmark_scoring, ci.LandmarkScoringConfig()]
)
def test_landmarks_use_the_paper_weights_by_default(scoring):
    assert dict(scoring.global_component_weights) == {
        "vScore": 0.50,
        "sScore": 0.30,
        "cScore": 0.10,
        "pScore": 0.10,
    }
    assert scoring.global_index_weights["2dvis"] == 0.30
    assert scoring.pragmatic_search_radius == 200.0
    assert ci.CityImageConfig().structural_kwargs["neighbours_radius"] == 150


def test_districts_that_cannot_be_assigned_are_left_out_with_a_warning(monkeypatch):
    import cityImage.pipeline as pipeline

    def cannot_amend(*args, **kwargs):
        raise ValueError("A pass amended no node")

    monkeypatch.setattr(pipeline, "amend_nodes_membership", cannot_amend)
    nodes, edges, _, _ = _layers()

    with pytest.warns(UserWarning, match="districts not computed: A pass amended no node"):
        image = ci.image_of_the_city(
            nodes=nodes, edges=edges, elements=("paths", "nodes", "districts"), config=CONFIG
        )

    assert image.districts is None
    assert image.paths is not None and image.nodes is not None
    assert "district" not in image.nodes.columns


def test_longitude_latitude_layers_are_projected_to_their_utm_zone():
    nodes, edges, _, _ = _layers()
    projected = ci.image_of_the_city(nodes=nodes, edges=edges, elements=("paths",), config=CONFIG)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        geographic = ci.image_of_the_city(
            nodes=nodes.to_crs(4326), edges=edges.to_crs(4326), elements=("paths",), config=CONFIG
        )

    assert geographic.paths.crs.is_projected
    assert geographic.metadata["crs"] != "EPSG:4326"
    # Measured in metres, the network cleans as the projected one does.
    assert len(geographic.paths) == len(projected.paths)
    assert geographic.paths.length.sum() == pytest.approx(projected.paths.length.sum(), rel=0.01)


def test_centralities_are_computed_only_for_the_elements_asked(monkeypatch):
    import cityImage.pipeline as pipeline

    measured = []
    real = pipeline.calculate_centrality

    def recording(graph, measure, **kwargs):
        measured.append((measure, kwargs.get("weight")))
        return real(graph, measure, **kwargs)

    monkeypatch.setattr(pipeline, "calculate_centrality", recording)
    nodes, edges, buildings, _ = _layers()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ci.image_of_the_city(
            nodes=nodes, edges=edges, buildings=buildings, elements=("landmarks",), config=CONFIG
        )
    assert measured == []

    ci.image_of_the_city(nodes=nodes, edges=edges, elements=("paths",), config=CONFIG)
    assert measured == [("betweenness", "rad")]
