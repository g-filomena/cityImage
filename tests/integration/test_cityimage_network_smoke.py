"""Unit tests for the package."""

import pytest

pytestmark = pytest.mark.network

ox = pytest.importorskip("osmnx")
gpd = pytest.importorskip("geopandas")
matplotlib = pytest.importorskip("matplotlib")
nx = pytest.importorskip("networkx")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

LinearSegmentedColormap = pytest.importorskip("matplotlib.colors").LinearSegmentedColormap

matplotlib.use("Agg")

import cityImage as ci  # noqa: E402

# define queries to use throughout tests

place = "Susa, Italy"
download_method = "OSMplace"
crs_york = "EPSG:2019"
crs_susa = "EPSG:3003"
OSMPolygon = "Susa (44287)"
address = "Susa, Corso Francia"
distance = 1500
location = (45.1383, 7.0509)

buildings_gdf = None
nodes_gdf, edges_gdf = None, None
nodes_gdf_y, edges_gdf_y = None, None
barriers_gdf = None
graph = None


def _graph_to_cityimage_gdfs(G, crs):
    """Convert an OSMnx graph to cityImage-standardized node/edge tables."""
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)

    if crs is not None:
        nodes = nodes.to_crs(crs)
        edges = edges.to_crs(crs)

    nodes = ci.standardize_nodes_gdf(nodes, validate=False)
    edges = ci.standardize_edges_gdf(edges, validate=False)
    return ci.reset_index_graph_gdfs(nodes, edges, nodeID="nodeID", edgeID="edgeID")


def _network_from_osmnx_place(query, crs):
    """Delegate live OSM network loading to OSMnx, then standardize for cityImage."""
    return _graph_to_cityimage_gdfs(ox.graph_from_place(query, network_type="all"), crs)


def _network_from_osmnx_address(query, crs, dist):
    """Delegate live address-distance OSM loading to OSMnx."""
    return _graph_to_cityimage_gdfs(ox.graph_from_address(query, dist=dist, network_type="all"), crs)


def _network_from_osmnx_polygon(poly, crs):
    """Delegate live polygon OSM loading to OSMnx."""
    return _graph_to_cityimage_gdfs(ox.graph_from_polygon(poly, network_type="all"), crs)


def _buildings_from_osmnx(query, method="place", crs=None, distance=1000):
    """Delegate live OSM building loading to OSMnx, then preserve cityImage semantics."""
    tags = {"building": True}
    if method == "place":
        buildings = ox.features_from_place(query, tags=tags)
    elif method == "address":
        buildings = ox.features_from_address(query, tags=tags, dist=distance)
    elif method == "point":
        buildings = ox.features_from_point(query, tags=tags, dist=distance)
    else:
        raise ValueError("method must be 'place', 'address', or 'point'")

    if crs is not None:
        buildings = buildings.to_crs(crs)
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    buildings = ci.derive_land_uses_raw_fromOSM(buildings, default="residential")
    buildings = ci.classify_land_uses_raws_into_OSMgroups(buildings)
    buildings = ci.gdf_multipolygon_to_polygon(buildings)
    return ci.standardize_buildings_gdf(buildings, add_area=True, validate=False)


def _ensure_osm_network():
    """Initialise the live OSM network once per module if tests run out of order."""
    global nodes_gdf
    global edges_gdf

    if nodes_gdf is None or edges_gdf is None:
        nodes_gdf, edges_gdf = _network_from_osmnx_place(place, crs_susa)


def _ensure_graph():
    """Initialise the primal graph if tests run out of order."""
    global graph

    _ensure_osm_network()
    if graph is None:
        graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column="nodeID")


def _ensure_centrality_metrics():
    """Ensure centrality columns used by plotting smoke tests exist."""
    global graph
    global nodes_gdf
    global edges_gdf

    _ensure_graph()
    required_node_columns = {"Bc", "Sc", "Rc", "Cc"}
    if required_node_columns.issubset(nodes_gdf.columns) and "Eb" in edges_gdf.columns:
        return

    weight = "length"
    services = ox.features_from_address(
        address,
        tags={"amenity": True},
        dist=distance,
    ).to_crs(crs_susa)
    services = services[services["geometry"].geom_type == "Point"]

    graph = ci.weight_nodes(nodes_gdf, services, graph, field_name="services", radius=50)

    Rc = ci.calculate_centrality(
        graph, measure="reach", weight=weight, radius=400, attribute="services"
    )
    Bc = ci.calculate_centrality(graph, measure="betweenness", weight=weight)
    Sc = ci.calculate_centrality(graph, measure="straightness", weight=weight)
    Cc = ci.calculate_centrality(graph, measure="closeness", weight=weight)

    for metric, column in zip([Bc, Sc, Rc, Cc], ["Bc", "Sc", "Rc", "Cc"], strict=False):
        nodes_gdf[column] = nodes_gdf.nodeID.map(metric)

    Eb = nx.edge_betweenness_centrality(graph, weight=weight, normalized=False)
    edges_gdf = ci.append_edges_metrics(edges_gdf, graph, [Eb], ["Eb"])


def _osm_features_from_place(tags):
    """Delegate OSM feature retrieval to OSMnx with a safe empty fallback."""
    try:
        if hasattr(ox, "features_from_place"):
            features = ox.features_from_place(place, tags=tags)
        else:
            features = ox.geometries_from_place(place, tags=tags)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if features.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return features


def _barriers_from_osmnx_features():
    """Build cityImage barriers from externally downloaded OSM feature layers."""
    return ci.barriers_from_osm_features(
        roads_gdf=_osm_features_from_place({"highway": True}),
        waterways_gdf=_osm_features_from_place({"waterway": True}),
        water_gdf=_osm_features_from_place({"natural": "water"}),
        coastline_gdf=_osm_features_from_place({"natural": "coastline"}),
        railways_gdf=_osm_features_from_place({"railway": True}),
        parks_gdf=_osm_features_from_place({"leisure": True}),
        crs=crs_susa,
        parks_min_area=200,
    )


def _ensure_barriers():
    """Ensure barrier data used by plotting smoke tests exists."""
    global barriers_gdf

    _ensure_osm_network()
    if barriers_gdf is None:
        barriers_gdf = _barriers_from_osmnx_features()


def _historic_buildings_from_osmnx():
    """Download historic/cultural OSM features directly with OSMnx.

    cityImage no longer owns the historic-OSM loading route after the hard
    landmarks extraction. This helper keeps the network smoke test focused on
    cityImage cultural scoring while delegating OSM feature retrieval to OSMnx.
    """
    tags = {"building": True, "historic": True, "heritage": True}

    if hasattr(ox, "features_from_place"):
        historic = ox.features_from_place(place, tags=tags)
    else:
        historic = ox.geometries_from_place(place, tags=tags)

    if historic.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs_susa)

    historic = historic[historic.geometry.notna()].copy()

    if historic.crs is None:
        historic = historic.set_crs("EPSG:4326")

    return historic.to_crs(crs_susa)


def _land_use_label(value):
    """Return a stable scalar label from list-like land-use cells."""
    if isinstance(value, (list, tuple, set)):
        labels = [str(item) for item in value if item is not None]
        return ";".join(labels) if labels else "unknown"
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass
    return str(value)


def test_loadOSM():
    global nodes_gdf
    global edges_gdf
    global crs_susa

    nodes_gdf, edges_gdf = _network_from_osmnx_place(place, crs_susa)
    polygon = edges_gdf.to_crs('EPSG:4326').geometry.union_all().convex_hull
    _, _ = _network_from_osmnx_polygon(polygon, crs_susa)
    _, _ = _network_from_osmnx_address(address, crs_susa, distance)


def test_loadFile_clean():
    global nodes_gdf_y
    global edges_gdf_y
    global crs_york
    input_path = "tests/input/York_street_network.shp"
    dict_columns = {
        "highway": "type",
        "oneway": "oneway",
        "lanes": None,
        "maxspeed": "maxspeed",
        "name": "name",
    }
    network_lines = gpd.read_file(input_path)
    nodes_gdf_y, edges_gdf_y = ci.network_from_lines(
        network_lines, crs_york, dict_columns=dict_columns, other_columns=[]
    )
    nodes_gdf_y, edges_gdf_y = ci.clean_network(
        nodes_gdf_y,
        edges_gdf_y,
        dead_ends=True,
        remove_islands=True,
        same_vertexes_edges=True,
        self_loops=True,
        fix_topology=True,
    )


def test_graph():
    _ensure_osm_network()
    global nodes_gdf
    global edges_gdf
    global graph
    global crs_susa

    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column="nodeID")
    _ = ci.multiGraph_fromGDF(nodes_gdf, edges_gdf, "nodeID")
    nodes_dual, edges_dual = ci.dual_gdf(
        nodes_gdf, edges_gdf, crs=crs_susa, oneway=False, angle="degree"
    )
    _ = ci.dual_graph_fromGDF(nodes_dual, edges_dual)


def test_angles():
    _ensure_osm_network()
    global edges_gdf

    nodeID = edges_gdf.iloc[0].u
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    line_geometryA = tmp.iloc[0].geometry
    line_geometryB = tmp.iloc[1].geometry

    _ = ci.angle_line_geometries(
        line_geometryA, line_geometryB, degree=True, calculation_type="vectors"
    )
    _ = ci.angle_line_geometries(
        line_geometryA, line_geometryB, degree=True, calculation_type="deflection"
    )
    _ = ci.angle_line_geometries(
        line_geometryA, line_geometryB, degree=True, calculation_type="angular_change"
    )


def test_centrality():
    _ensure_centrality_metrics()


def test_regions():
    _ensure_osm_network()
    global nodes_gdf
    global edges_gdf
    global crs_susa

    nodes_gdf, edges_gdf = ci.clean_network(
        nodes_gdf,
        edges_gdf,
        dead_ends=True,
        remove_islands=True,
        same_vertexes_edges=False,
        self_loops=False,
        fix_topology=False,
    )
    graph_susa = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column="nodeID")
    nodes_dual, edges_dual = ci.dual_gdf(
        nodes_gdf, edges_gdf, crs=crs_susa, oneway=False, angle="degree"
    )
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    dual_regions = ci.identify_regions(dual_graph, edges_gdf, weight=None)
    primal_regions = ci.identify_regions_primal(graph_susa, nodes_gdf, weight=None)

    polygons_gdf = ci.polygonise_partitions(dual_regions, "p_topo", convex_hull=False, buffer=30)
    polygons_gdf = ci.polygonise_partitions(dual_regions, "p_topo", convex_hull=True, buffer=30)
    _ = ci.districts_to_edges_from_nodes(primal_regions, edges_gdf, "p_topo")
    _ = ci.district_to_nodes_from_edges(nodes_gdf, dual_regions, "p_topo")

    nodes_gdf = ci.district_to_nodes_from_polygons(nodes_gdf, polygons_gdf, "p_topo")
    min_size_district = 10
    nodes_gdf = ci.amend_nodes_membership(
        nodes_gdf, edges_gdf, "p_topo", min_size_district=min_size_district
    )
    nodes_gdf = ci.find_gateways(nodes_gdf, edges_gdf, "p_topo")


def test_barriers():
    _ensure_osm_network()
    global edges_gdf
    global barriers_gdf
    global place
    global crs_susa

    barriers_gdf = _barriers_from_osmnx_features()
    # assign barriers to street network
    edges_gdf_updated = ci.along_within_parks(edges_gdf, barriers_gdf)
    edges_gdf_updated = ci.along_water(edges_gdf_updated, barriers_gdf)
    edges_gdf_updated = ci.along_within_parks(edges_gdf_updated, barriers_gdf)
    edges_gdf_updated = ci.assign_structuring_barriers(edges_gdf_updated, barriers_gdf)


def test_landmarks():
    _ensure_osm_network()

    global crs_susa
    global address
    global location
    global nodes_gdf
    global edges_gdf
    global buildings_gdf

    _ = _buildings_from_osmnx(address, method="address", crs=crs_susa, distance=1000)
    _ = _buildings_from_osmnx(location, method="point", crs=crs_susa, distance=1000)

    # weights
    global_indexes_weights = {
        "3dvis": 0.50,
        "fac": 0.30,
        "height": 0.20,
        "area": 0.30,
        "2dvis": 0.30,
        "neigh": 0.20,
        "road": 0.20,
    }
    global_components_weights = {"vScore": 0.50, "sScore": 0.30, "cScore": 0.10, "pScore": 0.10}

    local_indexes_weights = {
        "3dvis": 0.50,
        "fac": 0.30,
        "height": 0.20,
        "area": 0.40,
        "2dvis": 0.00,
        "neigh": 0.30,
        "road": 0.30,
    }
    local_components_weights = {"vScore": 0.25, "sScore": 0.35, "cScore": 0.10, "pScore": 0.30}

    buildings_gdf = _buildings_from_osmnx(place, method="place", crs=crs_susa)
    rng = np.random.default_rng(seed=42)
    buildings_gdf["height"] = rng.choice([10, 1, 50], buildings_gdf.shape[0])

    # testing with only 5 nodes, to avoid time issues
    sight_lines_pars = {"distance_along": 300, "min_observer_target_distance": 400}
    simplification_pars = {
        "simplified_target_buildings": gpd.GeoDataFrame,
        "consolidate": True,
        "consolidate_tolerance": 15,
    }
    processing_pars = {"sight_lines_chunk_size": 1000000, "num_workers": 1, "with_pyvista": False}

    sight_lines = ci.compute_3d_sight_lines(
        nodes_gdf=nodes_gdf.iloc[:7],
        target_buildings_gdf=buildings_gdf,
        obstructions_buildings_gdf=buildings_gdf,
        edges_gdf=edges_gdf,
        city_name="York",
        **sight_lines_pars,
        **simplification_pars,
        **processing_pars,
    )
    processing_pars = {"sight_lines_chunk_size": 1000000, "num_workers": 1, "with_pyvista": True}
    sight_lines = ci.compute_3d_sight_lines(
        nodes_gdf=nodes_gdf.iloc[:7],
        target_buildings_gdf=buildings_gdf,
        obstructions_buildings_gdf=buildings_gdf,
        edges_gdf=edges_gdf,
        city_name="York",
        **sight_lines_pars,
        **simplification_pars,
        **processing_pars,
    )
    # historical elements
    historic = _historic_buildings_from_osmnx()
    # scores
    buildings_gdf = ci.structural_score(
        buildings_gdf,
        buildings_gdf,
        edges_gdf,
        advance_vis_expansion_distance=100,
        neighbours_radius=100,
    )

    buildings_gdf = ci.cultural_score(buildings_gdf, historic_elements_gdf=historic, from_OSM=False)
    buildings_gdf = ci.pragmatic_score(buildings_gdf, search_radius=200)
    buildings_gdf = ci.visibility_score(buildings_gdf, sight_lines)

    buildings_gdf = ci.compute_global_scores(
        buildings_gdf, global_indexes_weights, global_components_weights
    )
    buildings_gdf = ci.compute_local_scores(
        buildings_gdf, local_indexes_weights, local_components_weights, rescaling_radius=1500
    )


def test_plot():
    global nodes_gdf
    global edges_gdf
    global barriers_gdf
    global buildings_gdf

    _ensure_centrality_metrics()
    _ensure_barriers()
    if buildings_gdf is None:
        test_landmarks()

    tmp_nodes = nodes_gdf.copy()
    base_map_dict = {
        "base_map_gdf": edges_gdf,
        "base_map_alpha": 0.4,
        "base_map_geometry_size": 1.1,
        "base_map_zorder": 0,
    }
    tmp_nodes["Bc_sc"] = ci.scaling_columnDF(tmp_nodes["Bc"])

    # Lynch's bins - only for variables from 0 to 1
    scheme_dict = {
        "column": "Bc_sc",
        "bins": [0.125, 0.25, 0.5, 0.75, 1.00],
        "scheme": "User_Defined",
    }
    cmap = ci.kindlmann()
    _ = ci.plot_gdf(
        title="Example",
        gdf=tmp_nodes,
        black_background=True,
        cmap=cmap,
        legend=True,
        axes_frame=True,
        geometry_size=25,
        **base_map_dict,
        **scheme_dict,
        figsize=(10, 5),
    )

    # Appending the attributes to the geodataframe
    columns = ["Bc", "Sc", "Cc", "Rc"]
    # 2x2 color bar
    cbar_dict = {
        "cbar": True,
        "cbar_ticks": 2,
        "cbar_max_symbol": True,
        "cbar_min_max": True,
        "cbar_shrink": 0.75,
    }
    _ = ci.plot_grid_gdf_columns(
        gdf=tmp_nodes,
        columns=columns,
        black_background=True,
        cmap=cmap,
        legend=False,
        ncols=2,
        nrows=2,
        figsize=(15, 5),
        geometry_size_factor=30,
        axes_frame=True,
        **cbar_dict,
        titles=columns,
    )

    # 2x2 legend
    scheme_dict = {"bins": [0.125, 0.25, 0.5, 0.75, 1.00], "scheme": "User_Defined"}
    _ = ci.plot_grid_gdf_columns(
        gdf=tmp_nodes,
        columns=columns,
        black_background=True,
        cmap=cmap,
        nrows=2,
        ncols=2,
        figsize=(15, 7),
        classes=5,
        legend=True,
        axes_frame=True,
        **scheme_dict,
        titles=columns,
    )

    # 4x2 white
    _ = ci.plot_grid_gdf_columns(
        gdf=tmp_nodes,
        columns=columns + columns,
        black_background=False,
        cmap=cmap,
        nrows=4,
        ncols=2,
        classes=6,
        scheme="quantiles",
        legend=True,
        figsize=(20, 10),
        axes_frame=True,
        fontsize=15,
        titles=columns + columns,
    )
    # 3 x 1
    columns = ["Bc", "Sc", "Cc"]
    _ = ci.plot_grid_gdf_columns(
        gdf=tmp_nodes,
        columns=columns,
        black_background=True,
        cmap=cmap,
        ncols=1,
        nrows=3,
        classes=5,
        legend=True,
        figsize=(9, 9),
        **scheme_dict,
        axes_frame=True,
        titles=columns,
    )
    # 1x3
    _ = ci.plot_grid_gdf_columns(
        gdf=tmp_nodes,
        columns=columns,
        black_background=True,
        cmap=cmap,
        ncols=3,
        nrows=1,
        classes=5,
        legend=True,
        figsize=(12, 4),
        **scheme_dict,
        axes_frame=True,
        titles=columns,
    )

    # edges
    _ = ci.plot_gdf(
        edges_gdf,
        column="Eb",
        black_background=True,
        scheme="Fisher_Jenks",
        cmap=cmap,
        norm=None,
        legend=False,
        **cbar_dict,
        figsize=(10, 5),
        title="Testing",
        axes_frame=True,
    )

    # multi_gdfs
    _ = ci.plot_grid_gdfs_column(
        gdfs=[edges_gdf, edges_gdf],
        column="length",
        black_background=True,
        ncols=2,
        nrows=1,
        figsize=(10, 5),
        scheme="Fisher_Jenks",
        cmap=cmap,
        **cbar_dict,
    )

    barriers_gdf.sort_values(by="barrier_type", ascending=False, inplace=True)
    colors = ["green", "red", "gray", "blue"]
    if "secondary_road" in list(barriers_gdf["barrier_type"].unique()):
        colors.append("gold")

    base_map_dict = {
        "base_map_gdf": buildings_gdf,
        "base_map_alpha": 1,
        "base_map_zorder": 0,
        "base_map_color": "yellow",
    }
    cmap = LinearSegmentedColormap.from_list("cmap", colors, N=len(colors))
    ci.plot_gdf(
        barriers_gdf,
        column="barrier_type",
        geometry_size=1.1,
        legend=True,
        axes_frame=True,
        black_background=True,
        cmap=cmap,
        figsize=(15, 5),
        title="Barriers",
        **base_map_dict,
    )

    buildings_for_plot = buildings_gdf.copy()
    if "land_uses" in buildings_for_plot.columns:
        buildings_for_plot["land_use_label"] = buildings_for_plot["land_uses"].apply(
            _land_use_label
        )
    elif "land_uses_raw" in buildings_for_plot.columns:
        buildings_for_plot["land_use_label"] = buildings_for_plot["land_uses_raw"].apply(
            _land_use_label
        )
    elif "land_use_raw" in buildings_for_plot.columns:
        buildings_for_plot["land_use_label"] = buildings_for_plot["land_use_raw"].apply(
            _land_use_label
        )
    else:
        buildings_for_plot["land_use_label"] = "unknown"

    cmap = ci.rand_cmap(nlabels=len(buildings_for_plot["land_use_label"].unique()))
    ci.plot_gdf(
        buildings_for_plot,
        column="land_use_label",
        black_background=True,
        legend=True,
        figsize=(25, 10),
    )


def test_landuse():

    crs = "EPSG:25832"
    input_path = "tests/input/Muenster_buildings.shp"
    buildings_raw = gpd.read_file(input_path).to_crs(crs)
    buildings_shp = ci.standardize_buildings_gdf(
        buildings_raw,
        land_uses_raw_column="land_use",
        add_area=True,
        validate=False,
    )
    if "height" not in buildings_shp.columns and "height" in buildings_raw.columns:
        buildings_shp["height"] = buildings_raw["height"]
    if "base" not in buildings_shp.columns and "base" in buildings_raw.columns:
        buildings_shp["base"] = buildings_raw["base"]
    attributes_gdf = gpd.read_file("tests/input/Muenster_buildings_attributes.shp").to_crs(crs)

    adult_entertainment = ["brothel", "casino", "swingerclub", "stripclub", "nightclub", "gambling"]
    agriculture = [
        "shed",
        "silo",
        "greenhouse",
        "stable",
        "agricultural and forestry",
        "greenhouse (botany)",
        "building in the botanical garden",
    ]
    attractions = [
        "attractions",
        "attraction",
        "aquarium",
        "monument",
        "gatehouse",
        "terrace",
        "tower",
        "attraction and Leisure",
        "information",
        "viewpoint",
        "tourist information center",
        "recreation and amusement park",
        "zoo",
        "exhibition hall, trade hall",
        "boathouse",
        "bath house, thermal baths",
        "entertainment hall",
        "sauna",
    ]
    business_services = [
        "bank",
        "service",
        "offices",
        "foundation",
        "office",
        "atm",
        "bureau_de_change",
        "post_office",
        "post_office;atm",
        "coworking_space",
        "conference_centre",
        "trade and services",
        "trade and services building",
        "customs office",
        "insurance",
        "tax_office",
        "post",
        "administrative building",
        "facility building",
        "residential building with trade and services",
        "data_center",
        "tax office",
    ]
    commercial = [
        "commercial",
        "retail",
        "pharmacy",
        "commercial;educa",
        "shop",
        "supermarket",
        "books",
        "commercial services",
        "commercial land",
        "car_wash",
        "internet_cafe",
        "driving_school",
        "marketplace",
        "fuel",
        "car_sharing",
        "commercial and industry buidling",
        "crematorium",
        "commercial building",
        "commercial and industry building",
        "commercial building to traffic facilities (general)",
        "funeral parlor",
        "gas station",
        "car wash",
        "pumping station",
        "boat_rental",
        "boat_sharing",
        "bicycle_rental",
        "car_rental",
        "dive_centre",
    ]
    culture = [
        "club_house",
        "gallery",
        "arts_centre",
        "cultural facility",
        "cultural_centre",
        "theatre",
        "cinema",
        "studio",
        "exhibition_centre",
        "music_school",
        "theater",
        "castle",
        "museum",
        "culture",
    ]
    eating_drinking = [
        "bbq",
        "restaurant",
        "fast_food",
        "cafe",
        "bar",
        "pub",
        "accommodation, eating and drinking",
        "ice_cream",
        "kitchen",
        "food_court",
        "cafe;restaurant",
        "biergarten",
    ]
    education_research = [
        "university",
        "research",
        "university building",
        "education and research",
        "research_institute",
    ]
    emergency_service = [
        "fire brigade",
        "fire_station",
        "police",
        "emergency service",
        "resque_station",
        "ranger_station",
        "security",
    ]
    general_education = [
        "school",
        "college",
        "kindergarten",
        "education",
        "education and health",
        "childcare",
        "language_school",
        "children home",
        "nursery",
        "general education school",
    ]
    hospitality = [
        "hotel",
        "hostel",
        "guest_house",
        "building for accommodation",
        "hotel, motel, pension",
        "refuge",
    ]
    industrial = [
        "industrial",
        "factory",
        "construction",
        "manufacturing and production",
        "gasometer",
        "workshop",
        "production building",
    ]
    medical_care = [
        "hospital",
        "doctors",
        "dentist",
        "clinic",
        "veterinary",
        "medical Care",
        "nursing_home",
        "sanatorium, nursing home",
        "retirement home",
        "healthcare",
    ]
    military_detainment = ["general aviation", "barracks", "military", "penitentiary", "prison"]
    other = [
        "toilets",
        "picnic_site",
        "hut",
        "storage_tank",
        " canopy",
        "toilet",
        "bunker, shelter",
        "warehouse",
        "converter",
        "garage",
        "garages",
        "parking",
    ]
    public = [
        "townhall",
        "public_building",
        "library",
        "civic",
        "courthouse",
        "public",
        "embassy",
        "public infrastructure",
        "community_centre",
        "court",
        "district government",
        "residential building with public facilities",
    ]
    religious = [
        "church",
        "place_of_worship",
        "convent",
        "rectory",
        "chapel",
        "religious building",
        "monastery",
        "nuns home",
        "vocational school",
        "cathedral",
    ]
    residential = [
        "apartments",
        None,
        "NaN",
        "residential",
        "flats",
        "houses",
        "building",
        "residential land",
        "residential building",
        "student dorm",
        "building usage mixed with living",
    ]
    social = [
        "social_facility",
        "community_centre",
        "community buidling",
        "dormitory",
        "social_centre",
        "social serives building",
        "social services",
        "community hall",
        "commercial social facility",
        "recreational",
    ]
    sport = [
        "stadium",
        "sport and entertainment",
        "sports or exercise facility",
        "gym",
        "sports building",
        "sports hall",
        "horse riding school",
        "swimming pool",
        "sport hall",
        "bowling hall",
        "indoor swimming pool",
    ]
    transport = [
        "transport",
        "road transport",
        "station",
        "subway_entrance",
        "bus_station",
        "shipping facility building",
        "train_station",
        "railway building",
    ]
    utilities = [
        "gas supply",
        "electricity supply",
        "electricity substation",
        "waste treatment building",
        "water supply",
        "waste water treatment plant",
        "smokestack",
        "supply systems",
        "waste management",
        "water works",
        "heating plant",
        "boiler house",
        "telecommunication",
    ]

    categories = [
        adult_entertainment,
        agriculture,
        attractions,
        business_services,
        commercial,
        culture,
        eating_drinking,
        education_research,
        emergency_service,
        general_education,
        hospitality,
        industrial,
        medical_care,
        military_detainment,
        other,
        public,
        religious,
        residential,
        social,
        sport,
        transport,
        utilities,
    ]
    strings = [
        "adult_entertainment",
        "agriculture",
        "attractions",
        "business_services",
        "commercial",
        "culture",
        "eating_drinking",
        "education_research",
        "emergency_service",
        "general_education",
        "hospitality",
        "industrial",
        "medical_care",
        "military_detainment",
        "other",
        "public",
        "religious",
        "residential",
        "social",
        "sport",
        "transport",
        "utilities",
    ]

    attributes_gdf = ci.classify_land_use(
        attributes_gdf,
        raw_land_use_column="lu_eng",
        new_land_use_column="land_use",
        categories=categories,
        strings=strings,
    )
    attributes_gdf["land_use"] = attributes_gdf["land_use"].str.lower()
    _ = attributes_gdf.explode(index_parts=False, ignore_index=True)
