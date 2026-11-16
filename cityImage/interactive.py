"""Interactive web map of the Image of the City with the symbols of Lynch's legend; needs the
``interactive`` extra (folium).

``explore_city_image`` draws the five elements of a ``CityImage`` (``image_of_the_city``) as Lynch
drew the images derived from the interviews in *The Image of the City* (1960, Figures 35-46),
black on white. His four frequency ranges (over 75%, 50-75%, 25-50%, 12.5-25% of the
interviewees) are the four Lynch classes of the scores (``lynch_class`` 1, the strongest, to 4):

- paths: a heavy bar, a medium bar, a dashed line, a dotted line;
- nodes: circles, dark, densely dotted, sparsely dotted, hatched;
- landmarks: a six-pointed star, a large open triangle, a small filled triangle, a small open
  triangle, all pointing down;
- edges: a band with ticks across it, drawn in screen pixels so that it keeps its spacing at
  every zoom;
- districts: freehand areas of dots or of hatching, a different one for neighbouring districts,
  under everything else. Each is drawn apart from its neighbours, with rounded corners and a
  wavering double outline; where two district polygons overlap, the overlap goes to the
  district of the nearest node.

Edges and districts are not ranked.
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box

INK = "#111111"
CLASSES = (1, 2, 3, 4)

# Paths by Lynch class: line weight (px) at zoom 16 and closer, and dash pattern. Zoomed out,
# the weights shrink (``_path_zoom_script``), down to half at zoom 13, so that a whole city is
# not loaded with heavy lines.
_PATH_STYLE: dict[int, dict[str, Any]] = {
    1: {"weight": 6, "dashArray": None},
    2: {"weight": 3.5, "dashArray": None},
    3: {"weight": 2.5, "dashArray": "10 6"},
    4: {"weight": 2.5, "dashArray": "1 7", "lineCap": "round"},
}
# Symbol sizes in metres, so that they scale with the map like a drawing.
_NODE_RADIUS = {1: 30.0, 2: 26.0, 3: 22.0, 4: 20.0}
_LANDMARK_SIDE = {1: 62.0, 2: 50.0, 3: 24.0, 4: 22.0}
_LANDMARK_WEIGHT = {1: 2.4, 2: 2.2, 3: 1.4, 4: 1.4}
# Transparency of edges and nodes, so that the paths and the districts under them show.
_EDGE_OPACITY = 0.6
_NODE_OPACITY = 0.75
# Districts, in metres: the white gap left between neighbours, the rounding of the corners.
_DISTRICT_GAP = 15.0
_DISTRICT_ROUNDING = 25.0
_DISTRICT_INK = "#808080"
_DISTRICT_OPACITY = 0.45


def explore_city_image(city_image: Any, **kwargs: Any) -> Any:
    """An interactive map of a ``CityImage`` with the symbols of Lynch's legend, on white.

    The layer control groups the layers by element, with one layer per Lynch class for paths,
    nodes and landmarks, each shown by its symbol (faded when the layer is off): classes 3 and 4
    of nodes and landmarks, usually many, start hidden, as does the street network underneath.
    Hovering a feature shows its score and Lynch class.

    Parameters
    ----------
    city_image
        The result of ``image_of_the_city``.
    **kwargs
        Passed to ``folium.Map`` (e.g. ``zoom_start``).

    Returns
    -------
    folium.Map
        Shown inline in a notebook; ``.save("map.html")`` writes it to a file.
    """
    try:
        import folium
        from branca.element import MacroElement
        from folium.plugins import CirclePattern, GroupedLayerControl, StripePattern
        from jinja2 import Template
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('explore() needs folium: pip install "cityImage[interactive]"') from exc

    layers = city_image.layers()
    if not layers:
        raise ValueError("the CityImage has no element to map")
    focus = [layers[name] for name in ("districts", "nodes") if name in layers] or list(
        layers.values()
    )
    bounds = gpd.GeoSeries(
        [layer.to_crs(4326).union_all().envelope for layer in focus]
    ).total_bounds

    # Fractional zoom, so that the study area fills the map rather than the zoom level below.
    options = {"zoom_snap": 0.25, "zoom_delta": 0.5, **kwargs}
    m = folium.Map(tiles=None, control_scale=True, **options)
    # The root of a map is a branca Figure, with a header and a body (typed as Element).
    root: Any = m.get_root()
    root.header.add_child(
        folium.Element(
            "<style>.leaflet-container { background: #ffffff !important; }"
            ".leaflet-tooltip, .leaflet-control-layers { font-family: Georgia, serif; }"
            # The symbol of each layer stands for its checkbox, faded when the layer is off.
            ".leaflet-control-layers-selector { display: none; }"
            ".leaflet-control-layers label { cursor: pointer; margin: 2px 0; }"
            ".leaflet-control-layers-selector:not(:checked) + span { opacity: 0.3; }"
            ".ci-symbol { vertical-align: middle; margin-right: 5px; }</style>"
        )
    )
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    # A pane under the default one of vector layers (z-index 400): districts stay under
    # everything, also after being switched off and on again.
    folium.map.CustomPane("districts", z_index=350, pointer_events=True).add_to(m)

    def pattern(p):
        p.add_to(m)
        return p

    def dots(spacing, radius, color=INK):
        return pattern(
            CirclePattern(
                width=spacing,
                height=spacing,
                radius=radius,
                weight=0,
                color=color,
                fill_color=color,
                opacity=1,
                fill_opacity=1,
            )
        )

    def stripes(angle, spacing, color=INK):
        return pattern(
            StripePattern(
                angle=angle, weight=1, space_weight=spacing, color=color, opacity=1, space_opacity=0
            )
        )

    grey = "#555555"
    district_fills = [
        dots(5, 0.8, grey),
        stripes(45, 5, grey),
        dots(9, 1.1, grey),
        stripes(-45, 5, grey),
        stripes(0, 6, grey),
        stripes(90, 6, grey),
    ]

    def add(gdf, group, style, tooltip=None, extra=(), **options):
        if gdf is None or gdf.empty:
            return
        fields = [f for f in (tooltip or []) if f in gdf.columns]
        columns = [*fields, *(c for c in extra if c not in fields), "geometry"]
        data = json.loads(gdf[columns].to_crs(4326).to_json(drop_id=True))
        return folium.GeoJson(
            data,
            style_function=style,
            tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None,
            **options,
        ).add_to(group)

    groups: dict[str, list[Any]] = {}

    def layer(element, name, show=True):
        name = f"{_symbol(element, name)}{name}"
        group = folium.FeatureGroup(name=name, show=show).add_to(m)
        groups.setdefault(element, []).append(group)
        return group

    paths = layers.get("paths")
    nodes = layers.get("nodes")
    landmarks = layers.get("landmarks")
    districts = layers.get("districts")
    edges = layers.get("edges")

    if districts is not None:
        drawn = _sketched_districts(districts, nodes)
        drawn["fill"] = _alternating_fills(drawn.geometry, len(district_fills))
        group = layer("Context", "Districts")
        add(
            drawn,
            group,
            lambda f: {
                "stroke": False,
                "fillPattern": district_fills[f["properties"]["fill"]],
                "fillOpacity": _DISTRICT_OPACITY,
            },
            tooltip=["district"],
            extra=["fill"],
            pane="districts",
        )
        outlines = gpd.GeoDataFrame(geometry=_wavering_outlines(drawn.geometry), crs=drawn.crs)
        add(
            outlines,
            group,
            lambda f: {"color": _DISTRICT_INK, "weight": 0.9, "opacity": 0.7},
            pane="districts",
        )

    if edges is not None:
        # A black band with a white core gives the two sides of Lynch's edge; a short dash
        # across the core gives its ticks. Widths and dashes are in pixels, so the symbol keeps
        # its spacing at every zoom.
        group = layer("Context", "Edges")
        for style in (
            {"color": INK, "weight": 7, "opacity": _EDGE_OPACITY},
            {"color": "#ffffff", "weight": 4.5, "opacity": 1},
            {
                "color": INK,
                "weight": 4.5,
                "opacity": _EDGE_OPACITY,
                "dashArray": "1.5 7",
                "lineCap": "butt",
            },
        ):
            add(edges, group, lambda f, style=style: style, tooltip=["barrier_type"])

    if paths is not None:
        add(
            paths,
            layer("Context", "Street network", show=False),
            lambda f: {"color": "#c8c8c8", "weight": 0.6, "opacity": 1},
        )

    path_layers: list[tuple[str, float]] = []
    if paths is not None and "lynch_class" in paths.columns:
        for lynch_class in reversed(CLASSES):
            style = {"color": INK, "opacity": 1, **_PATH_STYLE[lynch_class]}
            drawn_paths = add(
                paths[paths["lynch_class"] == lynch_class],
                layer("Paths", f"class {lynch_class}"),
                lambda f, style=style: style,
                tooltip=["name", "Ab", "Ab_sc", "lynch_class"],
            )
            if drawn_paths is not None:
                path_layers.append(
                    (drawn_paths.get_name(), float(_PATH_STYLE[lynch_class]["weight"]))
                )

    if nodes is not None and "lynch_class" in nodes.columns:
        fills: dict[int, dict[str, Any]] = {
            1: {"fillColor": "#3a3a3a"},
            2: {"fillPattern": dots(4, 0.9)},
            3: {"fillPattern": dots(7, 0.9)},
            4: {"fillPattern": stripes(45, 3)},
        }
        for lynch_class in reversed(CLASSES):
            ranked = nodes[nodes["lynch_class"] == lynch_class]
            circles = ranked.copy()
            circles["geometry"] = ranked.geometry.buffer(_NODE_RADIUS[lynch_class], 24)
            style = {
                "color": INK,
                "weight": 1.3,
                "opacity": _NODE_OPACITY,
                "fillOpacity": _NODE_OPACITY,
                **fills[lynch_class],
            }
            add(
                circles,
                layer("Nodes", f"class {lynch_class}", show=lynch_class <= 2),
                lambda f, style=style: style,
                tooltip=["nodeID", "Bc", "Bc_sc", "lynch_class", "district"],
            )

    if landmarks is not None and "lynch_class" in landmarks.columns:
        for lynch_class in reversed(CLASSES):
            ranked = landmarks[landmarks["lynch_class"] == lynch_class]
            symbols = ranked.copy()
            centres = ranked.geometry.centroid
            side = _LANDMARK_SIDE[lynch_class]
            symbols["geometry"] = (
                _stars(centres, side) if lynch_class == 1 else _triangles(centres, side)
            )
            filled = lynch_class == 3
            style = {
                "color": INK,
                "weight": _LANDMARK_WEIGHT[lynch_class],
                "opacity": 1,
                "fillColor": INK if filled else "#ffffff",
                "fillOpacity": 1 if filled else 0.9,
            }
            add(
                symbols,
                layer("Landmarks", f"class {lynch_class}", show=lynch_class <= 2),
                lambda f, style=style: style,
                tooltip=["buildingID", "gScore", "gScore_sc", "lynch_class"],
            )

    root.html.add_child(folium.Element(_LEGEND))
    # Classes 1-4 from the top, the strongest first; the layers were added weakest first, so
    # that the strongest are drawn on top.
    ordered = {
        element: (members if element == "Context" else members[::-1])
        for element, members in groups.items()
    }
    GroupedLayerControl(ordered, exclusive_groups=False, collapsed=False).add_to(m)
    if path_layers:
        # A child of the map, added last, so that its script runs after the layers exist.
        rescale = MacroElement()
        rescale._template = Template(
            "{% macro script(this, kwargs) %}"
            + _path_zoom_script(m.get_name(), path_layers)
            + "{% endmacro %}"
        )
        m.add_child(rescale)
    return m


def _path_zoom_script(map_name: str, layers: list[tuple[str, float]]) -> str:
    """JavaScript scaling the weight of the path layers with the zoom: full at zoom 16 and
    closer, down to half at zoom 13 and farther."""
    entries = ", ".join(f"[{name}, {weight}]" for name, weight in layers)
    return f"""
(function () {{
  var map = {map_name};
  var layers = [{entries}];
  function restyle() {{
    var factor = Math.min(1, Math.max(0.5, 1 - (16 - map.getZoom()) / 6));
    layers.forEach(function (l) {{ l[0].setStyle({{weight: l[1] * factor}}); }});
  }}
  map.on("zoomend", restyle);
  restyle();
}})();
"""


def _symbol(element: str, name: str) -> str:
    """A small SVG of the symbol of a layer, for the layer control; its patterns are those of
    the legend."""
    lynch_class = int(name[-1]) if name.startswith("class ") else 0
    shapes = {
        ("Paths", 1): '<line x1="2" y1="8" x2="26" y2="8" stroke="#111" stroke-width="5"/>',
        ("Paths", 2): '<line x1="2" y1="8" x2="26" y2="8" stroke="#111" stroke-width="3"/>',
        ("Paths", 3): '<line x1="2" y1="8" x2="26" y2="8" stroke="#111" stroke-width="2.5" '
        'stroke-dasharray="6 3"/>',
        ("Paths", 4): '<line x1="2" y1="8" x2="26" y2="8" stroke="#111" stroke-width="2.5" '
        'stroke-dasharray="0.1 5" stroke-linecap="round"/>',
        ("Nodes", 1): '<circle cx="14" cy="8" r="6.5" fill="#3a3a3a" stroke="#111"/>',
        ("Nodes", 2): '<circle cx="14" cy="8" r="6.5" fill="url(#ci-dense)" stroke="#111"/>',
        ("Nodes", 3): '<circle cx="14" cy="8" r="6.5" fill="url(#ci-sparse)" stroke="#111"/>',
        ("Nodes", 4): '<circle cx="14" cy="8" r="6.5" fill="url(#ci-hatch)" stroke="#111"/>',
        ("Landmarks", 1): '<path d="M14 0.5l6.5 11h-13z M14 15.5l6.5 -11h-13z" fill="#fff" '
        'stroke="#111" stroke-width="1.5"/>',
        ("Landmarks", 2): '<path d="M7.5 2h13l-6.5 12z" fill="#fff" stroke="#111" '
        'stroke-width="1.6"/>',
        ("Landmarks", 3): '<path d="M9.5 4h9l-4.5 8z" fill="#111" stroke="#111"/>',
        ("Landmarks", 4): '<path d="M9.5 4h9l-4.5 8z" fill="#fff" stroke="#111"/>',
    }
    shape = shapes.get((element, lynch_class)) or {
        "Districts": '<path d="M3 11 q1 -8 9 -9 q11 -1 13 5 q1 7 -10 7.5 q-10 1 -12 -3.5z" '
        'fill="url(#ci-district)" stroke="#808080"/>',
        "Edges": '<line x1="2" y1="8" x2="26" y2="8" stroke="#111" stroke-width="7" '
        'stroke-opacity="0.6"/><line x1="2" y1="8" x2="26" y2="8" stroke="#fff" '
        'stroke-width="4.5"/><line x1="2" y1="8" x2="26" y2="8" stroke="#111" '
        'stroke-width="4.5" stroke-opacity="0.6" stroke-dasharray="1.5 5"/>',
        "Street network": '<path d="M2 12 L12 6 L26 9 M12 6 L15 1" fill="none" '
        'stroke="#aaa" stroke-width="1"/>',
    }.get(name, "")
    return f'<svg class="ci-symbol" width="28" height="16">{shape}</svg>'


def _sketched_districts(
    districts: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame | None
) -> gpd.GeoDataFrame:
    """The district polygons as freehand areas: without overlaps, apart from each other, with
    rounded corners and a wavering outline.

    Where two polygons overlap, the overlap goes to the district of the nearest node (by the
    Voronoi cells of the nodes); each polygon is then shrunk by ``_DISTRICT_GAP``, its corners
    rounded by ``_DISTRICT_ROUNDING``, and its outline made to waver a little.
    """
    drawn = districts[["district", "geometry"]].reset_index(drop=True).copy()
    territory = _node_territories(drawn, nodes)
    geometries = list(drawn.geometry)
    shapes = []
    for i, (district, polygon) in enumerate(zip(drawn["district"], geometries, strict=True)):
        others = shapely.union_all([g for j, g in enumerate(geometries) if j != i])
        overlap = polygon.intersection(others)
        if not overlap.is_empty:
            own = territory.get(district)
            kept = polygon.difference(others)
            polygon = kept.union(overlap.intersection(own)) if own is not None else kept
        shapes.append(_freehand(polygon, seed=i))
    drawn["geometry"] = shapes
    return drawn[~drawn.geometry.is_empty]


def _node_territories(
    districts: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame | None
) -> dict[Any, Any]:
    """The area closer to the nodes of each district than to any other node: the union of the
    Voronoi cells of its nodes."""
    if nodes is None or "district" not in nodes.columns:
        return {}
    points = nodes[nodes["district"].isin(districts["district"])][["district", "geometry"]]
    if len(points) < 2:
        return {}
    extent = box(*districts.total_bounds).buffer(1000.0)
    cells = shapely.voronoi_polygons(
        shapely.multipoints(points.geometry.to_numpy()), extend_to=extent
    )
    cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=districts.crs)
    joined = gpd.sjoin(cells, points, predicate="contains")
    return joined.dissolve("district").geometry.to_dict()


def _freehand(polygon: Any, seed: int) -> Any:
    """``polygon`` shrunk, with rounded corners and a wavering outline; its holes and its slivers
    are left out."""
    rounding = _DISTRICT_ROUNDING
    shape = polygon.buffer(-_DISTRICT_GAP).buffer(-rounding).buffer(2 * rounding).buffer(-rounding)
    if shape.is_empty:
        shape = polygon
    parts = [p for p in getattr(shape, "geoms", [shape]) if isinstance(p, Polygon)]
    if not parts:
        return Polygon()
    largest = max(p.area for p in parts)
    rng = np.random.default_rng(seed)
    wavering = [_waver(p.exterior, rng) for p in parts if p.area >= 0.05 * largest]
    return shapely.union_all([Polygon(w) for w in wavering if len(w) >= 4]).buffer(0)


def _waver(ring: Any, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """The coordinates of ``ring`` pushed in and out along its normals by a smooth periodic wave,
    as a hand drawing it would: a few metres, more on a longer ring."""
    ring = shapely.segmentize(shapely.LinearRing(ring.coords), 8.0)
    xy = np.asarray(ring.coords)[:-1]
    if len(xy) < 3:
        return np.asarray(ring.coords)
    step = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    perimeter = step.sum()
    s = np.concatenate([[0.0], np.cumsum(step)[:-1]]) / perimeter
    amplitude = scale * float(np.clip(0.004 * perimeter, 3.0, 12.0))
    offset = np.zeros(len(xy))
    for wavelength, weight in ((180.0, 0.65), (70.0, 0.35)):
        k = max(1, round(perimeter / wavelength))
        offset += weight * np.sin(2 * np.pi * k * s + rng.uniform(0, 2 * np.pi))
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-9)[:, None]
    moved = xy + normal * (amplitude * offset)[:, None]
    return np.vstack([moved, moved[:1]])


def _wavering_outlines(polygons: gpd.GeoSeries) -> list[Any]:
    """Two strokes around each polygon, the second wavering on its own, as a pencil going over an
    outline twice."""
    lines = []
    for i, polygon in enumerate(polygons):
        rng = np.random.default_rng(10_000 + i)
        for part in getattr(polygon, "geoms", [polygon]):
            lines.append(LineString(part.exterior.coords))
            lines.append(LineString(_waver(part.exterior, rng, scale=0.4)))
    return lines


def _alternating_fills(polygons: gpd.GeoSeries, n_fills: int) -> list[int]:
    """A fill index for each polygon, different from its neighbours' where ``n_fills`` allow:
    greedy colouring of the polygons near each other, the most connected first."""
    geometries = polygons.reset_index(drop=True)
    near = geometries.buffer(2 * _DISTRICT_GAP + 10.0)
    left, right = geometries.sindex.query(near, predicate="intersects")
    neighbours: dict[int, set[int]] = {i: set() for i in range(len(geometries))}
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        if a != b:
            neighbours[a].add(b)
            neighbours[b].add(a)
    fills: dict[int, int] = {}
    for i in sorted(neighbours, key=lambda i: -len(neighbours[i])):
        taken = [fills[j] for j in neighbours[i] if j in fills]
        free = [k for k in range(n_fills) if k not in taken]
        fills[i] = free[0] if free else min(range(n_fills), key=taken.count)
    return [fills[i] for i in range(len(geometries))]


def _triangle(x: float, y: float, side: float, down: bool = True) -> Polygon:
    """An equilateral triangle of ``side`` centred on ``(x, y)``, pointing down or up."""
    half, height = side / 2, side * np.sqrt(3) / 2
    sign = -1 if down else 1
    return Polygon(
        [
            (x - half, y - sign * height / 3),
            (x + half, y - sign * height / 3),
            (x, y + sign * 2 * height / 3),
        ]
    )


def _triangles(points: gpd.GeoSeries, side: float) -> list[Polygon]:
    """A downward triangle of ``side`` metres centred on each point."""
    return [_triangle(x, y, side) for x, y in shapely.get_coordinates(points.to_numpy())]


def _stars(points: gpd.GeoSeries, side: float) -> list[MultiPolygon]:
    """A six-pointed star (two triangles) of ``side`` metres centred on each point."""
    return [
        MultiPolygon([_triangle(x, y, side, down=False), _triangle(x, y, side, down=True)])
        for x, y in shapely.get_coordinates(points.to_numpy())
    ]


_LEGEND = """
<div style="position: fixed; bottom: 30px; right: 20px; z-index: 9999; background: #fff;
            border: 1px solid #111; padding: 10px 14px 8px; color: #111;
            font: 11px/1.3 Georgia, 'Times New Roman', serif;">
  <div style="font-style: italic; font-size: 15px; margin-bottom: 6px;">
    The Image of the City</div>
  <svg width="430" height="172" style="display: block" font-family="Georgia, serif"
       font-size="10">
    <defs>
      <pattern id="ci-dense" width="4" height="4" patternUnits="userSpaceOnUse">
        <circle cx="2" cy="2" r="0.9" fill="#111"/></pattern>
      <pattern id="ci-sparse" width="7" height="7" patternUnits="userSpaceOnUse">
        <circle cx="3.5" cy="3.5" r="0.9" fill="#111"/></pattern>
      <pattern id="ci-district" width="5" height="5" patternUnits="userSpaceOnUse">
        <circle cx="2.5" cy="2.5" r="0.8" fill="#555"/></pattern>
      <pattern id="ci-district2" width="6" height="6" patternUnits="userSpaceOnUse"
               patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke="#555" stroke-width="1"/></pattern>
      <pattern id="ci-hatch" width="4" height="4" patternUnits="userSpaceOnUse"
               patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="4" stroke="#111" stroke-width="1"/></pattern>
    </defs>
    <g text-anchor="middle" letter-spacing="1">
      <text x="152" y="10">PATH</text><text x="212" y="10">EDGE</text>
      <text x="270" y="10">NODE</text><text x="330" y="10">DISTRICT</text>
      <text x="398" y="10">LANDMARK</text>
    </g>
    <g text-anchor="start">
      <text x="0" y="38">class 1 (over 0.75)</text>
      <text x="0" y="76">class 2 (0.5 to 0.75)</text>
      <text x="0" y="114">class 3 (0.25 to 0.5)</text>
      <text x="0" y="152">class 4 (0.125 to 0.25)</text>
    </g>
    <line x1="132" y1="34" x2="172" y2="34" stroke="#111" stroke-width="7"/>
    <line x1="132" y1="72" x2="172" y2="72" stroke="#111" stroke-width="4"/>
    <line x1="132" y1="110" x2="172" y2="110" stroke="#111" stroke-width="3"
          stroke-dasharray="7 4"/>
    <line x1="132" y1="148" x2="172" y2="148" stroke="#111" stroke-width="3"
          stroke-dasharray="0.1 6" stroke-linecap="round"/>
    <line x1="192" y1="34" x2="232" y2="34" stroke="#111" stroke-width="7"
          stroke-opacity="0.6"/>
    <line x1="192" y1="34" x2="232" y2="34" stroke="#fff" stroke-width="4.5"/>
    <line x1="192" y1="34" x2="232" y2="34" stroke="#111" stroke-width="4.5"
          stroke-opacity="0.6" stroke-dasharray="1.5 7"/>
    <text x="212" y="62" text-anchor="middle" font-style="italic" fill="#666">not ranked</text>
    <circle cx="270" cy="34" r="12" fill="#3a3a3a" stroke="#111"/>
    <circle cx="270" cy="72" r="11" fill="url(#ci-dense)" stroke="#111"/>
    <circle cx="270" cy="110" r="10" fill="url(#ci-sparse)" stroke="#111"/>
    <circle cx="270" cy="148" r="9" fill="url(#ci-hatch)" stroke="#111"/>
    <path d="M306 40 q2 -12 13 -13 q9 0 8 9 q-1 9 -11 9 q-9 0 -10 -5z"
          fill="url(#ci-district)" stroke="#808080" stroke-width="0.9"/>
    <path d="M305.5 39 q3 -11 14 -12.5 q8 1 7.5 9.5 q-2 8 -11 8.5 q-9 -1 -10.5 -5.5"
          fill="none" stroke="#808080" stroke-width="0.9"/>
    <path d="M330 41 q1 -11 11 -13 q10 -1 10 8 q0 9 -11 10 q-9 1 -10 -5z"
          fill="url(#ci-district2)" stroke="#808080" stroke-width="0.9"/>
    <path d="M330.5 40 q2 -10 11 -12.5 q9 0 9 8.5 q-1 8 -11 10 q-8 0 -9.5 -6"
          fill="none" stroke="#808080" stroke-width="0.9"/>
    <text x="330" y="62" text-anchor="middle" font-style="italic" fill="#666">not ranked</text>
    <path d="M398 20 l13 22.5 h-26z M398 48 l13 -22.5 h-26z" fill="#fff" stroke="#111"
          stroke-width="2"/>
    <path d="M386 63 h24 l-12 20.5z" fill="#fff" stroke="#111" stroke-width="2"/>
    <path d="M392 105 h12 l-6 10z" fill="#111" stroke="#111" stroke-width="1.3"/>
    <path d="M392 143 h12 l-6 10z" fill="#fff" stroke="#111" stroke-width="1.3"/>
  </svg>
</div>
"""
