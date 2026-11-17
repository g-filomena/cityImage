"""Build the cityImage documentation dataset (``cityImage-docs-data.zip``) and its provenance.

usage: python scripts/build_docs_data.py <current docs/data> <output dir> [--overpass-url URL]
       [--steps streets buildings landmarks tadoussac]

The Boston and Tadoussac files are rebuilt from their sources; the study area of Boston, its terrain
model and the Tripoli files are copied from ``<current docs/data>``. ``<output dir>`` receives
the files at their published paths, ``provenance.json`` (sources, dates, versions, counts,
SHA-256 of every file) and the archive. ``--steps`` rebuilds only the files named; the others are
kept from an earlier build in ``<output dir>``, with their provenance.

Boston, within 4 km of City Hall (the area of Filomena, Verstegen & Manley 2019):

- ``Boston_streetNetwork4000.gpkg``: the OpenStreetMap drive network (OSMnx), one line per street
  segment. ODbL, © OpenStreetMap contributors.
- ``Boston_obstructions.gpkg``: the buildings of the City of Boston (BPDA, "Boston Buildings with
  Roof Breaks", PDDL), the structures only (``IEL_TYPE`` "BLDG"), with ``height`` in metres (their
  2010 LiDAR height, from feet), ``bra_land_use`` (the city's land-use code) and
  ``osm_land_uses``: the OpenStreetMap land uses of the buildings (cityImage groups, from the OSM
  building polygons overlapping them by at least 20% and from the OSM points of interest on
  them), separated by ``"; "``. With the OSM land uses the file is under the ODbL.
- ``otherSources/Boston_historic_landmarks.gpkg``: the Boston Landmarks Commission landmarks
  (PDDL).

Tadoussac (``Tadoussac/Tadoussac_street_network.gpkg``): the OpenStreetMap ways tagged
``highway`` in a 3 km square around the village (Quebec), whole and clipped to it, as lines not split at
the junctions: a network with the topological errors that ``fix_topology`` repairs. ODbL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box

import cityImage as ci
from cityImage.landuse.derive import derive_land_uses_raw_fromOSM

CRS = "EPSG:26986"  # NAD83 / Massachusetts Mainland, metres
CITY_HALL = (-71.0580, 42.3603)  # lon, lat
RADIUS = 4000.0
FEET = 0.3048
CKAN = "https://data.boston.gov/api/3/action/package_show?id="
BUILDINGS_PACKAGE = "boston-buildings-with-roof-breaks"
LANDMARKS_PACKAGE = "boston-landmarks-commission-blc-landmarks"
# the ArcGIS GeoServices resources of the two packages
BUILDINGS_LAYER = (
    "https://gis.bostonplans.org/hosting/rest/services/Boston_Buildings/FeatureServer/9"
)
LANDMARKS_LAYER = "https://gisportal.boston.gov/arcgis/rest/services/ISD/landmarks_historic_districts_viewing/MapServer/0"
POI_TAGS = {key: True for key in ("amenity", "shop", "office", "tourism", "leisure", "craft")}
STREET_COLUMNS = ["osmid", "highway", "name", "oneway", "lanes", "maxspeed", "bridge", "tunnel"]
# 3 km square around the centre of Tadoussac (as geocoded by Nominatim: 48.14334, -69.71746)
TADOUSSAC_BOUNDS = (-69.73781, 48.12972, -69.69711, 48.15696)  # west, south, east, north
STEPS = ("streets", "buildings", "landmarks", "tadoussac")
COPIED = (
    "Boston/Boston_studyArea.gpkg",
    "Boston/Boston_DTM.tif",
    "Tripoli/Tripoli_edges.gpkg",
    "Tripoli/Tripoli_nodes.gpkg",
    "Tripoli/Tripoli_studyArea.gpkg",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ckan_package(package: str) -> dict:
    """The metadata of an Analyze Boston package."""
    with urllib.request.urlopen(CKAN + package, timeout=60) as response:
        result = json.load(response)["result"]
    return result


def _query_layer(layer_url: str, circle_wgs) -> tuple[gpd.GeoDataFrame, int]:
    """The features of an ArcGIS layer within the bounding box of ``circle_wgs``, page by page.

    The file downloads of Analyze Boston (pre-signed S3 links) answer 403 to scripts; its
    ArcGIS GeoServices resources, listed in the same packages, serve the same data.
    """
    with urllib.request.urlopen(layer_url + "?f=pjson", timeout=60) as response:
        page_size = int(json.load(response).get("maxRecordCount") or 1000)
    query = {
        "where": "1=1",
        "geometry": ",".join(str(v) for v in circle_wgs.bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "orderByFields": "OBJECTID",
        "resultRecordCount": page_size,
        "f": "geojson",
    }
    pages, offset = [], 0
    while True:
        url = layer_url + "/query?" + urllib.parse.urlencode({**query, "resultOffset": offset})
        with urllib.request.urlopen(url, timeout=300) as response:
            features = json.load(response).get("features", [])
        if not features:
            break
        pages.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))
        offset += len(features)
    gdf = pd.concat(pages, ignore_index=True)
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326").to_crs(CRS), offset


def _join(value) -> str | None:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return None if pd.isna(value) else str(value)


def street_network(circle_wgs, out: Path, record: dict) -> None:
    record["streets"] = {
        "source": "OpenStreetMap via OSMnx",
        "network_type": "drive",
        "downloaded": _now(),
    }
    graph = ox.graph_from_polygon(circle_wgs, network_type="drive")
    edges = ox.graph_to_gdfs(ox.convert.to_undirected(graph), nodes=False).reset_index(drop=True)
    edges = edges[[c for c in STREET_COLUMNS if c in edges.columns] + ["geometry"]].to_crs(CRS)
    for column in edges.columns.drop("geometry"):
        edges[column] = edges[column].apply(_join)
    edges.to_file(out, layer="boston_4000", driver="GPKG")
    record["streets"]["segments"] = len(edges)


def osm_land_uses(circle_wgs, buildings: gpd.GeoDataFrame, record: dict) -> pd.Series:
    """The OSM land uses (cityImage groups) of each building, joined by ``"; "``."""
    record["osm_land_uses"] = {"source": "OpenStreetMap via OSMnx", "downloaded": _now()}
    osm_buildings = ci.buildings_from_osm(
        circle_wgs, download_method="polygon", crs=CRS, min_area=None, default_land_use=""
    )
    osm_buildings = osm_buildings[osm_buildings["land_uses"].str.len() > 0]
    from_polygons = ci.land_use_from_other_gdf(
        buildings,
        osm_buildings,
        new_land_use_column="uses",
        other_land_use_column="land_uses",
        min_overlap_threshold=0.20,
        fill_default_when_no_match=False,
    )["uses"]

    pois = ci.features_from_osm(circle_wgs, POI_TAGS, download_method="polygon", crs=CRS)
    pois = pois[pois.geometry.geom_type == "Point"].copy()
    pois = derive_land_uses_raw_fromOSM(pois, default="")
    pois = ci.classify_land_uses_raws_into_OSMgroups(pois)
    pois = pois[pois["land_uses"].str.len() > 0]
    from_points = ci.land_use_from_other_gdf(
        buildings,
        pois[["land_uses", "geometry"]],
        new_land_use_column="uses",
        other_land_use_column="land_uses",
        fill_default_when_no_match=False,
    )["uses"]

    def combine(a, b):
        uses = [u for u in dict.fromkeys(list(a) + list(b)) if u != "UNCLASSIFIED"]
        return "; ".join(uses) if uses else None

    combined = pd.Series(
        [combine(a, b) for a, b in zip(from_polygons, from_points, strict=True)],
        index=buildings.index,
    )
    record["osm_land_uses"].update(
        osm_buildings_with_use=len(osm_buildings),
        points_of_interest=len(pois),
        buildings_with_osm_use=int(combined.notna().sum()),
    )
    return combined


def boston_buildings(circle, circle_wgs, out: Path, record: dict) -> None:
    package = _ckan_package(BUILDINGS_PACKAGE)
    record["buildings"] = {
        "source": package["title"],
        "publisher": "City of Boston (BPDA)",
        "licence": package.get("license_title"),
        "metadata_modified": package.get("metadata_modified"),
        "url": BUILDINGS_LAYER,
        "downloaded": _now(),
    }
    raw, downloaded = _query_layer(BUILDINGS_LAYER, circle_wgs)
    record["buildings"]["features_in_bounding_box"] = downloaded

    structures = raw[raw["IEL_TYPE"].astype(str).str.strip().str.lower().isin(["bldg", "buidling"])]
    structures = structures[structures.representative_point().within(circle)]
    buildings = gpd.GeoDataFrame(
        {
            "height": pd.to_numeric(structures["BLDG_HGT_2010"], errors="coerce") * FEET,
            "bra_land_use": structures["BRA_Land_Use"].str.strip(),
        },
        geometry=structures.geometry.values,
        crs=CRS,
    ).reset_index(drop=True)
    buildings.loc[buildings["height"] <= 0, "height"] = float("nan")
    buildings["osm_land_uses"] = osm_land_uses(circle_wgs, buildings, record)
    buildings.to_file(out, layer="Boston_obstructions", driver="GPKG")
    record["buildings"].update(
        structures_within_4km=len(buildings), with_height=int(buildings["height"].notna().sum())
    )


def boston_landmarks(circle, circle_wgs, out: Path, record: dict) -> None:
    package = _ckan_package(LANDMARKS_PACKAGE)
    record["landmarks"] = {
        "source": package["title"],
        "publisher": "City of Boston",
        "licence": package.get("license_title"),
        "metadata_modified": package.get("metadata_modified"),
        "url": LANDMARKS_LAYER,
        "downloaded": _now(),
    }
    raw, downloaded = _query_layer(LANDMARKS_LAYER, circle_wgs)
    landmarks = raw[raw.intersects(circle)]
    demolished = landmarks["Demolished"].astype(str).str.strip().str.lower().isin(["yes", "y"])
    landmarks = landmarks[~demolished]
    # the names of people (petitioners, the editors' user names) are left out
    personal = [c for c in landmarks.columns if c.endswith("_user") or c == "Petitioner"]
    landmarks = landmarks.drop(columns=personal)
    landmarks.reset_index(drop=True).to_file(out, layer="Boston_historic_landmarks", driver="GPKG")
    record["landmarks"].update(
        features_in_bounding_box=downloaded,
        demolished_left_out=int(demolished.sum()),
        within_4km=len(landmarks),
        status=landmarks["Status"].value_counts(dropna=False).to_dict(),
    )


def tadoussac_network(out: Path, record: dict) -> None:
    record["tadoussac"] = {
        "source": "OpenStreetMap via OSMnx",
        "tags": {"highway": True},
        "bounds_west_south_east_north": TADOUSSAC_BOUNDS,
        "downloaded": _now(),
    }
    area = box(*TADOUSSAC_BOUNDS)
    ways = ci.features_from_osm(area, {"highway": True}, download_method="polygon", crs="EPSG:4326")
    ways = ways[ways.geometry.geom_type == "LineString"]
    ways = gpd.clip(ways, area).explode(index_parts=False)
    ways = ways[ways.geometry.geom_type == "LineString"]

    def tag(column: str) -> pd.Series:
        return ways[column] if column in ways.columns else pd.Series(None, index=ways.index)

    ways_gdf = gpd.GeoDataFrame(
        {
            "osm_id": [int(i[1]) for i in ways.index],
            "name": tag("name").values,
            "ref": tag("ref").values,
            "type": tag("highway").values,
            "oneway": tag("oneway").isin(["yes", "true", "1", "-1"]).astype(int).values,
            "bridge": (tag("bridge").notna() & (tag("bridge") != "no")).astype(int).values,
            "maxspeed": pd.to_numeric(
                tag("maxspeed").astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
            ).values,
        },
        geometry=ways.geometry.values,
        crs="EPSG:4326",
    )
    ways_gdf.to_file(out, layer="Tadoussac_street_network", driver="GPKG")
    record["tadoussac"].update(ways=len(ways_gdf), types=ways_gdf["type"].value_counts().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("current", type=Path, help="the current docs/data folder")
    parser.add_argument("output", type=Path, help="where the dataset is written")
    parser.add_argument("--overpass-url", help="Overpass endpoint for OSMnx")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    args = parser.parse_args()
    if args.overpass_url:
        ox.settings.overpass_url = args.overpass_url
    ox.settings.requests_timeout = 600

    out = args.output
    for folder in ("Boston/otherSources", "Tadoussac"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    previous = out / "provenance.json"
    record: dict = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    record.update(
        {
            "built": _now(),
            "steps_built": args.steps,
            "cityImage": ci.__version__,
            "osmnx": version("osmnx"),
            "geopandas": version("geopandas"),
            "overpass_url": ox.settings.overpass_url,
            "area": {"centre_lon_lat": CITY_HALL, "radius_m": RADIUS, "crs": CRS},
        }
    )
    circle = gpd.GeoSeries([Point(CITY_HALL)], crs="EPSG:4326").to_crs(CRS).buffer(RADIUS).iloc[0]
    circle_wgs = gpd.GeoSeries([circle], crs=CRS).to_crs("EPSG:4326").iloc[0]

    if "streets" in args.steps:
        street_network(circle_wgs, out / "Boston/Boston_streetNetwork4000.gpkg", record)
    if "buildings" in args.steps:
        boston_buildings(circle, circle_wgs, out / "Boston/Boston_obstructions.gpkg", record)
    if "landmarks" in args.steps:
        boston_landmarks(
            circle, circle_wgs, out / "Boston/otherSources/Boston_historic_landmarks.gpkg", record
        )
    if "tadoussac" in args.steps:
        tadoussac_network(out / "Tadoussac/Tadoussac_street_network.gpkg", record)
    for name in COPIED:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.current / name, out / name)

    files = sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.suffix in {".gpkg", ".tif"}
    )
    record["files"] = {
        name: {"bytes": (out / name).stat().st_size, "sha256": _sha256(out / name)}
        for name in files
    }
    archive = out / "cityImage-docs-data.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (out / name).read_bytes())
    record["archive"] = {
        "name": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
    }
    (out / "provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["archive"], indent=2))


if __name__ == "__main__":
    main()
