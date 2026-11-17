"""Land-use triplets ("token:group:domain") into labels: OSM macro groups, DMA categories
(live, work, visit) or classes of a sparse attribute.

The triplets come from :mod:`cityImage.landuse.derive`, already resolved. UNCLASSIFIED tokens
can be listed with :func:`find_unclassified_tokens_OSM_groups` and assigned a group by hand with
:func:`apply_manual_triplet_overrides`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .tags import OSM_DOMAIN_GROUPS
from .utils import _to_list

UNCLASSIFIED = "UNCLASSIFIED"


def classify_land_uses_raws_into_OSMgroups(
    buildings_gdf, land_uses_raw_column: str = "land_uses_raw", new_group_column: str = "land_uses"
):
    """
    Convert per-row triplets into per-row macro-group labels.

    Input
    -----
    buildings_gdf[land_uses_raw_column]:
        Scalar or list-like of triplets formatted as:
            "<token>:<group>:<domain>"

    Output
    ------
    buildings_gdf[new_group_column]:
        list[str] of group labels, order-preserving and de-duplicated.
        UNCLASSIFIED is kept (never dropped).

    Notes
    -----
    - Malformed triplets (not 3-part) are ignored.
    - De-dup is by group string only (first occurrence wins).
    """
    buildings_gdf = buildings_gdf.copy()

    def _row_triplets_to_groups(cell: Any) -> list[str]:
        triplets = _to_list(cell)
        groups: list[str] = []
        seen: set[str] = set()

        for triplet in triplets:
            string = str(triplet).strip()
            parts = string.split(":", 2)
            if len(parts) != 3:
                continue

            _, group, _ = parts
            group = group.strip()

            if group in seen:
                continue
            seen.add(group)
            groups.append(group)

        return groups

    buildings_gdf[new_group_column] = buildings_gdf[land_uses_raw_column].apply(
        _row_triplets_to_groups
    )
    return buildings_gdf


def find_unclassified_tokens_OSM_groups(
    buildings_gdf,
    land_uses_raw_column: str = "land_uses_raw",
    return_counts: bool = True,
    mode: str = "token",  # "token" | "token_domain" | "triplet"
):
    """
    Find UNCLASSIFIED items in "<token>:<group>:<domain>" triplets.

    mode:
      - "token":        returns token only
      - "token_domain": returns "token:domain"
      - "triplet":      returns full triplet string
    """
    exploded = buildings_gdf[land_uses_raw_column].apply(_to_list).explode().dropna()

    def _parse(triplet: Any) -> tuple[str, str, str] | None:
        string = str(triplet).strip()
        parts = string.split(":", 2)
        if len(parts) != 3:
            return None
        return parts[0], parts[1], parts[2]

    parsed = exploded.map(_parse).dropna()
    unclassified = parsed[parsed.map(lambda x: x[1] == UNCLASSIFIED)]

    if mode == "token":
        values = unclassified.map(lambda x: x[0])
    elif mode == "token_domain":
        values = unclassified.map(lambda x: f"{x[0]}:{x[2]}")
    elif mode == "triplet":
        values = unclassified.map(lambda x: f"{x[0]}:{x[1]}:{x[2]}")
    else:
        raise ValueError("mode must be one of: 'token', 'token_domain', 'triplet'")

    return values.value_counts() if return_counts else sorted(set(values))


def apply_manual_triplet_overrides(
    buildings_gdf,
    overrides: dict[str, str],
    land_uses_raw_column: str = "land_uses_raw",
):
    """
    Override UNCLASSIFIED triplets using a user mapping token -> group (strict, domain-agnostic).

    Only triplets where group == UNCLASSIFIED are eligible.

    Triplet format
    -------------
        "<token>:<group>:<domain>"

    Validation (strict)
    -------------------
    - The target group must exist in the GLOBAL set of macro-groups
      (union of all OSM_DOMAIN_GROUPS[domain].keys()).
    - Domain is not used for validation (downstream consumes group only).
    - Malformed triplets are kept unchanged.

    Parameters
    ----------
    buildings_gdf
        GeoDataFrame containing the column of triplets.
    overrides : dict[str, str]
        Mapping {token -> group} where group must be in the global macro-group vocabulary.
    land_uses_raw_column : str, default "land_uses_raw"
        Column containing scalar or list-like triplets.

    Returns
    -------
    GeoDataFrame
        Copy with updated triplets in `land_uses_raw_column`.

    Raises
    ------
    ValueError
        - if overrides is empty / not a dict
        - if an override group is not in the global vocabulary
    """
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("overrides must be a non-empty dict[token -> group]")

    buildings_gdf = buildings_gdf.copy()

    allowed_groups: set[str] = {group for groups in OSM_DOMAIN_GROUPS.values() for group in groups}

    def _override_triplet(triplet: Any) -> str | None:
        if triplet is None:
            return None

        string = str(triplet).strip()
        if not string:
            return None

        parts = string.split(":", 2)
        if len(parts) != 3:
            return string  # malformed -> keep as-is

        token, group, domain = parts

        if group != UNCLASSIFIED:
            return string
        if token not in overrides:
            return string

        new_group = str(overrides[token]).strip()
        if new_group not in allowed_groups:
            raise ValueError(f"Invalid override: token '{token}' -> '{new_group}'")

        return f"{token}:{new_group}:{domain}"

    def _process_cell(cell: Any) -> list[str]:
        triplets = _to_list(cell)
        if not triplets:
            return []
        out: list[str] = []
        for triplet in triplets:
            updated = _override_triplet(triplet)
            if updated is not None:
                out.append(updated)
        return out

    buildings_gdf[land_uses_raw_column] = buildings_gdf[land_uses_raw_column].apply(_process_cell)
    return buildings_gdf


def classify_land_uses_intoDMAs(
    buildings_gdf,
    land_uses_column: str = "land_uses",
    macro_to_dma: dict[str, str] | None = None,
):
    """Classify cityImage land-use groups into DMA functional categories.

    The DMA categories are live, work, visit, and mixed combinations of those
    functions. Input labels should already be normalised cityImage land-use groups,
    for example values derived from OSM shop, amenity, office, tourism, leisure, or
    building tags.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        Building table containing normalised land-use labels.
    land_uses_column : str
        Column containing list-like land-use groups.
    macro_to_dma : mapping, optional
        Mapping from macro groups to DMA categories.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the building table with DMA classifications.
    """

    gdf = buildings_gdf.copy()

    if land_uses_column not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{land_uses_column}'")

    # Reasonable defaults for your OSM macro-groups
    default_macro_to_dma = {
        # LIVE
        "residential": "live",
        "accommodation": "live",
        # WORK
        "commercial": "work",
        "office": "work",
        "industrial": "work",
        "craft": "work",
        "agricultural": "work",
        "education": "work",
        "healthcare": "work",
        "public_service": "work",
        "transportation": "work",
        "waste_management": "work",
        "power_technical": "work",
        "storage": "work",
        "cars": "work",
        "civic_amenity": "work",
        # VISIT
        "tourism": "visit",
        "leisure": "visit",
        "sustenance": "visit",
        "entertainment_arts_culture": "visit",
        "financial": "visit",
        "religious": "visit",
        "place_of_worship": "visit",
        "sports": "visit",
    }

    if macro_to_dma is None:
        macro_to_dma = default_macro_to_dma

    def _dims_from_macros(cell: Any) -> set[str]:
        dims: set[str] = set()
        for macro in _to_list(cell):
            if macro is None:
                continue
            s = str(macro).strip()
            if not s or s == UNCLASSIFIED:
                continue
            if s.startswith("shop_"):
                dims.add("visit")
                continue
            d = macro_to_dma.get(s)
            if d in {"live", "work", "visit"}:
                dims.add(d)
        return dims

    def _dims_to_dma_label(dims: set[str]) -> str:
        if not dims:
            return "other"
        if dims == {"live"}:
            return "live"
        if dims == {"work"}:
            return "work"
        if dims == {"visit"}:
            return "visit"
        if dims == {"live", "work"}:
            return "live_work"
        if dims == {"live", "visit"}:
            return "live_visit"
        if dims == {"work", "visit"}:
            return "work_visit"
        if dims == {"live", "work", "visit"}:
            return "live_work_visit"
        return "other"  # defensive

    dims_series = gdf[land_uses_column].apply(_dims_from_macros)
    gdf["DMA"] = dims_series.apply(_dims_to_dma_label)
    return gdf


def classify_land_use(
    buildings_gdf,
    raw_land_use_column: str,
    new_land_use_column: str,
    categories: list[list[object]],
    strings: list[str],
):
    """Classify sparse attribute values using explicit category lists.

    For non-OSM land-use attributes, :func:`classify_sparse_land_uses` is preferable.

    `categories[i]` contains raw values and `strings[i]` is the target class.
    Scalar and list-like cells are supported. Unmatched values are preserved.
    """
    if len(categories) != len(strings):
        raise ValueError("categories and strings must have the same length")

    gdf = buildings_gdf.copy()

    lookup = {}
    for raw_values, target in zip(categories, strings, strict=False):
        for raw_value in raw_values:
            lookup[raw_value] = target
            if raw_value is not None:
                lookup[str(raw_value).strip().lower()] = target

    def _classify_one(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return lookup.get(value, lookup.get(str(value).strip().lower(), value))

    def _classify_cell(value):
        values = _to_list(value)
        if not values:
            return None
        if len(values) == 1:
            return _classify_one(values[0])

        out = []
        seen = set()
        for item in values:
            classified = _classify_one(item)
            if classified is None:
                continue
            if classified not in seen:
                seen.add(classified)
                out.append(classified)
        return out

    gdf[new_land_use_column] = gdf[raw_land_use_column].apply(_classify_cell)
    return gdf
