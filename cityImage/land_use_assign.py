"""
Stage 3. Assign land-use labels to building polygons by
spatially matching against another GeoDataFrame (points or polygons).

Assumptions (per your latest design)
------------------------------------
- `other_gdf[other_land_use_column]` is already:
  1) cleaned (no junk tokens),
  2) normalized/canonical, and
  3) classified (i.e., already the labels you want to propagate, typically macro-groups).

Therefore this module:
- does NOT normalize tokens,
- does NOT apply amenity exclusions,
- does NOT qualify duplicates,
- does NOT apply cross-source redundancy rules.

It only:
- collects labels from spatial matches,
- de-duplicates per building while preserving order (point mode),
- aggregates polygon overlaps into per-building label weights (polygon mode),
- optionally fills a default label when no match exists.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .land_use_utils import _to_list, _deduplicate_preserving_order

# =============================================================================
# Public API
# =============================================================================

def land_use_from_other_gdf(
    buildings_gdf,
    other_gdf,
    new_land_use_column: str,
    other_land_use_column: str,
    min_overlap_threshold: float = 0.20,
    overlap_column_name: str | None = None,
    fill_default_when_no_match: bool = True,
    default_land_use: str = "residential",
):
    """
    Assign land-use labels to buildings by spatially matching against `other_gdf`.

    Modes
    -----
    - Point mode:  other_gdf contains only Point/MultiPoint.
      For each building, collect labels from all points intersecting the building polygon.
      No overlap column is created in point mode.

    - Polygon mode: other_gdf contains only Polygon/MultiPolygon.
      For each building, intersect matching polygons and aggregate overlap fractions
      (intersection area / building area), then sum by label.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Target building footprints (polygons). CRS must match `other_gdf`.
    other_gdf : GeoDataFrame
        Source geometries (Points/MultiPoints or Polygons/MultiPolygons) with labels in `other_land_use_column`.
        Must be non-empty and have valid geometries. CRS must match `buildings_gdf`.
    new_land_use_column : str
        Output column name. Each cell will be list[str].
    other_land_use_column : str
        Label column in `other_gdf`. Values may be scalar or list-like (already clean/classified).
    min_overlap_threshold : float, default 0.20
        Polygon mode only. Minimum per-intersection fraction of building area required for a polygon match.
        Applied per polygon intersection before aggregation.
    overlap_column_name : str | None, default None
        Polygon mode only. If not None, creates this column as list[float] aligned 1:1 with `new_land_use_column`.
        Point mode ignores this parameter and never creates an overlap column.
    fill_default_when_no_match : bool, default True
        If True, buildings with no matches receive `[default_land_use]`. If False, they keep empty lists.
    default_land_use : str, default "residential"
        Default label used when `fill_default_when_no_match=True` and no matches are found.

    Returns
    -------
    GeoDataFrame
        Copy of `buildings_gdf` with `new_land_use_column` populated (list[str] per row).
        In polygon mode, if `overlap_column_name` is provided, also includes an overlap column
        (list[float] per row aligned 1:1 with the labels).

    Raises
    ------
    ValueError
        - CRS mismatch
        - other_gdf is empty or has no valid geometries
        - unsupported/mixed geometry types in `other_gdf`
    """
    if buildings_gdf.crs != other_gdf.crs:
        raise ValueError("CRS mismatch: buildings_gdf and other_gdf must have the same CRS")
    if other_gdf.empty:
        raise ValueError("other_gdf is empty: nothing to match against")

    geom_types = set(other_gdf.geometry.dropna().geom_type.unique())
    if not geom_types:
        raise ValueError("other_gdf has no valid geometries")

    is_point_mode = geom_types.issubset({"Point", "MultiPoint"})
    is_poly_mode = geom_types.issubset({"Polygon", "MultiPolygon"})
    if not (is_point_mode or is_poly_mode):
        raise ValueError(f"Unsupported/mixed geometry types in other_gdf: {sorted(geom_types)}")

    if is_point_mode:
        output_gdf = land_use_from_points(
            buildings_gdf=buildings_gdf,
            other_gdf=other_gdf,
            new_land_use_column=new_land_use_column,
            other_land_use_column=other_land_use_column,
        )
        effective_overlap_col = None
    else:
        output_gdf = land_use_from_polygons(
            buildings_gdf=buildings_gdf,
            other_gdf=other_gdf,
            new_land_use_column=new_land_use_column,
            other_land_use_column=other_land_use_column,
            min_overlap_threshold=min_overlap_threshold,
            overlap_column_name=overlap_column_name,
        )
        effective_overlap_col = overlap_column_name

        # Defensive: ensure overlap column exists when requested
        if effective_overlap_col is not None and effective_overlap_col not in output_gdf.columns:
            output_gdf[effective_overlap_col] = [[] for _ in range(len(output_gdf))]

    if not fill_default_when_no_match:
        return output_gdf

    # Apply defaults only where no labels were assigned (empty list)
    no_land_use_mask = output_gdf[new_land_use_column].apply(lambda v: len(_to_list(v)) == 0)
    if no_land_use_mask.any():
        idx = output_gdf.index[no_land_use_mask]

        # Safe assignment: keep list objects (avoid pandas "unboxing"/broadcast quirks)
        output_gdf[new_land_use_column] = output_gdf[new_land_use_column].astype("object")
        output_gdf.loc[idx, new_land_use_column] = pd.Series(
            [[default_land_use]] * len(idx),
            index=idx,
            dtype="object",
        )

        # In polygon mode: optionally also assign default overlap=1.0
        if effective_overlap_col is not None:
            empty_overlap_mask = output_gdf[effective_overlap_col].apply(lambda v: len(_to_list(v)) == 0)
            apply_default_overlap_mask = no_land_use_mask & empty_overlap_mask
            if apply_default_overlap_mask.any():
                idx2 = output_gdf.index[apply_default_overlap_mask]
                output_gdf[effective_overlap_col] = output_gdf[effective_overlap_col].astype("object")
                output_gdf.loc[idx2, effective_overlap_col] = pd.Series(
                    [[1.0]] * len(idx2),
                    index=idx2,
                    dtype="object",
                )

    return output_gdf


# =============================================================================
# Point mode
# =============================================================================

def land_use_from_points(
    buildings_gdf,
    other_gdf,
    new_land_use_column: str,
    other_land_use_column: str,
):
    """
    Collect labels from Point/MultiPoint geometries intersecting each building.

    Notes
    -----
    - Assumes other_gdf labels are already clean/normalized/classified.
    - De-duplicates collected labels per building while preserving first-seen order.
    - No overlap column is created in point mode.
    """
    if buildings_gdf.crs != other_gdf.crs:
        raise ValueError("CRS mismatch: buildings_gdf and other_gdf must have the same CRS")

    output_gdf = buildings_gdf.copy()
    output_gdf[new_land_use_column] = [[] for _ in range(len(output_gdf))]

    sindex = other_gdf.sindex

    def _collect_point_labels(building_geometry):
        # bbox filter via sindex, then exact predicate
        candidate_idx = list(sindex.intersection(building_geometry.bounds))
        if not candidate_idx:
            return []
        candidates = other_gdf.iloc[candidate_idx]
        matches = candidates[candidates.intersects(building_geometry)]
        if matches.empty:
            return []

        flat: list[str] = []
        for v in matches[other_land_use_column].tolist():
            flat.extend(_to_list(v))
        return _deduplicate_preserving_order(flat)

    output_gdf[new_land_use_column] = output_gdf.geometry.apply(_collect_point_labels)
    return output_gdf


# =============================================================================
# Polygon mode
# =============================================================================

def land_use_from_polygons(
    buildings_gdf,
    other_gdf,
    new_land_use_column: str,
    other_land_use_column: str,
    min_overlap_threshold: float = 0.20,
    overlap_column_name: str | None = None,
):
    """
    Intersect polygons with each building and compute *building-centric* land-use weights.

    Aggregation model (per building)
    -------------------------------
    1) Find candidate polygons via spatial index (bbox), then filter with intersects().
    2) Compute raw intersection fractions:
         frac = area(building ∩ polygon) / area(building)
    3) Drop polygon matches with frac < min_overlap_threshold.
    4) If a polygon carries multiple labels, split its frac evenly across its labels.
    5) Sum fractions by label.
    6) Rescale across labels so that the final overlaps sum to 1.0 for the building:
         norm = raw / sum(raw)

    Output ordering
    ---------------
    Labels are ordered by normalized overlap descending.
    Ties are broken deterministically by label ascending.

    Notes
    -----
    - Assumes other_gdf labels are already clean/normalized/classified.
    - If a polygon label-list contains duplicates, we de-duplicate within that polygon
      to avoid accidental double counting.
    - Overlaps returned are *mixture weights* (sum to 1.0), not raw "% of building area".
    """
    if buildings_gdf.crs != other_gdf.crs:
        raise ValueError("CRS mismatch: buildings_gdf and other_gdf must have the same CRS")

    output_gdf = buildings_gdf.copy()
    output_gdf[new_land_use_column] = [[] for _ in range(len(output_gdf))]
    if overlap_column_name is not None:
        output_gdf[overlap_column_name] = [[] for _ in range(len(output_gdf))]

    sindex = other_gdf.sindex

    def _safe_intersection_areas(poly_series, geom):
        # buffer(0) is a pragmatic “light fix” for occasional GEOS topology errors.
        try:
            return poly_series.intersection(geom).area
        except Exception:
            geom_fixed = geom.buffer(0)
            poly_fixed = poly_series.buffer(0)
            return poly_fixed.intersection(geom_fixed).area

    def _labels_list(v) -> list[str]:
        # other_gdf is already clean/classified; ensure list-like + de-dup within polygon.
        return _deduplicate_preserving_order(_to_list(v))

    def _renormalize_to_one(labels: list[str], weights: list[float], ndigits: int = 3):
        """
        Normalize weights so they sum to 1.0, then round.
        After rounding, adjust the last weight to make the rounded sum exactly 1.0 (if possible).
        """
        if not labels or not weights:
            return [], []

        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            return [], []

        norm = [float(w) / total for w in weights]

        # round
        norm_r = [round(w, ndigits) for w in norm]

        # make sum exactly 1.0 after rounding (deterministic: adjust last)
        s = round(sum(norm_r), ndigits)
        target = round(1.0, ndigits)
        delta = round(target - s, ndigits)

        if norm_r and delta != 0:
            norm_r[-1] = round(norm_r[-1] + delta, ndigits)
            # if adjustment makes it negative due to extreme rounding edge-case, fall back to unadjusted
            if norm_r[-1] < 0:
                norm_r = [round(w, ndigits) for w in norm]

        return labels, norm_r

    def _land_use_and_overlap(building_geometry):
        candidate_idx = list(sindex.intersection(building_geometry.bounds))
        if not candidate_idx:
            return [], []

        candidates = other_gdf.iloc[candidate_idx]
        matches = candidates[candidates.intersects(building_geometry)]
        if matches.empty:
            return [], []

        # Fast path: identical geometry => full cover split evenly across labels.
        identical = matches[matches.geometry.geom_equals(building_geometry)]
        if not identical.empty:
            land_uses = _labels_list(identical.iloc[0][other_land_use_column])
            if not land_uses:
                return [], []
            raw = [1.0 / len(land_uses)] * len(land_uses)
            return _renormalize_to_one(land_uses, raw)

        building_area = building_geometry.area
        if not np.isfinite(building_area) or building_area <= 0:
            return [], []

        inter_areas = _safe_intersection_areas(matches.geometry, building_geometry)
        fracs = inter_areas / building_area

        # Per-intersection filter: applied before any aggregation (by design).
        keep = fracs >= float(min_overlap_threshold)
        matches = matches[keep]
        fracs = fracs[keep]
        if matches.empty:
            return [], []

        matches = matches.assign(
            _frac=fracs,
            _labels=matches[other_land_use_column].apply(_labels_list),
        )

        # If a polygon has multiple labels, split its frac evenly across them.
        label_counts = matches["_labels"].str.len().replace(0, np.nan)
        matches["_frac"] = matches["_frac"] / label_counts

        exploded = matches.explode("_labels", ignore_index=True)
        exploded = exploded[exploded["_labels"].notna()].copy()
        if exploded.empty:
            return [], []

        exploded["_label"] = exploded["_labels"]
        agg = exploded.groupby("_label", sort=False)["_frac"].sum().reset_index()

        # Order by RAW overlap first (for stable deterministic ordering), then label asc.
        agg = agg.sort_values(["_frac", "_label"], ascending=[False, True], kind="mergesort")

        labels = agg["_label"].tolist()
        raw_overlaps = [float(v) for v in agg["_frac"].to_numpy()]

        # Renormalize so overlaps sum to 1.0 per building (your requested behavior).
        labels, overlaps = _renormalize_to_one(labels, raw_overlaps, ndigits=3)

        # Re-order by NORMALIZED overlap desc, then label asc (requested semantics).
        if labels:
            tmp = list(zip(labels, overlaps))
            tmp.sort(key=lambda x: (-x[1], x[0]))
            labels, overlaps = zip(*tmp)
            return list(labels), list(overlaps)

        return [], []

    results = output_gdf.geometry.apply(_land_use_and_overlap)

    output_gdf[new_land_use_column] = results.apply(lambda x: x[0])
    if overlap_column_name is not None:
        output_gdf[overlap_column_name] = results.apply(lambda x: x[1])

    return output_gdf
