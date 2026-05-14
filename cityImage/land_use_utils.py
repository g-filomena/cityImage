import re
from typing import Any

import numpy as np
import pandas as pd

FALSEY_OSM_TAG_VALUES = {"no", "false", "0"}
_COLON_RE = re.compile(r":+")
_WS_RE = re.compile(r"\s+")
_BAD_CHARS_RE = re.compile(r"[^a-z0-9_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")

# Tokens to drop after normalization.
_DROP = {"yes", "no", "true", "false", "1", "0", "unknown", "unclassified", "fixme", "none", ""}


def _is_missing_scalar(value: Any) -> bool:
    """True for None/NA/NaN scalars; False for list-like values."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return False
    return bool(pd.isna(value))


def _is_truthy_osm_tag_value(value: Any) -> bool:
    """Treat a tag as present unless missing or explicitly falsey ('no', 'false', '0')."""
    if _is_missing_scalar(value):
        return False
    return str(value).strip().lower() not in FALSEY_OSM_TAG_VALUES


def _deduplicate_preserving_order(values):
    """Fast de-duplication preserving first-seen order (assumes hashable tokens)."""
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _is_canonical_token(token: str) -> bool:
    """
    True if `token` already matches _normalize_token() output:
    - lowercase
    - no whitespace / hyphens
    - underscores not duplicated or dangling
    - characters limited to [a-z0-9_]
    """
    if not token:
        return False
    if token != token.lower():
        return False
    if any(character in token for character in (" ", "\t", "\r", "\n", "-")):
        return False
    if token.startswith("_") or token.endswith("_"):
        return False
    if "__" in token:
        return False
    return _BAD_CHARS_RE.search(token) is None


def _to_list(value: Any) -> list[Any]:
    """
    Normalize a value into a Python list.

    Rules:
      - None -> []
      - pandas/NumPy missing scalars (NaN, pd.NA, NaT) -> []
      - list -> value (as-is)
      - tuple/set -> list(value)
      - numpy.ndarray / pandas.Series -> value.tolist()
      - string -> [string]
      - anything else -> [value]
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if isinstance(value, (pd.Series, np.ndarray)):
        return value.tolist()

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [value]


def _normalize_token(value: object) -> str | None:
    """
    Normalize a raw OSM scalar into a canonical token.

    Policy:
      - lowercase
      - trim
      - remove ':' entirely (so raw noise like "fi:re_station" -> "fire_station")
      - collapse whitespace/hyphens/etc. into underscores
      - remove non [a-z0-9_]
      - collapse duplicate underscores
      - strip leading/trailing underscores
    """
    if value is None:
        return None

    string = str(value).strip().lower()
    if not string:
        return None

    # IMPORTANT: strip ':' at ingestion (qualifiers are added later by code)
    string = _COLON_RE.sub("", string)

    string = string.replace("-", "_")
    string = _WS_RE.sub("_", string)
    string = _BAD_CHARS_RE.sub("_", string)
    string = _MULTI_UNDERSCORE_RE.sub("_", string).strip("_")

    return string or None


def _clean_tokens(raw_tokens: Any) -> list[str]:
    """
    Clean + normalize tokens, splitting ';' multi-values, dropping junk, de-duping in order.

    Returns
    -------
    list[str]
        Normalized base tokens (no qualification is applied here).
    """
    if raw_tokens is None:
        return []

    if isinstance(raw_tokens, str):
        items = [raw_tokens]
    elif isinstance(raw_tokens, (list, tuple, set, np.ndarray, pd.Series)):
        items = list(raw_tokens)
    else:
        items = [raw_tokens]

    cleaned: list[str] = []
    seen: set[str] = set()

    def _add(token: str | None) -> None:
        if token and token not in _DROP and token not in seen:
            seen.add(token)
            cleaned.append(token)

    for item in items:
        if item is None:
            continue

        text = item.strip() if isinstance(item, str) else str(item).strip()
        if not text:
            continue

        if ";" not in text:
            token = text if _is_canonical_token(text) else _normalize_token(text)
            _add(token)
            continue

        for part in (p.strip() for p in text.split(";")):
            if part:
                _add(_normalize_token(part))

    return cleaned

def find_land_use_values_matching(
    buildings_gdf,
    land_uses_column: str = "land_uses",
    pattern: str = r"^shop",          # e.g. "^shop" or "shop"
    ignore_case: bool = True,
    return_counts: bool = True,
) -> pd.Series | list[str]:
    """
    Find land-use labels in buildings_gdf[land_uses_column] matching a regex pattern.

    Assumes cells contain either:
      - list[str] of labels, or
      - a scalar label (string), or
      - missing/empty.

    Parameters
    ----------
    buildings_gdf
        GeoDataFrame / DataFrame with the land use column.
    land_uses_column : str, default "land_uses"
        Column containing list-like or scalar land-use labels.
    pattern : str, default "^shop"
        Regex pattern to match labels (e.g. "^shop" matches "shop_*").
    ignore_case : bool, default True
        If True uses case-insensitive matching.
    return_counts : bool, default True
        If True returns value_counts() of matched labels.
        If False returns sorted unique matched labels.

    Returns
    -------
    pandas.Series | list[str]
        Counts (Series) or sorted unique labels.
    """
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(pattern, flags)

    exploded = buildings_gdf[land_uses_column].explode().dropna()
    matched = exploded.map(lambda v: str(v).strip()).loc[lambda s: s.map(lambda x: bool(rx.search(x)))]

    return matched.value_counts() if return_counts else sorted(set(matched))
