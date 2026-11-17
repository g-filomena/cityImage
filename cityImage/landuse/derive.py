"""Raw land-use triplets ("token:macro_group:domain") of buildings downloaded from OSM.

The tokens of each building are read from its tags, as single values or ";"-separated lists;
junk values (yes, no, ...) and the amenities in ``EXCLUDE_AMENITIES`` are dropped, and
duplicates removed in order. Within a domain, a group label "g:g:D" is dropped when one of its
members "x:g:D" is present (``OSM_DOMAIN_GROUPS``, ``OSM_DOMAIN_VALUE_TO_GROUP``). The triplets
are then resolved with ``RESOLUTION_RULES`` (:func:`apply_resolution_rules`). The registries
are in ``cityImage.landuse.tags``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .tags import (
    EXCLUDE_AMENITIES,
    OSM_DOMAIN_GROUPS,
    OSM_DOMAIN_VALUE_TO_GROUP,
    POW_RELIGIOUS_BUILDINGS,
    RESOLUTION_RULES,
)
from .utils import (
    _clean_tokens,
    _is_missing_scalar,
    _is_truthy_osm_tag_value,
    _normalize_token,
)

BUILDING_USE_PREFIX = "building:use:"

# OSM domains (ordered) come from tags registry
OSM_DOMAINS = tuple(OSM_DOMAIN_GROUPS)  # e.g. amenity, building, shop, industrial, craft, ...


def drop_redundant_group_label(triplets: list[str]) -> list[str]:
    """
    Remove macro-group label triplets when at least one concrete member exists
    for the same (macro_group, domain).

    Triplet format
    --------------
    "base:macro_group:domain"

    Rule
    ----
    Drop "g:g:D" if there exists any "x:g:D" with x != g.

    Intended effect
    ---------------
    Avoid keeping both a container/group label and a specific member value
    in the same domain bucket.

    Examples
    --------
    in:  ["trullo:accommodation:building", "accommodation:accommodation:building"]
    out: ["trullo:accommodation:building"]

    in:  ["accommodation:accommodation:building"]
    out: ["accommodation:accommodation:building"]   # group label remains if no member exists
    """

    if not triplets:
        return triplets

    member_groups: set[tuple[str, str]] = set()

    for triplet in triplets:
        try:
            base, macro_group, domain = triplet.split(":", 2)
        except ValueError:
            continue

        if macro_group and macro_group != "UNCLASSIFIED" and base != macro_group:
            member_groups.add((macro_group, domain))

    if not member_groups:
        return triplets

    out: list[str] = []
    for triplet in triplets:
        try:
            base, macro_group, domain = triplet.split(":", 2)
        except ValueError:
            out.append(triplet)
            continue

        is_group_label_triplet = (
            macro_group and macro_group != "UNCLASSIFIED" and base == macro_group
        )

        if is_group_label_triplet and (macro_group, domain) in member_groups:
            continue

        out.append(triplet)

    return out


def derive_land_uses_raw_fromOSM(buildings_gdf, default: str = "residential"):
    """Derive raw cityImage land-use candidates from OSM-style tag columns.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        Building table containing OSM-style tag columns.
    land_uses_raw_column : str
        Name of the output column containing raw land-use candidates.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the building table with derived raw land-use candidates.

    Notes
    -----
    The output is an intermediate representation used by the land-use classifier.
    It preserves OSM tag provenance before conversion to cityImage land-use groups.
    """

    buildings_gdf = buildings_gdf.copy()
    default_token = _normalize_token(default)

    # token -> first domain where token exists (OSM_DOMAINS order)
    # token_primary_domain: dict[str, str] = {}
    # for domain in OSM_DOMAINS:
    # for token in OSM_DOMAIN_VALUE_TO_GROUP[domain]:
    # token_primary_domain.setdefault(token, domain)
    # # or:

    # token -> first domain where token exists (values OR group labels), OSM_DOMAINS order
    token_primary_domain: dict[str, str] = {}
    for domain in OSM_DOMAINS:
        # registered values
        for token in OSM_DOMAIN_VALUE_TO_GROUP[domain]:
            token_primary_domain.setdefault(token, domain)
        # macro-group labels (top-level keys in the registry)
        for group_label in OSM_DOMAIN_GROUPS.get(domain, {}):
            token_primary_domain.setdefault(group_label, domain)

    # Precompute building:use:* suffix tokens once per GDF
    building_use_suffixes = []
    for col in buildings_gdf.columns:
        if isinstance(col, str) and col.startswith(BUILDING_USE_PREFIX):
            suffix = _normalize_token(col[len(BUILDING_USE_PREFIX) :])
            if suffix:
                building_use_suffixes.append((col, suffix))

    def _as_list(value: Any) -> list[Any]:
        if _is_missing_scalar(value):
            return []
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            return [v for v in value if not _is_missing_scalar(v)]
        return [value]

    def _build_token_macro_domain_triplet(token: str, source_domain: str) -> str:
        # choose domain
        if token in OSM_DOMAIN_VALUE_TO_GROUP[source_domain] or token in OSM_DOMAIN_GROUPS.get(
            source_domain, {}
        ):
            chosen_domain = source_domain
        else:
            # fallback: keep provenance (source_domain) if token is unknown everywhere
            chosen_domain = token_primary_domain.get(token, source_domain)

        # choose macro-group
        macro_group = OSM_DOMAIN_VALUE_TO_GROUP.get(chosen_domain, {}).get(token)

        if macro_group is None and token in OSM_DOMAIN_GROUPS.get(chosen_domain, {}):
            macro_group = token

        if not macro_group:
            macro_group = "UNCLASSIFIED"

        return f"{token}:{macro_group}:{chosen_domain}"

    def _clean_and_label(source_domain: str, raw_tokens: list[Any]) -> list[str]:
        cleaned = _clean_tokens(raw_tokens)  # ":" stripped in normalize_token
        cleaned = [token for token in cleaned if token and token not in EXCLUDE_AMENITIES]
        return [_build_token_macro_domain_triplet(token, source_domain) for token in cleaned]

    def _extract_row(row: pd.Series) -> list[str]:
        labeled_tokens = []

        # building:use:* -> source_domain="building"
        for col_name, suffix_token in building_use_suffixes:
            if _is_truthy_osm_tag_value(row.get(col_name)):
                labeled_tokens.append(_build_token_macro_domain_triplet(suffix_token, "building"))

        # domain columns (amenity/shop/...) in OSM_DOMAINS order
        for domain in OSM_DOMAINS:
            labeled_tokens.extend(_clean_and_label(domain, _as_list(row.get(domain))))

        # row-level de-dup, preserve first-seen order
        labeled_tokens = list(dict.fromkeys(labeled_tokens))
        labeled_tokens = drop_redundant_group_label(labeled_tokens)
        labeled_tokens = apply_resolution_rules(labeled_tokens)

        if not labeled_tokens and default_token:
            labeled_tokens = [_build_token_macro_domain_triplet(default_token, "building")]

        return labeled_tokens

    buildings_gdf["land_uses_raw"] = buildings_gdf.apply(_extract_row, axis=1)
    return buildings_gdf


def apply_resolution_rules(triplets: list[str]) -> list[str]:
    """Resolve the "token:macro_group:domain" triplets of a row with ``RESOLUTION_RULES``.

    A row with an UNCLASSIFIED macro group or domain in any triplet is only freed of exact
    duplicates: it is meant to be fixed by hand first (with overrides) and resolved again.
    In the other rows, in this order:

    1. "when" rules reclassify single triplets. Their only condition is ``domain_equals`` and
       their only action ``reclassify``; the first matching clause wins.
    2. A token with a canonical form takes it, and is kept once, at its first occurrence.
    3. When a religious building is present (a ``building`` token in
       ``POW_RELIGIOUS_BUILDINGS``), every ``place_of_worship`` triplet is dropped.
    4. Container rules drop a parent when one of its children is present, after steps 1 and 2.
    5. Exact duplicates are dropped; the triplets keep the input order.

    Triplets are assumed well formed ("token:macro_group:domain").
    """
    if not triplets:
        return triplets

    def parse(t: str) -> tuple[str, str, str]:
        token, group, domain = t.split(":", 2)
        return token, group, domain

    def make(token: str, group: str, domain: str) -> str:
        return f"{token}:{group}:{domain}"

    # Parse once, preserve original order via index
    parsed = [(i, *parse(t)) for i, t in enumerate(triplets)]

    # A row with any UNCLASSIFIED triplet is left as it is, only without exact duplicates.
    if any(group == "UNCLASSIFIED" or domain == "UNCLASSIFIED" for _, _, group, domain in parsed):
        return list(dict.fromkeys(triplets))

    # token -> (canonical macro_group, canonical domain)
    canonical_target: dict[str, tuple[str, str]] = {
        token: (rule["canonical"]["macro_group"], rule["canonical"]["domain"])
        for token, rule in RESOLUTION_RULES.items()
        if "canonical" in rule
    }

    # Pass 1: the "when" rules reclassify single triplets, before presence and canonical forms
    # are computed. Only `domain_equals` conditions and the `reclassify` action exist; the first
    # matching clause wins.
    reclassify_override: dict[int, tuple[str, str, str]] = {}

    for i, token, _group, domain in parsed:
        for clause in RESOLUTION_RULES.get(token, {}).get("when") or []:
            cond = clause.get("if") or {}
            if cond.get("domain_equals") != domain:
                continue

            then = clause.get("then") or {}
            if then.get("action") == "reclassify":
                to = then.get("to") or {}
                reclassify_override[i] = (
                    to.get("token", token),
                    to["macro_group"],
                    to["domain"],
                )
            break  # first matching clause wins

    def after_when(i: int, token: str, group: str, domain: str) -> tuple[str, str, str]:
        """The triplet after pass 1 (unchanged when no rule applies)."""
        return reclassify_override.get(i, (token, group, domain))

    # Pass 2: which tokens are present in each domain, after the "when" rules and the canonical
    # forms. The place-of-worship drop and the container rules read it.
    domain_to_tokens: dict[str, set[str]] = {}

    for i, token, group, domain in parsed:
        token, group, domain = after_when(i, token, group, domain)

        if token in canonical_target:
            group, domain = canonical_target[token]

        domain_to_tokens.setdefault(domain, set()).add(token)

    # A religious building drops every place_of_worship triplet of the row.
    religious_building_present = bool(
        domain_to_tokens.get("building", set()) & POW_RELIGIOUS_BUILDINGS
    )

    # Containers: the (parent_token, parent_domain) pairs to drop because a child is present.
    container_drops: set[tuple[str, str]] = set()

    for token, rule in RESOLUTION_RULES.items():
        container = rule.get("container")
        if not container:
            continue

        parent_domain = container["parent_domain"]

        # Only a parent present after pass 1 and the canonical forms can be dropped.
        if token not in domain_to_tokens.get(parent_domain, set()):
            continue

        for cond in container.get("drop_parent_if_child") or []:
            child_domain = cond["child_domain"]
            child_tokens = cond.get("child_tokens")  # None => any token in child_domain
            present_children = domain_to_tokens.get(child_domain, set())

            if child_tokens is None:
                if present_children:
                    container_drops.add((token, parent_domain))
                    break
            else:
                if present_children & set(child_tokens):
                    container_drops.add((token, parent_domain))
                    break

    # Output, in the input order: "when" reclassification, one canonical triplet per token,
    # the place-of-worship drop, the container drops, and no exact duplicates.
    out: list[str] = []
    seen: set[str] = set()  # exact triplet dedup
    emitted_canon: set[str] = set()  # token-level canonical dedup

    for i, tok0, grp0, dom0 in parsed:
        tok, grp, dom = after_when(i, tok0, grp0, dom0)

        # A token with a canonical form is emitted once, in that form, at its first occurrence.
        if tok in canonical_target:
            if tok in emitted_canon:
                continue
            grp, dom = canonical_target[tok]
            emitted_canon.add(tok)

        if religious_building_present and dom == "place_of_worship":
            continue

        # Container drop
        if (tok, dom) in container_drops:
            continue

        # Exact triplet dedup (order-preserving)
        t = make(tok, grp, dom)
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out
