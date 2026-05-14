"""
Stage 1: derive raw land-use triplets ("token:macro_group:domain") from an OSM-derived buildings GeoDataFrame.

Key features
------------
1) Normalize + clean tokens
   - accepts scalars and ";"-separated multi-values
   - drops junk values (yes/no/etc.) and excluded amenities
   - order-preserving de-duplication per row

2) Redundancy removal (registry-driven)
   - within-domain: drop group-label triplets "g:g:D" if any member "x:g:D" exists (x != g),
     using OSM_DOMAIN_GROUPS and OSM_DOMAIN_VALUE_TO_GROUP.

3) Rule-based resolution
   - applies RESOLUTION_RULES (canonical targets, conditional reclassify, container drops)
   - global: if any religious building is present, drop all domain=="place_of_worship" triplets.

Dependencies (from .land_use_tags)
----------------------------------
- EXCLUDE_AMENITIES: set[str]
- RESOLUTION_RULES: dict[str, ...]
- POW_RELIGIOUS_BUILDINGS: set[str]
- OSM_DOMAIN_GROUPS: dict[str, dict[str, list[str]]]
- OSM_DOMAIN_VALUE_TO_GROUP: dict[str, dict[str, str]]

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .land_use_tags import (
    EXCLUDE_AMENITIES,
    RESOLUTION_RULES,
    OSM_DOMAIN_GROUPS,
    OSM_DOMAIN_VALUE_TO_GROUP,
    POW_RELIGIOUS_BUILDINGS
)

from .land_use_utils import _normalize_token, _clean_tokens, _is_truthy_osm_tag_value, _is_missing_scalar

# -----------------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------------

BUILDING_USE_PREFIX = "building:use:"

# OSM domains (ordered) come from tags registry
OSM_DOMAINS = tuple(OSM_DOMAIN_GROUPS) # e.g. amenity, building, shop, industrial, craft, ...

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
            macro_group
            and macro_group != "UNCLASSIFIED"
            and base == macro_group
        )

        if is_group_label_triplet and (macro_group, domain) in member_groups:
            continue

        out.append(triplet)

    return out

def derive_land_uses_raw_fromOSM(buildings_gdf, default: str = "residential"):
    """
    Derive building-level land-use triplets from an OSM-derived GeoDataFrame.

    Output per row
    --------------
    - land_uses_raw: list[str] of "base:macro_group:domain"

    Extraction sources (merge priority)
    -----------------------------------
    1) building:use:* columns (truthy => include the suffix token; source domain "building")
    2) Domain tag columns for each domain in OSM_DOMAINS order (amenity, building, shop, ...)

    Domain assignment
    -----------------
    - If base token exists in the source domain registry -> chosen_domain = source_domain
    - Else if base token exists in any domain registry -> chosen_domain = first domain found in OSM_DOMAINS order
    - Else -> chosen_domain = "UNCLASSIFIED"

    Macro-group assignment
    ----------------------
    - If base token is a registered value within chosen_domain:
          macro_group = OSM_DOMAIN_VALUE_TO_GROUP[chosen_domain][base]
    - Else if base token is itself a macro-group label within chosen_domain:
          macro_group = base
      (e.g., "accommodation:accommodation:building")
    - Else:
          macro_group = "UNCLASSIFIED"

    Post-processing (per row)
    -------------------------
    - Exact triplet de-duplication (order-preserving)
    - Drop group-label triplets "g:g:D" if any member "x:g:D" exists
    - Apply RESOLUTION_RULES (canonicalization + containers + place_of_worship special case)
    - If still empty, emit default as:
          "<default_token>:<macro_group>:building" (domain selection uses same logic)

    Notes
    -----
    - normalize_token() strips ':' at ingestion; qualifiers are added only after normalization.
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
            suffix = _normalize_token(col[len(BUILDING_USE_PREFIX):])
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
        if token in OSM_DOMAIN_VALUE_TO_GROUP[source_domain] or token in OSM_DOMAIN_GROUPS.get(source_domain, {}):
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
    """
    Resolver for "token:macro_group:domain" triplets using RESOLUTION_RULES.

    Policy
    ------
    - If ANY triplet in the row has macro_group == "UNCLASSIFIED" OR domain == "UNCLASSIFIED",
      this function SKIPS all automatic resolution logic for the entire row and only performs:
        * exact triplet de-duplication (order-preserving)
      Rationale: UNCLASSIFIED rows are intended to be fixed manually first (via overrides),
      then re-run through this resolver once clean.

    Automatic resolution is applied ONLY when the row contains NO UNCLASSIFIED entries.

    Automatic resolution semantics (applied only on fully-classified rows)
    ---------------------------------------------------------------------
      1) when (PASS 1):
         - supports only condition: {"domain_equals": "<domain>"}
         - supports only action: "reclassify"
         - first matching clause wins
         - evaluated per-triplet before anything else

      2) canonical:
         - token-level canonicalization to (macro_group, domain)
         - token emitted at most once in canonical form (first occurrence wins)

      3) POW global drop:
         - if ANY religious building token exists in the row (domain="building" and token in POW_RELIGIOUS_BUILDINGS),
           drop ALL triplets with domain == "place_of_worship"

      4) containers:
         - rule-driven "drop parent if child present" evaluation
         - presence is computed AFTER (when + canonical)

      5) final output:
         - preserves input order
         - exact triplet dedup (first occurrence wins)

    Notes
    -----
    - Assumes no malformed triplets (always "token:macro_group:domain").
    - With the row-skip policy, UNCLASSIFIED triplets never participate in when/canonical/presence/containers/POW.
    """
    if not triplets:
        return triplets

    # ----------------------------
    # Parse / format helpers
    # ----------------------------
    def parse(t: str) -> tuple[str, str, str]:
        token, group, domain = t.split(":", 2)
        return token, group, domain

    def make(token: str, group: str, domain: str) -> str:
        return f"{token}:{group}:{domain}"

    # Parse once, preserve original order via index
    parsed = [(i, *parse(t)) for i, t in enumerate(triplets)]

    # ============================================================================
    # ROW-GUARD — if ANY UNCLASSIFIED exists, skip all automatic logic
    # ============================================================================
    if any(group == "UNCLASSIFIED" or domain == "UNCLASSIFIED" for _, _, group, domain in parsed):
        # exact-dedup, preserve order
        return list(dict.fromkeys(triplets))

    # ----------------------------
    # Canonical targets:
    # token -> (canonical_macro_group, canonical_domain)
    # ----------------------------
    canonical_target: dict[str, tuple[str, str]] = {
        token: (rule["canonical"]["macro_group"], rule["canonical"]["domain"])
        for token, rule in RESOLUTION_RULES.items()
        if "canonical" in rule
    }

    # ============================================================================
    # PASS 1 — "when" per-triplet reclassify
    #
    # Goal:
    #   override a specific triplet's (token, group, domain) before any presence/canonical.
    #
    # Notes:
    #   - only supports domain_equals
    #   - only supports action=reclassify
    #   - first matching clause wins
    # ============================================================================
    reclassify_override: dict[int, tuple[str, str, str]] = {}

    for i, token, group, domain in parsed:
        for clause in (RESOLUTION_RULES.get(token, {}).get("when") or []):
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
        """Triplet after PASS 1 (or unchanged if no override)."""
        return reclassify_override.get(i, (token, group, domain))

    # ============================================================================
    # PASS 2 — Presence AFTER (when + canonical)
    #
    # This drives:
    #   - POW global drop (religious building present)
    #   - container drop rules
    # ============================================================================
    domain_to_tokens: dict[str, set[str]] = {}

    for i, token, group, domain in parsed:
        token, group, domain = after_when(i, token, group, domain)

        # Presence reflects canonical endpoints
        if token in canonical_target:
            group, domain = canonical_target[token]

        domain_to_tokens.setdefault(domain, set()).add(token)

    # POW global drop condition:
    religious_building_present = bool(domain_to_tokens.get("building", set()) & POW_RELIGIOUS_BUILDINGS)

    # ============================================================================
    # Containers — rule-driven
    #
    # Output:
    #   container_drops contains (parent_token, parent_domain) pairs to remove.
    # ============================================================================
    container_drops: set[tuple[str, str]] = set()

    for token, rule in RESOLUTION_RULES.items():
        container = rule.get("container")
        if not container:
            continue

        parent_domain = container["parent_domain"]

        # Parent must be present (post-when + post-canonical) to be droppable
        if token not in domain_to_tokens.get(parent_domain, set()):
            continue

        for cond in (container.get("drop_parent_if_child") or []):
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

    # ============================================================================
    # FINAL EMISSION — preserve order and apply filters
    #
    # Order:
    #   - when reclassify
    #   - canonical uniqueness
    #   - POW global drop
    #   - container drops
    #   - exact triplet dedup
    # ============================================================================
    out: list[str] = []
    seen: set[str] = set()          # exact triplet dedup
    emitted_canon: set[str] = set() # token-level canonical dedup

    for i, tok0, grp0, dom0 in parsed:
        tok, grp, dom = after_when(i, tok0, grp0, dom0)

        # Canonical: force canonical endpoint and only emit first occurrence of that token
        if tok in canonical_target:
            if tok in emitted_canon:
                continue
            grp, dom = canonical_target[tok]
            emitted_canon.add(tok)

        # POW global drop
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
   