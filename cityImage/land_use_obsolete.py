


















# # -----------------------------------------------------------------------------
# # Duplicate qualification control (2-arg signature, module-level duplicates set)
# # -----------------------------------------------------------------------------

# def _determine_if_duplicate(raw_token: str, domain_name: str) -> str:
    # """
    # If raw_token is ambiguous and domain_name is a real domain, qualify as "<raw_token>:<domain_name>".
    # Otherwise return raw_token unchanged.
    # """
    # if raw_token in DUPLICATE_TOKENS and domain_name in OSM_DOMAINS:
        # return f"{raw_token}:{domain_name}"
    # return raw_token




# def _base_token(tok: str) -> str:
    # """
    # Return the base token from a qualified triplet.

    # Input format
    # ------------
    # tok: "base:macro_group:domain"

    # Examples
    # --------
    # - "trullo:accommodation:building" -> "trullo"
    # - "accommodation:accommodation:building" -> "accommodation"
    # - "pharmacy:shop_health_beauty:shop" -> "pharmacy"
    # """
    # return tok.split(":", 1)[0]









# # -----------------------------------------------------------------------------
# # Redundancy removal A: within-domain (derived from tags registries, no rules dict)
# # -----------------------------------------------------------------------------

# def apply_within_OSM_domains_redundancy(
    # tokens_by_domains: dict[str, list[str]],
# ) -> dict[str, list[str]]:
    # """
    # Within-domain redundancy removal computed from OSM_DOMAIN_GROUPS + OSM_DOMAIN_VALUE_TO_GROUP.

    # Input
    # -----
    # tokens_by_domains:
        # dict[domain -> list[str]]
        # - domains are OSM keys (e.g., "building", "shop", "amenity", "industrial", ...)
        # - tokens are already normalized strings; tokens may be qualified ("bakery:shop")
        # - evaluation is on BASE tokens only via _base_token()

    # Rule (per domain bucket D)
    # --------------------------
    # Drop a domain sub-group label token `g` from bucket D if:
      # - `g` is present in bucket D, AND
      # - at least one value token `v` is present in bucket D such that:
            # OSM_DOMAIN_VALUE_TO_GROUP[D].get(v) == g

    # Examples
    # --------
    # 1) building group label + member value
       # in:  {"building": ["accommodation", "trullo"]}
       # out: {"building": ["trullo"]}

    # 2) industrial group label + member value (qualified ok)
       # in:  {"industrial": ["industrial:industrial", "warehouse"]}
       # out: {"industrial": ["warehouse"]}

    # 3) group label alone stays
       # in:  {"building": ["accommodation"]}
       # out: {"building": ["accommodation"]}

    # Notes
    # -----
    # - Only domains present in OSM_DOMAIN_GROUPS/OSM_DOMAIN_VALUE_TO_GROUP are processed.
    # - Buckets not in those registries (e.g., "building_use") are ignored.
    # """
    # if not tokens_by_domains:
        # return tokens_by_domains

    # # Process only registered OSM domains, in stable precedence order.
    # domains = [d for d in OSM_DOMAINS if d in tokens_by_domains]

    # # Cache base-token sets per bucket.
    # present_by_domain: dict[str, set[str]] = {}
    # for domain in domains:
        # tokens = tokens_by_domains.get(domain, [])
        # present_by_domain[domain] = {_base_token(t) for t in tokens} if tokens else set()

    # to_drop_by_domain: dict[str, set[str]] = {domain: set() for domain in domains}

    # for domain in domains:
        # present = present_by_domain.get(domain, set())
        # if not present:
            # continue

        # groups = OSM_DOMAIN_GROUPS.get(domain)                 # dict[group_label -> list(values)]
        # value_to_group = OSM_DOMAIN_VALUE_TO_GROUP.get(domain) # dict[value -> group_label]
        # if not groups or not value_to_group:
            # continue

        # group_labels = set(groups.keys())
        # present_group_labels = present & group_labels
        # if not present_group_labels:
            # continue

        # # If any present value maps to a present group label, drop that group label.
        # for value in present:
            # group = value_to_group.get(value)
            # if group and group in present_group_labels:
                # to_drop_by_domain[domain].add(group)

    # if not any(to_drop_by_domain.values()):
        # return tokens_by_domains

    # new_tokens_by_domains: dict[str, list[str]] = {}
    # for domain, tokens in tokens_by_domains.items():
        # drops = to_drop_by_domain.get(domain, set())
        # if not drops:
            # new_tokens_by_domains[domain] = tokens
        # else:
            # new_tokens_by_domains[domain] = [
                # token for token in tokens
                # if _base_token(token) not in drops
            # ]

    # return new_tokens_by_domains


# def apply_cross_OSM_domains_redundancy(
    # tokens_by_domains: dict[str, list[str]],
# ) -> dict[str, list[str]]:
    # """
    # Cross-domain container removal for a small explicit whitelist (CROSS_DOMAIN_CONTAINER_RULES).

    # Input
    # -----
    # tokens_by_domains:
        # dict[domain -> list[str]]
        # - tokens are already normalized strings; tokens may be qualified ("bakery:shop")
        # - evaluation is on BASE tokens only via _base_token()

    # CROSS_DOMAIN_CONTAINER_RULES (from land_use_tags)
    # ------------------------------------------------
    # {
      # "<parent_domain>": {
        # "<parent_base_token>": {
          # "<child_domain>": None | set[str]
        # }
      # }
    # }

    # Semantics
    # ---------
    # - child_domain: None => any token in that domain triggers the container drop
    # - child_domain: set  => intersection on BASE tokens triggers the drop
    # - If triggered, drop <parent_base_token> from <parent_domain>

    # Example
    # -------
    # in:
      # {
        # "shop": ["bakery"],
        # "building": ["commercial"],
      # }
    # rule:
      # building.commercial drops if any shop token exists
    # out:
      # {
        # "shop": ["bakery"],
        # "building": [],
      # }
    # """
    # if not tokens_by_domains:
        # return tokens_by_domains

    # # Stable domain iteration for presence cache.
    # domains = [d for d in OSM_DOMAINS if d in tokens_by_domains]

    # present_by_domain: dict[str, set[str]] = {}
    # for domain in domains:
        # tokens = tokens_by_domains.get(domain, [])
        # present_by_domain[domain] = {_base_token(t) for t in tokens} if tokens else set()

    # to_drop_by_domain: dict[str, set[str]] = {domain: set() for domain in domains}

    # for parent_domain, parent_rules in CROSS_DOMAIN_CONTAINER_RULES.items():
        # parent_present = present_by_domain.get(parent_domain, set())
        # if not parent_present:
            # continue

        # for parent_token, triggers in parent_rules.items():
            # if parent_token not in parent_present:
                # continue

            # for child_domain, child_set in triggers.items():
                # child_present = present_by_domain.get(child_domain, set())
                # if not child_present:
                    # continue

                # # Any token in child domain triggers.
                # if child_set is None:
                    # to_drop_by_domain.setdefault(parent_domain, set()).add(parent_token)
                    # break

                # # Intersection on BASE tokens triggers.
                # if child_present & set(child_set):
                    # to_drop_by_domain.setdefault(parent_domain, set()).add(parent_token)
                    # break

    # if not any(to_drop_by_domain.values()):
        # return tokens_by_domains

    # new_tokens_by_domains: dict[str, list[str]] = {}
    # for domain, tokens in tokens_by_domains.items():
        # drops = to_drop_by_domain.get(domain, set())
        # if not drops:
            # new_tokens_by_domains[domain] = tokens
        # else:
            # new_tokens_by_domains[domain] = [
                # token for token in tokens
                # if _base_token(token) not in drops
            # ]

    # return new_tokens_by_domains

# def apply_OSM_container_redundancy(
    # tokens_by_domains: dict[str, list[str]],
# ) -> tuple[dict[str, list[str]], list[str]]:
    # """
    # Full redundancy pipeline:
      # 1) within-domain (computed from tags registries)
      # 2) cross-domain whitelist (CROSS_DOMAIN_CONTAINER_RULES)
      # 3) merge in source order, de-duplicate preserving order
    # """
    # tokens_by_domains = apply_within_OSM_domains_redundancy(tokens_by_domains)
    # tokens_by_domains = apply_cross_OSM_domains_redundancy(tokens_by_domains)

    # merged_in_order = [token for domain in OSM_DOMAINS for token in tokens_by_domains.get(domain, [])]
    # return tokens_by_domains, _remove_duplicates(merged_in_order)


# # -----------------------------------------------------------------------------
# # Stage 1: derive raw tokens from OSM
# # -----------------------------------------------------------------------------

# def derive_land_uses_raw_fromOSM(buildings_gdf, default: str = "residential"):
    # """
    # Derive building-level land-use descriptor tokens from an OSM-derived GeoDataFrame.

    # Output columns per row
    # ----------------------
    # - tokens_<key> for every key in SEMANTIC_DOMAINS (e.g., tokens_amenity, tokens_shop, ...)
    # - tokens_building: list[str]
    # - land_uses_raw: merged list[str] (order-preserving, de-duplicated)

    # Extraction sources (merge priority)
    # -----------------------------------
    # 1) building:use:* columns
    # 2) Semantic tag keys (SEMANTIC_DOMAINS)
    # 3) building=* fallback (excluding "yes")

    # Duplicate qualification
    # -----------------------
    # After exclusions, if a BASE token is known duplicate (appears in multiple domains),
    # and the current source is a qualifiable domain, token becomes "<base>:<source>".

    # Redundancy removal
    # ------------------
    # - within-domain (computed from tags registries)
    # - cross-domain whitelist (CROSS_DOMAIN_CONTAINER_RULES)
    # """
    # buildings_gdf = buildings_gdf.copy()
    # default_token = normalize_token(default)

    # # Precompute building:use:* suffix tokens once per GDF
    # building_use_suffixes = []
    # for col in buildings_gdf.columns:
        # if isinstance(col, str) and col.startswith(BUILDING_USE_PREFIX):
            # suffix = normalize_token(col[len(BUILDING_USE_PREFIX):])
            # if suffix:
                # building_use_suffixes.append((col, suffix))



    # def _clean_domain_tokens(domain: str, raw_tokens: list[Any]) -> list[str]:
        
        # cleaned = _clean_tokens(raw_tokens)
        # cleaned = [token for token in cleaned if _base_token(token) not in EXCLUDE_AMENITIES]
        # # qualify duplicates (BASE-only)
        # return [_determine_if_duplicate(_base_token(token), domain) for token in cleaned]

    # def _extract_row(row: pd.Series):
        # raw_tokens_by_domain = {name: [] for name in OSM_DOMAINS}

        # # (a) building:use:* columns -> building_use bucket
        # for col_name, suffix_token in building_use_suffixes:
            # if _is_truthy_osm_tag_value(row.get(col_name)):
                # raw_tokens_by_domain["building"].append(suffix_token)

        # # (b) semantic OSM tag keys
        # for key in OSM_DOMAIN_GROUPS:
            # raw_tokens_by_domain[key].extend(_as_list(row.get(key)))

        # # clean per bucket
        # tokens_by_domains = {
            # domain: _clean_domain_tokens(domain, raw_tokens)
            # for domain, raw_tokens in raw_tokens_by_domain.items()
        # }


        # # redundancy removal + merge (apply_OSM_container_redundancy must derive order internally)
        # tokens_by_domains, merged = apply_OSM_container_redundancy(tokens_by_domains)

        # if not merged:
            # merged = [default_token] if default_token else []

        # return tokens_by_domains, merged

    # extracted = buildings_gdf.apply(_extract_row, axis=1)

    # for domain in OSM_DOMAINS:
        # buildings_gdf[f"tokens_{domain}"] = extracted.apply(lambda r: r[0][domain])

    # buildings_gdf["land_uses_raw"] = extracted.apply(lambda r: r[1])
    # return buildings_gdf
    
    # def _remove_duplicates(values: list[str]) -> list[str]:
    # """De-duplicate preserving first-seen order (assumes hashable tokens)."""
    # seen: set[str] = set()
    # out: list[str] = []
    # for v in values:
        # if v not in seen:
            # seen.add(v)
            # out.append(v)
    # return out




# # =============================================================================
# # Qualified-token support
# # =============================================================================
# # Stage 2
# # In the pipeline, some *ambiguous* raw values are "qualified" at extraction time
# # to preserve provenance (the OSM key they came from):
# #
# #   "bakery:industrial"  -> raw value "bakery" that came from industrial=*
# #   "bakery:craft"       -> raw value "bakery" that came from craft=*
# #
# # Classification must therefore:
# #   1) Recognize "<value>:<domain>" when <domain> is one of the known OSM tag keys
# #   2) If qualified, force mapping using ONLY that domain's lookup
# #   3) If unqualified, apply a precedence order across domains
# #   4) If no mapping exists (qualified or unqualified), return UNCLASSIFIED (not the raw token).
# #
# # Important:
# # - We treat only the *last* ":" as a possible qualifier delimiter.
# # - If the suffix is not a known domain, we keep the token as-is (after normalize).
# #
# # =============================================================================

# def _split_qualified_token(raw_token: object) -> tuple[str | None, str | None]:
    # """
    # Split a possibly-qualified token into (base_value, domain_or_none).

    # - Derived tokens should contain ':' ONLY when *we* added a qualifier.
    # - If suffix after the last ':' is NOT a known domain, treat ':' as noise and remove it.
      # (aligns with the policy that ':' should have been removed at ingestion time)
    
    # "f:ire_station" should be already normalised and not get here.
    # Examples:
      # "bakery:industrial" -> ("bakery", "industrial")
      # "industrial"        -> ("industrial", None)
      # "bakery:building_use" -> ("bakery", None)  # robust fallback; but this should not be produced now
    # """

    # if raw_token is None:
        # return None, None

    # if ":" in raw_token:
        # base, suffix = raw_token.rsplit(":", 1)
        # if suffix in OSM_DOMAIN_VALUE_TO_GROUP:
            # return base, suffix
        # # unknown suffix => ignore it, keep base unqualified
        # return base, None

    # return raw_token, None


# # =============================================================================
# # OSM macro-group classifiers (simple wrappers around reverse maps)
# # =============================================================================
# #
# # Each lookup maps: value_token -> macro_group_label
# # e.g. "church" -> "religious", "pub" -> "sustenance", ...
# #
# # =============================================================================

# def _first_mapped_group(base: str) -> str | None:
    # """
    # Unqualified duplicates resolve by domain precedence (first match wins),
    # using OSM_DOMAIN_PRECEDENCE (dict insertion order).
    # """
    # for domain in OSM_DOMAIN_PRECEDENCE:
        # mapped = OSM_DOMAIN_VALUE_TO_GROUP[domain].get(base)
        # if mapped is not None:
            # return mapped
    # return None

# def _classify_OSM_macro_group(raw_token: str) -> str | None:
    # """
    # Map a raw token (qualified or not) to an OSM macro-group label.

    # Behavior
    # --------
    # 1) detect qualification:
         # base, domain = _split_qualified_token(raw_token)
    # 2) If base is already a macro label (e.g., "industrial", "education"), keep it.
    # 3) If qualified (domain is not None):
         # use ONLY that domain's lookup:
           # - "bakery:industrial" -> "industrial"
           # - "bakery:craft"      -> "craft"
         # If not found in that domain's lookup: return UNCLASSIFIED.
    # 4) If unqualified:
         # scan domains in OSM_DOMAIN_PRECEDENCE (first match wins).
    # 5) If still unmapped: return UNCLASSIFIED

    # Returns
    # -------
    # str | None
        # - macro-group label when a mapping exists, or
        # - UNCLASSIFIED when no mapping exists

    # Examples
    # --------
    # - "bakery:industrial" -> "industrial"    (forced domain lookup)
    # - "bakery"            -> depends on precedence (first match wins)
    # - "baretto"           -> UNCLASSIFIED
    # """
    # base, domain = _split_qualified_token(raw_token)
    # if base is None:
        # return None

    # if base in OSM_MACRO_GROUP_LABELS:
        # return base

    # if domain is not None:
        # return OSM_DOMAIN_VALUE_TO_GROUP[domain].get(base) or UNCLASSIFIED

    # mapped = _first_mapped_group(base)
    # return mapped if mapped is not None else UNCLASSIFIED

# # =============================================================================
# # Public APIs
# # =============================================================================

# def classify_land_use_into_OSM_groups(
    # buildings_gdf,
    # raw_land_use_column,
    # new_group_column,
    # overlap_column=None,
# ):
    # """
    # Map raw tokens to OSM macro-groups, supporting qualified tokens.

    # Examples
    # --------
    # - "bakery:industrial" -> "industrial"
    # - "bakery:craft"      -> "craft"
    # - "industrial"        -> "industrial"      (already a macro label)
    # - "church"            -> "religious"
    # - "unknown_token"     -> UNCLASSIFIED

    # Overlap handling
    # ---------------
    # If overlap_column is provided, overlap weights are aggregated by mapped macro-group
    # and stored in "new_<overlap_column>", aligned 1:1 with the mapped group list.
    # """
    # return _map_tokens_and_aggregate_overlaps(
        # buildings_gdf=buildings_gdf,
        # source_tokens_column=raw_land_use_column,
        # mapped_tokens_column=new_group_column,
        # token_mapper=_classify_OSM_macro_group, 
        # source_overlap_column=overlap_column,
    # )

# # =============================================================================
# # Mapping + (optional) overlap aggregation
# # =============================================================================

# def _map_tokens_and_aggregate_overlaps(
    # buildings_gdf,
    # source_tokens_column,
    # mapped_tokens_column,
    # token_mapper,
    # source_overlap_column=None,
# ):
    # """
    # Apply `token_mapper` to each token in `source_tokens_column` (row-wise),
    # de-duplicate mapped tokens preserving first-seen order, and optionally
    # aggregate overlap weights.

    # Mode A: mapping only
    # --------------------
    # - Input:  tokens list per row
    # - Output: mapped tokens list per row

    # Mode B: mapping + overlap aggregation
    # ------------------------------------
    # Data contract:
      # - `source_overlap_column` must contain list[float] aligned 1:1 with tokens.
      # - If lengths mismatch for a row, ValueError is raised (fail-fast).

    # Aggregation:
      # - If multiple source tokens map to the same mapped token, overlaps are summed.
      # - The aggregated overlaps are rounded to 3 decimals.
      # - Output overlaps are stored in "new_<source_overlap_column>" aligned with output tokens.
    # """
    # buildings_gdf = buildings_gdf.copy()

    # # -------------------------
    # # Mode A: mapping only
    # # -------------------------
    # if source_overlap_column is None:
        # def map_and_deduplicate_row(raw_tokens_value):
            # source_tokens = _to_list(raw_tokens_value)

            # mapped_tokens_in_order = []
            # seen_mapped_tokens = set()

            # for source_token in source_tokens:
                # mapped_token = token_mapper(source_token)
                # if mapped_token is None or mapped_token in seen_mapped_tokens:
                    # continue
                # seen_mapped_tokens.add(mapped_token)
                # mapped_tokens_in_order.append(mapped_token)

            # return mapped_tokens_in_order

        # buildings_gdf[mapped_tokens_column] = buildings_gdf[source_tokens_column].apply(map_and_deduplicate_row)
        # return buildings_gdf

    # # -------------------------
    # # Mode B: mapping + overlap aggregation
    # # -------------------------
    # output_overlap_column = f"new_{source_overlap_column}"

    # def map_and_aggregate_row(row):
        # source_tokens = _to_list(row[source_tokens_column])
        # source_overlaps = _to_list(row[source_overlap_column])

        # if not source_tokens:
            # return [], []

        # if len(source_tokens) != len(source_overlaps):
            # raise ValueError(
                # f"[row={row.name}] Token/overlap length mismatch: "
                # f"{len(source_tokens)} tokens vs {len(source_overlaps)} overlaps"
            # )

        # mapped_tokens_in_order = []
        # overlap_sum_by_token = {}

        # for source_token, overlap_value in zip(source_tokens, source_overlaps):
            # mapped_token = token_mapper(source_token)
            # if mapped_token is None:
                # continue

            # if mapped_token not in overlap_sum_by_token:
                # overlap_sum_by_token[mapped_token] = 0.0
                # mapped_tokens_in_order.append(mapped_token)

            # overlap_sum_by_token[mapped_token] += float(overlap_value)

        # if not mapped_tokens_in_order:
            # return [], []

        # aggregated_overlaps = [
            # round(overlap_sum_by_token[mapped_token], 3)
            # for mapped_token in mapped_tokens_in_order
        # ]

        # return mapped_tokens_in_order, aggregated_overlaps

    # mapped_results = buildings_gdf.apply(map_and_aggregate_row, axis=1)
    # buildings_gdf[mapped_tokens_column] = mapped_results.apply(lambda pair: pair[0])
    # buildings_gdf[output_overlap_column] = mapped_results.apply(lambda pair: pair[1])

    # return buildings_gdf

