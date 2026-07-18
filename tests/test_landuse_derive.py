"""Offline tests for the pure triplet helpers in ``cityImage.landuse.derive``."""

from __future__ import annotations

import cityImage as ci

U = "UNCLASSIFIED"


def test_drop_redundant_group_label_removes_group_when_member_present():
    out = ci.drop_redundant_group_label(
        ["trullo:accommodation:building", "accommodation:accommodation:building"]
    )
    assert out == ["trullo:accommodation:building"]  # group-label triplet dropped


def test_drop_redundant_group_label_keeps_group_label_without_a_member():
    triplets = ["accommodation:accommodation:building"]
    assert ci.drop_redundant_group_label(triplets) == triplets


def test_drop_redundant_group_label_empty_is_passthrough():
    assert ci.drop_redundant_group_label([]) == []


def test_apply_resolution_rules_empty_is_passthrough():
    assert ci.apply_resolution_rules([]) == []


def test_apply_resolution_rules_unclassified_row_only_deduplicates():
    # Row-guard: any UNCLASSIFIED entry skips automatic resolution and only exact-dedups, in order.
    triplets = [f"foo:{U}:amenity", "bar:commercial:shop", f"foo:{U}:amenity"]
    assert ci.apply_resolution_rules(triplets) == [f"foo:{U}:amenity", "bar:commercial:shop"]
