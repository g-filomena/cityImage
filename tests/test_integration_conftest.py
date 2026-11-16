"""Offline tests for how the live-OSM tests tell an unreachable Overpass endpoint from a failure."""

from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")
ox = pytest.importorskip("osmnx")

from tests.integration import conftest  # noqa: E402

ResponseStatusCodeError = ox._errors.ResponseStatusCodeError


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_busy_endpoint_is_unreachable(status):
    error = ResponseStatusCodeError(f"'overpass-api.de' responded: {status} Bad Gateway # busy")

    assert conftest._is_unreachable(error)
    assert conftest._describe(error) == f"ResponseStatusCodeError ({status})"


def test_a_connection_failure_is_unreachable():
    assert conftest._is_unreachable(requests.exceptions.ConnectTimeout())


@pytest.mark.parametrize(
    "error",
    [
        ResponseStatusCodeError("'overpass-api.de' responded: 400 Bad Request # bad query"),
        ValueError("a cityImage bug"),
    ],
)
def test_any_other_error_still_fails(error):
    assert not conftest._is_unreachable(error)
