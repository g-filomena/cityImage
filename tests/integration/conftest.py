"""Live-OSM test settings: a reachable Overpass endpoint, a shorter timeout, and an unreachable
service reported as a skip rather than a failure.

The public Overpass endpoint (overpass-api.de) often refuses connections from CI runners: each
attempt then waits out the whole connect timeout and the test fails although cityImage never ran.
A test that cannot reach any endpoint is skipped with the reason; any other error still fails.
"""

from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")
ox = pytest.importorskip("osmnx")

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.kumi.systems/api",
)
PROBE_TIMEOUT = 10  # seconds to wait for an endpoint to answer the probe
REQUESTS_TIMEOUT = 60  # seconds for each OSM request (OSMnx's default is 180)
UNREACHABLE = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def _reachable(endpoint: str) -> bool:
    """Whether an Overpass endpoint answers a tiny query in time."""
    try:
        response = requests.get(
            f"{endpoint}/interpreter",
            params={"data": "[out:json][timeout:5];node(1);out;"},
            timeout=PROBE_TIMEOUT,
        )
    except UNREACHABLE:
        return False
    return response.ok


@pytest.fixture(scope="session", autouse=True)
def overpass_endpoint():
    """Point OSMnx at the first endpoint that answers, with a shorter request timeout."""
    ox.settings.requests_timeout = REQUESTS_TIMEOUT
    for endpoint in OVERPASS_ENDPOINTS:
        if _reachable(endpoint):
            ox.settings.overpass_url = endpoint
            # A mirror answers OSMnx's /status slot check with an error page; the check then
            # sleeps before every request, so it is only kept for the main endpoint.
            ox.settings.overpass_rate_limit = endpoint == OVERPASS_ENDPOINTS[0]
            return endpoint
    return None  # every test that needs OSM will be skipped by pytest_runtest_call below


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Report a test that could not reach OSM as skipped, with the reason."""
    try:
        return (yield)
    except UNREACHABLE as error:
        pytest.skip(f"OSM (Overpass) unreachable: {type(error).__name__}")
