"""Live-OSM test settings: a reachable Overpass endpoint, a shorter timeout, and an unreachable
service reported as a skip rather than a failure.

The public Overpass endpoint (overpass-api.de) often refuses connections from CI runners, or
answers that it is at capacity (502, 429...): the test would fail although cityImage never ran.
A test that cannot reach OSM runs again on another endpoint, while one answers, then is skipped with
the reason; any other error still fails.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pytest

requests = pytest.importorskip("requests")
ox = pytest.importorskip("osmnx")

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.kumi.systems/api",
)
PROBE_TIMEOUT = 20  # seconds to wait for an endpoint to answer the probe
REQUESTS_TIMEOUT = 60  # seconds for each OSM request (OSMnx's default is 180)
UNREACHABLE = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
# An endpoint that answers with one of these statuses is busy or throttling, not broken.
BUSY_STATUSES = (429, 502, 503, 504)
ResponseStatusCodeError = getattr(getattr(ox, "_errors", None), "ResponseStatusCodeError", ())

# Why each endpoint was found unusable, in the order tried; reported when a test is skipped.
_failures: dict[str, str] = {}


def _host(endpoint: str) -> str:
    return urlsplit(endpoint).hostname or endpoint


def _probe(endpoint: str) -> str | None:
    """Why an Overpass endpoint does not answer a tiny query in time, or None if it does.

    The probe sends OSMnx's own headers: endpoints refuse a request without a meaningful
    User-Agent (overpass-api.de with 406, overpass.kumi.systems with 429), so a probe sent with
    the ``requests`` default would reject an endpoint OSMnx can use.
    """
    headers = {
        "User-Agent": ox.settings.http_user_agent,
        "referer": ox.settings.http_referer,
        "Accept-Language": ox.settings.http_accept_language,
    }
    try:
        response = requests.get(
            f"{endpoint}/interpreter",
            params={"data": "[out:json][timeout:5];node(1);out;"},
            headers=headers,
            timeout=PROBE_TIMEOUT,
        )
    except UNREACHABLE as error:
        return type(error).__name__
    return None if response.ok else f"HTTP {response.status_code}"


def _first_reachable(skip=()) -> str | None:
    """The first endpoint, in order and not in ``skip``, that answers."""
    for endpoint in OVERPASS_ENDPOINTS:
        if endpoint in skip:
            continue
        failure = _probe(endpoint)
        if failure is None:
            return endpoint
        _failures[endpoint] = f"{failure} on the probe"
    return None


def _use(endpoint: str) -> None:
    ox.settings.overpass_url = endpoint
    # A mirror answers OSMnx's /status slot check with an error page; the check then sleeps
    # before every request, so it is only kept for the main endpoint.
    ox.settings.overpass_rate_limit = endpoint == OVERPASS_ENDPOINTS[0]


@pytest.fixture(scope="session", autouse=True)
def overpass_endpoint():
    """Point OSMnx at the first endpoint that answers, with a shorter request timeout."""
    ox.settings.requests_timeout = REQUESTS_TIMEOUT
    endpoint = _first_reachable()
    if endpoint is not None:
        _use(endpoint)
    return endpoint  # None: every test that needs OSM is skipped by pytest_runtest_call below


def _busy_status(error: Exception) -> int | None:
    """The status of an OSMnx error for an endpoint that answered it is busy, else None."""
    if not isinstance(error, ResponseStatusCodeError):
        return None
    match = re.search(r"responded: (\d{3})", str(error))
    status = int(match.group(1)) if match else None
    return status if status in BUSY_STATUSES else None


def _is_unreachable(error: Exception) -> bool:
    """Whether ``error`` says the endpoint could not serve the request, not that cityImage failed."""
    return isinstance(error, UNREACHABLE) or _busy_status(error) is not None


def _describe(error: Exception) -> str:
    """The error's type and the host it could not reach."""
    status = _busy_status(error)
    if status is not None:
        return f"{type(error).__name__} ({status})"
    url = getattr(getattr(error, "request", None), "url", None)
    host = urlsplit(url).hostname if url else None
    return f"{type(error).__name__} from {host}" if host else type(error).__name__


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Run a test that could not reach OSM again on another endpoint, while one answers; then
    report it as skipped, with the reason.

    An endpoint that stops answering mid-run (overpass-api.de throttles a client that has sent many
    queries, by refusing its connections) rarely answers again within the run, so each new attempt
    goes to an endpoint not yet tried. Later tests stay on the endpoint that worked.
    """
    try:
        return (yield)
    except Exception as error:
        if not _is_unreachable(error):
            raise
        _failures[ox.settings.overpass_url] = f"{_describe(error)} in the test"
    tried = {ox.settings.overpass_url}
    while (endpoint := _first_reachable(skip=tried)) is not None:
        tried.add(endpoint)
        _use(endpoint)
        try:
            item.runtest()
            return None
        except Exception as error:
            if not _is_unreachable(error):
                raise
            _failures[endpoint] = f"{_describe(error)} in the test"
    reasons = "; ".join(
        f"{_host(endpoint)}: {_failures.get(endpoint, 'not tried')}"
        for endpoint in OVERPASS_ENDPOINTS
    )
    pytest.skip(f"OSM unreachable on every endpoint ({reasons})")
