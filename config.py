"""Shared configuration and HTTP client for the baseball bot."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

import httpx

logger = logging.getLogger("baseball_bot.config")

# ── Season ──────────────────────────────────────────────────────────

CURRENT_SEASON = datetime.now().year
PREVIOUS_SEASON = CURRENT_SEASON - 1

# ── API endpoints ───────────────────────────────────────────────────

MLB_API = "https://statsapi.mlb.com/api/v1"
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

# ── Model version ───────────────────────────────────────────────────
# Bump this whenever weights, thresholds, or calibration constants change.
MODEL_VERSION = "v3.1"  # v3: bullpen split (0.95), v3.1: all 30 parks + blended platoon + current statcast

# ── League baselines ────────────────────────────────────────────────

LEAGUE_AVG = 0.248
LEAGUE_BABIP = 0.300

# ── Tier thresholds (single source of truth) ───────────────────────

TIER_STRONG = 0.70   # STRONG HIT: 70%+
TIER_LEAN = 0.62     # LEAN HIT: 62-70%
TIER_TOSSUP = 0.55   # TOSS-UP: 55-62%
                      # FADE: <55%

# ── Shared async HTTP client ───────────────────────────────────────

_client: httpx.AsyncClient | None = None
_api_semaphore = asyncio.Semaphore(10)  # limit concurrent API calls


def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client with retry support."""
    global _client
    if _client is None:
        transport = httpx.AsyncHTTPTransport(retries=2)
        _client = httpx.AsyncClient(
            transport=transport,
            timeout=20,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"User-Agent": "Mozilla/5.0 (compatible; BaseballResearch/1.0)"},
            follow_redirects=True,
        )
    return _client


# ── Circuit breaker state ──────────────────────────────────────────
# Tracks failures per host. After 5 consecutive failures, stop calling
# that host for 60 seconds to avoid hammering a downed service.

_circuit_breaker: dict[str, dict] = {}
_CB_THRESHOLD = 5       # failures before opening circuit
_CB_COOLDOWN = 60       # seconds to wait before retrying


def _get_host(url: str) -> str:
    """Extract host from URL for circuit breaker tracking."""
    from urllib.parse import urlparse
    return urlparse(url).netloc


def _check_circuit(host: str) -> bool:
    """Returns True if the circuit is closed (OK to call). False if open (skip)."""
    state = _circuit_breaker.get(host)
    if not state:
        return True
    if state["failures"] >= _CB_THRESHOLD:
        elapsed = time.time() - state["last_failure"]
        if elapsed < _CB_COOLDOWN:
            return False  # circuit open, skip
        # Cooldown expired, allow one attempt (half-open)
        state["failures"] = _CB_THRESHOLD - 1
    return True


def _record_success(host: str) -> None:
    """Reset circuit breaker on successful call."""
    if host in _circuit_breaker:
        del _circuit_breaker[host]


def _record_failure(host: str) -> None:
    """Record a failure for circuit breaker."""
    if host not in _circuit_breaker:
        _circuit_breaker[host] = {"failures": 0, "last_failure": 0}
    _circuit_breaker[host]["failures"] += 1
    _circuit_breaker[host]["last_failure"] = time.time()


async def fetch_json(url: str, **kwargs) -> dict:
    """Fetch JSON with rate limiting, circuit breaker, and exponential backoff."""
    host = _get_host(url)
    if not _check_circuit(host):
        raise ConnectionError(f"Circuit breaker open for {host} (too many failures)")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with _api_semaphore:
                r = await get_client().get(url, **kwargs)
                r.raise_for_status()
                _record_success(host)
                return r.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == max_retries - 1:
                _record_failure(host)
                logger.warning(f"fetch_json failed after {max_retries} attempts: {url} -> {e}")
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.debug(f"fetch_json retry {attempt + 1}/{max_retries} for {url} (waiting {wait}s)")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def fetch_text(url: str, **kwargs) -> str:
    """Fetch raw text with rate limiting, circuit breaker, and exponential backoff."""
    host = _get_host(url)
    if not _check_circuit(host):
        raise ConnectionError(f"Circuit breaker open for {host} (too many failures)")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with _api_semaphore:
                r = await get_client().get(url, **kwargs)
                r.raise_for_status()
                _record_success(host)
                return r.text
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == max_retries - 1:
                _record_failure(host)
                logger.warning(f"fetch_text failed after {max_retries} attempts: {url} -> {e}")
                raise
            wait = 2 ** attempt
            logger.debug(f"fetch_text retry {attempt + 1}/{max_retries} for {url} (waiting {wait}s)")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


# ── Odds conversion (single source of truth) ───────────────────────


def prob_to_american(prob: float) -> str:
    """Convert probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "---"
    if prob >= 0.5:
        odds = -round(prob / (1 - prob) * 100)
        return str(odds)
    else:
        odds = round((1 - prob) / prob * 100)
        return f"+{odds}"


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5  # odds == 0 is undefined, return even
