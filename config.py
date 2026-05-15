"""Shared configuration and HTTP client for the baseball bot."""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

# ── Season ──────────────────────────────────────────────────────────

CURRENT_SEASON = datetime.now().year
PREVIOUS_SEASON = CURRENT_SEASON - 1

# ── API endpoints ───────────────────────────────────────────────────

MLB_API = "https://statsapi.mlb.com/api/v1"
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

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
            headers={"User-Agent": "baseball-bot/1.0"},
            follow_redirects=True,
        )
    return _client


async def fetch_json(url: str, **kwargs) -> dict:
    """Fetch JSON from a URL using the shared client with rate limiting."""
    async with _api_semaphore:
        r = await get_client().get(url, **kwargs)
        r.raise_for_status()
        return r.json()


async def fetch_text(url: str, **kwargs) -> str:
    """Fetch raw text (for CSV endpoints) using the shared client with rate limiting."""
    async with _api_semaphore:
        r = await get_client().get(url, **kwargs)
        r.raise_for_status()
        return r.text


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
