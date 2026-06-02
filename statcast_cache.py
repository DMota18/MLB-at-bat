"""Daily Statcast cache builder.

Fetches contact quality metrics (exit velo, hard hit rate, barrel rate)
from Baseball Savant for active MLB batters and writes to a local JSON
cache file. The production bot reads this cache instead of calling
Savant directly (which rate-limits VPS IPs).

Usage:
    python statcast_cache.py              # fetch top 300 batters by PA
    python statcast_cache.py --batters 50 # fetch top 50 only (testing)
    python statcast_cache.py --file cache.json  # custom output path

The cache file is a JSON dict keyed by batter_id (string) with values:
    {
        "avg_exit_velo": 89.5,
        "hard_hit_pct": 0.385,
        "barrel_pct": 0.072,
        "batted_balls": 142,
        "has_data": true,
        "xba": 0.267,
        "updated": "2026-06-01"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import sys
from datetime import date
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("statcast_cache")

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_API = "https://statsapi.mlb.com/api/v1"
CACHE_PATH = Path(__file__).parent / "statcast_cache.json"


async def get_active_batter_ids(client: httpx.AsyncClient, limit: int = 300) -> list[dict]:
    """Get batter IDs for players with the most PA this season."""
    season = date.today().year
    r = await client.get(
        f"{MLB_API}/stats",
        params={
            "stats": "season",
            "season": str(season),
            "group": "hitting",
            "sortStat": "plateAppearances",
            "order": "desc",
            "limit": str(limit),
            "sportId": "1",
        },
    )
    r.raise_for_status()
    data = r.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    return [
        {"id": s["player"]["id"], "name": s["player"]["fullName"], "pa": s["stat"]["plateAppearances"]}
        for s in splits
        if s.get("player", {}).get("id")
    ]


async def fetch_batter_statcast(
    client: httpx.AsyncClient, batter_id: int, season: int,
) -> dict | None:
    """Fetch one batter's Statcast contact quality from Savant."""
    date_gt = f"{season}-03-01"
    date_lt = f"{season}-11-01"

    params = {
        "all": "true",
        "player_type": "batter",
        "batters_lookup[]": str(batter_id),
        "game_date_gt": date_gt,
        "game_date_lt": date_lt,
        "type": "details",
        "sort_col": "pitches",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
    }

    try:
        r = await client.get(SAVANT_CSV, params=params)
        text = r.text
        if len(text) < 200:
            return None

        text = text.lstrip("﻿")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            return None

        header = [h.strip().strip('"') for h in header]
        ev_idx = header.index("launch_speed") if "launch_speed" in header else None
        la_idx = header.index("launch_angle") if "launch_angle" in header else None
        if ev_idx is None:
            return None

        exit_velos: list[float] = []
        hard_hits = 0
        barrels = 0
        batted_balls = 0

        for cols in reader:
            if len(cols) <= ev_idx:
                continue
            ev_str = cols[ev_idx].strip().strip('"')
            if not ev_str or ev_str == "null":
                continue
            try:
                ev = float(ev_str)
            except ValueError:
                continue
            if ev <= 0:
                continue

            batted_balls += 1
            exit_velos.append(ev)
            if ev >= 95:
                hard_hits += 1
            if la_idx is not None and la_idx < len(cols):
                la_str = cols[la_idx].strip().strip('"')
                try:
                    la = float(la_str)
                    if ev >= 98 and 26 <= la <= 30 + (ev - 98) * 2:
                        barrels += 1
                except (ValueError, TypeError):
                    pass

        if batted_balls < 10:
            return None

        from matchup_data import _xba_from_ev
        xba_sum = sum(_xba_from_ev(ev) for ev in exit_velos)

        return {
            "avg_exit_velo": round(sum(exit_velos) / len(exit_velos), 1),
            "hard_hit_pct": round(hard_hits / batted_balls, 3),
            "barrel_pct": round(barrels / batted_balls, 3),
            "batted_balls": batted_balls,
            "has_data": True,
            "xba": round(xba_sum / len(exit_velos), 3),
            "updated": date.today().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Savant fetch failed for {batter_id}: {e}")
        return None


async def build_cache(limit: int = 300, output: Path = CACHE_PATH) -> dict:
    """Fetch Statcast data for top batters and write cache file."""
    season = date.today().year

    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(
        transport=transport,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; BaseballResearch/1.0)"},
        follow_redirects=True,
    ) as client:
        logger.info(f"Fetching top {limit} batters by PA...")
        batters = await get_active_batter_ids(client, limit)
        logger.info(f"Found {len(batters)} batters")

        cache: dict[str, dict] = {}
        succeeded = 0
        failed = 0

        for i, b in enumerate(batters):
            result = await fetch_batter_statcast(client, b["id"], season)
            if result:
                cache[str(b["id"])] = result
                succeeded += 1
            else:
                failed += 1

            if (i + 1) % 25 == 0:
                logger.info(f"Progress: {i+1}/{len(batters)} ({succeeded} ok, {failed} failed)")

            await asyncio.sleep(0.4)

        logger.info(f"Done: {succeeded} cached, {failed} failed out of {len(batters)}")

        output.write_text(json.dumps(cache, indent=2))
        logger.info(f"Cache written to {output} ({len(cache)} entries)")

        return cache


def main():
    parser = argparse.ArgumentParser(description="Build Statcast cache")
    parser.add_argument("--batters", type=int, default=300, help="Number of batters to fetch")
    parser.add_argument("--file", type=str, default=str(CACHE_PATH), help="Output file path")
    args = parser.parse_args()

    cache = asyncio.run(build_cache(limit=args.batters, output=Path(args.file)))
    print(f"\nCached {len(cache)} batters to {args.file}")


if __name__ == "__main__":
    main()
