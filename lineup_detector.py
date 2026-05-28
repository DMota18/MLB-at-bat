"""
Lineup detector for the baseball bot.

Polls the MLB Stats API for today's games. When lineups are posted
(~2-3 hours before first pitch), triggers pregame analysis.

MLB API lineup endpoint:
  GET /api/v1/schedule?date=YYYY-MM-DD&sportId=1&hydrate=probablePitcher,lineups
"""

from __future__ import annotations

import logging
import time
from datetime import date
from dataclasses import dataclass, field

from config import MLB_API, CURRENT_SEASON, fetch_json

logger = logging.getLogger("baseball_bot.lineup")

# Module-level cache for player bio data (handedness, etc.)
_player_bios: dict[int, dict] = {}
_player_bios_loaded: bool = False
_player_bios_loaded_at: float = 0.0
_BIOS_TTL_SECONDS: int = 24 * 60 * 60  # refresh every 24 hours


@dataclass
class PlayerInfo:
    id: int
    name: str
    position: str
    batting_order: int
    bats: str  # L, R, S


@dataclass
class GameLineup:
    game_pk: int
    game_time: str  # ISO datetime
    status: str
    venue: str
    away_team: str
    home_team: str
    away_pitcher: str | None
    home_pitcher: str | None
    away_pitcher_throws: str | None  # L or R
    home_pitcher_throws: str | None
    away_lineup: list[PlayerInfo] = field(default_factory=list)
    home_lineup: list[PlayerInfo] = field(default_factory=list)
    officials: list[dict] = field(default_factory=list)  # raw officials data from API

    @property
    def lineups_posted(self) -> bool:
        """True if at least one team has a lineup posted."""
        return len(self.away_lineup) > 0 or len(self.home_lineup) > 0

    @property
    def both_lineups_posted(self) -> bool:
        return len(self.away_lineup) > 0 and len(self.home_lineup) > 0


async def _load_player_bios() -> None:
    """Load all MLB player bios in a single API call.

    Caches ID -> {bats, throws, fullName} for the session.
    Replaces ~270 individual API calls with 1.
    Refreshes automatically every 24 hours to pick up callups,
    trades, and roster moves.
    """
    global _player_bios_loaded, _player_bios_loaded_at
    if _player_bios_loaded and (time.time() - _player_bios_loaded_at) < _BIOS_TTL_SECONDS:
        return

    try:
        data = await fetch_json(
            f"{MLB_API}/sports/1/players?season={CURRENT_SEASON}&gameType=R"
        )
        for p in data.get("people", []):
            pid = p.get("id", 0)
            _player_bios[pid] = {
                "bats": p.get("batSide", {}).get("code", "R"),
                "throws": p.get("pitchHand", {}).get("code", "R"),
                "fullName": p.get("fullName", ""),
            }
        _player_bios_loaded = True
        _player_bios_loaded_at = time.time()
        logger.info(f"Cached {len(_player_bios)} player bios")
    except Exception as e:
        logger.error(f"Failed to load player bios: {e}")


def _get_bio(pid: int) -> dict:
    """Get cached player bio. Returns defaults if not found."""
    return _player_bios.get(pid, {"bats": "R", "throws": "R", "fullName": ""})


def _extract_pitcher(team_data: dict) -> tuple[str | None, str | None]:
    """Extract pitcher name and throws from team schedule data."""
    pitcher = team_data.get("probablePitcher", {})
    name = pitcher.get("fullName")
    pid = pitcher.get("id")
    throws = _get_bio(pid).get("throws") if pid else None
    return name, throws


def _extract_lineup(lineup_data: list) -> list[PlayerInfo]:
    """Extract lineup from the hydrated lineup data.

    Uses the pre-loaded player bio cache instead of per-player API calls.
    """
    players = []
    for idx, entry in enumerate(lineup_data):
        if "person" in entry:
            pid = entry["person"].get("id", 0)
            name = entry["person"].get("fullName", "Unknown")
        else:
            pid = entry.get("id", 0)
            name = entry.get("fullName", "Unknown")

        position = entry.get("primaryPosition", {}).get("abbreviation", "?")
        if "position" in entry and isinstance(entry["position"], dict):
            position = entry["position"].get("abbreviation", position)

        order_num = idx + 1
        bats = _get_bio(pid).get("bats", "R")

        players.append(PlayerInfo(
            id=pid, name=name, position=position,
            batting_order=order_num, bats=bats,
        ))

    return players


async def get_todays_games(game_date: str | None = None) -> list[GameLineup]:
    """Fetch today's games with lineup and pitcher data.

    Returns a list of GameLineup objects. Lineups may or may not be
    posted yet — check .lineups_posted on each game.
    """
    # Ensure player bios are loaded (single API call, cached for session)
    await _load_player_bios()

    target_date = game_date or date.today().isoformat()

    try:
        data = await fetch_json(
            f"{MLB_API}/schedule?date={target_date}&sportId=1"
            f"&hydrate=probablePitcher,lineups,venue,officials"
        )
    except Exception as e:
        logger.error(f"Failed to fetch schedule: {e}")
        return []

    dates = data.get("dates", [])
    if not dates:
        return []

    games = []
    for game in dates[0].get("games", []):
        game_pk = game.get("gamePk", 0)
        game_time = game.get("gameDate", "")
        status = game.get("status", {}).get("detailedState", "")
        venue = game.get("venue", {}).get("name", "Unknown")

        away = game.get("teams", {}).get("away", {})
        home = game.get("teams", {}).get("home", {})

        away_team = away.get("team", {}).get("name", "Unknown")
        home_team = home.get("team", {}).get("name", "Unknown")

        away_pitcher, away_throws = _extract_pitcher(away)
        home_pitcher, home_throws = _extract_pitcher(home)

        lineups = game.get("lineups", {})
        away_lineup_data = lineups.get("awayPlayers", [])
        home_lineup_data = lineups.get("homePlayers", [])

        away_lineup = _extract_lineup(away_lineup_data) if away_lineup_data else []
        home_lineup = _extract_lineup(home_lineup_data) if home_lineup_data else []

        games.append(GameLineup(
            game_pk=game_pk,
            game_time=game_time,
            status=status,
            venue=venue,
            away_team=away_team,
            home_team=home_team,
            away_pitcher=away_pitcher,
            home_pitcher=home_pitcher,
            away_pitcher_throws=away_throws,
            home_pitcher_throws=home_throws,
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            officials=game.get("officials", []),
        ))

    logger.info(f"Found {len(games)} games for {target_date}, "
                f"{sum(1 for g in games if g.lineups_posted)} with lineups")
    return games
