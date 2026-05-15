"""Unit tests for odds_api pure logic functions."""

import pytest

from odds_api import _parse_hit_props, find_best_odds, match_event_to_game, _team_keywords


# ── Fixture data ───────────────────────────────────────────────────────

def _sample_odds_response():
    """Simulated Odds API response with two bookmakers and multiple players."""
    return {
        "id": "event123",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_hits",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": -180, "point": 0.5},
                            {"name": "Under", "description": "Aaron Judge", "price": 140, "point": 0.5},
                            {"name": "Over", "description": "Juan Soto", "price": -160, "point": 0.5},
                            {"name": "Under", "description": "Juan Soto", "price": 125, "point": 0.5},
                            {"name": "Over", "description": "Aaron Judge", "price": 200, "point": 1.5},
                            {"name": "Under", "description": "Aaron Judge", "price": -260, "point": 1.5},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "batter_hits",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": -170, "point": 0.5},
                            {"name": "Under", "description": "Aaron Judge", "price": 135, "point": 0.5},
                            {"name": "Over", "description": "Juan Soto", "price": -175, "point": 0.5},
                            {"name": "Under", "description": "Juan Soto", "price": 140, "point": 0.5},
                        ],
                    }
                ],
            },
        ],
    }


def _sample_events():
    """Simulated events list from the Odds API."""
    return [
        {
            "id": "abc123",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
        },
        {
            "id": "def456",
            "home_team": "Los Angeles Dodgers",
            "away_team": "San Francisco Giants",
        },
        {
            "id": "ghi789",
            "home_team": "St. Louis Cardinals",
            "away_team": "Chicago Cubs",
        },
        {
            "id": "jkl012",
            "home_team": "Tampa Bay Rays",
            "away_team": "Kansas City Royals",
        },
    ]


# ── 1. _parse_hit_props ───────────────────────────────────────────────

def test_parse_hit_props_extracts_players():
    """Should extract all players from the response."""
    result = _parse_hit_props(_sample_odds_response())
    assert "Aaron Judge" in result
    assert "Juan Soto" in result


def test_parse_hit_props_multiple_books():
    """Each player should have entries from multiple bookmakers."""
    result = _parse_hit_props(_sample_odds_response())
    judge_05 = [e for e in result["Aaron Judge"] if e["line"] == 0.5]
    books = {e["book"] for e in judge_05}
    assert "FanDuel" in books
    assert "DraftKings" in books


def test_parse_hit_props_over_under_paired():
    """Over and Under for same book+line should be in the same entry."""
    result = _parse_hit_props(_sample_odds_response())
    judge_05 = [e for e in result["Aaron Judge"] if e["line"] == 0.5]
    for entry in judge_05:
        assert "over" in entry
        assert "under" in entry


def test_parse_hit_props_multiple_lines():
    """FanDuel has 0.5 and 1.5 lines for Judge; both should appear."""
    result = _parse_hit_props(_sample_odds_response())
    judge_lines = {e["line"] for e in result["Aaron Judge"]}
    assert 0.5 in judge_lines
    assert 1.5 in judge_lines


def test_parse_hit_props_empty_response():
    """Empty response should return empty dict."""
    result = _parse_hit_props({})
    assert result == {}


def test_parse_hit_props_no_batter_hits_market():
    """Bookmaker with non-batter_hits market should be ignored."""
    data = {
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {"name": "Over", "description": "Judge", "price": 300, "point": 0.5},
                        ],
                    }
                ],
            }
        ]
    }
    result = _parse_hit_props(data)
    assert result == {}


def test_parse_hit_props_skips_invalid_side():
    """Outcomes with name not 'Over' or 'Under' should be skipped."""
    data = {
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_hits",
                        "outcomes": [
                            {"name": "Push", "description": "Judge", "price": 100, "point": 0.5},
                        ],
                    }
                ],
            }
        ]
    }
    result = _parse_hit_props(data)
    assert result == {}


# ── 2. find_best_odds ─────────────────────────────────────────────────

def test_find_best_odds_picks_best_over():
    """Should pick the least negative over price (best for bettor)."""
    parsed = _parse_hit_props(_sample_odds_response())
    judge_odds = parsed["Aaron Judge"]

    best = find_best_odds(judge_odds)
    assert best is not None
    # DraftKings has -170 (better than FanDuel's -180) for 0.5 line
    assert best["best_over"] == -170
    assert best["best_book"] == "DraftKings"


def test_find_best_odds_implied_prob_negative():
    """Implied probability for negative odds should be calculated correctly."""
    odds = [{"book": "TestBook", "line": 0.5, "over": -200, "under": 150}]
    best = find_best_odds(odds)
    assert best is not None
    # -200 implied = 200 / (200 + 100) = 0.667
    assert abs(best["implied_prob"] - 0.667) < 0.001


def test_find_best_odds_implied_prob_positive():
    """Implied probability for positive odds should be calculated correctly."""
    odds = [{"book": "TestBook", "line": 0.5, "over": 150, "under": -200}]
    best = find_best_odds(odds)
    assert best is not None
    # +150 implied = 100 / (150 + 100) = 0.400
    assert abs(best["implied_prob"] - 0.400) < 0.001


def test_find_best_odds_empty_list():
    """Empty list should return None."""
    assert find_best_odds([]) is None


def test_find_best_odds_no_over_prices():
    """Entries with only under prices should return None."""
    odds = [{"book": "TestBook", "line": 0.5, "under": -150}]
    assert find_best_odds(odds) is None


def test_find_best_odds_all_books_list():
    """The all_books list should contain all books with over prices for the 0.5 line."""
    parsed = _parse_hit_props(_sample_odds_response())
    judge_odds = parsed["Aaron Judge"]
    best = find_best_odds(judge_odds)
    assert best is not None
    all_books = best["all_books"]
    # Should have 2 entries for 0.5 line (FanDuel + DraftKings)
    books_05 = [b for b in all_books if b["line"] == 0.5]
    assert len(books_05) == 2


def test_find_best_odds_fallback_to_non_half_line():
    """When no 0.5 line exists, should fall back to any available line."""
    odds = [{"book": "FanDuel", "line": 1.5, "over": 200, "under": -260}]
    best = find_best_odds(odds)
    assert best is not None
    assert best["best_over"] == 200


# ── 3. match_event_to_game ────────────────────────────────────────────

def test_match_event_yankees_red_sox():
    """Should match Yankees vs Red Sox by keyword."""
    events = _sample_events()
    event_id = match_event_to_game(events, "New York Yankees", "Boston Red Sox")
    assert event_id == "abc123"


def test_match_event_dodgers_giants():
    """Should match Dodgers vs Giants despite city name filtering."""
    events = _sample_events()
    event_id = match_event_to_game(events, "Los Angeles Dodgers", "San Francisco Giants")
    assert event_id == "def456"


def test_match_event_cardinals_cubs():
    """Should match Cardinals vs Cubs."""
    events = _sample_events()
    event_id = match_event_to_game(events, "St. Louis Cardinals", "Chicago Cubs")
    assert event_id == "ghi789"


def test_match_event_rays_royals():
    """Should match Rays vs Royals despite multi-word city filtering."""
    events = _sample_events()
    event_id = match_event_to_game(events, "Tampa Bay Rays", "Kansas City Royals")
    assert event_id == "jkl012"


def test_match_event_partial_names():
    """Should match using partial/nickname-style team names."""
    events = _sample_events()
    # Using just "Yankees" and "Red Sox" should still match
    event_id = match_event_to_game(events, "Yankees", "Red Sox")
    assert event_id == "abc123"


def test_match_event_no_match():
    """Non-existent team should return None."""
    events = _sample_events()
    event_id = match_event_to_game(events, "Miami Marlins", "Atlanta Braves")
    assert event_id is None


def test_match_event_empty_events():
    """Empty events list should return None."""
    assert match_event_to_game([], "Yankees", "Red Sox") is None


# ── _team_keywords helper ─────────────────────────────────────────────

def test_team_keywords_filters_city():
    """City words like 'new', 'york' should be filtered out."""
    kw = _team_keywords("New York Yankees")
    assert "yankees" in kw
    assert "new" not in kw
    assert "york" not in kw


def test_team_keywords_dodgers():
    """'Los Angeles Dodgers' -> {'dodgers'}."""
    kw = _team_keywords("Los Angeles Dodgers")
    assert kw == {"dodgers"}


def test_team_keywords_cardinals():
    """'St. Louis Cardinals' -> {'cardinals'}."""
    kw = _team_keywords("St. Louis Cardinals")
    assert kw == {"cardinals"}


def test_team_keywords_red_sox():
    """'Boston Red Sox' -> {'boston', 'red', 'sox'}."""
    kw = _team_keywords("Boston Red Sox")
    assert "red" in kw
    assert "sox" in kw
    assert "boston" in kw
