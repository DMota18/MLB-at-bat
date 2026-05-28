"""Tests for tracker.py — prediction storage, result checking, paper betting."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from predictor import HitPrediction, PredictionFactors


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path):
    """Redirect tracker to a temporary database for every test."""
    import tracker
    db_path = tmp_path / "test.db"
    with patch("tracker.DB_PATH", db_path):
        tracker._tables_initialized = False
        yield db_path
        tracker._tables_initialized = False


def _make_prediction(**overrides) -> HitPrediction:
    defaults = dict(
        batter_name="Aaron Judge",
        batter_id=592450,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        venue="Yankee Stadium",
        game_pk=717001,
        game_date="2026-05-01",
        hit_probability=0.72,
        confidence="medium",
        prediction="HIT",
        factors=PredictionFactors(),
        edge="platoon+ | hot",
    )
    defaults.update(overrides)
    return HitPrediction(**defaults)


# ── save_predictions ────────────────────────────────────────────────


def test_save_predictions_inserts_rows():
    from tracker import save_predictions, _get_db

    preds = [
        _make_prediction(batter_name="Judge", batter_id=1),
        _make_prediction(batter_name="Soto", batter_id=2),
    ]
    save_predictions(preds)

    db = _get_db()
    rows = db.execute("SELECT * FROM predictions").fetchall()
    db.close()
    assert len(rows) == 2
    assert rows[0]["batter_name"] == "Judge"
    assert rows[1]["batter_name"] == "Soto"


def test_save_predictions_stores_probability():
    from tracker import save_predictions, _get_db

    save_predictions([_make_prediction(hit_probability=0.68)])

    db = _get_db()
    row = db.execute("SELECT hit_probability FROM predictions").fetchone()
    db.close()
    assert abs(row["hit_probability"] - 0.68) < 0.001


def test_save_predictions_empty_list():
    from tracker import save_predictions, _get_db

    save_predictions([])

    db = _get_db()
    count = db.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    db.close()
    assert count == 0


# ── check_results ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_results_no_pending():
    from tracker import check_results

    result = await check_results("2026-05-01")
    assert result["checked"] == 0
    assert "No pending" in result["message"]


def test_check_results_updates_predictions():
    """Verify that check_results marks predictions with actual results."""
    from tracker import save_predictions, _get_db

    # Save a prediction, then manually mark it as having a result
    # (we can't call the real MLB API in tests)
    save_predictions([_make_prediction(game_pk=717001, batter_id=592450)])

    db = _get_db()
    # Simulate: manually update as if box score showed 2-for-4
    db.execute("""
        UPDATE predictions SET actual_result = '2-for-4', got_hit = 1, at_bats = 4, hits = 2
        WHERE batter_id = 592450
    """)
    db.commit()

    row = db.execute("SELECT * FROM predictions WHERE batter_id = 592450").fetchone()
    db.close()
    assert row["got_hit"] == 1
    assert row["hits"] == 2
    assert row["at_bats"] == 4


# ── get_overall_stats ───────────────────────────────────────────────


def test_overall_stats_empty():
    from tracker import get_overall_stats

    stats = get_overall_stats()
    assert stats["total_predictions"] == 0
    assert stats["accuracy"] == 0


def test_overall_stats_counts_correctly():
    from tracker import save_predictions, get_overall_stats, _get_db

    # Save 3 predictions with known outcomes
    preds = [
        _make_prediction(batter_id=1, prediction="HIT", hit_probability=0.70, confidence="medium"),
        _make_prediction(batter_id=2, prediction="HIT", hit_probability=0.65, confidence="medium"),
        _make_prediction(batter_id=3, prediction="HIT", hit_probability=0.55, confidence="low"),
    ]
    save_predictions(preds)

    # Manually set results
    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='1-for-4', got_hit=1 WHERE batter_id=1")
    db.execute("UPDATE predictions SET actual_result='0-for-3', got_hit=0 WHERE batter_id=2")
    db.execute("UPDATE predictions SET actual_result='1-for-3', got_hit=1 WHERE batter_id=3")
    db.commit()
    db.close()

    stats = get_overall_stats()
    assert stats["total_predictions"] == 3
    assert stats["correct"] == 2  # batter 1 HIT+hit, batter 3 HIT+hit
    assert stats["accuracy"] == pytest.approx(66.7, abs=0.1)
    assert stats["hit_predictions"]["total"] == 3
    assert stats["hit_predictions"]["correct"] == 2


def test_overall_stats_by_confidence():
    from tracker import save_predictions, get_overall_stats, _get_db

    preds = [
        _make_prediction(batter_id=1, confidence="high", prediction="HIT"),
        _make_prediction(batter_id=2, confidence="high", prediction="HIT"),
        _make_prediction(batter_id=3, confidence="medium", prediction="HIT"),
    ]
    save_predictions(preds)

    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='1-for-4', got_hit=1 WHERE batter_id=1")
    db.execute("UPDATE predictions SET actual_result='0-for-3', got_hit=0 WHERE batter_id=2")
    db.execute("UPDATE predictions SET actual_result='1-for-3', got_hit=1 WHERE batter_id=3")
    db.commit()
    db.close()

    stats = get_overall_stats()
    assert stats["by_confidence"]["high"]["total"] == 2
    assert stats["by_confidence"]["high"]["correct"] == 1
    assert stats["by_confidence"]["medium"]["total"] == 1
    assert stats["by_confidence"]["medium"]["correct"] == 1


def test_overall_stats_ignores_dnp():
    from tracker import save_predictions, get_overall_stats, _get_db

    save_predictions([_make_prediction(batter_id=1)])

    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='DNP' WHERE batter_id=1")
    db.commit()
    db.close()

    stats = get_overall_stats()
    assert stats["total_predictions"] == 0  # DNP excluded


# ── get_recent_predictions ──────────────────────────────────────────


def test_recent_predictions_limit():
    from tracker import save_predictions, get_recent_predictions

    preds = [
        _make_prediction(batter_id=1, batter_name="First"),
        _make_prediction(batter_id=2, batter_name="Second"),
        _make_prediction(batter_id=3, batter_name="Third"),
    ]
    save_predictions(preds)

    recent = get_recent_predictions(2)
    assert len(recent) == 2

    all_preds = get_recent_predictions(10)
    assert len(all_preds) == 3


# ── save_book_odds ──────────────────────────────────────────────────


def test_save_book_odds():
    from tracker import save_book_odds, get_odds_for_date

    odds_entries = [
        {"book": "DraftKings", "over": -180, "under": 140, "line": 0.5},
        {"book": "FanDuel", "over": -170, "under": 130, "line": 0.5},
    ]
    save_book_odds(
        game_pk=717001, game_date="2026-05-01",
        batter_name="Aaron Judge", batter_id=592450,
        model_prob=0.72, odds_entries=odds_entries,
    )

    odds = get_odds_for_date("2026-05-01")
    assert len(odds) == 2
    assert odds[0]["book"] in ("DraftKings", "FanDuel")


def test_save_book_odds_skips_none_over():
    from tracker import save_book_odds, get_odds_for_date

    odds_entries = [
        {"book": "DraftKings", "over": None, "under": 140, "line": 0.5},
    ]
    save_book_odds(
        game_pk=717001, game_date="2026-05-01",
        batter_name="Judge", batter_id=1,
        model_prob=0.70, odds_entries=odds_entries,
    )

    odds = get_odds_for_date("2026-05-01")
    assert len(odds) == 0


def test_save_book_odds_calculates_edge():
    from tracker import save_book_odds, _get_db

    save_book_odds(
        game_pk=717001, game_date="2026-05-01",
        batter_name="Judge", batter_id=1,
        model_prob=0.72,
        odds_entries=[{"book": "BetMGM", "over": -180, "under": 140, "line": 0.5}],
    )

    db = _get_db()
    row = db.execute("SELECT * FROM book_odds").fetchone()
    db.close()
    # -180 implied = 180/280 = 0.6429
    assert row["implied_prob"] == pytest.approx(0.6429, abs=0.001)
    assert row["edge"] == pytest.approx(0.72 - 0.6429, abs=0.001)


# ── place_paper_bets ────────────────────────────────────────────────


def test_place_paper_bets_picks_best():
    from tracker import save_predictions, save_book_odds, place_paper_bets, _get_db

    # Save a prediction with high probability
    save_predictions([_make_prediction(
        batter_id=1, batter_name="Judge", game_pk=717001,
        prediction="HIT", hit_probability=0.72, confidence="medium",
        edge="platoon+",
    )])

    # Save odds with edge
    save_book_odds(
        game_pk=717001, game_date="2026-05-01",
        batter_name="Judge", batter_id=1,
        model_prob=0.72,
        odds_entries=[{"book": "DraftKings", "over": -170, "under": 130, "line": 0.5}],
    )

    bets = place_paper_bets("2026-05-01")
    assert len(bets) == 1
    assert bets[0]["batter_name"] == "Judge"

    # Verify it's in the database
    db = _get_db()
    rows = db.execute("SELECT * FROM paper_bets").fetchall()
    db.close()
    assert len(rows) == 1
    assert rows[0]["stake"] == 100


def test_place_paper_bets_no_duplicates():
    from tracker import save_predictions, save_book_odds, place_paper_bets

    save_predictions([_make_prediction(batter_id=1, batter_name="Judge", prediction="HIT", hit_probability=0.72)])
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])

    bets1 = place_paper_bets("2026-05-01")
    bets2 = place_paper_bets("2026-05-01")  # second call
    assert len(bets1) == 1
    assert len(bets2) == 0  # no duplicates


def test_place_paper_bets_filters_low_prob():
    from tracker import save_predictions, save_book_odds, place_paper_bets

    # Prediction below 0.64 threshold
    save_predictions([_make_prediction(batter_id=1, batter_name="Weak", prediction="HIT", hit_probability=0.55)])
    save_book_odds(717001, "2026-05-01", "Weak", 1, 0.55,
                   [{"book": "DK", "over": +120, "under": -140, "line": 0.5}])

    bets = place_paper_bets("2026-05-01")
    assert len(bets) == 0


# ── settle_paper_bets ───────────────────────────────────────────────


def test_settle_paper_bets_win():
    from tracker import save_predictions, save_book_odds, place_paper_bets, settle_paper_bets, _get_db

    save_predictions([_make_prediction(batter_id=1, batter_name="Judge", prediction="HIT", hit_probability=0.72)])
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    # Mark prediction as hit
    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='1-for-3', got_hit=1 WHERE batter_id=1")
    db.commit()
    db.close()

    result = settle_paper_bets("2026-05-01")
    assert result["wins"] == 1
    assert result["losses"] == 0
    # -170 odds: win = 100 * (100/170) = 58.82
    assert result["pnl"] == pytest.approx(58.82, abs=0.1)


def test_settle_paper_bets_loss():
    from tracker import save_predictions, save_book_odds, place_paper_bets, settle_paper_bets, _get_db

    save_predictions([_make_prediction(batter_id=1, batter_name="Judge", prediction="HIT", hit_probability=0.72)])
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    # Mark prediction as no hit
    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='0-for-4', got_hit=0 WHERE batter_id=1")
    db.commit()
    db.close()

    result = settle_paper_bets("2026-05-01")
    assert result["wins"] == 0
    assert result["losses"] == 1
    assert result["pnl"] == -100.0


def test_settle_paper_bets_no_results_yet():
    from tracker import save_predictions, save_book_odds, place_paper_bets, settle_paper_bets

    save_predictions([_make_prediction(batter_id=1, batter_name="Judge", prediction="HIT", hit_probability=0.72)])
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    # Don't set results — settle should skip
    result = settle_paper_bets("2026-05-01")
    assert result["settled"] == 0


# ── get_paper_summary ───────────────────────────────────────────────


def test_paper_summary_empty():
    from tracker import get_paper_summary

    ps = get_paper_summary()
    assert ps["total_bets"] == 0
    assert ps["total_pnl"] == 0


def test_paper_summary_after_settlement():
    from tracker import save_predictions, save_book_odds, place_paper_bets, settle_paper_bets, get_paper_summary, _get_db

    # Two bets: one win, one loss
    save_predictions([
        _make_prediction(batter_id=1, batter_name="Winner", prediction="HIT", hit_probability=0.72, game_pk=717001),
        _make_prediction(batter_id=2, batter_name="Loser", prediction="HIT", hit_probability=0.70, game_pk=717002),
    ])
    save_book_odds(717001, "2026-05-01", "Winner", 1, 0.72,
                   [{"book": "DK", "over": -180, "under": 140, "line": 0.5}])
    save_book_odds(717002, "2026-05-01", "Loser", 2, 0.70,
                   [{"book": "BetMGM", "over": -160, "under": 130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='2-for-4', got_hit=1 WHERE batter_id=1")
    db.execute("UPDATE predictions SET actual_result='0-for-3', got_hit=0 WHERE batter_id=2")
    db.commit()
    db.close()

    settle_paper_bets("2026-05-01")

    ps = get_paper_summary()
    assert ps["total_bets"] == 2
    assert ps["wins"] == 1
    assert ps["losses"] == 1
    assert ps["days"] == 1
    assert len(ps["daily"]) == 1


# ── get_paper_bets_for_date ─────────────────────────────────────────


def test_paper_bets_for_date():
    from tracker import save_predictions, save_book_odds, place_paper_bets, get_paper_bets_for_date

    save_predictions([_make_prediction(batter_id=1, batter_name="Judge", prediction="HIT", hit_probability=0.72)])
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    bets = get_paper_bets_for_date("2026-05-01")
    assert len(bets) == 1

    empty = get_paper_bets_for_date("2026-05-02")
    assert len(empty) == 0


# ── games_with_odds ─────────────────────────────────────────────────


def test_games_with_odds():
    from tracker import save_book_odds, games_with_odds

    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    save_book_odds(717002, "2026-05-01", "Soto", 2, 0.68,
                   [{"book": "DK", "over": -150, "under": 120, "line": 0.5}])

    fetched = games_with_odds("2026-05-01")
    assert fetched == {717001, 717002}

    empty = games_with_odds("2026-05-02")
    assert empty == set()


# ── Positive odds payout ────────────────────────────────────────────


def test_settle_positive_odds_payout():
    from tracker import save_predictions, save_book_odds, place_paper_bets, settle_paper_bets, _get_db

    save_predictions([_make_prediction(batter_id=1, batter_name="Underdog", prediction="HIT", hit_probability=0.66)])
    save_book_odds(717001, "2026-05-01", "Underdog", 1, 0.66,
                   [{"book": "BetMGM", "over": 110, "under": -130, "line": 0.5}])
    place_paper_bets("2026-05-01")

    db = _get_db()
    db.execute("UPDATE predictions SET actual_result='1-for-3', got_hit=1 WHERE batter_id=1")
    db.commit()
    db.close()

    result = settle_paper_bets("2026-05-01")
    # +110 odds: win = 100 * (110/100) = 110
    assert result["pnl"] == pytest.approx(110.0, abs=0.1)


# ── CLV tracking ────────────────────────────────────────────────────


def test_update_closing_odds_positive_clv():
    """When closing line moves toward our position, CLV is positive."""
    from tracker import save_book_odds, update_closing_odds, _get_db

    # Opening: -170 (implied 63%)
    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])

    # Closing: -200 (implied 67%) — line moved our way
    update_closing_odds(717001, "2026-05-01", "Judge", "DK", -200)

    db = _get_db()
    row = db.execute("SELECT * FROM book_odds WHERE batter_name = 'Judge'").fetchone()
    db.close()
    assert row["closing_over_price"] == -200
    assert row["closing_implied_prob"] == pytest.approx(0.6667, abs=0.01)
    # Opening: -170 implied = 0.6296, Closing: -200 implied = 0.6667
    # CLV = closing - opening = 0.6667 - 0.6296 = +0.037 (positive = we got value)
    assert row["clv"] == pytest.approx(0.037, abs=0.01)


def test_update_closing_odds_negative_clv():
    """When closing line moves against us, CLV is negative."""
    from tracker import save_book_odds, update_closing_odds, _get_db

    # Opening: -170 (implied 63%)
    save_book_odds(717001, "2026-05-01", "Soto", 2, 0.70,
                   [{"book": "BetMGM", "over": -170, "under": 130, "line": 0.5}])

    # Closing: -140 (implied 58%) — line moved against us (got cheaper)
    update_closing_odds(717001, "2026-05-01", "Soto", "BetMGM", -140)

    db = _get_db()
    row = db.execute("SELECT * FROM book_odds WHERE batter_name = 'Soto'").fetchone()
    db.close()
    assert row["closing_over_price"] == -140
    # CLV = closing 0.5833 - opening 0.6296 = -0.046 (negative = line moved against us)
    assert row["clv"] == pytest.approx(-0.046, abs=0.01)


def test_get_clv_stats_empty():
    from tracker import get_clv_stats

    stats = get_clv_stats()
    assert stats["total"] == 0


def test_get_clv_stats_with_data():
    from tracker import save_book_odds, update_closing_odds, get_clv_stats

    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    save_book_odds(717002, "2026-05-01", "Soto", 2, 0.70,
                   [{"book": "DK", "over": -180, "under": 140, "line": 0.5}])

    update_closing_odds(717001, "2026-05-01", "Judge", "DK", -200)
    update_closing_odds(717002, "2026-05-01", "Soto", "DK", -150)

    stats = get_clv_stats()
    assert stats["total"] == 2
    assert "avg_clv" in stats
    assert "by_tier" in stats


def test_games_needing_closing_odds():
    from tracker import save_book_odds, games_needing_closing_odds

    save_book_odds(717001, "2026-05-01", "Judge", 1, 0.72,
                   [{"book": "DK", "over": -170, "under": 130, "line": 0.5}])
    save_book_odds(717002, "2026-05-01", "Weak", 2, 0.50,
                   [{"book": "DK", "over": +120, "under": -140, "line": 0.5}])

    # Only game with 64%+ model prob should be returned
    needs = games_needing_closing_odds("2026-05-01")
    assert 717001 in needs
    assert 717002 not in needs
