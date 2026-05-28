"""Tests for A/B testing framework."""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from dataclasses import dataclass

from ab_testing import (
    register_shadow_model,
    get_shadow_model,
    clear_shadow_model,
    run_shadow_prediction,
    settle_shadow_predictions,
    compare_models,
    format_ab_report,
    _compute_model_metrics,
)
from predictor import HitPrediction, PredictionFactors


# ── Helpers ──────────────────────────────────────────────────────────


def _mock_predict(**kwargs) -> HitPrediction:
    """A trivial shadow model that always predicts 60%."""
    return HitPrediction(
        batter_name=kwargs.get("batter_name", "Test"),
        batter_id=kwargs.get("batter_id", 1),
        pitcher_name=kwargs.get("pitcher_name", "Pitcher"),
        pitcher_id=kwargs.get("pitcher_id", 2),
        venue=kwargs.get("venue", "Park"),
        game_pk=kwargs.get("game_pk", 100),
        game_date=kwargs.get("game_date", "2026-05-18"),
        hit_probability=0.60,
        confidence="medium",
        prediction="HIT",
        factors=PredictionFactors(),
        edge="test-model",
    )


# ── Shadow model registry tests ──────────────────────────────────────


def test_register_and_get_shadow():
    clear_shadow_model()
    assert get_shadow_model() is None

    register_shadow_model("test", "v0.1", _mock_predict, "test model")
    model = get_shadow_model()
    assert model is not None
    assert model["name"] == "test"
    assert model["version"] == "v0.1"

    clear_shadow_model()
    assert get_shadow_model() is None


def test_run_shadow_no_model():
    """run_shadow_prediction should return None when no model registered."""
    clear_shadow_model()
    result = run_shadow_prediction(
        batter_data={}, pitcher_data={}, batter_bats="R",
        pitcher_throws="R", venue="Park", park_factors={"runs": 1.0},
    )
    assert result is None


def test_run_shadow_with_model():
    """run_shadow_prediction should call the shadow model and save result."""
    clear_shadow_model()
    register_shadow_model("test", "v0.1", _mock_predict, "test model")

    with patch("ab_testing._save_shadow_prediction") as mock_save:
        result = run_shadow_prediction(
            batter_data={}, pitcher_data={}, batter_bats="R",
            pitcher_throws="R", venue="Park", park_factors={"runs": 1.0},
            batter_name="Batter", batter_id=1, pitcher_name="Pitcher",
            pitcher_id=2, game_pk=100, game_date="2026-05-18",
        )
        assert result is not None
        assert result.hit_probability == 0.60
        mock_save.assert_called_once()

    clear_shadow_model()


def test_run_shadow_handles_error():
    """Shadow model failures should be caught, not propagated."""
    def bad_predict(**kwargs):
        raise ValueError("model exploded")

    clear_shadow_model()
    register_shadow_model("broken", "v0.0", bad_predict)

    result = run_shadow_prediction(
        batter_data={}, pitcher_data={}, batter_bats="R",
        pitcher_throws="R", venue="Park", park_factors={"runs": 1.0},
    )
    assert result is None
    clear_shadow_model()


# ── _compute_model_metrics tests ──────────────────────────────────────


def test_metrics_empty():
    assert _compute_model_metrics([])["n"] == 0


def test_metrics_basic():
    rows = [
        {"hit_probability": 0.70, "got_hit": 1},
        {"hit_probability": 0.70, "got_hit": 0},
        {"hit_probability": 0.40, "got_hit": 0},
        {"hit_probability": 0.40, "got_hit": 1},
    ]
    m = _compute_model_metrics(rows)
    assert m["n"] == 4
    assert 0 < m["brier"] < 1
    assert m["hit_rate"] == 0.5


def test_metrics_tiers():
    rows = [
        {"hit_probability": 0.75, "got_hit": 1},
        {"hit_probability": 0.65, "got_hit": 1},
        {"hit_probability": 0.58, "got_hit": 0},
        {"hit_probability": 0.45, "got_hit": 0},
    ]
    m = _compute_model_metrics(rows)
    assert m["tiers"]["STRONG HIT"]["total"] == 1
    assert m["tiers"]["STRONG HIT"]["hits"] == 1
    assert m["tiers"]["FADE"]["total"] == 1
    assert m["tiers"]["FADE"]["hits"] == 0


# ── compare_models tests ──────────────────────────────────────────────


def test_compare_no_shadow_data():
    """compare_models returns message when no shadow predictions exist."""
    # Create a temporary in-memory DB with no shadow_predictions
    with patch("ab_testing._get_db") as mock_db:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE shadow_predictions (
                id INTEGER PRIMARY KEY, game_pk INTEGER, game_date TEXT,
                batter_name TEXT, batter_id INTEGER, pitcher_name TEXT,
                pitcher_id INTEGER, venue TEXT, prediction TEXT,
                hit_probability REAL, confidence TEXT, edge TEXT,
                model_name TEXT, model_version TEXT, factors_json TEXT,
                actual_result TEXT, got_hit INTEGER, created_at TEXT
            )
        """)
        mock_db.return_value = conn
        result = compare_models()
        assert "message" in result
        conn.close()


# ── format_ab_report tests ────────────────────────────────────────────


def test_format_no_data():
    result = {"message": "No shadow predictions found."}
    report = format_ab_report(result)
    assert "No shadow predictions" in report


def test_format_with_results():
    result = {
        "v0.1": {
            "name": "test-model",
            "version": "v0.1",
            "n": 50,
            "primary": {
                "n": 50, "brier": 0.2300, "log_loss": 0.6500, "hit_rate": 0.600,
                "tiers": {
                    "STRONG HIT": {"hits": 8, "total": 10, "rate": 0.800},
                    "LEAN HIT": {"hits": 7, "total": 12, "rate": 0.583},
                    "TOSS-UP": {"hits": 8, "total": 15, "rate": 0.533},
                    "FADE": {"hits": 5, "total": 13, "rate": 0.385},
                },
            },
            "shadow": {
                "n": 50, "brier": 0.2200, "log_loss": 0.6400, "hit_rate": 0.600,
                "tiers": {
                    "STRONG HIT": {"hits": 9, "total": 10, "rate": 0.900},
                    "LEAN HIT": {"hits": 6, "total": 12, "rate": 0.500},
                    "TOSS-UP": {"hits": 9, "total": 15, "rate": 0.600},
                    "FADE": {"hits": 4, "total": 13, "rate": 0.308},
                },
            },
            "head_to_head": {
                "primary_closer": 18,
                "shadow_closer": 25,
                "ties": 7,
            },
            "disagreements": {
                "shadow_right": 5,
                "primary_right": 3,
                "both_wrong": 2,
            },
        }
    }
    report = format_ab_report(result)
    assert "test-model" in report
    assert "Brier" in report
    assert "Head-to-head" in report
    assert "BETTER" in report  # should have a verdict
