"""Tests for calibration drift detection."""

import sqlite3
import pytest
from unittest.mock import patch
from pathlib import Path

from drift import (
    check_drift,
    format_drift_report,
    _compute_metrics,
    BRIER_DRIFT_THRESHOLD,
    TIER_SEPARATION_MIN,
    MIN_PREDICTIONS_FOR_CHECK,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_rows(probs_and_outcomes: list[tuple[float, int]]) -> list:
    """Create fake prediction rows for testing."""
    return [
        {"hit_probability": p, "got_hit": y, "game_date": "2026-05-18", "model_version": "v3.1"}
        for p, y in probs_and_outcomes
    ]


# ── _compute_metrics tests ───────────────────────────────────────────


def test_compute_metrics_empty():
    result = _compute_metrics([])
    assert result == {}


def test_compute_metrics_basic():
    rows = _make_rows([(0.70, 1), (0.70, 0), (0.40, 0), (0.40, 1)])
    m = _compute_metrics(rows)
    assert m["n"] == 4
    assert 0 < m["brier"] < 1
    assert 0 < m["log_loss"] < 2
    assert m["hit_rate"] == 0.5


def test_compute_metrics_perfect_predictions():
    """Perfect predictions should have very low Brier score."""
    rows = _make_rows([(0.95, 1)] * 20 + [(0.05, 0)] * 20)
    m = _compute_metrics(rows)
    assert m["brier"] < 0.01
    assert m["hit_rate"] == 0.5


def test_compute_metrics_terrible_predictions():
    """Inverted predictions should have high Brier score."""
    rows = _make_rows([(0.90, 0)] * 20 + [(0.10, 1)] * 20)
    m = _compute_metrics(rows)
    assert m["brier"] > 0.5


def test_compute_metrics_tier_rates():
    """Verify tier classification and hit rates."""
    rows = _make_rows([
        (0.75, 1), (0.75, 1), (0.75, 0),  # STRONG: 2/3
        (0.65, 1), (0.65, 0),              # LEAN: 1/2
        (0.58, 0), (0.58, 0),              # TOSS-UP: 0/2
        (0.45, 0), (0.45, 0), (0.45, 1),   # FADE: 1/3
    ])
    m = _compute_metrics(rows)
    assert m["tier_rates"]["STRONG HIT"] == pytest.approx(2 / 3, abs=0.01)
    assert m["tier_rates"]["LEAN HIT"] == pytest.approx(1 / 2, abs=0.01)
    assert m["tier_rates"]["TOSS-UP"] == pytest.approx(0.0, abs=0.01)
    assert m["tier_rates"]["FADE"] == pytest.approx(1 / 3, abs=0.01)


# ── check_drift tests ────────────────────────────────────────────────


def test_drift_insufficient_data():
    """Should return insufficient_data when not enough predictions."""
    with patch("drift._fetch_settled_predictions", return_value=[]):
        result = check_drift()
        assert result["status"] == "insufficient_data"


def test_drift_healthy():
    """Consistent model should report healthy."""
    # Same distribution for lifetime and recent
    rows = _make_rows(
        [(0.70, 1)] * 30 + [(0.70, 0)] * 10 +
        [(0.50, 1)] * 15 + [(0.50, 0)] * 15 +
        [(0.35, 0)] * 20 + [(0.35, 1)] * 10
    )

    with patch("drift._fetch_settled_predictions", return_value=rows), \
         patch("drift.check_feature_coverage", return_value={}):
        result = check_drift()
        assert result["status"] == "healthy"
        assert result["alerts"] == []
        assert result["lifetime"]["n"] == 100


def test_drift_degraded_brier():
    """Should alert when recent Brier score is much worse than lifetime."""
    good_rows = _make_rows(
        [(0.75, 1)] * 40 + [(0.40, 0)] * 40 +
        [(0.60, 1)] * 10 + [(0.60, 0)] * 10
    )
    bad_rows = _make_rows(
        [(0.80, 0)] * 15 + [(0.30, 1)] * 15  # inverted predictions
    )

    def mock_fetch(days_back=None, model_version=None):
        if days_back is not None and days_back <= 7:
            return bad_rows
        return good_rows + bad_rows

    with patch("drift._fetch_settled_predictions", side_effect=mock_fetch), \
         patch("drift.check_feature_coverage", return_value={}):
        result = check_drift()
        assert result["status"] in ("warning", "degraded")
        assert any("Brier" in a for a in result["alerts"])


# ── format_drift_report tests ────────────────────────────────────────


def test_format_insufficient():
    result = {"status": "insufficient_data", "message": "Need 50+ predictions", "alerts": []}
    report = format_drift_report(result)
    assert "insufficient" in report.lower() or "50+" in report


def test_format_healthy():
    result = {
        "status": "healthy",
        "alerts": [],
        "lifetime": {
            "n": 100, "brier": 0.2300, "log_loss": 0.6500,
            "hit_rate": 0.610,
            "tier_rates": {"STRONG HIT": 0.75, "LEAN HIT": 0.65, "TOSS-UP": 0.58, "FADE": 0.45},
            "calibration": {},
        },
        "recent_7d": {
            "n": 50, "brier": 0.2280, "log_loss": 0.6480,
            "hit_rate": 0.620,
            "tier_rates": {"STRONG HIT": 0.76, "LEAN HIT": 0.64, "TOSS-UP": 0.59, "FADE": 0.44},
            "calibration": {},
        },
        "recent_3d": {},
    }
    report = format_drift_report(result)
    assert "HEALTHY" in report
    assert "No drift" in report
    assert "Lifetime" in report


def test_format_warning_with_alerts():
    result = {
        "status": "warning",
        "alerts": ["Brier drift (7d): 0.2800 vs 0.2300 (+0.0500)"],
        "lifetime": {
            "n": 200, "brier": 0.2300, "log_loss": 0.6500,
            "hit_rate": 0.610,
            "tier_rates": {"STRONG HIT": 0.75, "LEAN HIT": 0.65, "TOSS-UP": 0.58, "FADE": 0.45},
            "calibration": {},
        },
        "recent_7d": {
            "n": 50, "brier": 0.2800, "log_loss": 0.7000,
            "hit_rate": 0.580,
            "tier_rates": {"STRONG HIT": 0.68, "LEAN HIT": 0.60, "TOSS-UP": 0.55, "FADE": 0.48},
            "calibration": {},
        },
        "recent_3d": {},
    }
    report = format_drift_report(result)
    assert "WARNING" in report
    assert "Brier" in report
    assert "Alerts" in report
