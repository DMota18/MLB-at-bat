"""Shadow model variant: lighter calibration squeeze.

Identical to the primary predictor.predict_hit except:
  - Correlation discount: 0.80 (was 0.75) — less AB deflation
  - Calibration squeeze: 0.85 (was 0.75) — wider probability spread

Hypothesis: the primary model's squeeze is too aggressive, compressing
everything into 45-67% and killing the STRONG HIT tier. This variant
should produce more predictions above 70% while still being calibrated.

This model runs as a shadow via ab_testing — it never affects paper
bets, alerts, or the primary predictions table.
"""

from __future__ import annotations

from predictor import (
    predict_hit as _primary_predict,
    PredictionFactors,
    HitPrediction,
    hit_tier,
    TIER_EMOJI,
    _safe_float,
    LEAGUE_AVG,
    LEAGUE_BABIP,
    LEAGUE_GAME_HIT_RATE,
)
from config import TIER_STRONG, TIER_LEAN, TIER_TOSSUP


# ── Shadow constants (the only differences from primary) ─────────
SHADOW_CORRELATION_DISCOUNT = 0.80   # primary: 0.75
SHADOW_CALIBRATION_SQUEEZE = 0.85    # primary: 0.75


def predict_hit_shadow(
    batter_data: dict,
    pitcher_data: dict,
    batter_bats: str,
    pitcher_throws: str | None,
    venue: str,
    park_factors: dict,
    game_pk: int = 0,
    game_date: str = "",
    batter_name: str = "",
    batter_id: int = 0,
    pitcher_name: str = "",
    pitcher_id: int = 0,
    h2h_data: dict | None = None,
    arsenal_matchup: dict | None = None,
    batting_order: int = 5,
    bullpen_avg: float = 0.248,
    pitcher_recent: dict | None = None,
    batter_statcast: dict | None = None,
    weather_adj: float = 0.0,
) -> HitPrediction:
    """Shadow model: same logic as primary but with lighter calibration.

    Duplicates only the final scaling section. All factor computation,
    weights, per-AB probability, and edge tags are identical to primary.
    We call the primary model, then re-compute only the full-game
    scaling with different constants.
    """
    # Run the primary model to get all the computed factors
    primary = _primary_predict(
        batter_data=batter_data, pitcher_data=pitcher_data,
        batter_bats=batter_bats, pitcher_throws=pitcher_throws,
        venue=venue, park_factors=park_factors,
        game_pk=game_pk, game_date=game_date,
        batter_name=batter_name, batter_id=batter_id,
        pitcher_name=pitcher_name, pitcher_id=pitcher_id,
        h2h_data=h2h_data, arsenal_matchup=arsenal_matchup,
        batting_order=batting_order, bullpen_avg=bullpen_avg,
        pitcher_recent=pitcher_recent, batter_statcast=batter_statcast,
        weather_adj=weather_adj,
    )

    f = primary.factors

    # ── Re-compute full-game scaling with shadow constants ──────

    # Reconstruct bullpen per-AB prob from factors
    batter_avg = f.batter_avg
    pf_runs = f.park_factor
    bullpen_per_ab = (batter_avg * 0.60 + f.bullpen_avg_against * 0.40)
    bullpen_per_ab *= pf_runs
    if f.platoon_edge:
        bullpen_per_ab *= 1.02
    bullpen_per_ab = max(0.150, min(0.350, bullpen_per_ab))

    # AB splits (same as primary)
    per_ab = f.per_ab_prob
    total_ab = f.total_expected_ab

    if batting_order <= 2:
        starter_ab, bullpen_ab = 3.075, 1.425
    elif batting_order <= 4:
        starter_ab, bullpen_ab = 2.870, 1.330
    elif batting_order <= 6:
        starter_ab, bullpen_ab = 2.375, 1.425
    elif batting_order <= 8:
        starter_ab, bullpen_ab = 2.170, 1.330
    else:
        starter_ab, bullpen_ab = 1.965, 1.235

    # Shadow correlation discount (0.80 vs primary's 0.75)
    eff_starter = starter_ab * SHADOW_CORRELATION_DISCOUNT
    eff_bullpen = bullpen_ab * SHADOW_CORRELATION_DISCOUNT

    p_no_hit = (1 - per_ab) ** eff_starter * (1 - bullpen_per_ab) ** eff_bullpen
    hit_prob = 1 - p_no_hit

    # Shadow calibration squeeze (0.85 vs primary's 0.75)
    hit_prob = 0.60 + (hit_prob - 0.60) * SHADOW_CALIBRATION_SQUEEZE
    hit_prob = max(0.25, min(0.90, hit_prob))

    # Prediction + confidence (same thresholds as primary)
    prediction = "HIT" if hit_prob >= 0.46 else "NO HIT"

    pa = f.sample_size
    distance = abs(hit_prob - 0.52)
    data_quality = (
        (1 if pa >= 50 else 0)
        + (1 if f.platoon_pa >= 50 else 0)
        + (1 if f.h2h_ab >= 10 else 0)
        + (1 if f.arsenal_has_data else 0)
    )
    if data_quality >= 3 and distance > 0.12:
        confidence = "high"
    elif data_quality >= 2 and distance > 0.06:
        confidence = "medium"
    elif pa > 0:
        confidence = "low"
    else:
        confidence = "insufficient"

    return HitPrediction(
        batter_name=batter_name, batter_id=batter_id,
        pitcher_name=pitcher_name, pitcher_id=pitcher_id,
        venue=venue, game_pk=game_pk, game_date=game_date,
        hit_probability=hit_prob, confidence=confidence,
        prediction=prediction, factors=f, edge=primary.edge,
    )
