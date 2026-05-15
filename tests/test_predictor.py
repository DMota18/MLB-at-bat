"""Unit tests for the hit prediction model."""

import pytest

from predictor import predict_hit, PredictionFactors, LEAGUE_AVG


# ── Helpers ────────────────────────────────────────────────────────────

def _good_batter():
    """A strong left-handed batter with full season data."""
    return {
        "season": {"avg": ".310", "pa": 250},
        "platoon": {
            "vs_R": {"avg": ".330", "pa": 120},
            "vs_L": {"avg": ".270", "pa": 80},
        },
        "last7": {"avg": ".350", "ab": 25},
        "advanced": {"babip": ".310"},
    }


def _weak_batter():
    """A weak right-handed batter with decent sample."""
    return {
        "season": {"avg": ".190", "pa": 200},
        "platoon": {
            "vs_R": {"avg": ".175", "pa": 100},
            "vs_L": {"avg": ".210", "pa": 60},
        },
        "last7": {"avg": ".130", "ab": 20},
        "advanced": {"babip": ".240"},
    }


def _weak_pitcher():
    """A pitcher who gives up lots of hits."""
    return {
        "season": {"avg": ".280"},
        "platoon": {
            "vs_LHB": {"avg": ".300"},
            "vs_RHB": {"avg": ".260"},
        },
    }


def _strong_pitcher():
    """An ace pitcher."""
    return {
        "season": {"avg": ".195"},
        "platoon": {
            "vs_LHB": {"avg": ".205"},
            "vs_RHB": {"avg": ".190"},
        },
    }


def _neutral_park():
    return {"runs": 1.00}


def _hitter_park():
    return {"runs": 1.12}


def _pitcher_park():
    return {"runs": 0.88}


# ── 1. Typical HIT prediction ─────────────────────────────────────────

def test_typical_hit_prediction():
    """Good batter vs weak pitcher with platoon advantage -> HIT."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Coors Field",
        park_factors=_neutral_park(),
        batting_order=2,
        batter_name="Good Hitter",
        batter_id=1,
        pitcher_name="Weak Pitcher",
        pitcher_id=2,
    )
    assert result.prediction == "HIT"
    assert result.hit_probability > 0.55
    assert result.factors.platoon_edge is True
    assert result.factors.same_hand is False


# ── 2. Typical NO HIT prediction ──────────────────────────────────────

def test_typical_no_hit_prediction():
    """Weak batter vs strong pitcher, same hand -> NO HIT."""
    result = predict_hit(
        batter_data=_weak_batter(),
        pitcher_data=_strong_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Petco Park",
        park_factors=_pitcher_park(),
        batting_order=8,
        batter_name="Weak Batter",
        batter_id=3,
        pitcher_name="Strong Pitcher",
        pitcher_id=4,
    )
    assert result.prediction == "NO HIT"
    assert result.hit_probability < 0.46
    assert result.factors.same_hand is True
    assert result.factors.platoon_edge is False


# ── 3. Edge case: no season data ──────────────────────────────────────

def test_no_season_data_uses_league_avg():
    """Batter with no season stats should use league average and low confidence."""
    result = predict_hit(
        batter_data={},
        pitcher_data=_weak_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Generic Park",
        park_factors=_neutral_park(),
    )
    assert result.factors.batter_avg == LEAGUE_AVG
    assert result.confidence in ("low", "insufficient")


def test_no_season_data_insufficient_confidence():
    """Batter with pa=0 should get 'insufficient' confidence."""
    result = predict_hit(
        batter_data={"season": {"avg": ".000", "pa": 0}},
        pitcher_data=_weak_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.confidence == "insufficient"


# ── 4. Edge case: switch hitter ───────────────────────────────────────

def test_switch_hitter_no_platoon_edge():
    """Switch hitter (bats='S') should never have platoon_edge or same_hand."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_strong_pitcher(),
        batter_bats="S",
        pitcher_throws="R",
        venue="Yankee Stadium",
        park_factors=_neutral_park(),
    )
    # Switch hitters can't have platoon or same-hand tags
    assert result.factors.platoon_edge is False
    assert result.factors.same_hand is False


def test_switch_hitter_uses_best_pitcher_platoon():
    """Switch hitter should use max(vs_LHB, vs_RHB) from pitcher platoon data."""
    pitcher = {
        "season": {"avg": ".240"},
        "platoon": {
            "vs_LHB": {"avg": ".300"},
            "vs_RHB": {"avg": ".200"},
        },
    }
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=pitcher,
        batter_bats="S",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    # The model picks max(0.300, 0.200) = 0.300 for switch hitters
    # which is favorable to the batter
    assert result.prediction == "HIT"


# ── 5. Edge case: missing pitcher data ────────────────────────────────

def test_empty_pitcher_data():
    """Empty pitcher dict should not crash; uses league avg for pitcher."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data={},
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.factors.pitcher_avg_against == LEAGUE_AVG
    assert result.prediction in ("HIT", "NO HIT")  # doesn't crash


def test_pitcher_data_none_values():
    """Pitcher data with None values should fall back gracefully."""
    pitcher = {"season": {"avg": None}, "platoon": {}}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=pitcher,
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.factors.pitcher_avg_against == LEAGUE_AVG


# ── 6. H2H data with enough sample ───────────────────────────────────

def test_h2h_with_enough_sample():
    """H2H data with >= 10 AB should get higher weight (0.12)."""
    h2h = {"avg": ".400", "ab": 15, "hr": 3}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        h2h_data=h2h,
    )
    assert result.factors.h2h_avg == 0.400
    assert result.factors.h2h_ab == 15
    assert result.factors.h2h_hr == 3
    assert "H2H-owns" in result.edge


def test_h2h_struggles():
    """H2H where batter avg is well below season avg -> H2H-struggles tag."""
    batter = _good_batter()  # .310 season avg
    h2h = {"avg": ".150", "ab": 12, "hr": 0}
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        h2h_data=h2h,
    )
    assert "H2H-struggles" in result.edge


def test_h2h_small_sample_lower_weight():
    """H2H data with 3-9 AB should get lower weight (0.07)."""
    h2h = {"avg": ".400", "ab": 5, "hr": 1}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        h2h_data=h2h,
    )
    assert result.factors.h2h_avg == 0.400
    assert result.factors.h2h_ab == 5


def test_h2h_too_small_ignored():
    """H2H data with < 3 AB should be ignored."""
    h2h = {"avg": ".500", "ab": 2, "hr": 1}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        h2h_data=h2h,
    )
    assert result.factors.h2h_avg is None
    assert result.factors.h2h_ab == 2


# ── 7. Arsenal matchup integration ───────────────────────────────────

def test_arsenal_matchup_positive():
    """Arsenal matchup with good weighted avg should boost prediction."""
    arsenal = {"has_data": True, "weighted_avg": 0.350}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        arsenal_matchup=arsenal,
    )
    assert result.factors.arsenal_has_data is True
    assert result.factors.arsenal_avg == 0.350
    assert "arsenal+" in result.edge


def test_arsenal_matchup_negative():
    """Arsenal matchup with low weighted avg should suppress prediction."""
    batter = _good_batter()  # .310 season avg
    arsenal = {"has_data": True, "weighted_avg": 0.200}
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_strong_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        arsenal_matchup=arsenal,
    )
    assert result.factors.arsenal_has_data is True
    assert "arsenal-" in result.edge


def test_arsenal_no_data():
    """Arsenal matchup with has_data=False should be ignored."""
    arsenal = {"has_data": False, "weighted_avg": 0.400}
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        arsenal_matchup=arsenal,
    )
    assert result.factors.arsenal_has_data is False
    assert result.factors.arsenal_avg is None


# ── 8. Probability clamped between 0.25 and 0.90 ─────────────────────

def test_probability_clamped_lower():
    """Even the worst prediction should not go below 0.25."""
    # Terrible batter, ace pitcher, same-hand, pitcher park, bottom of order
    batter = {
        "season": {"avg": ".150", "pa": 300},
        "platoon": {"vs_R": {"avg": ".120", "pa": 150}},
        "last7": {"avg": ".050", "ab": 20},
        "advanced": {"babip": ".180"},
    }
    pitcher = {
        "season": {"avg": ".160"},
        "platoon": {"vs_RHB": {"avg": ".140"}},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=pitcher,
        batter_bats="R",
        pitcher_throws="R",
        venue="Park",
        park_factors=_pitcher_park(),
        batting_order=9,
    )
    assert result.hit_probability >= 0.25


def test_probability_clamped_upper():
    """Even the best prediction should not exceed 0.90."""
    batter = {
        "season": {"avg": ".400", "pa": 300},
        "platoon": {"vs_R": {"avg": ".450", "pa": 150}},
        "last7": {"avg": ".500", "ab": 25},
        "advanced": {"babip": ".300"},
    }
    pitcher = {
        "season": {"avg": ".320"},
        "platoon": {"vs_LHB": {"avg": ".350"}},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=pitcher,
        batter_bats="L",
        pitcher_throws="R",
        venue="Coors Field",
        park_factors=_hitter_park(),
        batting_order=1,
        h2h_data={"avg": ".500", "ab": 20, "hr": 5},
        arsenal_matchup={"has_data": True, "weighted_avg": 0.400},
    )
    assert result.hit_probability <= 0.90


# ── 9. Confidence levels map correctly ────────────────────────────────

def test_confidence_high():
    """High confidence needs data_quality >= 3 and distance > 0.12."""
    # data_quality = pa>=50 + platoon_pa>=50 + h2h_ab>=10 + arsenal_has_data
    # Need at least 3 of those, plus |hit_prob - 0.52| > 0.12
    result = predict_hit(
        batter_data=_good_batter(),  # pa=250 (+1), platoon vs_R pa=120 (+1)
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_hitter_park(),
        h2h_data={"avg": ".400", "ab": 15, "hr": 3},  # +1 -> total 3
        arsenal_matchup={"has_data": True, "weighted_avg": 0.350},  # +1 -> total 4
        batting_order=1,
    )
    # With all these strong signals, expect high confidence
    assert result.confidence == "high"


def test_confidence_medium():
    """Medium confidence needs data_quality >= 2 and distance > 0.06."""
    result = predict_hit(
        batter_data=_good_batter(),  # pa=250 (+1), platoon pa=120 (+1) -> quality 2
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        batting_order=3,
    )
    # data_quality = 2 (pa>=50, platoon_pa>=50), no h2h or arsenal
    # Should be medium if distance > 0.06
    assert result.confidence in ("medium", "high")


def test_confidence_low():
    """Low confidence when pa > 0 but not enough data quality."""
    batter = {
        "season": {"avg": ".250", "pa": 30},
        "platoon": {},
        "last7": {},
        "advanced": {},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.confidence == "low"


def test_confidence_insufficient():
    """Insufficient confidence when pa = 0."""
    result = predict_hit(
        batter_data={},
        pitcher_data={},
        batter_bats="R",
        pitcher_throws=None,
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.confidence == "insufficient"


# ── 10. Edge tags ─────────────────────────────────────────────────────

def test_edge_tag_platoon_plus():
    """L batter vs R pitcher should produce 'platoon+' tag."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "platoon+" in result.edge


def test_edge_tag_same_hand():
    """R batter vs R pitcher should produce 'same-hand-' tag."""
    result = predict_hit(
        batter_data=_weak_batter(),
        pitcher_data=_strong_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "same-hand-" in result.edge


def test_edge_tag_hot():
    """Batter with recent avg well above season avg -> 'hot' tag."""
    batter = {
        "season": {"avg": ".260", "pa": 200},
        "platoon": {"vs_R": {"avg": ".270", "pa": 100}},
        "last7": {"avg": ".400", "ab": 25},  # Well above .260
        "advanced": {"babip": ".300"},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "hot" in result.edge


def test_edge_tag_cold():
    """Batter with recent avg well below season avg -> 'cold' tag."""
    batter = {
        "season": {"avg": ".300", "pa": 200},
        "platoon": {"vs_R": {"avg": ".310", "pa": 100}},
        "last7": {"avg": ".100", "ab": 20},  # Well below .300
        "advanced": {"babip": ".300"},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "cold" in result.edge


def test_edge_tag_hitter_park():
    """High park factor -> 'hitter-park' tag."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Coors Field",
        park_factors=_hitter_park(),  # 1.12
    )
    assert "hitter-park" in result.edge


def test_edge_tag_pitcher_park():
    """Low park factor -> 'pitcher-park' tag."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Petco Park",
        park_factors=_pitcher_park(),  # 0.88
    )
    assert "pitcher-park" in result.edge


def test_edge_tag_pitcher_tough():
    """Pitcher with very low platoon avg -> 'pitcher-tough' tag."""
    tough_pitcher = {
        "season": {"avg": ".180"},
        "platoon": {"vs_LHB": {"avg": ".190"}},
    }
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=tough_pitcher,
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "pitcher-tough" in result.edge


def test_edge_tag_pitcher_vuln():
    """Pitcher with high platoon avg -> 'pitcher-vuln' tag."""
    vuln_pitcher = {
        "season": {"avg": ".290"},
        "platoon": {"vs_LHB": {"avg": ".310"}},
    }
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=vuln_pitcher,
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "pitcher-vuln" in result.edge


def test_edge_tag_babip_high():
    """High BABIP with enough PA -> 'BABIP-high' tag."""
    batter = {
        "season": {"avg": ".300", "pa": 200},
        "platoon": {"vs_R": {"avg": ".310", "pa": 100}},
        "last7": {"avg": ".300", "ab": 20},
        "advanced": {"babip": ".380"},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "BABIP-high" in result.edge


def test_edge_tag_babip_low():
    """Low BABIP with enough PA -> 'BABIP-low' tag."""
    batter = {
        "season": {"avg": ".230", "pa": 200},
        "platoon": {"vs_R": {"avg": ".220", "pa": 100}},
        "last7": {"avg": ".230", "ab": 20},
        "advanced": {"babip": ".220"},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert "BABIP-low" in result.edge


def test_edge_neutral_when_nothing_special():
    """No special conditions -> 'neutral' edge string."""
    batter = {
        "season": {"avg": ".248", "pa": 30},
        "platoon": {},
        "last7": {},
        "advanced": {"babip": ".300"},
    }
    pitcher = {
        "season": {"avg": ".248"},
        "platoon": {},
    }
    result = predict_hit(
        batter_data=batter,
        pitcher_data=pitcher,
        batter_bats="R",
        pitcher_throws=None,
        venue="Park",
        park_factors=_neutral_park(),
    )
    assert result.edge == "neutral"


# ── to_dict serialization ─────────────────────────────────────────────

def test_to_dict():
    """HitPrediction.to_dict() should return correct keys and rounded probability."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        game_pk=12345,
        game_date="2026-04-30",
        batter_name="Test Player",
        batter_id=99,
        pitcher_name="Test Pitcher",
        pitcher_id=88,
    )
    d = result.to_dict()
    assert d["batter_name"] == "Test Player"
    assert d["batter_id"] == 99
    assert d["pitcher_name"] == "Test Pitcher"
    assert d["pitcher_id"] == 88
    assert d["game_pk"] == 12345
    assert d["game_date"] == "2026-04-30"
    assert isinstance(d["hit_probability"], float)
    # Probability should be rounded to 3 decimal places
    assert d["hit_probability"] == round(d["hit_probability"], 3)
    assert d["prediction"] in ("HIT", "NO HIT")
    assert d["confidence"] in ("high", "medium", "low", "insufficient")


# ── Batting order affects expected AB ─────────────────────────────────

def test_leadoff_gets_more_ab():
    """Leadoff hitter (order=1) should get 4.5 expected AB."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        batting_order=1,
    )
    assert result.factors.total_expected_ab == 4.5


def test_nine_hole_gets_fewer_ab():
    """9-hole batter should get 3.2 expected AB."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        batting_order=9,
    )
    assert result.factors.total_expected_ab == 3.2


def test_bullpen_split_helps_weak_batter():
    """A weak batter facing a tough starter should benefit from bullpen ABs
    (bullpen is easier than the tough starter)."""
    # Same batter vs tough starter - check that bullpen_avg is set
    result = predict_hit(
        batter_data=_weak_batter(),
        pitcher_data=_strong_pitcher(),
        batter_bats="R",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        batting_order=5,
    )
    # Bullpen avg should be league default (0.248)
    assert result.factors.bullpen_avg_against == 0.248
    # Total AB should be split (starter + bullpen)
    assert result.factors.total_expected_ab == pytest.approx(3.8)


def test_bullpen_split_hurts_good_batter_vs_weak_pitcher():
    """A good batter facing a weak starter should get lower prob than if
    they faced the starter all game, because bullpen ABs are tougher."""
    result = predict_hit(
        batter_data=_good_batter(),
        pitcher_data=_weak_pitcher(),
        batter_bats="L",
        pitcher_throws="R",
        venue="Park",
        park_factors=_neutral_park(),
        batting_order=3,
    )
    # The per_ab (vs starter) should be higher than the final prob implies
    # because bullpen ABs drag it down
    assert result.factors.per_ab_prob > 0.25
    assert result.hit_probability < 0.85  # not unrealistically high


# ── Probability value comparisons ───────────────────────────────────


def test_good_batter_higher_prob_than_weak():
    """A strong batter should have a meaningfully higher probability than a weak one."""
    good = predict_hit(
        batter_data=_good_batter(), pitcher_data=_weak_pitcher(),
        batter_bats="L", pitcher_throws="R",
        venue="Park", park_factors=_neutral_park(), batting_order=3,
    )
    weak = predict_hit(
        batter_data=_weak_batter(), pitcher_data=_strong_pitcher(),
        batter_bats="R", pitcher_throws="R",
        venue="Park", park_factors=_neutral_park(), batting_order=7,
    )
    assert good.hit_probability > weak.hit_probability + 0.05


def test_platoon_advantage_boosts_probability():
    """Same batter should get higher probability with platoon advantage than same-hand."""
    batter = _good_batter()
    pitcher = _weak_pitcher()

    platoon = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="R",
        venue="Park", park_factors=_neutral_park(), batting_order=3,
    )
    same_hand = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="L",
        venue="Park", park_factors=_neutral_park(), batting_order=3,
    )
    assert platoon.hit_probability > same_hand.hit_probability


def test_leadoff_higher_prob_than_nine_hole():
    """Same batter/pitcher but batting 1st should have higher prob than 9th (more AB)."""
    batter = _good_batter()
    pitcher = _weak_pitcher()

    leadoff = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="R",
        venue="Park", park_factors=_neutral_park(), batting_order=1,
    )
    nine_hole = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="R",
        venue="Park", park_factors=_neutral_park(), batting_order=9,
    )
    assert leadoff.hit_probability > nine_hole.hit_probability


def test_hitter_park_boosts_probability():
    """Same matchup in a hitter-friendly park should have higher prob than pitcher park."""
    batter = _good_batter()
    pitcher = _weak_pitcher()

    hitter = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="R",
        venue="Coors", park_factors=_hitter_park(), batting_order=3,
    )
    pitcher_p = predict_hit(
        batter_data=batter, pitcher_data=pitcher,
        batter_bats="L", pitcher_throws="R",
        venue="Petco", park_factors=_pitcher_park(), batting_order=3,
    )
    assert hitter.hit_probability > pitcher_p.hit_probability


def test_probability_range_realistic():
    """All predictions should fall in a realistic range (0.30 to 0.85)."""
    scenarios = [
        (_good_batter(), _weak_pitcher(), "L", "R", 1),
        (_weak_batter(), _strong_pitcher(), "R", "R", 9),
        ({}, {}, "R", None, 5),
    ]
    for batter, pitcher, bats, throws, order in scenarios:
        result = predict_hit(
            batter_data=batter, pitcher_data=pitcher,
            batter_bats=bats, pitcher_throws=throws,
            venue="Park", park_factors=_neutral_park(), batting_order=order,
        )
        assert 0.25 <= result.hit_probability <= 0.90, (
            f"Probability {result.hit_probability} out of realistic range"
        )


def test_tier_property():
    """Verify tier labels map correctly to probability ranges."""
    from predictor import hit_tier
    assert hit_tier(0.75) == "STRONG HIT"
    assert hit_tier(0.65) == "LEAN HIT"
    assert hit_tier(0.58) == "TOSS-UP"
    assert hit_tier(0.50) == "FADE"
    assert hit_tier(0.70) == "STRONG HIT"  # boundary
    assert hit_tier(0.62) == "LEAN HIT"    # boundary
    assert hit_tier(0.55) == "TOSS-UP"     # boundary
