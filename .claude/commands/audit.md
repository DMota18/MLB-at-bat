Run a full diagnostic audit of the baseball bot's prediction quality and pipeline health.

1. Run the test suite first: `pytest tests/ -v`
2. Check model drift and calibration by reading `drift.py` and running the drift check logic
3. Check pipeline health — verify Statcast data coverage, API circuit breaker state
4. Review recent prediction accuracy by tier (STRONG/LEAN/TOSS-UP/FADE)
5. Compare shadow model performance vs primary (head-to-head win rate, Brier score delta)
6. Check paper betting P&L and CLV trends
7. Report findings as a structured scorecard with actionable recommendations

Focus on: Is the model getting better or worse? Is data coverage stable? Are tiers still separating cleanly?
