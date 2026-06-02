Run backtesting analysis on the prediction model.

1. Read `backtest.py` and `walk_forward.py` to understand available backtesting strategies
2. Run the specified backtest: `python backtest.py $ARGUMENTS`
3. Analyze the output:
   - Overall accuracy and Brier score
   - Tier-level accuracy (STRONG/LEAN/TOSS-UP/FADE)
   - Calibration curve analysis
   - Any time periods where the model underperformed
4. Compare against live prediction performance from `tracker.py`
5. If the backtest reveals a gap vs live performance, investigate why (data pipeline differences, sample composition, etc.)

Default: run against the most recent 500 predictions if no arguments specified.
