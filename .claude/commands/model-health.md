Quick model health check — is the bot performing as expected?

1. Run `python drift.py` or read the drift detection logic to check:
   - Brier score trend (7-day rolling vs 30-day baseline)
   - Tier separation (is STRONG still >> FADE?)
   - Calibration buckets (are 70% predictions hitting at ~67%?)
   - Data coverage (what % of predictions have Statcast data?)
2. Check the paper betting P&L trend (last 7 days vs overall)
3. Check if any circuit breakers are currently open (API failures)
4. Report a traffic-light status: GREEN (healthy) / YELLOW (watch) / RED (action needed)

This is a quick check — for deep analysis use `/project:audit`.
