Compare shadow model performance against the primary prediction model.

1. Read `ab_testing.py` to understand the current shadow model configuration
2. Read `predictor_shadow.py` and `predictor_v4.py` to understand what each shadow tests differently
3. Query the predictions database to get head-to-head comparison stats:
   - Overlapping settled prediction count
   - Brier score: primary vs shadow
   - Log loss: primary vs shadow
   - Head-to-head "closer to outcome" win rate
   - Tier accuracy comparison (which model's tiers separate better)
   - Calibration comparison (ECE for each)
4. Provide a recommendation: Should the shadow be promoted to primary? What's the confidence level?
5. If promoting, list exactly what constants/logic would change and flag any risks

$ARGUMENTS can specify which shadow model to compare (default: all registered).
