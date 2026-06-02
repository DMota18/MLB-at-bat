Pre-deployment validation checklist for the baseball bot.

Run these checks in order and report pass/fail for each:

1. **Lint**: `ruff check .` — must pass with zero errors
2. **Type check**: `mypy *.py --ignore-missing-imports` — report any new errors
3. **Tests**: `pytest tests/ -v` — all 118+ tests must pass
4. **Deploy script sync**: Check that every `.py` module imported by `bot.py` is listed in the `scp` line of `deploy.sh`. Flag any missing files.
5. **Requirements sync**: Check that every `import` in the codebase has a matching entry in `requirements.txt` or is a stdlib module
6. **Model version**: Confirm `config.MODEL_VERSION` was bumped if any prediction weights or calibration constants changed since last deploy
7. **.env.example**: Confirm all env vars used in the code are documented in `.env.example`

Report a go/no-go decision with any blockers listed.
