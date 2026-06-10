Pull a stats report from the baseball bot prediction database.

Arguments: optional date range (e.g. "2026-06-01 2026-06-05") or number of days (e.g. "7"). If no arguments, defaults to last 7 days of available data.

Steps:
1. Sync the DB from EC2: `python stats.py --sync`
2. Run the report with the requested date range: `python stats.py --start <start> --end <end>`
3. Present the output to the user — do NOT reformat it, the script output is the report.

If the user asks for "last N days", use `python stats.py --sync --days N`.
If the user gives specific dates, use `python stats.py --sync --start YYYY-MM-DD --end YYYY-MM-DD`.
