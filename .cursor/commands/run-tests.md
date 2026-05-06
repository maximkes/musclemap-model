# /run-tests

1. ruff check src/ scripts/ app/ tests/ — fix all errors first
2. pytest -q --tb=short
3. If failures: fix code (not tests), rerun
4. Report: N passed, N failed
