.PHONY: check lint format typecheck
VENV := .venv/bin

check: lint typecheck
	@echo "✓ All checks passed"

lint:
	$(VENV)/ruff check .

format:
	$(VENV)/ruff format .

typecheck:
	$(VENV)/pyright .
