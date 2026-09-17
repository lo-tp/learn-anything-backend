.PHONY: check lint format typecheck test coverage db-up db-down db-clean migrate revision install-hooks
VENV := .venv/bin

check: lint typecheck
	@echo "✓ All checks passed"

lint:
	$(VENV)/ruff check .

format:
	$(VENV)/ruff format .

typecheck:
	$(VENV)/pyright .

test:
	$(VENV)/python -m pytest tests/ -q

coverage:
	$(VENV)/python -m pytest tests/ --cov=graphs --cov-report=term-missing -q
	@echo "✓ Coverage report above"

db-up:
	podman compose up db

db-down:
	podman compose down

migrate:
	$(VENV)/alembic upgrade head

db-clean:
	podman exec learn-anything-backend-db-1 psql -U postgres -d learn_anything -c "TRUNCATE step_progress, step_materials, slide_contents, graph_stage_timings, probe_questions, plans, sessions CASCADE;"

revision:
	$(VENV)/alembic revision --autogenerate -m "$(m)"

install-hooks:
	@ln -sf ../../scripts/pre-push .git/hooks/pre-push
	@echo "✓ pre-push hook installed (symlink → scripts/pre-push)"
