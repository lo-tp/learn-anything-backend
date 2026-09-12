.PHONY: check lint format typecheck db-up db-down db-clean migrate revision
VENV := .venv/bin

check: lint typecheck
	@echo "✓ All checks passed"

lint:
	$(VENV)/ruff check .

format:
	$(VENV)/ruff format .

typecheck:
	$(VENV)/pyright .

db-up:
	podman compose up db

db-down:
	podman compose down

migrate:
	$(VENV)/alembic upgrade head

db-clean:
	podman exec learn-anything-backend-db-1 psql -U postgres -d learn_anything -c "TRUNCATE step_progress, step_materials, slide_contents, probe_questions, plans, sessions CASCADE;"

revision:
	$(VENV)/alembic revision --autogenerate -m "$(m)"
