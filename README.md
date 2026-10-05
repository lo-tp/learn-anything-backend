# Learn Anything Backend

A FastAPI + LangGraph backend that generates personalized learning plans, probes learner knowledge, and builds interactive slide materials.

## Prerequisites

- **Python 3.13+**
- **uv** (package manager) — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** or **Podman** (for the Postgres database)
- An **OpenAI API key** (for LLM calls)

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd learn-anythin-backend

# 2. Create a virtual environment and install dependencies
uv sync

# 3. Set up environment variables
cp .env.example .env
# Edit .env — at minimum set DATABASE_URL (see below)

# 4. Start the database
make db-up

# 5. Run migrations
make migrate

# 6. Install the pre-push hook (ruff + pyright + pytest gate)
make install-hooks

# 7. Run the server
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | `postgresql+psycopg://postgres:postgres@localhost:5432/learn_anything` |
| `PORT` | API server port | `8000` |
| `FRONTEND_DOMAIN` | Allowed CORS origin | `http://localhost:3000` |
| `SANDBOX_URL` | Esbuild sandbox URL for slide compilation | `http://localhost:8080` |
| `SANDBOX_SERVICE_TOKEN` | Shared token for the internal `/slides` gate (#103) | — |
| `JWT_SECRET` | Shared sign-in token secret; **must match the frontend** | — |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | OpenAI-compatible LLM endpoint | `gpt-4o-mini` at api.openai.com |
| `DEV_MODE` | `true`/`1` skips the sign-in gate on session endpoints | off |
| `MOCK_LLM` | `true`/`1` runs the pre-material phases without an LLM, on an in-memory DB | off |
| `LA_PROMPTS_STUB` | `1` uses the public prompt stubs when `prompts/` is not checked out | off |
| `MAX_PROBE_QUESTIONS` | Max questions per probe session | `10` |
| `MAX_MATERIAL_ATTEMPTS` | Max compile retry attempts per slide | `3` |
| `LOG_LEVEL` | Logging level (`INFO`, `DEBUG`, etc.) | `INFO` |

## Database

Postgres 17 runs in a container via `docker-compose.yml` (or `podman-compose`):

```bash
make db-up       # start the DB
make db-down     # stop the DB
make db-clean    # truncate all tables (dev only)
make migrate     # run alembic migrations
make revision    # autogenerate a new migration (pass -m "message")
```

## Deploy (Render)

`render.yaml` is a [Render Blueprint](https://render.com/docs/blueprint-spec): one
web service plus a Postgres database. Apply it once with
`https://dashboard.render.com/blueprint/new?repo=https://github.com/lo-tp/learn-anything-backend`.
Deploys stay manual on purpose (`autoDeploy: false`): after pushing to `main`, publish
with **Manual Deploy → Deploy latest commit** on the service page.

The build runs `scripts/build.sh`, which does three things:

1. **Fetches the private prompt templates.** `prompts/` is a submodule of a
   private repo, so host clone credentials cannot reach it. `build.sh` reads the
   pinned commit out of this repo's tree and fetches exactly that commit with
   `PROMPTS_TOKEN` — a GitHub token with **read-only** access to
   `lo-tp/learn-anything-prompts` and nothing else.
2. Installs runtime dependencies with `uv sync --frozen --no-install-project --no-dev`.
3. Applies migrations (`alembic upgrade head`). Render only offers
   `preDeployCommand` on paid plans, so this lives in the build instead.

Fill the `sync: false` values in the Render Dashboard before the first deploy:
`PROMPTS_TOKEN`, `JWT_SECRET` (same value as the frontend), `SANDBOX_SERVICE_TOKEN`,
`SANDBOX_URL`, `FRONTEND_DOMAIN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`.
A plain `postgresql://...` from any host works: the `+psycopg` driver is added on
read (`db.models.database_url_from_env`). `GET /health` is the health-check probe.


## Make Targets

| Target | Description |
|---|---|
| `make check` | Run lint + typecheck |
| `make lint` | Ruff lint |
| `make format` | Ruff format |
| `make typecheck` | Pyright type check |
| `make test` | Run unit tests |
| `make coverage` | Run tests with coverage report |
| `make db-up` / `db-down` | Start/stop Postgres container |
| `make migrate` | Apply DB migrations |
| `make revision` | Generate a new alembic revision |
| `make install-hooks` | Symlink the pre-push hook into `.git/hooks/` |

## Pre-Push Hook

After running `make install-hooks`, every `git push` will run:

1. `ruff check .` — lint
2. `pyright` — type check
3. `pytest tests/ -x -q` — unit tests

If any check fails, the push is blocked. On a fresh clone, run `make install-hooks` once.

## Project Structure

```
├── main.py              # FastAPI app entry point
├── llm.py              # LLM client factory
├── language.py         # Language detection & localization helpers
├── graphs/             # LangGraph workflows
│   ├── common.py       # Shared helpers (structured_invoke, etc.)
│   ├── clarify/        # Goal clarification graph
│   ├── probe/          # Knowledge probing graph
│   ├── plan/           # Learning plan generation graph
│   └── material/       # Slide material generation graph
├── routers/            # FastAPI route handlers
├── db/                 # SQLAlchemy models & session
├── alembic/            # Database migrations
├── scripts/pre-push    # Tracked pre-push hook script
├── tests/              # Unit tests (mirrors graphs/ structure)
└── Makefile            # Dev workflow targets
```

## Tests

```bash
make test       # run all tests
make coverage   # run with coverage report
```

Tests live in `tests/graphs/<name>/` and mirror the `graphs/<name>/` implementation structure.
