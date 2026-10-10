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
| `TAVILY_API_KEY` | Web search for the plan graph's `research_topic` step, through Tavily's MCP server (#166). Unset — or `MOCK_LLM` — means no external search: planning proceeds as before | — |
| `TAVILY_MCP_URL` | Override the hosted Tavily MCP endpoint | `https://mcp.tavily.com/mcp/` |
| `DEV_MODE` | `true`/`1` skips the sign-in gate on the Session **write** endpoints (the reads are public — #178) | off |
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

## Deploy

This repository builds the artifact; it does not decide where it runs. That lives in
**[lo-tp/learn-anything-infra](https://github.com/lo-tp/learn-anything-infra)**:
the production overlay pins this image **by digest**, a migration Job gates every
rollout (`PLAN.md` M5), and the environment variables the app reads come from
Secret Manager, rendered into cluster Secrets at deploy time. Values are never
stored in this repository.

What happens here is `.github/workflows/build-image.yml`: lint, typecheck and the
test suite; then build the image, **run it and assert its first routes** (a smoke
step), and only then publish it under a `sha-<commit>` tag. An image that never
answered a request never gets a tag.

Only `release` publishes, and `release` moves only by merging `main` into it — that
merge is the act that ships: the image is built, smoke-tested, published, pinned by
digest in the infrastructure repository, and deployed, in one chain. `main` is
checked and never builds an image.

The API is served at `https://api.lotp.xyz`; `GET /health` is the probe the
platform watches. It stays dependency-free on purpose — a slow database shows up as
request errors, not a restart loop.

Two build-time facts worth knowing:

1. **`prompts/` is a submodule of a private repository**, so a clone's credentials
   cannot reach it. CI initialises it from the pinned commit with `PROMPTS_TOKEN`,
   a GitHub token with read-only access to `lo-tp/learn-anything-prompts` and
   nothing else; locally, `git submodule update --init prompts`. The Dockerfile
   fails the build if the directory is missing rather than shipping an image that
   500s on every prompt.
2. **A plain `postgresql://…` URL is accepted** from any host: the `+psycopg`
   driver is added on read (`db.models.database_url_from_env`), which is how the
   same string works for the app and for `alembic`.


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
