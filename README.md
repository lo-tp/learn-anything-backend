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

### …and reachable from another device

Running the app on a laptop and using it on a phone needs two settings, because
the browser checks two different things.

```bash
# 1. the server must answer the network, not just the laptop
uv run uvicorn main:app --reload --host 0.0.0.0 --port $PORT   # or: python main.py, which already binds 0.0.0.0

# 2. the browser's origin must be allowed
CORS_ORIGINS=http://192.168.1.5:3000
```

`CORS_ORIGINS` adds to `FRONTEND_DOMAIN`; it never replaces it — the origin a
deployment is served from is the one that signs itself in, and swapping it out is
how a deploy quietly stops being able to. Each allowed origin shows up once in the
startup log (`CORS: answering 2 origin(s): …`), because a CORS refusal is reported
by the browser as a problem with the page, and the developer's phone is not the
laptop. A value with no scheme, or a `*`, is named in the log and dropped: a bare
`host:3000` can never match what a browser sends, and a wildcard is rejected by
browsers on a response that carries credentials — which every signed-in request
here does.

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | `postgresql+psycopg://postgres:postgres@localhost:5432/learn_anything` |
| `PORT` | API server port | `8000` |
| `FRONTEND_DOMAIN` | The origin this API answers first — the one the frontend is served from | `http://localhost:3000` |
| `CORS_ORIGINS` | Comma-separated origins **added** to it: another device on your network, a preview deploy | — |
| `SANDBOX_URL` | Esbuild sandbox URL for slide compilation | `http://localhost:8080` |
| `SANDBOX_SERVICE_TOKEN` | Shared token for the internal `/slides` gate (#103) | — |
| `JWT_SECRET` | Shared sign-in token secret; **must match the frontend** | — |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | OpenAI-compatible LLM endpoint | `gpt-4o-mini` at api.openai.com |
| `LLM_REQUEST_TIMEOUT_SECONDS` | How long one model request may take before this app gives up on it; read once at startup, because it belongs to the client (#194) | `120` |
| `LLM_MAX_RETRIES` | How many times a request that never answered is asked again; read once at startup | `2` |
| `TAVILY_API_KEY` | Web search for the plan graph's `research_topic` step, through Tavily's MCP server (#166). Unset — or `MOCK_LLM` — means no external search: planning proceeds as before | — |
| `TAVILY_MCP_URL` | Override the hosted Tavily MCP endpoint | `https://mcp.tavily.com/mcp/` |
| `DEV_MODE` | `true`/`1` skips the sign-in gate on the Session **write** endpoints (the reads are public — #178) | off |
| `MOCK_LLM` | `true`/`1` runs the pre-material phases without an LLM, on an in-memory DB seeded with one account — `mock@example.com` / `12345678`. Sign in with it first: mock mode opens the phase gates, but creating a Session still asks for a sign-in cookie (#145) | off |
| `LA_PROMPTS_STUB` | `1` uses the public prompt stubs when `prompts/` is not checked out | off |
| `MAX_PROBE_QUESTIONS` | Max questions per probe session | `10` |
| `MAX_MATERIAL_ATTEMPTS` | Max compile retry attempts per slide | `3` |
| `MAX_SLIDE_SELF_REPAIR_TURNS` | Model-driven repair turns one `write_slide` attempt makes against `validate_jsx` before the outer retry; `0` disables the inner loop (#165) | `2` |
| `LOG_LEVEL` | Logging level (`INFO`, `DEBUG`, etc.) | `INFO` |

### Model-call knobs — for every call, or for one stage

These are read **per call**, through `core/settings.py`: the plain name applies to
every model call, and the `<NAME>_<GRAPH>_<NODE>` form applies to one stage —
`plan.design_plan` → `…_PLAN_DESIGN_PLAN`, `material.write_slide` →
`…_MATERIAL_WRITE_SLIDE`. The per-stage name wins over the global one, and a
value the backend would reject (an effort outside the five it accepts, a cap that
is not a number, a temperature of 5) is logged and treated as unset. A stage is
named by its graph and node; the stages are the table in `core/reasoning.py` plus
the two slide-writing calls.

A sampling value may also be **a list**, which is how a stage that repeats its
call is retuned one repeat at a time — `material.write_slide` writes a slide
again, each attempt deliberately less creative, and a list is that ladder written
in `.env` instead of in `graphs/material/sandbox.py`:

```bash
LLM_TEMPERATURE_MATERIAL_WRITE_SLIDE=0.4,0.2,0.1   # attempt 1, 2, 3
LLM_TOP_K_MATERIAL_WRITE_SLIDE=25,15,10
```

The values are positional: a slide-writing call states which attempt it is and is
answered at that rung; a call past the end of the list keeps the last value, and
one below the start gets the first — the same clamping the code-stated ladder
already does. Only `write_slide` states a position today, so a list written for
another stage is logged and treated as unset — including a stage a graph loop asks
again round after round (`probe.generate_batch`), which would need its node to
carry the round the way that node carries its attempt. One value is a list of one:
the scalar and the list forms are the same mechanism.

| Variable | What the request carries | Default when unset |
|---|---|---|
| `REASONING_EFFORT` | `reasoning_effort` | the per-stage table in `core/reasoning.py` (`minimal` for the stages that only restate a decision, `low` for the ones that judge) |
| `LLM_MAX_OUTPUT_TOKENS` | `max_tokens` | the per-stage caps in `core/reasoning.py`, and the slide caps in `graphs/material/sandbox.py` |
| `LLM_TEMPERATURE` | `temperature` | nothing sent — the backend's own default, which on this llama.cpp server is already deterministic: measured, two identical requests with nothing in the body answered the same |
| `LLM_TOP_P` | `top_p` | nothing sent |
| `LLM_TOP_K` | `extra_body.top_k` | nothing sent |
| `LLM_MIN_P` | `extra_body.min_p` | nothing sent |
| `LLM_REPEAT_PENALTY` | `extra_body.repeat_penalty` | nothing sent |
| `LLM_PRESENCE_PENALTY` | `presence_penalty` | nothing sent |
| `LLM_FREQUENCY_PENALTY` | `frequency_penalty` | nothing sent |

The three nested under `extra_body` are not OpenAI parameters; the client
forwards them into the request body, which is what this project's llama.cpp
backend reads (`core/sampling.py` records which side of the wire each one travels
on, and measured what it found there).

Two things this app does **not** offer, on purpose: `LLM_SEED`, because measured
against this backend two requests with the same seed returned different
completions; and a knob per write *attempt* named one attempt at a time
(`…_MATERIAL_WRITE_SLIDE_2`) — a list is the whole ladder at once, which is the
shape the ladder actually is.

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
├── scripts/*.py        # One-shot DB analysis scripts (run directly: python scripts/<name>.py)
├── tests/              # Unit tests (mirrors graphs/ structure)
└── Makefile            # Dev workflow targets
```

## Tests

```bash
make test       # run all tests
make coverage   # run with coverage report
```

Tests live in `tests/graphs/<name>/` and mirror the `graphs/<name>/` implementation structure.
Tests for the one-shot scripts live in `tests/scripts/`; real sampled output checked in
for them lives in `tests/fixtures/`.
