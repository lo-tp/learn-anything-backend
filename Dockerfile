# syntax=docker/dockerfile:1
#
# The Learn Anything API.
#
# What this file deliberately does *not* do:
#
# * **Fetch anything at build time except Python packages.** `prompts/` is a git
#   submodule pointing at a private repository, and it arrives with the source —
#   CI initialises it from the pinned commit, a local checkout already has it. The
#   alternative (cloning it inside the build with a token) needs `git` installed,
#   which means an `apt-get` step: 13 MB of Debian package index before a single
#   line of this app is looked at. On a normal network that is noise; here it was
#   most of the build. So the build trusts the tree and *checks* it, loudly, below.
# * **Run migrations.** An earlier host ran `alembic upgrade head` inside the
#   build because its free plan had no pre-deploy command. Here, migrations belong
#   to a Job that gates the rollout, so building an image never touches the
#   database. See lo-tp/learn-anything-infra, PLAN.md M5.

# ── Dependencies ───────────────────────────────────────────────
FROM python:3.13-slim AS deps

# Where uv creates the virtualenv the runtime stage copies. Without it uv builds
# `.venv` at `/` and the copy below finds nothing.
WORKDIR /app

# From uv.lock, nothing dev, the project itself left uninstalled because the app
# is run from source (`uvicorn main:app`). `uv` itself is not pinned — `--frozen`
# is what makes the result deterministic, so a newer uv changes the resolver, not
# the package set.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-install-project --no-dev

# ── Runtime ────────────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# No compiler toolchain on purpose: psycopg uses its [binary] extra and
# argon2-cffi ships wheels, so build-essential would be most of this image and
# nothing in it would be compiled.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY . /app/

# The guard is the whole point of the design above: if the submodule did not come
# with the source, fail here rather than shipping an image that 500s on every
# prompt render. The same guard a local build needs: init the submodule first.
RUN test -f prompts/probe.py || { \
      echo "build context has no prompts/ — the submodule was not initialised."; \
      echo "locally: git submodule update --init prompts"; \
      echo "in CI:   the checkout step must initialise it (see .github/workflows/build-image.yml)"; \
      exit 1; }

RUN chown -R app:app /app

# Numeric, not `app`. Kubernetes' `runAsNonRoot: true` verifies the image's user
# numerically and refuses a name it cannot resolve ("image has non-numeric user
# (app), cannot verify user is non-root"), which turns a working container into a
# pod stuck in CreateContainerConfigError. The uid is already fixed by useradd
# above; naming it here too would only be a second spelling of the same fact.
USER 10001

EXPOSE 8000

# `exec` so uvicorn, not a shell, is PID 1: SIGTERM from Kubernetes has to reach
# the server or the graceful shutdown in a rollout never happens.
# No HEALTHCHECK — liveness and readiness are declared once, in the manifests.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
