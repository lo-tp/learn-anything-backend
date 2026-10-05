#!/usr/bin/env bash
# Build steps for a hosted deploy (Render runs this as the build command).
#
# Three steps, in order:
#   1. fetch the private prompt templates (see the note below)
#   2. install runtime dependencies from uv.lock (dev tools excluded)
#   3. apply alembic migrations
#
# Step 3 runs here rather than as a Render `preDeployCommand` because that
# Blueprint field is unavailable on free plans; the build container has the
# same DATABASE_URL and network access as the runtime container.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# --- 1. Private prompt templates -------------------------------------------
# ``prompts/`` is a submodule pointing at a private repository. Host clone
# credentials do not reach that repo, but the pinned commit is recorded in
# this repo's tree, so fetch exactly that commit with a read-only token.
# Skipped when the templates are already present (local builds, or a host
# that did initialise the submodule).
if [ ! -f prompts/probe.py ]; then
  sha="$(git rev-parse :prompts)"
  : "${PROMPTS_TOKEN:?prompts/ is empty and PROMPTS_TOKEN is not set. \
Provide a GitHub token with read-only access to lo-tp/learn-anything-prompts.}"
  prompts_repo="${PROMPTS_REPO:-github.com/lo-tp/learn-anything-prompts}"

  rm -rf prompts
  git init -q prompts
  git -C prompts remote add origin "https://x-access-token:${PROMPTS_TOKEN}@${prompts_repo}.git"
  git -C prompts fetch -q --depth 1 origin "$sha"
  git -C prompts checkout -q FETCH_HEAD
  echo "✓ prompts at ${sha:0:10}"
fi

# --- 2. Dependencies -------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH; installing it for this build."
  pip install --quiet uv
fi
uv sync --frozen --no-install-project --no-dev

# --- 3. Migrations ---------------------------------------------------------
.venv/bin/alembic upgrade head
