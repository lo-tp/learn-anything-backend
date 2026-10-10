# Agent Instructions

When the user says "issues" or "the issues", they mean the GitHub issues at:
https://github.com/lo-tp/learn-anything-frontend/issues

All issues — including backend ones — are filed in `lo-tp/learn-anything-frontend`. Use the `gh` CLI against that repository unless the user specifies otherwise. Example: `gh issue view 155 --repo lo-tp/learn-anything-frontend`

## Domain docs

**All context and ADR documentation lives in `lo-tp/learn-anything-frontend` — not in this backend repo.** (The old `lo-tp/learn-anything` repo is archived; it was moved there.)

- `CONTEXT.md` — the domain glossary
- `docs/adr/` — architecture decision records

Fetch and edit them via `gh` against `lo-tp/learn-anything-frontend` (e.g. `gh api repos/lo-tp/learn-anything-frontend/contents/CONTEXT.md`). Do **not** create `CONTEXT.md` or `docs/adr/` in this backend repo. The infrastructure context has its own glossary and ADRs in `lo-tp/learn-anything-infra`.

## Shipping

This repo follows Git Flow, adapted to its pipeline:

- `main` — the stable branch. Checked (ruff, pyright, pytest); never builds
  an image. What lands here is ready to ship.
- `develop` — the integration branch. All development work happens here;
  feature work never lands on `main` directly.
- `feature/<slug>` — cut from `develop` for each feature or ticket; merged
  back into `develop` when done.
- `hotfix/<slug>` — cut from `main` when what is serving is broken; merged
  back into `main` and into `develop`.
- `release` — this repo's release step is a persistent protected branch,
  not a per-release branch: merging `main` into `release` is the act that
  ships. The image is built, its smoke step asserts `/health`, it is
  published under a `sha-<commit>` tag, and `lo-tp/learn-anything-infra`
  pins the digest and deploys it — migrations first, without a human step
  in between.

So: feature work flows `feature/<slug>` → `develop` → `main`, hotfixes flow
`hotfix/<slug>` → `main` + `develop`, nothing lands on `release` directly, and
never expect a push to `main` to change what is serving. What production runs is recorded in that infrastructure repository's
`manifests/overlays/prod/kustomization.yaml`; reverting that pin commit is the
rollback. A migration that has run does not run backwards — a rollback across a
migration is a fix-forward.
