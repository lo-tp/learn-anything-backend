# Agent Instructions

When the user says "issues" or "the issues", they mean the GitHub issues at:
https://github.com/lo-tp/learn-anything/issues

Use the `gh` CLI to fetch issues from that repository unless the user specifies otherwise. Example: `gh issue view 64 --repo lo-tp/learn-anything`

## Domain docs

**All context and ADR documentation lives in `lo-tp/learn-anything-frontend` — not in this backend repo.** (The old `lo-tp/learn-anything` repo is archived; it was moved there.)

- `CONTEXT.md` — the domain glossary
- `docs/adr/` — architecture decision records

Fetch and edit them via `gh` against `lo-tp/learn-anything-frontend` (e.g. `gh api repos/lo-tp/learn-anything-frontend/contents/CONTEXT.md`). Do **not** create `CONTEXT.md` or `docs/adr/` in this backend repo. The infrastructure context has its own glossary and ADRs in `lo-tp/learn-anything-infra`.

## Shipping

`main` is checked (ruff, pyright, pytest) and never builds an image. Merging `main`
into `release` is the act that ships: the image is built, its smoke step asserts
`/health`, it is published under a `sha-<commit>` tag, and
`lo-tp/learn-anything-infra` pins the digest and deploys it — migrations first,
without a human step in between. `release` is protected and moves only by merging
`main` into it.

So: never commit directly to `release`, and never expect a push to `main` to change
what is serving. What production runs is recorded in that infrastructure repository's
`manifests/overlays/prod/kustomization.yaml`; reverting that pin commit is the
rollback. A migration that has run does not run backwards — a rollback across a
migration is a fix-forward.
