# Agent Instructions

When the user says "issues" or "the issues", they mean the GitHub issues at:
https://github.com/lo-tp/learn-anything/issues

Use the `gh` CLI to fetch issues from that repository unless the user specifies otherwise. Example: `gh issue view 64 --repo lo-tp/learn-anything`

## Domain docs

**All context and ADR documentation lives in the main repo `lo-tp/learn-anything` — not in this backend repo.**

- `CONTEXT.md` — the domain glossary
- `docs/adr/` — architecture decision records

Fetch and edit them via `gh` against `lo-tp/learn-anything` (e.g. `gh api repos/lo-tp/learn-anything/contents/CONTEXT.md`). Do **not** create `CONTEXT.md` or `docs/adr/` in this backend repo.
