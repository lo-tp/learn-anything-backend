"""Slide failure-class distribution over persisted failed compile attempts.

Reads ``failed_slides`` — one row per failed slide compile attempt, written by
the material graph's ``compile_slide`` node — reduces each stored error to the
message the sandbox actually reported, and maps it to exactly one class of
``SLIDE_FAILURE_CLASSES`` (``syntax_error / unknown_component /
truncated_output / sandbox_timeout / content_spec_invalid / other``).

This is the first honest answer to "how frequent are slide generation failures,
and why". The class lives in ``graphs/material/sandbox.py`` so the live graph
tags its own failures the same way; the report's real output is checked in as a
test fixture (``--dump-fixture``) and seeds the later slide-completeness work
(#165 self-repair, #175 few-shot exemplars, #168 preference capture).

Usage:
    .venv/bin/python scripts/classify_slide_failures.py
    .venv/bin/python scripts/classify_slide_failures.py --session <session_id>
    .venv/bin/python scripts/classify_slide_failures.py --since 2026-09-01
    .venv/bin/python scripts/classify_slide_failures.py \\
        --dump-fixture tests/fixtures/slide_failure_attempts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable regardless of the CWD.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env before importing the models (which read DATABASE_URL).
load_dotenv(_ROOT / ".env")

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from db.models import DBSession, FailedSlide, SessionFactory
from graphs.material.sandbox import (
    MAX_MATERIAL_ATTEMPTS,
    SLIDE_FAILURE_CLASSES,
    classify_compile_error,
    compile_error_for_feedback,
)

# How much of a failed slide's JSX to keep in a checked-in fixture: enough to
# recognise the shape of the failure (prose instead of a module, a bare
# identifier, a truncated module) without carrying whole slides into git.
FIXTURE_JSX_EXCERPT_CHARS = 120

# A slide is identified by (session_id, step_id, slide_index) — the same key
# ``scripts/slide_analysis.py`` uses.
SlideKey = tuple[str, str, int]


@dataclass(frozen=True)
class FailedAttempt:
    """One persisted failed slide compile attempt (a ``failed_slides`` row)."""

    id: int
    session_id: str
    step_id: str
    slide_index: int
    created_at: datetime
    error: str
    jsx: str

    @property
    def slide_key(self) -> SlideKey:
        return (self.session_id, self.step_id, self.slide_index)


@dataclass(frozen=True)
class ClassifiedAttempt:
    """A persisted attempt plus the one class it maps to.

    ``reduced_error`` is kept alongside so the report can show what actually
    sits behind a class, and a fixture can be replayed without the database.
    """

    attempt: FailedAttempt
    failure_class: str
    reduced_error: str


def classify_error(error: str) -> tuple[str, str]:
    """Map a persisted (raw, transport-wrapped) error to one failure class.

    Returns ``(failure_class, reduced_error)``. The reduction is the same one
    the graph uses for retry feedback, so the offline report and the live trace
    classify the same string.
    """
    reduced = compile_error_for_feedback(error)
    return classify_compile_error(reduced), reduced


def classify_attempts(attempts: list[FailedAttempt]) -> list[ClassifiedAttempt]:
    """Classify every attempt. Exactly one class per attempt, always."""
    out: list[ClassifiedAttempt] = []
    for attempt in attempts:
        failure_class, reduced = classify_error(attempt.error)
        out.append(
            ClassifiedAttempt(
                attempt=attempt, failure_class=failure_class, reduced_error=reduced
            )
        )
    return out


def fetch_attempts(
    factory: sessionmaker[DBSession],
    session_id: str | None = None,
    since: datetime | None = None,
) -> list[FailedAttempt]:
    """Read the persisted failed attempts, oldest row first.

    Row id order is write order, so a slide's attempts read 1 → 2 → 3.
    """
    with factory() as db:
        q = select(FailedSlide).order_by(FailedSlide.id)
        if session_id:
            q = q.where(FailedSlide.session_id == session_id)
        if since:
            q = q.where(FailedSlide.created_at >= since)
        rows = list(db.scalars(q))
    return [
        FailedAttempt(
            id=r.id,
            session_id=r.session_id,
            step_id=r.step_id,
            slide_index=r.slide_index,
            created_at=r.created_at,
            error=r.error,
            jsx=r.jsx,
        )
        for r in rows
    ]


# --- Report ---


def _pct(part: int, whole: int) -> str:
    return f"{(part / whole * 100) if whole else 0.0:.1f}%"


def _describe_filters(session_id: str | None, since: datetime | None) -> str:
    parts = ["table=failed_slides"]
    if session_id:
        parts.append(f"session={session_id}")
    if since:
        parts.append(f"since={since:%Y-%m-%d}")
    return ", ".join(parts)


def _slide_outcomes(
    rows: list[ClassifiedAttempt],
) -> dict[SlideKey, tuple[int, ClassifiedAttempt]]:
    """Per slide: how many failed rows it has and its LAST failed attempt."""
    counts: Counter[SlideKey] = Counter()
    last: dict[SlideKey, ClassifiedAttempt] = {}
    # ``rows`` is in id order, so the last row seen per key is the latest.
    for row in rows:
        key = row.attempt.slide_key
        counts[key] += 1
        last[key] = row
    return {key: (counts[key], last[key]) for key in counts}


def build_report(
    rows: list[ClassifiedAttempt], session_id: str | None, since: datetime | None
) -> str:
    """The failure-class distribution report (what the script prints)."""
    out: list[str] = []
    total = len(rows)
    out.append(
        f"=== Slide failure classes ({_describe_filters(session_id, since)}) ===\n"
    )
    if not rows:
        out.append("No failed attempts found.")
        return "\n".join(out)

    slides = _slide_outcomes(rows)
    sessions = {r.attempt.session_id for r in rows}
    first = min(r.attempt.created_at for r in rows)
    latest = max(r.attempt.created_at for r in rows)

    out.append(
        f"{total} failed attempt(s) across {len(slides)} slide(s) in "
        f"{len(sessions)} session(s)"
    )
    out.append(f"  span: {first:%Y-%m-%d %H:%M} → {latest:%Y-%m-%d %H:%M}\n")

    # --- 1. Distribution per attempt (and how many slides each class touched) ---
    by_class: Counter[str] = Counter(r.failure_class for r in rows)
    slides_by_class: dict[str, set[SlideKey]] = defaultdict(set)
    for r in rows:
        slides_by_class[r.failure_class].add(r.attempt.slide_key)

    unclassified = sorted(c for c in by_class if c not in SLIDE_FAILURE_CLASSES)
    verdict = (
        "yes" if not unclassified else "NO — " + ", ".join(unclassified)
    )
    out.append(f"Every attempt mapped to exactly one class: {verdict} ({total}/{total})\n")

    out.append("1. Failure-class distribution (per attempt)")
    out.append(f"   {'class':<22} {'attempts':>9} {'share':>7} {'slides touched':>15}")
    ranked = sorted(
        SLIDE_FAILURE_CLASSES,
        key=lambda c: (-by_class.get(c, 0), c),
    )
    for failure_class in ranked:
        count = by_class.get(failure_class, 0)
        out.append(
            f"   {failure_class:<22} {count:>9} {_pct(count, total):>7} "
            f"{len(slides_by_class.get(failure_class, ())):>15}"
        )
    out.append(f"   {'TOTAL':<22} {total:>9} {'100.0%':>7} {len(slides):>15}")
    out.append(
        "   (a slide whose attempts failed with different classes is counted "
        "under each;\n    the TOTAL slide count is distinct slides.)\n"
    )

    # --- 2. What sits behind each class ---
    out.append("2. The errors behind each class (reduced, most common first)")
    errors_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        errors_by_class[r.failure_class][" ".join(r.reduced_error.split())[:110]] += 1
    for failure_class in sorted(by_class, key=lambda c: -by_class[c]):
        out.append(f"   {failure_class} ({by_class[failure_class]})")
        for error, count in errors_by_class[failure_class].most_common(5):
            out.append(f"     {count:>3}x  {error}")
    out.append("")

    # --- 3. Slides, by whether they ever got a slide ---
    exhausted = {k: v for k, v in slides.items() if v[0] >= MAX_MATERIAL_ATTEMPTS}
    recovered = {k: v for k, v in slides.items() if v[0] < MAX_MATERIAL_ATTEMPTS}
    out.append("3. Slides, by outcome of their last failed attempt")
    out.append(f"   slides with failed attempts: {len(slides)}")
    exhausted_label = (
        f"exhausted all {MAX_MATERIAL_ATTEMPTS} attempts (placeholder/skipped)"
    )
    recovered_label = "fewer failures than attempts (a later attempt produced a slide)"
    for label, group in ((exhausted_label, exhausted), (recovered_label, recovered)):
        out.append(f"   {label}: {len(group)}")
        final = Counter(row.failure_class for _, row in group.values())
        for failure_class, count in final.most_common():
            out.append(f"     {failure_class:<22} {count}")
    out.append(
        "   note: failed_slides is append-only across regenerations, so a step\n"
        "   regenerated after the fact adds rows under the same slide key."
    )
    return "\n".join(out)


# --- Fixture ---


def dump_fixture(rows: list[ClassifiedAttempt], path: Path) -> None:
    """Write a real sample of classified attempts as a test fixture.

    What is checked in is the whole real signal set, not a hand-picked subset:
    ``distribution``/``attempt_count``/``slide_count`` are the counts over every
    persisted attempt, and ``attempts`` carries one entry per **distinct stored
    error** with a ``count`` of how many attempts shared it — so the counts add
    back up to ``attempt_count`` and a test re-classifying those errors pins the
    real distribution without 152 near-identical rows in git.

    The ``failure_class`` recorded per entry is what the shipped classifier
    produced on real data: a regression pin, not a hand-labelled oracle. The raw
    transport-wrapped error is kept so a test exercises the full reduce →
    classify path. The JSX is kept only as a short excerpt and its length —
    enough to recognise the shape of a failure (prose instead of a module, a
    bare identifier, a module that stops mid-tag); the later work that needs
    whole slides (#168, #175) captures those itself.
    """
    distribution = Counter(r.failure_class for r in rows)
    seen: dict[str, tuple[ClassifiedAttempt, int]] = {}
    for row in rows:
        first, count = seen.get(row.attempt.error, (row, 0))
        seen[row.attempt.error] = (first, count + 1)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": (
            "failed_slides rows from the local development database, read by "
            "scripts/classify_slide_failures.py --dump-fixture"
        ),
        "note": (
            "Real output, not hand-written: one entry per distinct stored "
            "error, with count = how many attempts shared it, so the counts "
            "sum to attempt_count. failure_class is what the shipped "
            "classifier produced on real data — a regression pin. "
            "jsx_excerpt is the first "
            f"{FIXTURE_JSX_EXCERPT_CHARS} characters of what the model "
            "returned, kept to show the shape of the failure; the prompts are "
            "left out (learner-specific and large)."
        ),
        "attempt_count": len(rows),
        "slide_count": len(_slide_outcomes(rows)),
        "distribution": {
            failure_class: distribution.get(failure_class, 0)
            for failure_class in SLIDE_FAILURE_CLASSES
        },
        "attempts": [
            {
                "id": r.attempt.id,
                "session_id": r.attempt.session_id,
                "step_id": r.attempt.step_id,
                "slide_index": r.attempt.slide_index,
                "created_at": r.attempt.created_at.isoformat(),
                "error": r.attempt.error,
                "failure_class": r.failure_class,
                "count": count,
                "jsx_chars": len(r.attempt.jsx),
                "jsx_excerpt": " ".join(r.attempt.jsx.split())[
                    :FIXTURE_JSX_EXCERPT_CHARS
                ],
            }
            for r, count in seen.values()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(payload['attempts'])} distinct stored error(s) covering "
        f"{len(rows)} attempt(s) to {path}"
    )


def _parse_since(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        sys.exit(f"Invalid --since date: {value!r} (expected YYYY-MM-DD)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify persisted failed slide compile attempts and "
        "report the failure-class distribution.",
    )
    parser.add_argument("--session", help="restrict to one session_id")
    parser.add_argument(
        "--since", help="only attempts created on/after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dump-fixture",
        metavar="PATH",
        help="also write the classified sample to PATH as a test fixture",
    )
    args = parser.parse_args()

    since = _parse_since(args.since) if args.since else None
    rows = classify_attempts(fetch_attempts(SessionFactory, args.session, since))
    print(build_report(rows, args.session, since))
    if args.dump_fixture:
        dump_fixture(rows, Path(args.dump_fixture))


if __name__ == "__main__":
    main()
