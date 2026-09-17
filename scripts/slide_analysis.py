"""Slide-generation time/failure stats from the database.

Reads ``graph_stage_timings`` (per-stage wall-clock durations) and
``failed_slides`` (one row per failed compile attempt) and prints:

  1. Duration stats per stage (count / min / mean / p50 / p90 / max).
  2. write_slide duration by attempt number (1st = most creative, ...).
  3. Retry/failure stats: how many slides needed 1, 2, 3+ attempts,
     overall failure rate, and exhausted slides.
  4. Total time per step's material (generate_step rows).
  5. Most common compile errors. 6. Failed attempts over time.

Usage:
    .venv/bin/python scripts/slide_analysis.py
    .venv/bin/python scripts/slide_analysis.py --session <session_id>
    .venv/bin/python scripts/slide_analysis.py --graph material --since 2026-09-01
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable regardless of the CWD.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env before importing the models (which read DATABASE_URL).
load_dotenv(_ROOT / ".env")

from sqlalchemy import select

from db.models import (
    FailedSlide,
    GraphStageTiming,
    SessionFactory,
)
from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a non-empty list."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _fmt_duration(s: float) -> str:
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.1f}s"


def _duration_stats(durations: list[float]) -> str:
    if not durations:
        return "  (no data)"
    return (
        f"n={len(durations):<4} "
        f"min={_fmt_duration(min(durations)):>7} "
        f"mean={_fmt_duration(sum(durations) / len(durations)):>7} "
        f"p50={_fmt_duration(_percentile(durations, 50)):>7} "
        f"p90={_fmt_duration(_percentile(durations, 90)):>7} "
        f"max={_fmt_duration(max(durations)):>7}"
    )


def _slide_key(row: GraphStageTiming) -> tuple[str, str, int | None]:
    """(session_id, step_id, slide_index) identity of a slide."""
    ctx = row.context or {}
    return (row.session_id, str(ctx.get("step_id", "")), ctx.get("slide_index"))


def analyze(session_id: str | None, graph: str, since: datetime | None) -> None:
    with SessionFactory() as db:
        q = select(GraphStageTiming).where(GraphStageTiming.graph == graph)
        if session_id:
            q = q.where(GraphStageTiming.session_id == session_id)
        if since:
            q = q.where(GraphStageTiming.created_at >= since)
        # Row (id) order = execution order, so per-step "latest row wins"
        # for re-run history is well-defined.
        q = q.order_by(GraphStageTiming.id)
        timings: list[GraphStageTiming] = list(db.scalars(q))

        qf = select(FailedSlide)
        if session_id:
            qf = qf.where(FailedSlide.session_id == session_id)
        if since:
            qf = qf.where(FailedSlide.created_at >= since)
        failures: list[FailedSlide] = list(db.scalars(qf))

    filter_desc = ", ".join(
        filter(
            None,
            [
                f"session={session_id}" if session_id else None,
                f"graph={graph}",
                f"since={since:%Y-%m-%d}" if since else None,
            ],
        )
    )
    print(f"=== Slide generation analysis ({filter_desc}) ===\n")

    if not timings and not failures:
        print("No rows found.")
        return

    # --- 1. Duration stats per stage ---
    by_stage: dict[str, list[float]] = {}
    for t in timings:
        by_stage.setdefault(t.stage, []).append(t.duration_seconds)

    print("1. Duration by stage")
    print(f"  {'stage':<24} {'stats'}")
    for stage in sorted(by_stage, key=lambda s: -sum(by_stage[s])):
        print(f"  {stage:<24} {_duration_stats(by_stage[stage])}")
    print()

    # --- 2. write_slide duration by attempt number ---
    write_by_attempt: dict[int, list[float]] = {}
    for t in timings:
        if t.stage != "write_slide":
            continue
        attempt = (t.context or {}).get("attempt") or 1
        write_by_attempt.setdefault(int(attempt), []).append(t.duration_seconds)

    print("2. write_slide duration by attempt number")
    for attempt in sorted(write_by_attempt):
        print(f"  attempt {attempt}: {_duration_stats(write_by_attempt[attempt])}")
    if not write_by_attempt:
        print("  (no write_slide rows)")
    print()

    # --- 3. Retry / failure stats ---
    # A "slide" is identified by (session_id, step_id, slide_index); the
    # attempt number on its write_slide rows tells us how many drafts it
    # needed. Slides with failed_slides rows but no matching write_slide
    # timing row (e.g. truncated data) still count as failed slides.
    write_attempts: dict[tuple[str, str, int | None], int] = {}
    for t in timings:
        if t.stage != "write_slide":
            continue
        key = _slide_key(t)
        attempt = int((t.context or {}).get("attempt") or 1)
        write_attempts[key] = max(write_attempts.get(key, 0), attempt)

    failed_slides: set[tuple[str, str, int | None]] = {
        (f.session_id, f.step_id, f.slide_index) for f in failures
    }
    all_slides = set(write_attempts) | failed_slides

    total_slides = len(all_slides)
    first_try = sum(
        1
        for k in all_slides
        if write_attempts.get(k, 1) == 1 and k not in failed_slides
    )
    attempts_dist = Counter(write_attempts.get(k, 1) for k in all_slides)
    # Slides that had at least one failed compile attempt.
    failed_count = len(failed_slides)

    # Slides that exhausted every attempt (never generated) vs slides that
    # eventually succeeded on a retry.
    failed_rows_by_slide = Counter(
        (f.session_id, f.step_id, f.slide_index) for f in failures
    )
    exhausted = sum(
        1
        for k in all_slides
        if failed_rows_by_slide.get(k, 0) >= MAX_MATERIAL_ATTEMPTS
    )

    print("3. Retry / failure stats")
    print(f"  total slides:        {total_slides}")
    print(f"  failed slides:       {failed_count} "
          f"({(failed_count / total_slides * 100) if total_slides else 0:.1f}%)")
    print(f"  exhausted slides:    {exhausted} "
          f"({(exhausted / total_slides * 100) if total_slides else 0:.1f}%) "
          f"(failed all {MAX_MATERIAL_ATTEMPTS} tries, never generated)")
    print(f"  failed attempts:     {len(failures)} rows in failed_slides")
    print("  slides by attempts needed:")
    for n in sorted(set(attempts_dist) | {1, 2, 3}):
        count = attempts_dist.get(n, 0)
        if count or (n == 1 and first_try):
            label = f"{n}" if n < 3 else "3+"
            print(f"    {label}: {count}")
    print()

    # --- 4. Total time per step's material ---
    # One generate_step row per completed step, written by the driver with
    # the wall time from first graph invoke to finalization.
    step_totals: dict[tuple[str, str], float] = {}
    for t in timings:
        if t.stage != "generate_step":
            continue
        ctx = t.context or {}
        key = (t.session_id, str(ctx.get("step_id", "")))
        # A re-run appends a fresh row; show the latest one per step.
        step_totals[key] = t.duration_seconds

    if step_totals:
        print("4. Total time per step's material (generate_step)")
        total = sum(step_totals.values())
        for (sid, step_id), dur in sorted(
            step_totals.items(), key=lambda kv: -kv[1]
        ):
            print(f"  session={sid[:8]}  step={step_id:<20} {_fmt_duration(dur)}")
        print(f"  {'':<13} {len(step_totals)} steps, {_fmt_duration(total)} total")
        print()

    # --- 5. Most common compile errors ---
    if failures:
        print("5. Top compile errors (truncated to 100 chars)")
        errors = Counter(
            " ".join(f.error.split())[:100] for f in failures
        )
        for error, count in errors.most_common(10):
            print(f"  {count:>3}x  {error}")
        print()

    # --- 6. Failures over time ---
    if failures:
        print("6. Failed attempts over time (newest first)")
        latest = max(f.created_at for f in failures)
        span = latest - min(f.created_at for f in failures)
        print(f"  span: {min(f.created_at for f in failures):%Y-%m-%d %H:%M} → "
              f"{latest:%Y-%m-%d %H:%M} ({_fmt_duration(span.total_seconds())})")
        for f in sorted(failures, key=lambda f: f.created_at, reverse=True)[:10]:
            print(
                f"  {f.created_at:%Y-%m-%d %H:%M}  "
                f"session={f.session_id[:8]}  step={f.step_id[:20]}  "
                f"slide={f.slide_index}  err={' '.join(f.error.split())[:60]}"
            )


def _parse_since(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        sys.exit(f"Invalid --since date: {value!r} (expected YYYY-MM-DD)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="restrict to one session_id")
    parser.add_argument(
        "--graph", default="material",
        help="graph name to filter (default: material)",
    )
    parser.add_argument(
        "--since", help="only rows created on/after this date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    analyze(
        session_id=args.session,
        graph=args.graph,
        since=_parse_since(args.since) if args.since else None,
    )


if __name__ == "__main__":
    main()
