"""Plan phase endpoints: generate, adjust, approve, and material polling.

The writes here are gated per route (#178); ``GET /sessions/{session_id}/materials``
— the generated deck — is a public read and declares no gate. Ownership begins
at the write, not the read.

Approving the plan schedules ``generate_materials`` as a FastAPI background
task: a per-step driver that commits one ``StepMaterial`` row (plus one
``SlideContent`` row per slide and ``FailedSlide`` rows for failed attempts)
per step. The per-slide retry loop lives inside the material graph
(``graphs/material/``); the driver simply invokes the graph once per step
and persists the results. The domain DB is the resume point — a step with an
existing ``StepMaterial`` row is skipped; a failure lands the session in the
``error`` phase (terminal, per design).

One slide, on its own, is the other half of that storage. Each ``SlideContent``
row keeps the content spec it was written from, so
``POST /sessions/{session_id}/slides/{slide_id}/regenerate`` can schedule
``regenerate_slide_material``: a background task that rebuilds everything
``write_slide`` needs from this database and runs the graph's own nodes over one
spec — no graph, no checkpointer, no interrupt (#197).
"""

import logging
import os
import threading
import time
from collections import deque
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from core.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    has_meaningful_signal,
    localize_status,
)
from core.llm import llm
from core.mock_llm import is_mock_mode
from core.security import is_dev_mode, require_auth
from db import (
    FailedSlide,
    GraphStageTiming,
    Phase,
    Plan,
    Session,
    SessionFactory,
    SlideContent,
    StepMaterial,
    get_db,
)
from graphs import (
    checkpointer,
    graph_config,
    material_graph,
    plan_graph,
    pre_material_llm,
)
from graphs.material.regenerate import regenerate_slide
from services.stage_timings import (
    empty_stage_timings,
    record_stage_timings,
    stage_timing_rows,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan"])

# The plan writes keep the DEV_MODE-bounded gate (#89): a local run can walk
# the phases without a cookie, production asks for one. It is declared per route
# rather than on the router because this router also serves the deck read —
# `GET /sessions/{session_id}/materials` — which is a public read (#178).
GATE_THE_WRITES = [Depends(require_auth)]

# One lock per session: a double-approve must not race check-then-insert on
# the shared in-memory SQLite connection. The guard protects lock creation.
_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


# --- Schemas ---


class AdjustIn(BaseModel):
    adjustment: str


class StepOut(BaseModel):
    id: str
    letter: str = ""  # display index (A, B, C, ...); empty for legacy plans
    title: str
    description: str
    depends_on: list[str]
    depth: int


class PlanBody(BaseModel):
    prose_summary: str
    dependency_dag: str
    steps: list[StepOut]


class PlanOut(BaseModel):
    phase: Phase
    plan: PlanBody


class ApproveOut(BaseModel):
    phase: Phase
    message: str


class RegenerateIn(BaseModel):
    # What the learner wants different about this slide. Optional, and it shapes
    # how the slide is rendered — not what the deck's plan asks it to cover.
    feedback: str | None = None


class RegenerateOut(BaseModel):
    slide_id: str
    message: str


class SummaryOut(BaseModel):
    step_id: str
    title: str
    key_points: list[str]


class SlideItem(BaseModel):
    type: Literal["slide"]
    slide_id: str


class QuestionItem(BaseModel):
    type: Literal["question"]
    id: str
    text: str
    options: list[str]
    correct_index: int
    explanation: str


MaterialItem = Annotated[
    SlideItem | QuestionItem,
    Field(discriminator="type"),
]


class MaterialOut(BaseModel):
    step_id: str
    summary: SummaryOut
    items: list[MaterialItem]


class MaterialsOut(BaseModel):
    phase: Phase
    generated_steps: list[MaterialOut]


def _session_lock(session_id: str) -> threading.Lock:
    """Get (or create) the per-session generation lock."""
    with _locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


# --- Background material generation driver ---


def generate_materials(session_id: str) -> None:
    """Generate material for every plan step, one committed row per step.

    Plain ``def`` so FastAPI runs it in the thread pool after the approve
    response is sent. Opens its OWN db session (the request's is closed by
    then). The DB is the resume point: a ``StepMaterial`` row for a step means
    it is done — skip it (but keep its summary in the accumulated context).

    Defensive guard (#118): mock mode never schedules this — a no-op even if
    it is somehow invoked.

    The per-slide retry loop lives inside the material graph; the driver
    streams the graph per step and persists each slide to the DB immediately
    after it compiles, so a crash mid-step loses at most the in-flight
    slide. Orphan SlideContent/FailedSlide rows from a previous partial run
    are cleaned up before re-invoking.
    """
    if is_mock_mode():
        return
    with _session_lock(session_id):
        started = time.monotonic()
        db = SessionFactory()
        try:
            session = db.get(Session, session_id)
            if session is None:
                logger.warning(
                    "Material generation: session %s not found, aborting",
                    session_id,
                )
                return
            steps = session.plan.steps if session.plan is not None else None
            boundary_map = session.boundary_map
            if not steps or boundary_map is None:
                logger.warning(
                    "Material generation: session %s has no plan or "
                    "boundary map, aborting",
                    session_id,
                )
                return

            # All materials are generated in the learner's language.
            language = session.language or DEFAULT_LANGUAGE
            logger.info(
                "Material generation started: session=%s, steps=%d",
                session_id,
                len(steps),
            )
            summaries: list[dict] = []
            for step in steps:  # already in dependency order
                existing = (
                    db.query(StepMaterial)
                    .filter_by(session_id=session_id, step_id=step["id"])
                    .first()
                )
                if existing is not None and existing.is_complete:
                    # Already generated — resume past it, but keep its summary.
                    logger.debug(
                        "Material generation: step %s already generated, skipping",
                        step["id"],
                    )
                    summaries.append(existing.summary)
                    continue

                step_started = time.monotonic()
                logger.debug(
                    "Material generation: generating step %s "
                    "(established concepts: %d)",
                    step["id"],
                    len(summaries),
                )
                # Clean up from a previous partial run of this step: a
                # provisional StepMaterial row (crashed before completing)
                # plus its orphan slide/failed rows. The step is re-run from
                # scratch.
                if existing is not None:
                    db.delete(existing)
                db.query(SlideContent).filter_by(
                    session_id=session_id,
                    step_id=step["id"],
                ).delete()
                db.query(FailedSlide).filter_by(
                    session_id=session_id,
                    step_id=step["id"],
                ).delete()
                db.commit()

                # Create a provisional StepMaterial row up front so the FE can
                # see this step appear and its slides land one by one. It is
                # filled in as slides compile and finalized (questions, summary,
                # is_complete=True) when the step finishes.
                provisional = StepMaterial(
                    session_id=session_id,
                    step_id=step["id"],
                    slides=[],
                    questions=[],
                    summary={
                        "step_id": step["id"],
                        "title": step.get("title", ""),
                        "key_points": [],
                    },
                    is_complete=False,
                )
                db.add(provisional)
                db.commit()

                # Run the graph with interrupt/resume: the graph pauses after
                # each successful compile (pause_after_compile); the driver
                # saves the slide to the DB and resumes. Each compiled slide is
                # also appended to the provisional row so the FE sees it.
                # Stage timings from state are persisted the same way: each
                # not-yet-saved entry becomes a GraphStageTiming row in the
                # same transaction as the slide commit (append-only — prior
                # attempt rows from a re-run are kept).
                config = graph_config(session_id, f"material:{step['id']}")
                state: dict = material_graph.invoke(
                    {
                        "step": step,
                        "established_concepts": summaries,
                        "learner_context": boundary_map,
                        "language": language,
                    },
                    config,
                )
                slide_ids: list[str] = []
                timings_persisted = 0
                while state.get("__interrupt__"):
                    slides = state.get("slides") or []
                    n = len(slides)
                    slide_id = f"{session_id}_{step['id']}_slide_{n}"
                    slide_ids.append(slide_id)
                    specs = state.get("slide_contents") or []
                    # The spec this slide was written from (#197). The graph's
                    # own cursor is the one that says which spec produced the
                    # JSX just compiled: the two disagree when a slide was
                    # skipped entirely, because the skipped one added nothing to
                    # `slides` and every later row shifts down by one.
                    spec_index = state.get("slide_index", 0) - 1
                    spec = specs[spec_index] if 0 <= spec_index < len(specs) else None
                    db.add(
                        SlideContent(
                            slide_id=slide_id,
                            session_id=session_id,
                            step_id=step["id"],
                            content=slides[-1],
                            content_spec=spec,
                            is_placeholder=state.get(
                                "current_slide_is_placeholder", False
                            ),
                        )
                    )
                    # Persist any stage timings that have accumulated since the
                    # last commit (one row per stage execution, append-only).
                    new_rows = stage_timing_rows(
                        session_id, "material", (state.get("stage_timings") or [])[timings_persisted:]
                    )
                    db.add_all(new_rows)
                    timings_persisted += len(new_rows)
                    # Publish the new slide on the provisional row (reassign a
                    # fresh list so SQLAlchemy's JSON column detects the change).
                    provisional.slides = provisional.slides + [slide_id]
                    db.commit()
                    state = material_graph.invoke(Command(resume=True), config)
                result = state

                # Finalize the provisional row and persist failed attempts and
                # any remaining stage timings in one transaction (the slides
                # are already committed above).
                failed = result.get("failed_attempts") or []
                for fa in failed:
                    db.add(
                        FailedSlide(
                            session_id=session_id,
                            step_id=step["id"],
                            slide_index=fa["index"] + 1,  # 1-based for DB
                            prompt=fa.get("prompt", ""),
                            jsx=fa["jsx"],
                            error=fa["error"],
                        )
                    )
                db.add_all(
                    stage_timing_rows(
                        session_id,
                        "material",
                        (result.get("stage_timings") or [])[timings_persisted:],
                    )
                )
                provisional.slides = slide_ids
                provisional.questions = result["questions"]
                provisional.summary = result["summary"]
                provisional.is_complete = True
                # Total wall time to generate this step's material: from the
                # first graph invoke to finalization, including the slide
                # commits between interrupts. Persisted as a generate_step
                # timing row alongside the per-stage rows (same append-only
                # history policy — a re-run appends a fresh row).
                step_total = time.monotonic() - step_started
                db.add(
                    GraphStageTiming(
                        session_id=session_id,
                        graph="material",
                        stage="generate_step",
                        context={
                            "step_id": step["id"],
                            "slide_index": None,
                            "attempt": None,
                        },
                        duration_seconds=step_total,
                    )
                )
                db.commit()
                summaries.append(result["summary"])
                logger.info(
                    "Material generation: step %s done (%d slides, %d failed "
                    "attempts, %d questions, %.1fs)",
                    step["id"],
                    len(slide_ids),
                    len(failed),
                    len(result["questions"]),
                    step_total,
                )

            session.phase = Phase.EXECUTING.value
            db.commit()
            logger.info(
                "Material generation complete: session=%s, %.1fs total",
                session_id,
                time.monotonic() - started,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "Material generation failed for session %s after %.1fs",
                session_id,
                time.monotonic() - started,
            )
            try:
                # Re-fetch: the rollback expired the objects loaded above.
                errored = db.get(Session, session_id)
                if errored is not None:
                    errored.phase = Phase.ERROR.value
                    db.commit()
            except Exception:
                logger.exception(
                    "Failed to record error state for session %s", session_id
                )
        finally:
            db.close()


# --- Single-slide regeneration driver (#197) ---


def _regeneration_inputs(
    db: DBSession, session: Session, slide: SlideContent
) -> tuple[dict, list[dict]] | None:
    """What this slide was written from, rebuilt from the domain DB.

    Returns ``(step, established_concepts)`` — the plan step the slide belongs
    to, and the summaries of every step before it, in plan order: exactly the
    values the material generation driver passed the graph on the way to this
    step. ``None`` when the plan no longer has that step, which is the case a
    plan adjustment can leave behind: a slide whose step is gone has no title,
    description or position to be rewritten from.

    A prior step with no row contributes nothing: a step that was never
    generated never established anything.
    """
    if session.plan is None:
        return None
    steps = session.plan.steps
    order = {step["id"]: i for i, step in enumerate(steps)}
    target = order.get(slide.step_id)
    if target is None:
        return None

    rank = len(order)
    rows = [row for row in db.query(StepMaterial).filter_by(session_id=session.session_id)]
    before = sorted(
        (row for row in rows if order.get(row.step_id, rank) < target),
        key=lambda row: (order.get(row.step_id, rank), row.id),
    )
    return steps[target], [row.summary for row in before]


def regenerate_slide_material(
    session_id: str, slide_id: str, feedback: str | None
) -> None:
    """Rewrite one slide from the content spec its own row stores.

    Plain ``def`` so FastAPI runs it in the thread pool after the 202 is sent,
    with its OWN db session. It opens with the one thing that used to be lost —
    ``SlideContent.content_spec`` — rebuilds the rest of what ``write_slide``
    needs from the plan, the prior steps' summaries, the session's boundary map
    and its language, and hands the result to :func:`graphs.material.regenerate`
    (which runs the material graph's own nodes over that one spec).

    The row is only replaced by a slide that actually compiled: a regeneration
    that exhausted its attempts ends with the placeholder stand-in, and
    exchanging a bad slide for a placeholder is not an improvement the learner
    asked for. The failed attempts and the stage timings are kept either way —
    append-only history, the same policy the deck driver follows.

    Takes the per-session generation lock: a slide is not rewritten while the
    deck it belongs to is being written, and two requests for the same slide do
    not both answer it.
    """
    if is_mock_mode():
        return
    with _session_lock(session_id):
        started = time.monotonic()
        db = SessionFactory()
        try:
            slide = db.get(SlideContent, slide_id)
            if slide is None or slide.session_id != session_id:
                # The deck changed under the request (a dev regeneration wipes
                # slides); there is nothing left to rewrite.
                logger.warning(
                    "Slide regeneration: slide %s is not part of session %s",
                    slide_id, session_id,
                )
                return
            session = db.get(Session, session_id)
            if session is None:
                logger.warning(
                    "Slide regeneration: session %s not found", session_id
                )
                return
            inputs = _regeneration_inputs(db, session, slide)
            if inputs is None:
                logger.warning(
                    "Slide regeneration: session %s has no plan step %s",
                    session_id, slide.step_id,
                )
                return
            step, established_concepts = inputs
            if slide.content_spec is None:
                # A row from before the spec was stored (#197): the only route
                # back for this slide is regenerating the whole deck.
                logger.warning(
                    "Slide regeneration: slide %s has no stored content spec",
                    slide_id,
                )
                return

            result = regenerate_slide(
                llm,
                step=step,
                content_spec=slide.content_spec,
                established_concepts=established_concepts,
                learner_context=session.boundary_map or {},
                language=session.language,
                feedback=feedback,
            )

            for attempt in result["failed_attempts"]:
                db.add(
                    FailedSlide(
                        session_id=session_id,
                        step_id=slide.step_id,
                        slide_index=attempt["index"] + 1,  # 1-based for DB
                        prompt=attempt.get("prompt", ""),
                        jsx=attempt["jsx"],
                        error=attempt["error"],
                    )
                )
            db.add_all(
                stage_timing_rows(session_id, "material", result["stage_timings"])
            )

            jsx = result["jsx"]
            if jsx is not None and not result["is_placeholder"]:
                slide.content = jsx
                # A placeholder row that regenerated into a real slide is now a
                # real slide, and the deck shows it again.
                slide.is_placeholder = False
                logger.info(
                    "Slide regenerated: session=%s slide=%s attempts=%d "
                    "(%d chars, %.1fs)",
                    session_id, slide_id, len(result["failed_attempts"]) + 1,
                    len(jsx), time.monotonic() - started,
                )
            else:
                logger.warning(
                    "Slide regeneration produced nothing usable for session=%s "
                    "slide=%s after %d attempt(s); keeping the slide the learner "
                    "has",
                    session_id, slide_id, len(result["failed_attempts"]),
                )
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Slide regeneration failed for session %s slide %s after %.1fs",
                session_id, slide_id, time.monotonic() - started,
            )
        finally:
            db.close()


# --- Helpers ---


def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _ordered_materials(session: Session) -> list[StepMaterial]:
    """StepMaterial rows in plan step order (deterministic).

    Rows whose step_id is not in the plan sort after the plan-ordered rows
    (by row id); if the session has no plan, rows keep insertion (row-id)
    order. Ordering is applied here, endpoint-side — the relationship has
    no order_by.
    """
    materials = list(session.materials)
    plan_steps = session.plan.steps if session.plan is not None else None
    if plan_steps is None:
        return sorted(materials, key=lambda m: m.id)
    order = {s["id"]: i for i, s in enumerate(plan_steps)}
    return sorted(
        materials,
        key=lambda m: (
            order[m.step_id] if m.step_id in order else len(plan_steps),
            m.id,
        ),
    )


def _cleanup_thread(session_id: str) -> None:
    """Best-effort: clear the plan checkpoint thread so the client can retry."""
    try:
        checkpointer.delete_thread(f"{session_id}:plan")
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to clean up plan thread")


def validate_plan(plan: dict) -> None:
    """Validate a rendered plan: non-empty steps, valid dep IDs, acyclic."""
    steps = plan.get("steps", [])
    if not steps:
        raise HTTPException(status_code=500, detail="Generated plan failed validation")

    ids = {s["id"] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            if dep not in ids:
                raise HTTPException(
                    status_code=500, detail="Generated plan failed validation"
                )

    # Kahn's algorithm for cycle detection
    in_degree: dict[str, int] = {s["id"]: 0 for s in steps}
    adj: dict[str, list[str]] = {s["id"]: [] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            adj[dep].append(s["id"])
            in_degree[s["id"]] += 1

    queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(ids):
        raise HTTPException(status_code=500, detail="Generated plan failed validation")


def _persist_plan(
    db: DBSession,
    session: Session,
    plan: dict,
    version: int,
    adjustments: list[str] | None = None,
) -> None:
    """Insert or update the Plan row for this session."""
    existing = session.plan
    if existing is None:
        row = Plan(
            session_id=session.session_id,
            version=version,
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=plan["steps"],
            adjustments=adjustments or [],
        )
        db.add(row)
    else:
        existing.version = version
        existing.prose_summary = plan["prose_summary"]
        existing.dependency_dag = plan["dependency_dag"]
        existing.steps = plan["steps"]
        if adjustments is not None:
            existing.adjustments = adjustments
    db.commit()


# --- Routes ---


@router.post(
    "/sessions/{session_id}/plan/generate",
    response_model=PlanOut,
    dependencies=GATE_THE_WRITES,
)
def generate_plan(session_id: str, db: DBSession = Depends(get_db)) -> PlanOut:
    """Trigger plan generation."""
    session = _get_session_or_404(db, session_id)
    if session.phase == Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail="Session is already reviewing — use /plan/adjust instead",
        )
    if session.phase != Phase.PLANNING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'planning'",
        )

    config = graph_config(session_id, "plan")

    try:
        result = plan_graph.invoke(
            {
                "goal": session.narrowed_goal or session.goal,
                "language": session.language or DEFAULT_LANGUAGE,
                "boundary_map": session.boundary_map or {},
                "research": None,
                "current_plan": None,
                "adjustment": None,
                "pass_count": 0,
                "stage_timings": empty_stage_timings(),
            },
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan generation failed")

    record_stage_timings(db, session_id, "plan", result)

    if "__interrupt__" not in result:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan generation did not pause as expected"
        )

    plan = result.get("current_plan")
    if plan is None:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan generation did not produce a plan"
        )

    try:
        validate_plan(plan)
    except HTTPException:
        _cleanup_thread(session_id)
        raise

    _persist_plan(db, session, plan, version=1)
    session.phase = Phase.REVIEWING.value
    db.commit()

    return PlanOut(
        phase=Phase.REVIEWING,
        plan=PlanBody(
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=[StepOut(**s) for s in plan["steps"]],
        ),
    )


@router.post(
    "/sessions/{session_id}/plan/adjust",
    response_model=PlanOut,
    dependencies=GATE_THE_WRITES,
)
def adjust_plan(
    session_id: str, body: AdjustIn, db: DBSession = Depends(get_db)
) -> PlanOut:
    """Submit a free-text adjustment and get the regenerated plan."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'reviewing'",
        )

    config = graph_config(session_id, "plan")

    # Follow the learner's language if the adjustment carries enough signal.
    if has_meaningful_signal(body.adjustment):
        session.language = detect_language(body.adjustment, pre_material_llm)
        db.commit()
    try:
        result = plan_graph.invoke(
            Command(
                resume={"action": "adjust", "text": body.adjustment},
                update={
                    "language": session.language or DEFAULT_LANGUAGE,
                    "stage_timings": empty_stage_timings(),
                },
            ),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan adjustment failed")

    record_stage_timings(db, session_id, "plan", result)

    if "__interrupt__" not in result:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan adjustment did not pause as expected"
        )

    plan = result.get("current_plan")
    if plan is None:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan adjustment did not produce a plan"
        )

    try:
        validate_plan(plan)
    except HTTPException:
        _cleanup_thread(session_id)
        raise

    existing = session.plan
    new_version = existing.version + 1 if existing else 1
    new_adjustments = (
        list(existing.adjustments) + [body.adjustment]
        if existing
        else [body.adjustment]
    )
    _persist_plan(db, session, plan, version=new_version, adjustments=new_adjustments)
    db.commit()

    return PlanOut(
        phase=Phase.REVIEWING,
        plan=PlanBody(
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=[StepOut(**s) for s in plan["steps"]],
        ),
    )


@router.post(
    "/sessions/{session_id}/plan/approve",
    response_model=ApproveOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def approve_plan(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
) -> ApproveOut:
    """Approve the plan, then start material generation in the background."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'reviewing'",
        )

    config = graph_config(session_id, "plan")

    try:
        result = plan_graph.invoke(
            Command(
                resume={"action": "approve"},
                update={"stage_timings": empty_stage_timings()},
            ),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan approval failed")

    record_stage_timings(db, session_id, "plan", result)

    plan = result.get("current_plan")
    if plan is None:
        raise HTTPException(
            status_code=500, detail="Plan approval did not produce a plan"
        )

    # Re-persist the final plan (idempotent)
    existing = session.plan
    if existing is not None:
        _persist_plan(
            db,
            session,
            plan,
            version=existing.version,
            adjustments=existing.adjustments,
        )

    if is_mock_mode():
        # #118: mock mode never advances into slide generation — the session
        # stays in REVIEWING and generate_materials is not scheduled.
        return ApproveOut(
            phase=Phase.REVIEWING,
            message="Plan approved. Slide generation is disabled in mock mode.",
        )

    session.phase = Phase.GENERATING.value
    db.commit()

    # Runs in the thread pool after the 202 is sent; opens its own db session.
    background_tasks.add_task(generate_materials, session_id)

    return ApproveOut(
        phase=Phase.GENERATING,
        # The confirmation is a user-facing reply, so it is in the learner's
        # language (falls back to the English text if translation fails).
        message=localize_status(
            pre_material_llm,
            session.language or DEFAULT_LANGUAGE,
            "Plan approved. Material generation started in the background.",
        ),
    )


@router.post(
    "/dev/sessions/{session_id}/regenerate",
    response_model=ApproveOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def dev_regenerate(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
) -> ApproveOut:
    """Dev-only: wipe existing materials and re-run generation.

    Requires the session to already have a plan and boundary_map
    (i.e. it has passed the planning phase). Skips all upstream phases.
    """
    # #118: dev regeneration is refused in mock mode (slide generation is
    # disabled there).
    if is_mock_mode():
        raise HTTPException(
            status_code=400, detail="Dev regeneration is disabled in mock mode."
        )
    if not is_dev_mode():
        raise HTTPException(status_code=404, detail="Not found")

    session = _get_session_or_404(db, session_id)
    if session.plan is None or session.boundary_map is None:
        raise HTTPException(
            status_code=409,
            detail="Session has no plan or boundary_map — cannot regenerate",
        )

    # Wipe existing materials so generation starts fresh.
    for m in list(session.materials):
        db.delete(m)
    for s in list(session.slide_contents):
        db.delete(s)

    # Delete material checkpoint threads so stale in-memory state
    # (e.g. old slide_contents/slides) does not resume into the fresh run.
    steps = session.plan.steps
    for step in steps:
        try:
            checkpointer.delete_thread(f"{session_id}:material:{step['id']}")
        except Exception:  # noqa: BLE001 — best-effort cleanup
            logger.warning(
                "Failed to delete material checkpoint thread for step %s",
                step["id"],
            )

    session.phase = Phase.GENERATING.value
    db.commit()

    background_tasks.add_task(generate_materials, session_id)
    return ApproveOut(
        phase=Phase.GENERATING,
        message="Material generation restarted (dev).",
    )


@router.post(
    "/sessions/{session_id}/slides/{slide_id}/regenerate",
    response_model=RegenerateOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def request_slide_regeneration(
    session_id: str,
    slide_id: str,
    background_tasks: BackgroundTasks,
    body: RegenerateIn | None = None,
    db: DBSession = Depends(get_db),
) -> RegenerateOut:
    """Regenerate one slide of a deck, leaving every other slide alone.

    A gated write (#178): it changes the deck, and the request that asks for it
    belongs to whoever is working on that Session.

    Everything needed to write the slide again is already in this backend: the
    content spec on the slide's own row (#197), the step in the plan, the prior
    steps' summaries, the boundary map, the language. That is what makes a
    single slide a self-contained operation instead of a reason to wipe the deck
    and re-run generation (``POST /dev/sessions/{id}/regenerate``).

    Answers ``202`` because writing a slide is several model calls: the work runs
    in the background and the row is replaced when a new slide compiles. A
    regeneration that exhausts its attempts changes nothing — the slide the
    learner has stays.
    """
    # #118: mock mode never generates slides, so there is no slide to rewrite.
    if is_mock_mode():
        raise HTTPException(
            status_code=400, detail="Slide generation is disabled in mock mode."
        )

    session = _get_session_or_404(db, session_id)
    slide = db.get(SlideContent, slide_id)
    if slide is None or slide.session_id != session_id:
        raise HTTPException(status_code=404, detail="Slide not found")
    # A placeholder slide is invisible outside dev mode here as everywhere else
    # in this API (GET /slides/{id}, the deck read): in production it was never
    # in the deck, so there is nothing on screen to ask about.
    if slide.is_placeholder and not is_dev_mode():
        raise HTTPException(status_code=404, detail="Slide not found")
    if session.phase == Phase.GENERATING.value:
        raise HTTPException(
            status_code=409,
            detail="Material generation is still running for this session",
        )
    if session.plan is None or _regeneration_inputs(db, session, slide) is None:
        raise HTTPException(
            status_code=409,
            detail="Session has no plan step for this slide — cannot regenerate",
        )
    if slide.content_spec is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This slide has no stored content spec — it was generated "
                "before specs were stored. Regenerate the whole deck instead."
            ),
        )

    feedback = ((body.feedback or "") if body is not None else "").strip()
    background_tasks.add_task(
        regenerate_slide_material, session_id, slide_id, feedback or None
    )
    return RegenerateOut(
        slide_id=slide_id,
        message=(
            "Regeneration started. The slide is replaced when a new one "
            "compiles."
        ),
    )


@router.get(
    "/sessions/{session_id}/materials",
    response_model=MaterialsOut,
)
def get_materials(session_id: str, db: DBSession = Depends(get_db)) -> MaterialsOut:
    """Poll material generation progress with full content.

    A public read (#178): this is the deck itself, and a Visitor opens it by
    address (ADR-0004). No sign-in dependency is declared on this route.
    """
    session = _get_session_or_404(db, session_id)

    try:
        phase = Phase(session.phase)
    except ValueError:
        logger.error("Session %s has unknown phase %r", session_id, session.phase)
        raise HTTPException(
            status_code=500,
            detail=f"Session {session_id} has unknown phase {session.phase!r}",
        )

    # Placeholder slides (a failed slide's stand-in) are always generated
    # and persisted, but are only returned here in dev mode — in production
    # the slot is omitted from the deck.
    if os.getenv("DEV_MODE") == "1":
        placeholder_ids: set[str] = set()
    else:
        placeholder_ids = {
            s.slide_id for s in session.slide_contents if s.is_placeholder
        }

    generated_steps = [
        MaterialOut(
            step_id=m.step_id,
            summary=SummaryOut(**m.summary),
            items=[
                *[
                    SlideItem(type="slide", slide_id=sid)
                    for sid in m.slides
                    if sid not in placeholder_ids
                ],
                *[QuestionItem(type="question", **q) for q in m.questions],
            ],
        )
        for m in _ordered_materials(session)
    ]

    return MaterialsOut(phase=phase, generated_steps=generated_steps)
