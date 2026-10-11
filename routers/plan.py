"""Plan phase endpoints: generate, adjust, approve, and material polling.

The writes here are gated per route (#178); ``GET /sessions/{session_id}/materials``
— the generated deck — is a public read and declares no gate. Ownership begins
at the write, not the read.

Approving a plan that went through the adjust loop also records one
``plan_preferences`` row (#167): the version the learner rejected, the version
they chose, their adjustment text, and the prompt of the design pass that
produced the chosen version's content. That table is offline training data — no
response, phase or message in this file depends on it, and a failed write never
fails an approve.

Approving the plan schedules ``generate_materials`` as a FastAPI background
task: a per-step driver that commits one ``StepMaterial`` row (plus one
``SlideContent`` row per slide and ``FailedSlide`` rows for failed attempts)
per step. The per-slide retry loop lives inside the material graph
(``graphs/material/``); the driver simply invokes the graph once per step
and persists the results. The domain DB is the resume point — a step with an
existing ``StepMaterial`` row is skipped; a failure lands the session in the
``error`` phase (terminal, per design).

A slide that failed to compile and was then fixed also leaves one
``slide_preferences`` row (#168): the prompt context, the rejected JSX and the
JSX that compiled, with the sandbox as the annotator. That table is offline
training data — no response, phase or message depends on it, and its write is
best-effort so it can never cost a learner their material.
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
from core.mock_llm import is_mock_mode
from core.security import is_dev_mode, require_auth
from db import (
    FailedSlide,
    GraphStageTiming,
    Phase,
    Plan,
    PlanPreference,
    Session,
    SessionFactory,
    SlideContent,
    SlidePreference,
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


def _capture_slide_preferences(
    db: DBSession, session_id: str, step_id: str, pairs: list[dict]
) -> int:
    """Store the preference pairs this step's compiles annotated (#168).

    ``compile_slide`` pairs every slide that failed to compile with the attempt
    that fixed it — the esbuild sandbox is the annotator, so the pair needs no
    labelling. One row per pair: the prompt each half was written from, the two
    JSX candidates, and the sandbox error that chose between them. It is
    training data for offline preference optimization; a first-attempt success
    rejected nothing and leaves no pair to store.

    Called at each compile pause with the pairs not stored yet, in its own
    transaction and best-effort: nothing in the serving path reads this table,
    so it must not cost a learner the slide committed alongside it. Returns the
    number of rows written, so the caller only advances past pairs it actually
    stored.
    """
    try:
        for p in pairs:
            db.add(
                SlidePreference(
                    session_id=session_id,
                    step_id=step_id,
                    slide_index=p["index"] + 1,  # 1-based for DB, as failed_slides
                    rejected_attempt=p["rejected_attempt"],
                    chosen_attempt=p["chosen_attempt"],
                    prompt_context=p["prompt_context"],
                    chosen_prompt=p["chosen_prompt"],
                    rejected_jsx=p["rejected_jsx"],
                    rejected_error=p["rejected_error"],
                    chosen_jsx=p["chosen_jsx"],
                )
            )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Material generation: could not store slide_preferences for "
            "session %s step %s",
            session_id,
            step_id,
        )
        return 0
    return len(pairs)



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
    are cleaned up before re-invoking. ``slide_preferences`` rows are not: a
    pair that really happened stays valid training data even if the step is
    generated again (same append-only policy as the timing rows).
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
                # attempt rows from a re-run are kept). Preference pairs are
                # persisted at the same pause, but in their own best-effort
                # transaction, so a pair that fails to store cannot take the
                # slide with it (#168).
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
                pairs_persisted = 0
                while state.get("__interrupt__"):
                    slides = state.get("slides") or []
                    n = len(slides)
                    slide_id = f"{session_id}_{step['id']}_slide_{n}"
                    slide_ids.append(slide_id)
                    db.add(
                        SlideContent(
                            slide_id=slide_id,
                            session_id=session_id,
                            step_id=step["id"],
                            content=slides[-1],
                            is_placeholder=state.get(
                                "current_slide_is_placeholder", False
                            ),
                        )
                    )
                    # Persist any stage timings that have accumulated since the
                    # last commit (one row per stage execution, append-only).
                    timings = state.get("stage_timings") or []
                    for t in timings[timings_persisted:]:
                        db.add(
                            GraphStageTiming(
                                session_id=session_id,
                                graph="material",
                                stage=t["stage"],
                                context=t.get("context"),
                                duration_seconds=t["duration_seconds"],
                            )
                        )
                    timings_persisted = len(timings)
                    # Publish the new slide on the provisional row (reassign a
                    # fresh list so SQLAlchemy's JSON column detects the change).
                    provisional.slides = provisional.slides + [slide_id]
                    db.commit()
                    # A slide that failed and was just fixed is a preference
                    # pair the compiler annotated. Stored here, next to the
                    # slide it describes, so it survives the step crashing
                    # later; the pair's prompt is in hand at this moment and
                    # nowhere else once the next attempt overwrites it.
                    pairs = list(state.get("slide_preferences") or [])
                    if len(pairs) > pairs_persisted:
                        pairs_persisted += _capture_slide_preferences(
                            db, session_id, step["id"], pairs[pairs_persisted:]
                        )
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
                timings = result.get("stage_timings") or []
                for t in timings[timings_persisted:]:
                    db.add(
                        GraphStageTiming(
                            session_id=session_id,
                            graph="material",
                            stage=t["stage"],
                            context=t.get("context"),
                            duration_seconds=t["duration_seconds"],
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
                    "attempts, %d preference pairs, %d questions, %.1fs)",
                    step["id"],
                    len(slide_ids),
                    len(failed),
                    pairs_persisted,
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


def _capture_plan_preference(db: DBSession, plan_row: Plan, result: dict) -> None:
    """Record the preference pair an adjust loop produced, at approve (#167).

    A pair is the plan version the learner rejected, the one they chose, the
    adjustment text that got them there, and the prompt of the design pass that
    produced the chosen version's content — training data for offline preference
    optimization. (The render call that formatted that content into the stored
    body is a separate prompt and is not recorded; what matters for training is
    that the learner's instruction and the rejected plan are in it.) Nothing
    user-visible reads this table, so the write is best-effort: a failed insert
    is logged, never turned into a failed approve.

    No pair exists when nothing was rejected: ``prior_plan`` is in the graph
    state only for a pass that superseded an earlier version, so approving an
    unadjusted plan writes nothing. Nor when nothing was said: the pair is
    instruction-conditioned, and an adjustment carrying no words regenerated the
    plan without the learner asking for anything, so there is no instruction to
    pair with it. The phase gate means a session approves once, but mock mode
    leaves it in ``reviewing``, so a repeat approve must not record the same pair
    twice.
    """
    rejected = result.get("prior_plan")
    chosen = result.get("current_plan")
    adjustment = result.get("adjustment") or ""
    if rejected is None or chosen is None or not adjustment.strip():
        return
    try:
        if (
            db.query(PlanPreference)
            .filter_by(session_id=plan_row.session_id)
            .first()
            is not None
        ):
            return
        db.add(
            PlanPreference(
                session_id=plan_row.session_id,
                rejected_version=plan_row.version - 1,
                chosen_version=plan_row.version,
                prompt_context=result.get("prompt_context") or [],
                adjustment=adjustment,
                rejected_plan=rejected,
                chosen_plan=chosen,
            )
        )
        db.commit()
    except Exception:  # training data must not fail an approve
        db.rollback()
        logger.exception(
            "Could not record the plan preference pair for session %s",
            plan_row.session_id,
        )


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
            },
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan generation failed")

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
                update={"language": session.language or DEFAULT_LANGUAGE},
            ),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan adjustment failed")

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
            Command(resume={"action": "approve"}),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan approval failed")

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
        # #167: if this plan arrived at through the adjust loop, the loop
        # produced a preference pair. Capture it; the response is untouched.
        _capture_plan_preference(db, existing, result)

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
