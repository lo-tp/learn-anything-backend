"""Plan phase endpoints: generate, adjust, approve, and material polling.

Approving the plan schedules ``generate_materials`` as a FastAPI background
task: a per-step driver that commits one ``StepMaterial`` row (plus one
``SlideContent`` row per slide) per step. The domain DB is the resume point —
a step with an existing row is skipped; a failure lands the session in the
``error`` phase (terminal, per design).
"""

import logging
import threading
import time
from collections import deque

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, Plan, Session, SessionFactory, SlideContent, StepMaterial, get_db
from graphs import checkpointer, graph_config, material_graph, plan_graph

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan"])

# One lock per session: a double-approve must not race check-then-insert on
# the shared in-memory SQLite connection. The guard protects lock creation.
_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


# --- Schemas ---


class AdjustIn(BaseModel):
    adjustment: str


class StepOut(BaseModel):
    id: str
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


class QuestionOut(BaseModel):
    id: str
    text: str
    options: list[str]
    correct_index: int
    explanation: str


class MaterialOut(BaseModel):
    step_id: str
    slides: list[str]
    questions: list[QuestionOut]


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
    """
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

            logger.info(
                "Material generation started: session=%s, steps=%d",
                session_id, len(steps),
            )
            summaries: list[dict] = []
            for step in steps:  # already in dependency order
                existing = (
                    db.query(StepMaterial)
                    .filter_by(session_id=session_id, step_id=step["id"])
                    .first()
                )
                if existing is not None:
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
                    step["id"], len(summaries),
                )
                result = material_graph.invoke(
                    {
                        "step": step,
                        "established_concepts": summaries,
                        "learner_context": boundary_map,
                    }
                )

                # One SlideContent row per slide + one StepMaterial row, in the
                # same transaction — one durable checkpoint per step.
                slide_ids: list[str] = []
                for n, html in enumerate(result["slides"], start=1):
                    slide_id = f"{step['id']}_slide_{n}"
                    slide_ids.append(slide_id)
                    db.add(
                        SlideContent(
                            slide_id=slide_id,
                            session_id=session_id,
                            step_id=step["id"],
                            content=html,
                        )
                    )
                db.add(
                    StepMaterial(
                        session_id=session_id,
                        step_id=step["id"],
                        slides=slide_ids,
                        questions=result["questions"],
                        summary=result["summary"],
                    )
                )
                db.commit()
                summaries.append(result["summary"])
                logger.info(
                    "Material generation: step %s done (%d slides, %d questions, %.1fs)",
                    step["id"],
                    len(slide_ids),
                    len(result["questions"]),
                    time.monotonic() - step_started,
                )

            session.phase = Phase.EXECUTING.value
            db.commit()
            logger.info(
                "Material generation complete: session=%s, %.1fs total",
                session_id, time.monotonic() - started,
            )
        except Exception as exc:
            db.rollback()
            logger.exception(
                "Material generation failed for session %s after %.1fs",
                session_id, time.monotonic() - started,
            )
            try:
                # Re-fetch: the rollback expired the objects loaded above.
                errored = db.get(Session, session_id)
                if errored is not None:
                    errored.phase = Phase.ERROR.value
                    errored.error = f"Material generation failed: {exc}"
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
        raise HTTPException(
            status_code=500, detail="Generated plan failed validation"
        )

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
        raise HTTPException(
            status_code=500, detail="Generated plan failed validation"
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
)
def generate_plan(
    session_id: str, db: DBSession = Depends(get_db)
) -> PlanOut:
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
        raise HTTPException(
            status_code=500, detail="Plan generation failed"
        )

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

    try:
        result = plan_graph.invoke(
            Command(resume={"action": "adjust", "text": body.adjustment}),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan adjustment failed"
        )

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
    new_adjustments = list(existing.adjustments) + [body.adjustment] if existing else [body.adjustment]
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
        raise HTTPException(
            status_code=500, detail="Plan approval failed"
        )

    plan = result.get("current_plan")
    if plan is None:
        raise HTTPException(
            status_code=500, detail="Plan approval did not produce a plan"
        )

    # Re-persist the final plan (idempotent)
    existing = session.plan
    if existing is not None:
        _persist_plan(
            db, session, plan, version=existing.version,
            adjustments=existing.adjustments,
        )
    session.phase = Phase.GENERATING.value
    db.commit()

    # Runs in the thread pool after the 202 is sent; opens its own db session.
    background_tasks.add_task(generate_materials, session_id)

    return ApproveOut(
        phase=Phase.GENERATING,
        message=(
            "Plan approved. Material generation started in the background."
        ),
    )


@router.get(
    "/sessions/{session_id}/materials",
    response_model=MaterialsOut,
)
def get_materials(
    session_id: str, db: DBSession = Depends(get_db)
) -> MaterialsOut:
    """Poll material generation progress with full content."""
    session = _get_session_or_404(db, session_id)

    materials = [
        MaterialOut(
            step_id=m.step_id,
            slides=m.slides,
            questions=[QuestionOut(**q) for q in m.questions],
        )
        for m in session.materials
    ]

    return MaterialsOut(phase=Phase(session.phase), generated_steps=materials)
