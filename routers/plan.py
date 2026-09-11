"""Plan phase endpoints: generate, adjust, approve, and material polling."""

from collections import deque

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, Plan, Session, get_db
from graphs import checkpointer, graph_config, plan_graph

router = APIRouter(tags=["plan"])


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
    session_id: str, db: DBSession = Depends(get_db)
) -> ApproveOut:
    """Approve the plan and transition to generating."""
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

    return ApproveOut(phase=Phase.GENERATING, message="Plan approved.")


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
