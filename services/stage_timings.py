"""Per-stage timing persistence for the graphs that pause between requests (#194).

The material graph runs a whole step in one background task, so its driver
records each stage's wall time as the graph pauses (``routers/plan.py``). Clarify,
Probe and Plan run one request at a time, and the checkpoint carries the state
from the previous ones — including the timings already recorded. So their drivers
need two things:

``empty_stage_timings()``
    The state value that empties the accumulated channel, so a request records
    what it ran and nothing it recorded already.

``record_stage_timings()``
    One ``graph_stage_timings`` row per stage this request ran.

Rows are append-only history, same policy as the material rows: a re-run appends
a fresh set rather than replacing anything. ``scripts/slide_analysis.py --graph
clarify|probe|plan`` reads them.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Overwrite
from sqlalchemy.orm import Session as DBSession

from db import GraphStageTiming


def empty_stage_timings() -> Any:
    """The state write that starts a timed run: an emptied timing channel.

    A checkpoint carries the state from the previous request, including the
    timings it recorded, and those rows are already written. Every driver of a
    graph that pauses between requests puts this value in the state it passes —
    the input of a first request, the ``update`` of a resume — so what the run
    returns is exactly the work this request did.

    A LangGraph ``Overwrite`` is what bypasses the state's accumulating reducer:
    the channel accumulates *within* a run on purpose (a run has several stages)
    and is emptied *between* runs for the same reason.

    The return is untyped because the value written for a ``list[dict]`` channel
    is not a list — that is the point of it.
    """
    return Overwrite([])


def record_stage_timings(
    db: DBSession, session_id: str, graph: str, result: dict[str, Any]
) -> int:
    """Record what each stage of this request cost.

    Returns the number of rows written. A result with no timings (a request that
    only resumed an interrupt and ended) writes nothing, which is the point: the
    rows describe work this service did, not time the learner spent.
    """
    rows = 0
    for timing in result.get("stage_timings") or []:
        db.add(
            GraphStageTiming(
                session_id=session_id,
                graph=graph,
                stage=timing["stage"],
                context=timing.get("context"),
                duration_seconds=timing["duration_seconds"],
            )
        )
        rows += 1
    if rows:
        db.commit()
    return rows
