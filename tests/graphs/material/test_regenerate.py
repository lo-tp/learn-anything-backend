"""Tests for graphs/material/regenerate.py — one stored slide, rewritten.

The seam is :func:`regenerate_slide` itself: the material graph's own
``write_slide``/``compile_slide`` nodes run over one stored content spec, with
no graph, no checkpointer and no interrupt. The model call and the sandbox
compile are patched at the node boundary, exactly as
``tests/graphs/material/test_nodes.py`` patches them, so these tests read as a
specification of what a single-slide regeneration does.

``MAX_SLIDE_SELF_REPAIR_TURNS`` is pinned to 0 for the whole file: the
self-repair loop belongs to ``write_slide`` and is tested there. What this
module composes is the *outer* attempt ladder, and pinning the inner one is
what makes the attempt counts here describe that ladder rather than a product
of two loops.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphs.material.regenerate import regenerate_slide
from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS
from graphs.material.schemas import SlideOut

STEP = {"id": "s1", "title": "Newton's second law", "description": "F = ma"}
SPEC = {"title": "Force and mass", "key_points": ["F = ma"], "visual_hint": "a cart"}


@pytest.fixture(autouse=True)
def _no_inner_repair_loop(monkeypatch):
    """Turn off write_slide's self-repair turns (see the module docstring)."""
    monkeypatch.setattr("graphs.material.nodes.MAX_SLIDE_SELF_REPAIR_TURNS", 0)


def _validate_ok(source: str) -> dict:
    return {
        "ok": True, "compiled_code": f"compiled:{source}", "source": source,
        "error": "", "repairs": [],
    }


def _validate_fail(source: str, error: str) -> dict:
    return {
        "ok": False, "compiled_code": None, "source": source,
        "error": error, "repairs": [],
    }


def _compiles_for(validates: list[dict], extra: list[dict] | None = None):
    """Sandbox outcomes that agree with the validation results scripted above.

    ``write_slide`` validates a slide and ``compile_slide`` then compiles the
    same source; if the two disagreed the test would describe a state the real
    nodes could never be in. ``extra`` covers a compile ``compile_slide`` makes
    for itself — the placeholder it substitutes when an attempt is exhausted.
    """
    outcomes = [
        (f"compiled:{v['source']}", "") if v["ok"] else (None, v["error"])
        for v in validates
    ]
    for result in extra or []:
        outcomes.append(
            (f"compiled:{result['source']}", "")
            if result["ok"]
            else (None, result["error"])
        )
    return outcomes


def _run(
    writes: list[SlideOut],
    validates: list[dict],
    *,
    extra_compiles: list[dict] | None = None,
    feedback: str | None = None,
):
    """Run one regeneration with a scripted model and sandbox.

    Returns ``(outcome, invoke_mock)`` — ``invoke_mock`` is the slide-writing
    call, so a test can count attempts and read the request that was sent.
    """
    llm = MagicMock()
    with (
        patch(
            "graphs.material.nodes.structured_invoke_messages",
            side_effect=writes,
        ) as invoke,
        patch("graphs.material.nodes.validate_jsx", side_effect=validates),
        patch(
            "graphs.material.nodes._compile_slide",
            side_effect=_compiles_for(validates, extra_compiles),
        ),
    ):
        outcome = regenerate_slide(
            llm,
            step=STEP,
            content_spec=SPEC,
            established_concepts=[{"title": "Prior", "key_points": ["fact"]}],
            learner_context={"strand": {"floor": "already knows x"}},
            language="English",
            feedback=feedback,
        )
    return outcome, invoke


class TestRegenerateSlide:
    def test_one_stored_spec_rewrites_one_slide(self):
        """The whole of #197: the stored spec is enough to write the slide
        again, on its own, with nothing else in the deck touched."""
        outcome, invoke = _run(
            [SlideOut(code="new slide")], [_validate_ok("new slide")]
        )

        assert outcome["jsx"] == "compiled:new slide"
        assert outcome["is_placeholder"] is False
        assert outcome["failed_attempts"] == []
        assert invoke.call_count == 1

    def test_the_request_is_the_one_that_would_have_written_it_anyway(self):
        """Everything the graph had is rebuilt from what the backend stored:
        the spec, the boundary map, and the prior steps' established concepts."""
        _, invoke = _run([SlideOut(code="new slide")], [_validate_ok("new slide")])
        human = invoke.call_args[0][2][-1].content

        assert SPEC["title"] in human
        assert "F = ma" in human
        assert SPEC["visual_hint"] in human
        assert "Prior" in human
        assert "already knows x" in human

    def test_what_the_learner_asks_for_is_carried_into_the_rewrite(self):
        _, invoke = _run(
            [SlideOut(code="new slide")],
            [_validate_ok("new slide")],
            feedback="make it more visual",
        )

        assert "make it more visual" in invoke.call_args[0][2][-1].content

    def test_a_slide_that_does_not_compile_retries_the_way_the_graph_does(self):
        """Same ladder, same bound, same step down the sampling profiles: the
        attempt counter is the graph's, not a second policy invented here."""
        attempts = MAX_MATERIAL_ATTEMPTS
        writes = [SlideOut(code=f"try {n}") for n in range(attempts)]
        validates = [
            _validate_fail(f"try {n}", "syntax error at line 2")
            for n in range(attempts - 1)
        ] + [_validate_ok(f"try {attempts - 1}")]

        outcome, invoke = _run(writes, validates)

        assert invoke.call_count == attempts
        assert outcome["jsx"] == f"compiled:try {attempts - 1}"
        assert outcome["is_placeholder"] is False
        # The attempts that failed are the record a developer debugs from.
        assert len(outcome["failed_attempts"]) == attempts - 1
        assert all(
            fa["error"] == "syntax error at line 2"
            for fa in outcome["failed_attempts"]
        )
        assert all(fa["prompt"] for fa in outcome["failed_attempts"])

    def test_every_attempt_available_is_used_before_a_regeneration_gives_up(self):
        attempts = MAX_MATERIAL_ATTEMPTS
        writes = [SlideOut(code=f"try {n}") for n in range(attempts)]
        validates = [
            _validate_fail(f"try {n}", "syntax error at line 2")
            for n in range(attempts)
        ]

        outcome, invoke = _run(
            writes,
            validates,
            # The last attempt exhausted, so the node substitutes the
            # placeholder slide and compiles that instead.
            extra_compiles=[_validate_ok("placeholder")],
        )

        assert invoke.call_count == attempts
        assert len(outcome["failed_attempts"]) == attempts
        assert outcome["is_placeholder"] is True
        assert outcome["jsx"] == "compiled:placeholder"

    def test_each_stage_of_a_regeneration_records_what_it_cost(self):
        """A regeneration runs the stages deck generation runs, and #194's rule
        is that a stage's cost is written down."""
        outcome, _ = _run([SlideOut(code="new slide")], [_validate_ok("new slide")])

        stages = [t["stage"] for t in outcome["stage_timings"]]
        assert stages == ["write_slide", "compile_slide"]
        assert all(t["duration_seconds"] >= 0 for t in outcome["stage_timings"])
        assert all(t["context"]["step_id"] == "s1" for t in outcome["stage_timings"])
        # A row that does not say which run it came from cannot be told apart
        # from a deck generation afterwards.
        assert all(
            t["context"]["regenerated"] is True for t in outcome["stage_timings"]
        )

    def test_a_model_call_that_fails_is_not_silently_a_finished_slide(self):
        """An exception from the model reaches the caller.

        A regeneration that never wrote a slide must not arrive as one that
        wrote a bad one — the caller keeps the slide the learner already has.
        """
        llm = MagicMock()
        with patch(
            "graphs.material.nodes.structured_invoke_messages",
            side_effect=RuntimeError("model is down"),
        ), pytest.raises(RuntimeError, match="model is down"):
            regenerate_slide(
                llm,
                step=STEP,
                content_spec=SPEC,
                established_concepts=[],
                learner_context={},
                language="English",
            )
