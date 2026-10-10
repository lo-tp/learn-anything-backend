"""Tests for graphs/material/nodes.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphs.material.nodes import (
    make_compile_slide,
    make_plan_slide_contents,
    make_summarize_step,
    make_write_questions,
    make_write_slide,
)
from graphs.material.schemas import (
    QuestionDraft,
    QuestionsOut,
    SlideContentsOut,
    SlideContentSpec,
    SlideOut,
    SummaryOut,
)
from graphs.material.state import MaterialState


def _make_llm() -> MagicMock:
    """Create a mock LLM that records structured output calls."""
    llm = MagicMock()
    return llm


# --- make_plan_slide_contents ---


class TestPlanSlideContents:
    def test_returns_slide_contents_and_index(self):
        llm = _make_llm()
        mock_out = SlideContentsOut(
            slide_contents=[
                SlideContentSpec(title="A", key_points=["k1"], visual_hint="v1"),
                SlideContentSpec(title="B", key_points=["k2"], visual_hint="v2"),
                SlideContentSpec(title="C", key_points=["k3"], visual_hint="v3"),
            ]
        )
        with patch(
            "graphs.material.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_plan_slide_contents(llm)
            result = node(
                {
                    "step": {"id": "s1", "title": "Step 1", "description": "desc"},
                    "established_concepts": [],
                    "learner_context": {},
                    "language": "English",
                }
            )

        assert result["slide_index"] == 0
        assert result["attempts_by_slide"] == [0, 0, 0]
        assert len(result["slide_contents"]) == 3
        assert result["slide_contents"][0]["title"] == "A"

    def test_includes_established_concepts_in_prompt(self):
        llm = _make_llm()
        mock_out = SlideContentsOut(
            slide_contents=[
                SlideContentSpec(title="A", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="B", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="C", key_points=["k"], visual_hint="v"),
            ]
        )
        with patch(
            "graphs.material.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_plan_slide_contents(llm)
            node(
                {
                    "step": {"id": "s2", "title": "Step 2", "description": "d"},
                    "established_concepts": [
                        {"title": "Prior", "key_points": ["fact1", "fact2"]}
                    ],
                    "learner_context": {},
                    "language": "English",
                }
            )
        # The human message (4th arg) should contain the established concepts
        assert "Prior" in str(mock_invoke.call_args)

    def test_empty_step_is_handled(self):
        llm = _make_llm()
        mock_out = SlideContentsOut(
            slide_contents=[
                SlideContentSpec(title="A", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="B", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="C", key_points=["k"], visual_hint="v"),
            ]
        )
        with patch("graphs.material.nodes.structured_invoke", return_value=mock_out):
            node = make_plan_slide_contents(llm)
            result = node({})
        assert result["slide_index"] == 0

    def test_records_stage_timing(self):
        llm = _make_llm()
        mock_out = SlideContentsOut(
            slide_contents=[
                SlideContentSpec(title="A", key_points=["k1"], visual_hint="v1"),
                SlideContentSpec(title="B", key_points=["k2"], visual_hint="v2"),
                SlideContentSpec(title="C", key_points=["k3"], visual_hint="v3"),
            ]
        )
        with patch("graphs.material.nodes.structured_invoke", return_value=mock_out):
            node = make_plan_slide_contents(llm)
            result = node(
                {
                    "step": {"id": "s1", "title": "Step 1", "description": "desc"},
                    "established_concepts": [],
                    "learner_context": {},
                    "language": "English",
                }
            )

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "plan_slide_contents"
        assert timing["context"] == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timing["duration_seconds"] >= 0

    def test_plans_with_the_minimum_reasoning_effort(self):
        """Slide planning is a slide-generation call: it binds minimal reasoning."""
        from graphs.material.sandbox import SLIDE_REASONING_EFFORT

        llm = _make_llm()
        mock_out = SlideContentsOut(
            slide_contents=[
                SlideContentSpec(title="A", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="B", key_points=["k"], visual_hint="v"),
                SlideContentSpec(title="C", key_points=["k"], visual_hint="v"),
            ]
        )
        with patch(
            "graphs.material.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_plan_slide_contents(llm)
            node({"step": {"id": "s1"}, "language": "English"})

        assert SLIDE_REASONING_EFFORT == "minimal"
        assert mock_invoke.call_args.kwargs == {
            "reasoning_effort": SLIDE_REASONING_EFFORT,
        }


# --- make_write_slide ---


class TestWriteSlide:
    def _state(self, **overrides) -> MaterialState:
        base: MaterialState = {
            "step": {"id": "s1", "title": "Step 1", "description": "d"},
            "slide_contents": [
                {"title": "Slide 1", "key_points": ["kp1"], "visual_hint": "v"}
            ],
            "slide_index": 0,
            "attempts_by_slide": [0],
            "established_concepts": [],
            "learner_context": {},
            "language": "English",
        }
        base.update(overrides)  # type: ignore[arg-type]
        return base

    @staticmethod
    def _validate_ok(source: str) -> dict:
        return {
            "ok": True, "compiled_code": "compiled", "source": source,
            "error": "", "repairs": [],
        }

    @staticmethod
    def _validate_fail(error: str, source: str, repairs=None) -> dict:
        return {
            "ok": False, "compiled_code": None, "source": source,
            "error": error, "repairs": repairs or [],
        }

    def _run(self, state: MaterialState, writes, validates):
        """Run write_slide with scripted LLM writes and validate_jsx results.

        Returns ``(result, invoke_mock)`` so tests can assert on the LLM calls
        (one call per write/repair turn) and on the node output.
        """
        with (
            patch(
                "graphs.material.nodes.structured_invoke_messages",
                side_effect=writes,
            ) as mock_invoke,
            patch("graphs.material.nodes.validate_jsx", side_effect=validates),
        ):
            node = make_write_slide(_make_llm())
            result = node(state)
        return result, mock_invoke

    def test_returns_jsx_and_resets_error(self):
        result, invoke = self._run(
            self._state(last_compile_error="old error"),
            writes=[SlideOut(code="export default function S() {}")],
            validates=[self._validate_ok("export default function S() {}")],
        )
        assert result["current_slide_jsx"] == "export default function S() {}"
        assert result["last_compile_error"] is None
        assert result["attempts_by_slide"] == [1]
        assert result["current_slide_prompt"] is not None
        # A first-pass compile needs no repair turn.
        assert invoke.call_count == 1

    def test_increments_attempts(self):
        result, _ = self._run(
            self._state(attempts_by_slide=[2]),
            writes=[SlideOut(code="export default function S() {}")],
            validates=[self._validate_ok("export default function S() {}")],
        )
        assert result["attempts_by_slide"] == [3]

    def _kwargs_for_attempt(self, attempts_by_slide: list[int]):
        _, invoke = self._run(
            self._state(attempts_by_slide=attempts_by_slide),
            writes=[SlideOut(code="export default function S() {}")],
            validates=[self._validate_ok("export default function S() {}")],
        )
        return invoke.call_args.kwargs

    def test_first_attempt_uses_creative_profile(self):
        from graphs.material.sandbox import slide_sampling_bind_kwargs

        assert self._kwargs_for_attempt([0]) == slide_sampling_bind_kwargs(1)

    def test_second_attempt_uses_middle_profile(self):
        from graphs.material.sandbox import slide_sampling_bind_kwargs

        assert self._kwargs_for_attempt([1]) == slide_sampling_bind_kwargs(2)

    def test_third_attempt_uses_reliable_profile(self):
        from graphs.material.sandbox import slide_sampling_bind_kwargs

        assert self._kwargs_for_attempt([2]) == slide_sampling_bind_kwargs(3)

    def test_attempt_beyond_profiles_uses_last_profile(self):
        from graphs.material.sandbox import (
            SLIDE_SAMPLING_PROFILES,
            slide_sampling_bind_kwargs,
        )

        n = len(SLIDE_SAMPLING_PROFILES)
        assert self._kwargs_for_attempt([n]) == slide_sampling_bind_kwargs(n + 1)

    def test_non_openai_params_are_nested_in_extra_body(self):
        from graphs.material.sandbox import slide_sampling_bind_kwargs

        kwargs = self._kwargs_for_attempt([0])
        # Standard params stay top-level...
        assert "temperature" in kwargs
        assert "top_p" in kwargs
        # ...while OpenAI-incompatible params are nested under extra_body...
        assert "extra_body" in kwargs
        for param in ("top_k", "min_p", "repeat_penalty"):
            assert param in kwargs["extra_body"]
            assert param not in kwargs
        # ...and the result matches the shared helper exactly.
        assert kwargs == slide_sampling_bind_kwargs(1)

    def test_writes_with_the_minimum_reasoning_effort(self):
        """Every slide-writing attempt asks for the smallest reasoning budget."""
        from graphs.material.sandbox import SLIDE_REASONING_EFFORT

        for attempt_index in (0, 1, 2):
            kwargs = self._kwargs_for_attempt([attempt_index])
            assert kwargs["reasoning_effort"] == SLIDE_REASONING_EFFORT == "minimal"
            assert "reasoning_effort" not in kwargs.get("extra_body", {})

    def _human_content(self, **state_overrides) -> str:
        _, invoke = self._run(
            self._state(**state_overrides),
            writes=[SlideOut(code="export default function S() {}")],
            validates=[self._validate_ok("export default function S() {}")],
        )
        return invoke.call_args[0][2][-1].content

    def test_retry_prompt_includes_sanitized_compile_error(self):
        human_content = self._human_content(
            attempts_by_slide=[1],
            last_compile_error=(
                "Transport/HTTP error: Client error '400 code must declare "
                "`export default`' for url 'http://localhost:3001/api/compile' "
                "For more information check: https://developer.mozilla.org"
            ),
        )
        # The carried-over error from a prior attempt is fed back...
        assert "failed to compile" in human_content
        assert "code must declare `export default`" in human_content
        # ...but the raw transport wrapper is not echoed.
        assert "Transport/HTTP error" not in human_content
        assert "localhost:3001" not in human_content

    def test_no_compile_error_in_first_attempt_prompt(self):
        human_content = self._human_content()
        assert "failed to compile" not in human_content

    def test_records_stage_timing(self):
        result, _ = self._run(
            self._state(),
            writes=[SlideOut(code="export default function S() {}")],
            validates=[self._validate_ok("export default function S() {}")],
        )

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "write_slide"
        assert timing["context"] == {
            "step_id": "s1", "slide_index": 1, "attempt": 1,
        }
        assert timing["duration_seconds"] >= 0

    # --- Bounded self-repair loop ---

    def test_repairs_then_succeeds(self):
        """First validation fails, a repair turn is issued, second passes."""
        result, invoke = self._run(
            self._state(),
            writes=[
                SlideOut(code="broken v1"),
                SlideOut(code="fixed v2"),
            ],
            validates=[
                self._validate_fail("syntax error at line 2", "broken v1"),
                self._validate_ok("fixed v2"),
            ],
        )
        assert result["current_slide_jsx"] == "fixed v2"
        assert result["last_compile_error"] is None
        # One initial write + one repair turn.
        assert invoke.call_count == 2

    def test_exhausts_self_repair_and_carries_error(self):
        """All validations fail: one initial write + MAX repair turns, then
        the last reduced error is carried to the outer retry routing."""
        from graphs.material.sandbox import MAX_SLIDE_SELF_REPAIR_TURNS

        n = MAX_SLIDE_SELF_REPAIR_TURNS
        result, invoke = self._run(
            self._state(),
            writes=[SlideOut(code=f"v{i}") for i in range(n + 1)],
            validates=[
                self._validate_fail(f"error {i}", f"v{i}")
                for i in range(n + 1)
            ],
        )
        assert result["last_compile_error"] == f"error {n}"
        # Final JSX is the last repair's output.
        assert result["current_slide_jsx"] == f"v{n}"
        assert invoke.call_count == n + 1

    def test_deterministic_repair_source_is_stored(self):
        """When validate_jsx applies a deterministic repair, the repaired
        source it reports is what gets stored (not the raw model output)."""
        result, _ = self._run(
            self._state(),
            writes=[SlideOut(code="```tsx\nraw\n```")],
            validates=[{
                "ok": True, "compiled_code": "compiled", "source": "raw",
                "error": "", "repairs": ["strip_markdown_fences"],
            }],
        )
        assert result["current_slide_jsx"] == "raw"
        assert result["last_compile_error"] is None

    def test_no_repair_turn_when_first_validation_passes(self):
        _, invoke = self._run(
            self._state(),
            writes=[SlideOut(code="good")],
            validates=[self._validate_ok("good")],
        )
        assert invoke.call_count == 1

    def test_repair_turn_prompt_includes_reduced_error_and_jsx(self):
        _, invoke = self._run(
            self._state(),
            writes=[SlideOut(code="broken"), SlideOut(code="fixed")],
            validates=[
                self._validate_fail(
                    "400 code must declare `export default`", "broken"
                ),
                self._validate_ok("fixed"),
            ],
        )
        # The repair turn is the second LLM call; its human message carries
        # the reduced error, the failed JSX, and a repair instruction.
        repair_human = invoke.call_args_list[1][0][2][-1].content
        assert "failed to validate" in repair_human
        assert "code must declare `export default`" in repair_human
        assert "broken" in repair_human

    def test_repair_turn_prompt_reports_deterministic_repairs(self):
        _, invoke = self._run(
            self._state(),
            writes=[SlideOut(code="broken"), SlideOut(code="fixed")],
            validates=[
                self._validate_fail(
                    "syntax error at line 2", "stripped", repairs=["balance_braces"]
                ),
                self._validate_ok("fixed"),
            ],
        )
        repair_human = invoke.call_args_list[1][0][2][-1].content
        assert "balance_braces" in repair_human

    def test_includes_established_concepts_in_initial_prompt(self):
        _, invoke = self._run(
            self._state(
                established_concepts=[{"title": "Prior", "key_points": ["fact1"]}]
            ),
            writes=[SlideOut(code="x")],
            validates=[self._validate_ok("x")],
        )
        human = invoke.call_args[0][2][-1].content
        assert "Prior" in human
        assert "fact1" in human

    def test_repair_turn_includes_established_concepts(self):
        _, invoke = self._run(
            self._state(
                established_concepts=[{"title": "Prior", "key_points": ["fact1"]}]
            ),
            writes=[SlideOut(code="broken"), SlideOut(code="fixed")],
            validates=[
                self._validate_fail("syntax error", "broken"),
                self._validate_ok("fixed"),
            ],
        )
        repair_human = invoke.call_args_list[1][0][2][-1].content
        assert "Prior" in repair_human

    def test_odd_backticks_are_tolerated_not_fatal(self):
        # An odd number of backticks (possible broken template literal) is
        # logged, not fatal: the JSX is still returned.
        result, _ = self._run(
            self._state(),
            writes=[SlideOut(code="const x = `a;")],  # one backtick
            validates=[self._validate_ok("const x = `a;")],
        )
        assert result["current_slide_jsx"] == "const x = `a;"


# --- make_compile_slide ---


class TestCompileSlide:
    def _state(self, **overrides) -> MaterialState:
        base: MaterialState = {
            "step": {"id": "s1", "title": "Step 1"},
            "slide_contents": [
                {"title": "Slide 1", "key_points": [], "visual_hint": ""},
                {"title": "Slide 2", "key_points": [], "visual_hint": ""},
            ],
            "slide_index": 0,
            "current_slide_jsx": "export default function S() {}",
            "current_slide_prompt": "[]",
            "attempts_by_slide": [1],
        }
        base.update(overrides)  # type: ignore[arg-type]
        return base

    @patch("graphs.material.nodes._compile_slide")
    def test_success(self, mock_compile):
        mock_compile.return_value = ("compiled_code", "")
        node = make_compile_slide(MagicMock())
        result = node(self._state())

        assert result["compile_result"] == "success"
        assert result["slides"] == ["compiled_code"]
        assert result["slide_index"] == 1
        assert result["current_slide_is_placeholder"] is False

    @patch("graphs.material.nodes._compile_slide")
    def test_retry_on_first_attempt(self, mock_compile):
        mock_compile.return_value = (None, "syntax error")
        node = make_compile_slide(MagicMock())
        result = node(self._state())

        assert result["compile_result"] == "retry"
        assert result["last_compile_error"] == "syntax error"
        assert "slide_index" not in result  # stays at 0 for retry
        assert len(result["failed_attempts"]) == 1
        assert result["failed_attempts"][0]["error"] == "syntax error"

    @patch("graphs.material.nodes._compile_slide")
    def test_failed_attempt_is_tagged_with_failure_class(self, mock_compile):
        # The failure class rides along on the failed attempt so skip rates
        # are measurable by class (C1.2 builds on this).
        mock_compile.return_value = (None, "could not resolve `./Widget`")
        node = make_compile_slide(MagicMock())
        result = node(self._state())

        assert result["failed_attempts"][0]["failure_class"] == "unknown_component"

    @patch("graphs.material.nodes._compile_slide")
    def test_exhausted_when_placeholder_also_fails(self, mock_compile):
        # Both the real slide and the placeholder fail to compile.
        mock_compile.return_value = (None, "still broken")
        node = make_compile_slide(MagicMock())
        from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS

        state = self._state(attempts_by_slide=[MAX_MATERIAL_ATTEMPTS])
        result = node(state)

        assert mock_compile.call_count == 2  # real slide + placeholder
        assert result["compile_result"] == "exhausted"
        assert "slides" not in result
        assert result["slide_index"] == 1  # moves past this slide

    @patch("graphs.material.nodes._compile_slide")
    def test_exhausted_last_slide_routes_to_questions(self, mock_compile):
        mock_compile.return_value = (None, "broken")
        node = make_compile_slide(MagicMock())
        from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS

        state = self._state(
            slide_contents=[{"title": "Only", "key_points": [], "visual_hint": ""}],
            slide_index=0,
            attempts_by_slide=[MAX_MATERIAL_ATTEMPTS],
        )
        result = node(state)
        assert result["compile_result"] == "exhausted"
        assert result["slide_index"] == 1  # == len(slide_contents)

    @patch("graphs.material.nodes._compile_slide")
    def test_placeholder_on_exhaust(self, mock_compile):
        from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS

        # First call is the real slide (fails), second is the placeholder
        # (succeeds). The placeholder is generated regardless of DEV_MODE.
        mock_compile.side_effect = [
            (None, "broken"),
            ("placeholder_code", ""),
        ]
        node = make_compile_slide(MagicMock())
        state = self._state(attempts_by_slide=[MAX_MATERIAL_ATTEMPTS])
        result = node(state)
        assert result["compile_result"] == "success"
        assert result["slides"] == ["placeholder_code"]
        assert result["current_slide_is_placeholder"] is True

    @patch("graphs.material.nodes._compile_slide")
    def test_records_stage_timing(self, mock_compile):
        mock_compile.return_value = ("compiled_code", "")
        node = make_compile_slide(MagicMock())
        result = node(self._state())

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "compile_slide"
        assert timing["context"] == {
            "step_id": "s1", "slide_index": 1, "attempt": 1,
        }
        assert timing["duration_seconds"] >= 0

    @patch("graphs.material.nodes._compile_slide")
    def test_records_stage_timing_on_retry(self, mock_compile):
        mock_compile.return_value = (None, "syntax error")
        node = make_compile_slide(MagicMock())
        result = node(self._state(attempts_by_slide=[2]))

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "compile_slide"
        # The retry attempt number is the one just taken (attempts[i] = 2).
        assert timing["context"] == {
            "step_id": "s1", "slide_index": 1, "attempt": 2,
        }
        assert timing["duration_seconds"] >= 0


# --- make_write_questions ---


class TestWriteQuestions:
    def test_returns_formatted_questions(self):
        llm = MagicMock()
        mock_out = QuestionsOut(
            questions=[
                QuestionDraft(
                    text="Q1?", options=["A", "B"], correct_index=0, explanation="e"
                ),
                QuestionDraft(
                    text="Q2?", options=["C", "D"], correct_index=1, explanation="e"
                ),
                QuestionDraft(
                    text="Q3?", options=["E", "F"], correct_index=0, explanation="e"
                ),
            ]
        )
        with (
            patch("graphs.material.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts + ["I don't know"],
            ),
        ):
            node = make_write_questions(llm)
            result = node(
                {
                    "step": {"id": "s1", "title": "Step 1", "description": "d"},
                    "established_concepts": [],
                    "learner_context": {},
                    "language": "English",
                }
            )

        assert len(result["questions"]) == 3
        q = result["questions"][0]
        assert q["id"] == "s1_q1"
        assert q["text"] == "Q1?"
        assert q["correct_index"] == 0
        assert "I don't know" in q["options"]

    def test_includes_established_concepts_in_prompt(self):
        llm = MagicMock()
        mock_out = QuestionsOut(
            questions=[
                QuestionDraft(
                    text=f"Q{i}?", options=["A", "B"], correct_index=0, explanation="e"
                )
                for i in range(3)
            ]
        )
        with (
            patch("graphs.material.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts,
            ),
        ):
            node = make_write_questions(llm)
            node(
                {
                    "step": {"id": "s1", "title": "T", "description": "d"},
                    "established_concepts": [
                        {"title": "Prior", "key_points": ["fact1"]}
                    ],
                    "learner_context": {},
                    "language": "English",
                }
            )

    def test_question_ids_are_sequential(self):
        llm = MagicMock()
        mock_out = QuestionsOut(
            questions=[
                QuestionDraft(
                    text=f"Q{i}?", options=["A", "B"], correct_index=0, explanation="e"
                )
                for i in range(4)
            ]
        )
        with (
            patch("graphs.material.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts + ["I don't know"],
            ),
        ):
            node = make_write_questions(llm)
            result = node({"step": {"id": "s9"}, "language": "English"})

        assert [q["id"] for q in result["questions"]] == [
            "s9_q1",
            "s9_q2",
            "s9_q3",
            "s9_q4",
        ]

    def test_records_stage_timing(self):
        llm = MagicMock()
        mock_out = QuestionsOut(
            questions=[
                QuestionDraft(
                    text=f"Q{i}?", options=["A", "B"], correct_index=0, explanation="e"
                )
                for i in range(3)
            ]
        )
        with (
            patch("graphs.material.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts,
            ),
        ):
            node = make_write_questions(llm)
            result = node({"step": {"id": "s1"}, "language": "English"})

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "write_questions"
        assert timing["context"] == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timing["duration_seconds"] >= 0


# --- make_summarize_step ---


class TestSummarizeStep:
    def test_returns_summary(self):
        llm = MagicMock()
        mock_out = SummaryOut(key_points=["kp1", "kp2", "kp3"])
        with patch("graphs.material.nodes.structured_invoke", return_value=mock_out):
            node = make_summarize_step(llm)
            result = node(
                {
                    "step": {"id": "s1", "title": "Step 1", "description": "d"},
                    "slide_contents": [
                        {"title": "A", "key_points": ["k1"]},
                        {"title": "B", "key_points": ["k2"]},
                    ],
                    "language": "English",
                }
            )

        assert result["summary"]["step_id"] == "s1"
        assert result["summary"]["title"] == "Step 1"
        assert result["summary"]["key_points"] == ["kp1", "kp2", "kp3"]

    def test_empty_slide_contents(self):
        llm = MagicMock()
        mock_out = SummaryOut(key_points=["only point"])
        with patch("graphs.material.nodes.structured_invoke", return_value=mock_out):
            node = make_summarize_step(llm)
            result = node({"step": {"id": "s1", "title": "T"}, "language": "English"})
        assert result["summary"]["key_points"] == ["only point"]

    def test_records_stage_timing(self):
        llm = MagicMock()
        mock_out = SummaryOut(key_points=["kp1"])
        with patch("graphs.material.nodes.structured_invoke", return_value=mock_out):
            node = make_summarize_step(llm)
            result = node(
                {"step": {"id": "s1", "title": "T"}, "language": "English"}
            )

        (timing,) = result["stage_timings"]
        assert timing["stage"] == "summarize_step"
        assert timing["context"] == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timing["duration_seconds"] >= 0
