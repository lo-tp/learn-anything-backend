r"""The notation policy for the two stages whose output is read as text (#198).

The fix this guards: `summarize_step` and `write_questions` used to be told to
"Use LaTeX for math ($ inline, $$ display)". Their output is displayed to a
learner exactly as written — step key points and quiz text have no math renderer
— so a majority of it was shown as LaTeX source, and a lone backslash in that
text is consumed as a JSON escape on the way through the structured call
(`\text{…}` arriving as a tab plus "ext", `\bar` as a backspace). Slides are a
deliberate second notation: AsciiMath through `<MathJax>`, unchanged here.

Two layers, because the contract has two parts: the shared block in the private
prompt package (prompts/material.py), and the system prompt each node actually
sends, which is that block plus the response-language instruction.

These tests read the private prompt package, so they skip when the submodule is
not checked out (see tests/conftest.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.prompts import _JSX_SYSTEM_PROMPT
from graphs.material.nodes import make_summarize_step, make_write_questions
from graphs.material.schemas import QuestionDraft, QuestionsOut, SummaryOut

pytestmark = pytest.mark.real_prompts

# One import, at collection time: the module only runs where the private
# package exists, and core.prompts deliberately does not re-export internal
# prompt pieces.
prompts_material = pytest.importorskip("prompts.material")
PLAIN_MATH_NOTATION = prompts_material.PLAIN_MATH_NOTATION

STAGES = {
    "summarize_step": SummaryOut(key_points=["c is invariant in every inertial frame"]),
    "write_questions": QuestionsOut(
        questions=[
            QuestionDraft(text="Q1?", options=["A", "B"], correct_index=0, explanation="e"),
            QuestionDraft(text="Q2?", options=["C", "D"], correct_index=1, explanation="e"),
            QuestionDraft(text="Q3?", options=["E", "F"], correct_index=0, explanation="e"),
        ]
    ),
}


def _sent_system_prompt(stage: str) -> str:
    """The exact system string a material node hands to its structured call."""
    state = {
        "step": {"id": "s1", "title": "Time dilation", "description": "derive it"},
        "slide_contents": [{"title": "The light clock", "key_points": ["moving ticks run slow"]}],
        "established_concepts": [],
        "learner_context": {},
        "language": "English",
    }
    factory = make_summarize_step if stage == "summarize_step" else make_write_questions
    with (
        patch("graphs.material.nodes.structured_invoke", return_value=STAGES[stage]) as call,
        patch(
            "graphs.material.nodes.with_unknown_option",
            side_effect=lambda llm, lang, opts: list(opts),
        ),
    ):
        factory(MagicMock())(state)  # type: ignore[arg-type]
    return str(call.call_args[0][2])


class TestSharedNotationRule:
    """One named block, shared by both text-returning stages."""

    def test_both_stages_carry_the_same_block(self) -> None:
        for stage in ("summarize_step", "write_questions"):
            prompt = getattr(prompts_material, f"{stage.upper()}_SYSTEM")
            assert PLAIN_MATH_NOTATION in prompt, f"{stage} does not carry the shared rule"

    def test_it_names_the_forms_a_learner_needs(self) -> None:
        """Forbidding LaTeX without naming the replacement leaves the model to guess."""
        assert "x^2" in PLAIN_MATH_NOTATION, "powers"
        assert "a/b" in PLAIN_MATH_NOTATION, "fractions"
        assert "sqrt(" in PLAIN_MATH_NOTATION, "roots"
        assert "v_0" in PLAIN_MATH_NOTATION, "subscripts"
        assert "≈" in PLAIN_MATH_NOTATION and "×" in PLAIN_MATH_NOTATION, "operators"

    def test_it_names_the_forms_it_forbids(self) -> None:
        for forbidden in ("$…$", "$$…$$", "\\frac", "\\sqrt", "\\times", "\\text"):
            assert forbidden in PLAIN_MATH_NOTATION, f"{forbidden} is not prohibited by name"


class TestLatexIsNoLongerRequested:
    """The exact mandates the issue was filed about."""

    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_no_prompt_asks_for_latex(self, stage: str) -> None:
        prompt = getattr(prompts_material, f"{stage.upper()}_SYSTEM")
        assert "Use LaTeX" not in prompt
        assert "$ inline" not in prompt
        assert "$$ display" not in prompt

    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_latex_is_forbidden_by_name(self, stage: str) -> None:
        assert "do not write latex" in getattr(prompts_material, f"{stage.upper()}_SYSTEM").lower()


class TestWhatTheNodesActuallySend:
    """The seam a model sees: the shared block plus the response-language instruction."""

    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_the_sent_system_prompt_carries_the_rule(self, stage: str) -> None:
        sent = _sent_system_prompt(stage)
        assert PLAIN_MATH_NOTATION in sent
        assert "RESPONSE LANGUAGE: Write ALL user-facing text in English" in sent

    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_the_sent_system_prompt_asks_for_no_latex(self, stage: str) -> None:
        assert "Use LaTeX" not in _sent_system_prompt(stage)


class TestSlideNotationIsUnchanged:
    """The other half of the decision: AsciiMath is what the slide renderer reads."""

    def test_the_asciimath_loader_is_still_required(self) -> None:
        assert 'load: ["input/asciimath"]' in _JSX_SYSTEM_PROMPT

    def test_latex_is_still_forbidden_in_slides(self) -> None:
        assert "Do NOT use LaTeX" in _JSX_SYSTEM_PROMPT

    def test_the_two_notations_remain_deliberately_different(self) -> None:
        """A learner crosses between them, so both sides have to state their own rule
        rather than one prompt happening to be right."""
        assert "AsciiMath" in _JSX_SYSTEM_PROMPT
        assert "AsciiMath" not in PLAIN_MATH_NOTATION
