r"""The notation policy for text surfaces that no renderer reads (#198, option A).

Two prompts used to ask for LaTeX — `summarize_step` and `write_questions` — and
the surfaces that show their output (step key points, quiz text) have no math
renderer, so a majority of that text was displayed to learners as LaTeX source,
with `\gamma`-style commands surviving only because the model doubled the
backslash, and a single backslash arriving as a tab or a stray letter.

The rule this guards is one block shared by both prompts. The slide path is a
different decision — AsciiMath through `<MathJax>` — and is asserted unchanged,
because the whole point of the policy is that the two notations are *different
on purpose* rather than an accident of two prompts.

These tests read the private prompt package, so they skip when the submodule is
not checked out (see tests/conftest.py).
"""

from __future__ import annotations

import pytest

from core.prompts import (
    _JSX_SYSTEM_PROMPT,
    SUMMARIZE_STEP_SYSTEM,
    WRITE_QUESTIONS_SYSTEM,
)

pytestmark = pytest.mark.real_prompts

TEXT_SURFACES = {
    "summarize_step": SUMMARIZE_STEP_SYSTEM,
    "write_questions": WRITE_QUESTIONS_SYSTEM,
}


class TestSharedNotationRule:
    """One named rule, used by both text-returning stages."""

    def test_the_rule_exists_and_is_shared_by_both_stages(self) -> None:
        from prompts.material import PLAIN_MATH_NOTATION

        for name, prompt in TEXT_SURFACES.items():
            assert PLAIN_MATH_NOTATION in prompt, f"{name} does not carry the shared rule"

    def test_it_states_that_nothing_renders_this_text(self) -> None:
        from prompts.material import PLAIN_MATH_NOTATION

        assert "no math renderer" in PLAIN_MATH_NOTATION.lower()

    def test_it_names_the_forms_a_learner_needs(self) -> None:
        """Forbidding LaTeX without offering the replacement is how a model ends up
        writing `x squared` or a bare caret-free guess."""
        from prompts.material import PLAIN_MATH_NOTATION

        assert "x^2" in PLAIN_MATH_NOTATION, "powers"
        assert "a/b" in PLAIN_MATH_NOTATION, "fractions"
        assert "sqrt(" in PLAIN_MATH_NOTATION, "roots"
        assert "v_0" in PLAIN_MATH_NOTATION, "subscripts"
        assert "≈" in PLAIN_MATH_NOTATION and "×" in PLAIN_MATH_NOTATION, "operators"

    def test_it_explains_why_a_lone_backslash_is_not_safe_here(self) -> None:
        """The transport reason is the one an editor is most likely to undo."""
        from prompts.material import PLAIN_MATH_NOTATION

        assert "backslash" in PLAIN_MATH_NOTATION.lower()


class TestLatexIsNoLongerRequested:
    """The exact mandates this issue was filed about."""

    @pytest.mark.parametrize("name", sorted(TEXT_SURFACES))
    def test_no_prompt_asks_for_latex(self, name: str) -> None:
        prompt = TEXT_SURFACES[name]
        assert "Use LaTeX" not in prompt
        assert "$ inline" not in prompt
        assert "$$ display" not in prompt

    @pytest.mark.parametrize("name", sorted(TEXT_SURFACES))
    def test_latex_is_forbidden_by_name(self, name: str) -> None:
        assert "do not write latex" in TEXT_SURFACES[name].lower()


class TestSlideNotationIsUnchanged:
    """Option A leaves the slide path alone: AsciiMath is what that renderer reads."""

    def test_the_asciimath_loader_is_still_required(self) -> None:
        assert 'load: ["input/asciimath"]' in _JSX_SYSTEM_PROMPT

    def test_latex_is_still_forbidden_in_slides(self) -> None:
        assert "Do NOT use LaTeX" in _JSX_SYSTEM_PROMPT

    def test_the_two_notations_remain_deliberately_different(self) -> None:
        """A learner crosses between them, so the policy has to be explicit on both
        sides rather than one prompt that happens to be right."""
        from prompts.material import PLAIN_MATH_NOTATION

        assert "AsciiMath" in _JSX_SYSTEM_PROMPT
        assert "AsciiMath" not in PLAIN_MATH_NOTATION
