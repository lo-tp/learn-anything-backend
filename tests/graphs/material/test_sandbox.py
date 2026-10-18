"""Tests for graphs/material/sandbox.py."""

from __future__ import annotations

import logging
from itertools import pairwise
from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.reasoning import stage_name
from graphs.material.sandbox import (
    MATHJAX_CONTEXT_RULE,
    MATHJAX_MATH_RULE,
    MAX_BRACE_APPEND,
    MAX_MATERIAL_ATTEMPTS,
    MAX_SLIDE_SELF_REPAIR_TURNS,
    SLIDE_CONTENTS_MAX_OUTPUT_TOKENS,
    SLIDE_EXTRA_BODY_PARAMS,
    SLIDE_FAILURE_CLASSES,
    SLIDE_MAX_OUTPUT_TOKENS,
    SLIDE_REASONING_EFFORT,
    SLIDE_SAMPLING_PROFILES,
    _compile_slide,
    _latex_in_mathjax,
    _mathjax_without_context,
    _placeholder_slide_jsx,
    _strip_markdown_fences,
    _unbalanced_open_braces,
    classify_compile_error,
    compile_error_for_feedback,
    deterministic_repair,
    slide_max_output_tokens,
    slide_reasoning_effort,
    slide_sampling_bind_kwargs,
    slide_sampling_for_attempt,
    validate_jsx,
    wrap_mathjax_context,
)

# The stage every slide-writing call belongs to, spelled the way
# graphs.material.nodes.WRITE_SLIDE spells it (that module's constant is the one
# the graph actually uses; tests/core/test_reasoning.py and
# tests/graphs/material/test_nodes.py are where the two are kept agreeing).
WRITE_SLIDE = stage_name("material", "write_slide")


@pytest.mark.real_prompts
class TestPlaceholderSlideJsx:
    def test_embeds_title(self):
        jsx = _placeholder_slide_jsx("My Slide")
        assert '"My Slide"' in jsx

    def test_embeds_attempts(self):
        jsx = _placeholder_slide_jsx("Test")
        assert str(MAX_MATERIAL_ATTEMPTS) in jsx

    def test_empty_title_uses_default(self):
        jsx = _placeholder_slide_jsx("")
        assert '"This slide"' in jsx

    def test_none_title_uses_default(self):
        jsx = _placeholder_slide_jsx(None)  # type: ignore[arg-type]
        assert '"This slide"' in jsx

    def test_escapes_special_chars(self):
        jsx = _placeholder_slide_jsx('He said "hi" \\ there')
        # json.dumps escapes quotes and backslashes
        assert '\\"hi\\"' in jsx
        assert "\\\\" in jsx

    def test_is_valid_jsx_structure(self):
        jsx = _placeholder_slide_jsx("Test")
        assert jsx.startswith("export default function SlidePlaceholder()")
        assert jsx.endswith("}")


class TestSlideSamplingForAttempt:
    def test_first_attempt_is_most_creative(self):
        p1 = slide_sampling_for_attempt(1)
        assert p1 == SLIDE_SAMPLING_PROFILES[0]
        assert p1["temperature"] == max(
            p["temperature"] for p in SLIDE_SAMPLING_PROFILES
        )
        assert p1["top_p"] == max(p["top_p"] for p in SLIDE_SAMPLING_PROFILES)

    def test_last_attempt_is_most_deterministic(self):
        n = len(SLIDE_SAMPLING_PROFILES)
        p_last = slide_sampling_for_attempt(n)
        assert p_last == SLIDE_SAMPLING_PROFILES[-1]
        assert p_last["temperature"] == min(
            p["temperature"] for p in SLIDE_SAMPLING_PROFILES
        )

    def test_profiles_step_down_monotonically(self):
        temps = [
            slide_sampling_for_attempt(i)["temperature"]
            for i in range(1, len(SLIDE_SAMPLING_PROFILES) + 1)
        ]
        # Non-increasing across attempts; plateaus allowed.
        assert all(a >= b for a, b in pairwise(temps))

    def test_clamps_beyond_profile_count(self):
        n = len(SLIDE_SAMPLING_PROFILES)
        assert slide_sampling_for_attempt(n + 5) == slide_sampling_for_attempt(n)

    def test_clamps_below_one(self):
        assert slide_sampling_for_attempt(0) == slide_sampling_for_attempt(1)

    def test_returns_copy(self):
        a = slide_sampling_for_attempt(1)
        a["temperature"] = 99.0
        assert slide_sampling_for_attempt(1)["temperature"] != 99.0


class TestSlideSamplingBindKwargs:
    def test_splits_standard_and_extra_body_params(self):
        kwargs = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        profile = slide_sampling_for_attempt(1)
        # Standard params are top-level with their profile values.
        for key in ("temperature", "top_p"):
            assert kwargs[key] == profile[key]
        # Non-OpenAI params are nested under extra_body with their values.
        assert set(kwargs["extra_body"]) == SLIDE_EXTRA_BODY_PARAMS
        for key in SLIDE_EXTRA_BODY_PARAMS:
            assert kwargs["extra_body"][key] == profile[key]
            assert key not in kwargs

    def test_covers_every_profile_param_exactly_once(self):
        kwargs = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        profile = slide_sampling_for_attempt(1)
        top_level = set(kwargs) - {"extra_body"}
        # reasoning_effort and max_tokens are not profile entries; the bind
        # factory adds both to every slide-writing call (see
        # test_reasoning_effort_is_minimal and TestSlideOutputCaps).
        assert top_level | set(kwargs["extra_body"]) == set(profile) | {
            "reasoning_effort",
            "max_tokens",
        }
        assert not (top_level & set(kwargs["extra_body"]))

    def test_reasoning_effort_is_minimal_on_every_attempt(self):
        """Slide generation always runs with the smallest reasoning budget."""
        for attempt in range(1, len(SLIDE_SAMPLING_PROFILES) + 2):
            kwargs = slide_sampling_bind_kwargs(attempt, WRITE_SLIDE)
            assert kwargs["reasoning_effort"] == SLIDE_REASONING_EFFORT == "minimal"

    def test_reasoning_effort_stays_out_of_extra_body(self):
        """Nesting it is the vLLM dialect; llama.cpp only reads the top level.

        Measured against the project's llama.cpp backend: a top-level
        `reasoning_effort: "minimal"` cut the completion to 4 tokens with no
        reasoning, while `chat_template_kwargs: {"thinking": false}` was
        ignored and the model kept thinking.
        """
        kwargs = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        assert "reasoning_effort" not in kwargs["extra_body"]
        assert "reasoning_effort" in kwargs

    def test_tracks_attempt(self):
        # The bind kwargs follow the graded ladder: each attempt binds its
        # own profile, and temperature steps down monotonically toward the
        # most deterministic profile.
        kwargs = [
            slide_sampling_bind_kwargs(i, WRITE_SLIDE)
            for i in range(1, len(SLIDE_SAMPLING_PROFILES) + 1)
        ]
        for i, kw in enumerate(kwargs, start=1):
            profile = SLIDE_SAMPLING_PROFILES[i - 1]
            assert kw["temperature"] == profile["temperature"]
            assert kw["top_p"] == profile["top_p"]
            assert kw["extra_body"]["top_k"] == profile["top_k"]
        temps = [kw["temperature"] for kw in kwargs]
        assert all(a > b for a, b in pairwise(temps))

    def test_returns_fresh_dict(self):
        a = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        a["extra_body"]["top_k"] = 999
        assert slide_sampling_bind_kwargs(1, WRITE_SLIDE)["extra_body"]["top_k"] != 999


class TestTheSlideCallsReadTheSameNamesEveryOtherStageReads:
    """#194's knobs, for the two stages that were left out of its table.

    The slide defaults live beside the sampling ladder because that is where the
    measurements are; they are still only defaults. ``REASONING_EFFORT`` and
    ``LLM_MAX_OUTPUT_TOKENS`` — with or without ``_MATERIAL_WRITE_SLIDE`` on the
    end — reach these calls, so a run that turns thinking on for everything can
    measure what that costs on slides, and one stage can be put back afterwards.
    """

    def test_a_stage_can_be_put_back_to_the_measured_minimum(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "low")
        monkeypatch.setenv("REASONING_EFFORT_MATERIAL_WRITE_SLIDE", "minimal")
        assert slide_reasoning_effort(WRITE_SLIDE) == "minimal"
        bound = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        assert bound["reasoning_effort"] == "minimal"

    def test_the_global_setting_reaches_a_slide_call(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "medium")
        bound = slide_sampling_bind_kwargs(2, WRITE_SLIDE)
        assert bound["reasoning_effort"] == "medium"

    def test_an_effort_the_backend_would_reject_leaves_the_measured_default(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("REASONING_EFFORT_MATERIAL_WRITE_SLIDE", "extra-high")
        with caplog.at_level(logging.WARNING):
            assert slide_reasoning_effort(WRITE_SLIDE) == SLIDE_REASONING_EFFORT
        assert "extra-high" in caplog.text

    def test_a_slide_cap_can_be_raised_on_its_own(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS_MATERIAL_WRITE_SLIDE", "8192")
        assert slide_max_output_tokens(WRITE_SLIDE) == 8192
        assert slide_sampling_bind_kwargs(1, WRITE_SLIDE)["max_tokens"] == 8192

    def test_a_stage_named_cap_does_not_move_the_pre_slide_stages(self, monkeypatch):
        from core.reasoning import max_output_tokens_for

        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS_MATERIAL_WRITE_SLIDE", "9000")
        assert max_output_tokens_for("plan.design_plan") == 4096


class TestCompileErrorForFeedback:
    def test_strips_client_error_wrapper(self):
        raw = (
            "Transport/HTTP error: Client error '400 code must declare "
            "`export default`' for url 'http://localhost:3001/api/compile' "
            "For more information check: https://developer.mozilla.org"
        )
        assert compile_error_for_feedback(raw) == (
            "400 code must declare `export default`"
        )

    def test_strips_server_error_wrapper_with_nested_quotes(self):
        raw = (
            "Transport/HTTP error: Server error '500 Build failed with 1 error: "
            "compile.tsx:162:74: ERROR: Expected \"}\" but found \")\"' for url "
            "'http://localhost:3001/api/compile' For more information check: ..."
        )
        assert compile_error_for_feedback(raw) == (
            '500 Build failed with 1 error: compile.tsx:162:74: '
            'ERROR: Expected "}" but found ")"'
        )

    def test_plain_errors_are_whitespace_collapsed(self):
        assert compile_error_for_feedback("syntax  error\nat\nline 3") == (
            "syntax error at line 3"
        )


class TestCompileSlide:
    @patch("graphs.material.sandbox.httpx.Client")
    def test_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = lambda _: mock_client
        mock_client_cls.return_value.__exit__ = lambda *a: None
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"code": "compiled_code_here"}
        mock_client.post.return_value = mock_response

        code, error = _compile_slide("some jsx")
        assert code == "compiled_code_here"
        assert error == ""

    @patch("graphs.material.sandbox.httpx.Client")
    def test_sandbox_returns_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = lambda _: mock_client
        mock_client_cls.return_value.__exit__ = lambda *a: None
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "syntax error at line 3"}
        mock_client.post.return_value = mock_response

        code, error = _compile_slide("bad jsx")
        assert code is None
        assert "syntax error" in error

    @patch("graphs.material.sandbox.httpx.Client")
    def test_transport_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = lambda _: mock_client
        mock_client_cls.return_value.__exit__ = lambda *a: None
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        code, error = _compile_slide("some jsx")
        assert code is None
        assert "Transport/HTTP error" in error
        assert "connection refused" in error

    @patch("graphs.material.sandbox.httpx.Client")
    def test_http_error_raises_transport(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = lambda _: mock_client
        mock_client_cls.return_value.__exit__ = lambda *a: None
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_client.post.return_value = mock_response

        code, error = _compile_slide("some jsx")
        assert code is None
        assert "Transport/HTTP error" in error

    @patch("graphs.material.sandbox.httpx.Client")
    def test_empty_code_falls_back_to_input(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = lambda _: mock_client
        mock_client_cls.return_value.__exit__ = lambda *a: None
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"code": ""}
        mock_client.post.return_value = mock_response

        code, error = _compile_slide("original")
        assert code == "original"
        assert error == ""


# --- Deterministic repairs (run before a failure is reported) ---


class TestStripMarkdownFences:
    def test_no_fences_unchanged(self):
        code = "export default function S() { return null; }"
        assert _strip_markdown_fences(code) == code

    def test_strips_fenced_block_with_language_tag(self):
        code = "```tsx\nexport default function S() { return null; }\n```"
        assert _strip_markdown_fences(code) == (
            "export default function S() { return null; }"
        )

    def test_strips_fenced_block_without_language_tag(self):
        code = "```\nexport default function S() { return null; }\n```"
        assert _strip_markdown_fences(code) == (
            "export default function S() { return null; }"
        )

    def test_drops_prose_before_the_block(self):
        code = "Here is the slide:\n```tsx\nconst x = 1;\n```"
        assert _strip_markdown_fences(code) == "const x = 1;"

    def test_drops_prose_after_the_block(self):
        code = "```tsx\nconst x = 1;\n```\nHope that helps!"
        assert _strip_markdown_fences(code) == "const x = 1;"

    def test_single_unterminated_fence_unchanged(self):
        code = "const x = `a`;\nconst y = 1;"
        # One backtick-delimited template literal, no real fence pair.
        assert _strip_markdown_fences(code) == code

    def test_only_one_fence_marker_unchanged(self):
        code = "const x = 1;\n```"
        # A lone trailing ``` with no opening block is not a pair.
        assert _strip_markdown_fences(code) == code

    def test_empty_input(self):
        assert _strip_markdown_fences("") == ""


class TestUnbalancedOpenBraces:
    def test_balanced_is_zero(self):
        assert _unbalanced_open_braces("{ a: { b: 1 } }") == 0

    def test_one_unclosed_open(self):
        assert _unbalanced_open_braces("function S() { return <div>") == 1

    def test_excess_closers_is_negative(self):
        assert _unbalanced_open_braces("} } {") < 0

    def test_ignores_braces_in_single_quotes(self):
        assert _unbalanced_open_braces("const s = '}';") == 0

    def test_ignores_braces_in_double_quotes(self):
        assert _unbalanced_open_braces('const s = "{";') == 0

    def test_ignores_braces_in_template_literals(self):
        assert _unbalanced_open_braces("const s = `a ${b} c`; ") == 0

    def test_ignores_line_comment_braces(self):
        assert _unbalanced_open_braces("const x = 1; // { ") == 0

    def test_ignores_block_comment_braces(self):
        assert _unbalanced_open_braces("/* } } } */ const x = 1;") == 0

    def test_unclosed_inside_a_component(self):
        code = "export default function S() { return <div className='a'>"
        # One open for the function body, plus the JSX has no unclosed brace
        # here (className is a string). Expect 1.
        assert _unbalanced_open_braces(code) == 1

    def test_line_comment_ending_at_newline_is_skipped(self):
        # A line comment holding a brace, then balanced code after the newline.
        code = "const a = 1; // {\nconst b = {x: 1};"
        assert _unbalanced_open_braces(code) == 0

    def test_escaped_quote_in_string_is_skipped(self):
        # An escaped quote does not end the string, so the trailing brace is
        # still matched.
        code = 'const s = "a\\"b"; const o = {}'
        assert _unbalanced_open_braces(code) == 0


class TestDeterministicRepair:
    def test_clean_code_is_unchanged(self):
        code = "export default function S() { return <div />; }"
        repaired, repairs = deterministic_repair(code)
        assert repaired == code
        assert repairs == []

    def test_strips_fences_and_reports(self):
        code = "```tsx\nexport default function S() { return <div />; }\n```"
        repaired, repairs = deterministic_repair(code)
        assert repaired == "export default function S() { return <div />; }"
        assert "strip_markdown_fences" in repairs

    def test_appends_missing_braces_and_reports(self):
        code = "export default function S() { return <div>"
        repaired, repairs = deterministic_repair(code)
        assert repaired.endswith("}")
        assert "balance_braces" in repairs

    def test_does_not_touch_balanced_braces(self):
        code = "export default function S() { return <div />; }"
        _, repairs = deterministic_repair(code)
        assert "balance_braces" not in repairs

    def test_both_repairs_together(self):
        code = "```tsx\nexport default function S() { return <div>\n```"
        # Fence strip yields `export default function S() { return <div>`;
        # that has one unclosed brace, which balance_braces then appends.
        repaired, repairs = deterministic_repair(code)
        assert "strip_markdown_fences" in repairs
        assert "balance_braces" in repairs
        assert repaired.endswith("}")

    def test_brace_balance_is_capped(self):
        # Far more open than close: only up to MAX_BRACE_APPEND are appended.
        code = "function f("
        code += "{" * (MAX_BRACE_APPEND + 5)
        repaired, _ = deterministic_repair(code)
        appended = repaired.count("}") - code.count("}")
        assert appended == MAX_BRACE_APPEND

    def test_does_not_repair_excess_closers(self):
        # Excess closers are not (dangerously) deleted.
        code = "} const x = 1;}"
        _, repairs = deterministic_repair(code)
        assert "balance_braces" not in repairs

    def test_returns_fresh_string(self):
        code = "```tsx\nx\n```"
        a, _ = deterministic_repair(code)
        # Strings are immutable; ensure we did not mutate the input object.
        assert deterministic_repair(code)[0] == a


# --- validate_jsx: the self-repair tool bound to write_slide ---


class TestValidateJsx:
    @patch("graphs.material.sandbox._compile_slide")
    def test_success(self, mock_compile):
        mock_compile.return_value = ("compiled", "")
        code = "export default function S() { return <div />; }"
        result = validate_jsx(code)
        assert result["ok"] is True
        assert result["compiled_code"] == "compiled"
        assert result["source"] == code
        assert result["error"] == ""
        assert result["repairs"] == []

    @patch("graphs.material.sandbox._compile_slide")
    def test_success_after_deterministic_repair(self, mock_compile):
        # The raw (fenced) code fails to compile; the fence-stripped repair
        # then compiles. Two sandbox calls, in that order.
        mock_compile.side_effect = [(None, "syntax error"), ("compiled", "")]
        code = "```tsx\nexport default function S() { return <div />; }\n```"
        result = validate_jsx(code)
        assert result["ok"] is True
        # The source is the fence-stripped version that was compiled.
        assert result["source"] == "export default function S() { return <div />; }"
        assert "strip_markdown_fences" in result["repairs"]

    @patch("graphs.material.sandbox._compile_slide")
    def test_valid_slide_with_apostrophe_in_text_is_not_corrupted(self, mock_compile):
        """An apostrophe in JSX *text* (e.g. ``Let's``) is not a string
        delimiter. The raw code is valid, so validate_jsx must compile it as-is
        and return it unchanged — never appending a spurious closing brace.
        """
        code = "export default function S() { return <div>Let's go</div>; }"
        mock_compile.return_value = ("compiled", "")
        result = validate_jsx(code)
        assert result["ok"] is True
        assert result["source"] == code  # not corrupted
        assert result["repairs"] == []
        # Compiled the raw code once — no repair recompile.
        assert mock_compile.call_count == 1
        assert mock_compile.call_args[0][0] == code

    @patch("graphs.material.sandbox._compile_slide")
    def test_failure_returns_reduced_error(self, mock_compile):
        mock_compile.return_value = (None, "syntax error at line 3")
        result = validate_jsx("bad jsx")
        assert result["ok"] is False
        assert result["compiled_code"] is None
        assert result["error"] == "syntax error at line 3"
        assert result["source"] == "bad jsx"

    @patch("graphs.material.sandbox._compile_slide")
    def test_failure_reduces_transport_wrapper(self, mock_compile):
        raw = (
            "Transport/HTTP error: Client error '400 code must declare "
            "`export default`' for url 'http://localhost:3001/api/compile' "
            "For more information check: https://developer.mozilla.org"
        )
        mock_compile.return_value = (None, raw)
        result = validate_jsx("bad")
        assert result["ok"] is False
        assert result["error"] == "400 code must declare `export default`"
        # The raw transport wrapper is not echoed.
        assert "Transport/HTTP error" not in result["error"]

    @patch("graphs.material.sandbox._compile_slide")
    def test_records_repairs_on_failure(self, mock_compile):
        mock_compile.return_value = (None, "still broken")
        code = "```tsx\nfunction S() { return <div>\n```"
        result = validate_jsx(code)
        assert result["ok"] is False
        assert "strip_markdown_fences" in result["repairs"]
        assert "balance_braces" in result["repairs"]


# --- MathJax context contract (renders blank, compiles green) ---

_MATHJAX_SLIDE = (
    'export default function Slide() {\n'
    '  return <div><MathJax inline>{"`x^2`"}</MathJax></div>;\n'
    '}'
)
_WRAPPED_SLIDE = (
    'export default function Slide() {\n'
    '  return <MathJaxContext config={{ loader: { load: ["input/asciimath"] } }}>'
    '<div><MathJax inline>{"`x^2`"}</MathJax></div></MathJaxContext>;\n'
    '}'
)
_ARROW_SLIDE = 'export default () => <div><MathJax inline>{"`x^2`"}</MathJax></div>;'


class TestMathjaxWithoutContextDetection:
    def test_detects_mathjax_tag_without_context(self):
        assert _mathjax_without_context(_MATHJAX_SLIDE) is True
        assert _mathjax_without_context("<MathJax/>") is True
        assert _mathjax_without_context("<MathJax\n dynamic>") is True

    def test_context_tag_closes_the_contract(self):
        # <MathJaxContext> must never match the <MathJax> tag pattern.
        assert _mathjax_without_context(_WRAPPED_SLIDE) is False
        assert _mathjax_without_context("<MathJaxContext>") is False
        assert _mathjax_without_context("<MathJaxContext config={{}}>\n") is False

    def test_no_math_or_prose_mentions_are_not_math(self):
        assert _mathjax_without_context("export default function S(){return <div/>;}") is False
        assert _mathjax_without_context(
            'export default function S(){return <p>MathJax is a library</p>;}'
        ) is False


class TestWrapMathjaxContext:
    def test_named_default_export_is_renamed_and_wrapped(self):
        wrapped, applied = wrap_mathjax_context(_MATHJAX_SLIDE)
        assert applied is True
        # The original component becomes a plain function; the model's content
        # is untouched — nothing moved, nothing deleted.
        assert "function Slide() {" in wrapped
        assert 'return <div><MathJax inline>{"`x^2`"}</MathJax></div>;' in wrapped
        # The new default export renders it inside the mandated wrapper, with
        # the exact config the prompt mandates.
        assert (
            'export default function __SlideWithMathJaxContext() {\n'
            '  return <MathJaxContext config={{ loader: { load: ["input/asciimath"] } }}>'
            '<Slide /></MathJaxContext>;\n'
            '}' in wrapped
        )
        assert wrapped.count("export default") == 1

    def test_anonymous_default_export_gains_a_name(self):
        anon = 'export default function () {\n  return <MathJax inline>{"`x`"}</MathJax>;\n}'
        wrapped, applied = wrap_mathjax_context(anon)
        assert applied is True
        assert "function __SlideInner ()" in wrapped
        assert "<__SlideInner />" in wrapped
        assert wrapped.count("export default") == 1

    def test_unpinned_module_shape_is_declined_not_guessed(self):
        # The prompt pins "export default function"; an arrow default is not
        # safely rewritable, so the violation must be reported instead.
        unchanged, applied = wrap_mathjax_context(_ARROW_SLIDE)
        assert applied is False
        assert unchanged == _ARROW_SLIDE

    def test_declined_when_it_would_collide_with_an_existing_name(self):
        colliding = _MATHJAX_SLIDE.replace(
            "function Slide()", "function __SlideWithMathJaxContext()"
        )
        _, applied = wrap_mathjax_context(colliding)
        assert applied is False

    def test_noop_when_contract_is_met_or_absent(self):
        assert wrap_mathjax_context(_WRAPPED_SLIDE) == (_WRAPPED_SLIDE, False)
        plain = "export default function S(){return <div/>;}"
        assert wrap_mathjax_context(plain) == (plain, False)


class TestValidateJsxMathjaxContract:
    @staticmethod
    def _compile_by_context(code: str):
        # Distinguishes the raw slide from the wrapped variant by content.
        return ("compiled-wrapped", "") if "MathJaxContext" in code else ("compiled-raw", "")

    @patch("graphs.material.sandbox._compile_slide")
    def test_compiled_slide_missing_context_is_deterministically_wrapped(self, mock_compile):
        mock_compile.side_effect = self._compile_by_context
        result = validate_jsx(_MATHJAX_SLIDE)
        assert result["ok"] is True
        assert result["repairs"] == ["wrap_mathjax_context"]
        # The stored source is the wrapped module, and its own compiled output
        # is what is returned — not the raw slide's.
        assert _mathjax_without_context(result["source"]) is False
        assert result["compiled_code"] == "compiled-wrapped"

    @patch("graphs.material.sandbox._compile_slide")
    def test_compliant_slide_compiles_once_and_is_untouched(self, mock_compile):
        mock_compile.side_effect = self._compile_by_context
        result = validate_jsx(_WRAPPED_SLIDE)
        assert result["ok"] is True
        assert result["source"] == _WRAPPED_SLIDE
        assert result["repairs"] == []
        assert mock_compile.call_count == 1

    @patch("graphs.material.sandbox._compile_slide")
    def test_unwritable_shape_reports_the_contract(self, mock_compile):
        mock_compile.side_effect = self._compile_by_context
        result = validate_jsx(_ARROW_SLIDE)
        assert result["ok"] is False
        assert "<MathJaxContext>" in result["error"]
        assert "renders blank" in result["error"]
        assert result["repairs"] == []

    @patch("graphs.material.sandbox._compile_slide")
    def test_wrap_that_does_not_compile_falls_back_to_the_model(self, mock_compile):
        mock_compile.side_effect = [("compiled-raw", ""), (None, "Build failed with 1 error")]
        result = validate_jsx(_MATHJAX_SLIDE)
        assert result["ok"] is False
        assert "Build failed with 1 error" in result["error"]
        # The contract wording rides along so the repair turn fixes both.
        assert "<MathJaxContext>" in result["error"]
        assert result["repairs"] == []
        # The second compile call was the wrapped variant.
        assert "MathJaxContext" in mock_compile.call_args_list[1][0][0]

    @patch("graphs.material.sandbox._compile_slide")
    def test_failed_compile_also_gets_the_wrap_repair(self, mock_compile):
        # The raw slide does not compile; after the deterministic wrap the
        # wrapped variant does.
        def fake(code):
            if code == _MATHJAX_SLIDE:
                return (None, "syntax error")
            return self._compile_by_context(code)
        mock_compile.side_effect = fake
        result = validate_jsx(_MATHJAX_SLIDE)
        assert result["ok"] is True
        assert result["repairs"] == ["wrap_mathjax_context"]

    @patch("graphs.material.sandbox._compile_slide")
    def test_repaired_but_unwritable_slide_still_reports_the_contract(self, mock_compile):
        # A truncated arrow-default slide: its one unclosed brace gets
        # balanced (a repair it genuinely needs) and the repaired source
        # compiles — it must *still* fail the contract, reporting both the
        # applied repairs and the contract wording to the repair turn.
        truncated = 'export default () => { return <MathJax inline>{"`x`"}</MathJax>; '

        def fake(code):
            if code == truncated:
                return (None, "unexpected end of input")
            return ("compiled", "")
        mock_compile.side_effect = fake
        result = validate_jsx(truncated)
        assert result["ok"] is False
        assert "balance_braces" in result["repairs"]
        assert "<MathJaxContext>" in result["error"]


class TestMathjaxContractClassification:
    def test_contract_violation_is_content_spec_invalid(self):
        # A slide that compiles but breaks the slide contract: the honest
        # class is the content gate's, not a syntax error.
        assert classify_compile_error(MATHJAX_CONTEXT_RULE) == "content_spec_invalid"

    def test_syntax_errors_are_not_swallowed_by_the_new_token(self):
        assert (
            classify_compile_error('500 Build failed with 1 error: Unexpected ">"')
            == "syntax_error"
        )


class TestMathSyntaxClassification:
    def test_math_rule_is_math_notation(self):
        assert classify_compile_error(MATHJAX_MATH_RULE) == "math_notation"

    def test_flagged_macros_do_not_change_the_class(self):
        assert (
            classify_compile_error(f"{MATHJAX_MATH_RULE} Flagged: \\sqrt, \\frac")
            == "math_notation"
        )

    def test_math_rule_wins_over_a_generic_syntax_error(self):
        # The self-repair turn fixes the math first; the generic parser
        # complaint riding along does not steal the class.
        assert (
            classify_compile_error("syntax error at line 3 " + MATHJAX_MATH_RULE)
            == "math_notation"
        )

    def test_syntax_errors_are_not_swallowed_by_the_math_token(self):
        assert (
            classify_compile_error('500 Build failed with 1 error: Unexpected ">"')
            == "syntax_error"
        )

    def test_both_contracts_classify_as_content_spec_invalid(self):
        # A slide that breaks both render-time contracts carries both rule
        # wordings; the context token is checked first, matching how a
        # compile error plus the context rule has always classified.
        combined = f"{MATHJAX_CONTEXT_RULE} {MATHJAX_MATH_RULE} Flagged: \\sqrt"
        assert classify_compile_error(combined) == "content_spec_invalid"


# --- Math-syntax contract: math inside <MathJax> must be AsciiMath ---
#
# A backslash macro inside a <MathJax> string compiles green and passes the
# sandbox's headless gate, but the browser harness loads only the AsciiMath
# input, where a backslash has no meaning — the macro renders as literal
# text (measured: 526 of 606 stored components carry one, #198 problem 2).

_CLEAN_MATH_SLIDE = (
    "export default function Slide() {\n"
    '  return <MathJaxContext config={{ loader: { load: ["input/asciimath"] } }}>'
    '<div>'
    '<MathJax>{"`I(V(I)) = sqrt(I)`"}</MathJax>'
    "</div></MathJaxContext>;\n"
    "}"
)
# The source form a model actually writes: a backslash inside a JS string is
# escaped, so the file carries ``\\sqrt`` (two backslash characters).
_LATEX_MATH_SLIDE = (
    "export default function Slide() {\n"
    '  return <MathJaxContext config={{ loader: { load: ["input/asciimath"] } }}>'
    '<div>'
    '<MathJax>{"`\\\\sqrt{I}`"}</MathJax>'
    "</div></MathJaxContext>;\n"
    "}"
)
_LATEX_MATH_NO_CONTEXT_SLIDE = (
    "export default function Slide() {\n"
    '  return <div><MathJax>{"`\\\\sqrt{I}`"}</MathJax></div>;\n'
    "}"
)
_LATEX_MATH_ARROW_SLIDE = (
    # No context, unwritable (arrow) shape.
    "export default () => <div><MathJax>{\"`\\\\sqrt{I}`\"}</MathJax></div>;\n"
)
_PROSE_BACKSLASH_SLIDE = (
    "export default function Slide() {\n"
    '  return <MathJaxContext config={{ loader: { load: ["input/asciimath"] } }}>'
    '<div><p>the \\sqrt notation in prose</p>'
    '<MathJax>{"`sqrt(I)`"}</MathJax></div></MathJaxContext>;\n'
    "}"
)


class TestLatexInMathjaxDetection:
    def test_finds_escaped_macros_in_math_blocks(self):
        assert _latex_in_mathjax(_LATEX_MATH_SLIDE) == ["\\sqrt"]

    def test_single_backslash_form_is_found(self):
        # A source carrying a lone (unescaped) backslash in a math string.
        code = '<div><MathJax>{"`\\sqrt{I}`"}</MathJax></div>'
        assert _latex_in_mathjax(code) == ["\\sqrt"]

    def test_multiple_macros_keep_first_seen_order_and_dedupe(self):
        code = '<MathJax>{"`\\\\frac{a}{b} \\sqrt{I} \\\\frac{c}{d}`"}</MathJax>'
        assert _latex_in_mathjax(code) == ["\\frac", "\\sqrt"]

    def test_delimiter_macros_are_named(self):
        code = '<MathJax>{"`\\\\(x\\\\)`"}</MathJax>'
        assert _latex_in_mathjax(code) == ["\\(", "\\)"]

    def test_clean_asciimath_is_absent(self):
        assert _latex_in_mathjax(_CLEAN_MATH_SLIDE) == []

    def test_backslash_outside_math_blocks_is_not_math(self):
        assert _latex_in_mathjax(_PROSE_BACKSLASH_SLIDE) == []

    def test_unnamed_backslash_is_still_reported(self):
        # A backslash with no macro letters after it is still a violation —
        # AsciiMath has no backslash at all.
        code = '<MathJax>{"`a \\ b`"}</MathJax>'
        assert _latex_in_mathjax(code) == ["\\"]


class TestValidateJsxMathSyntax:
    @patch("graphs.material.sandbox._compile_slide")
    def test_compiled_latex_math_is_rejected_and_named(self, mock_compile):
        mock_compile.return_value = ("compiled", "")
        result = validate_jsx(_LATEX_MATH_SLIDE)
        assert result["ok"] is False
        assert result["compiled_code"] is None
        assert MATHJAX_MATH_RULE in result["error"]
        # The macro the model must fix is named, in canonical single-backslash
        # form.
        assert "\\sqrt" in result["error"]
        # No deterministic repair exists for this contract: the source is
        # untouched and the violation feeds a self-repair turn.
        assert result["source"] == _LATEX_MATH_SLIDE
        assert result["repairs"] == []
        assert mock_compile.call_count == 1

    @patch("graphs.material.sandbox._compile_slide")
    def test_clean_asciimath_slide_is_untouched(self, mock_compile):
        mock_compile.return_value = ("compiled", "")
        result = validate_jsx(_CLEAN_MATH_SLIDE)
        assert result["ok"] is True
        assert result["source"] == _CLEAN_MATH_SLIDE
        assert result["repairs"] == []

    @patch("graphs.material.sandbox._compile_slide")
    def test_backslash_in_prose_is_not_flagged(self, mock_compile):
        # The contract is on the math: a backslash in a text node is a
        # different (transport) problem, not this gate's.
        mock_compile.return_value = ("compiled", "")
        result = validate_jsx(_PROSE_BACKSLASH_SLIDE)
        assert result["ok"] is True
        assert result["repairs"] == []

    @patch("graphs.material.sandbox._compile_slide")
    def test_failed_compile_carries_the_math_rule_too(self, mock_compile):
        mock_compile.return_value = (None, "syntax error at line 3")
        result = validate_jsx(_LATEX_MATH_SLIDE)
        assert result["ok"] is False
        assert "syntax error at line 3" in result["error"]
        assert MATHJAX_MATH_RULE in result["error"]
        assert "\\sqrt" in result["error"]

    @patch("graphs.material.sandbox._compile_slide")
    def test_wrappable_slide_with_latex_math_is_rejected_after_the_wrap(self, mock_compile):
        # The context wrap is applied (and recompiled) but cannot fix the
        # math: the slide must still fail, naming the macro.
        mock_compile.side_effect = [
            ("compiled-raw", ""),
            ("compiled-wrapped", ""),
        ]
        result = validate_jsx(_LATEX_MATH_NO_CONTEXT_SLIDE)
        assert result["ok"] is False
        assert result["repairs"] == ["wrap_mathjax_context"]
        assert "MathJaxContext" in result["source"]
        assert MATHJAX_MATH_RULE in result["error"]
        assert "\\sqrt" in result["error"]

    @patch("graphs.material.sandbox._compile_slide")
    def test_unwritable_slide_reports_both_contracts(self, mock_compile):
        # No context and LaTeX math, in a shape no deterministic wrap can
        # fix: the repair turn must see both contracts it has to repair.
        mock_compile.return_value = ("compiled", "")
        result = validate_jsx(_LATEX_MATH_ARROW_SLIDE)
        assert result["ok"] is False
        assert "<MathJaxContext>" in result["error"]
        assert MATHJAX_MATH_RULE in result["error"]
        assert "\\sqrt" in result["error"]
        assert result["repairs"] == []


# --- Failure-class classifier (supports the skip-rate measurement) ---


class TestClassifyCompileError:
    def test_sandbox_timeout(self):
        assert classify_compile_error("Request timed out after 30s") == "sandbox_timeout"
        assert classify_compile_error("504 Gateway Timeout") == "sandbox_timeout"

    def test_unknown_component(self):
        assert (
            classify_compile_error("could not resolve `./MyWidget`")
            == "unknown_component"
        )
        assert (
            classify_compile_error("no matching export `Card` from `lib`")
            == "unknown_component"
        )
        assert (
            classify_compile_error("`Foo` is not defined") == "unknown_component"
        )

    def test_truncated_output(self):
        assert classify_compile_error("Unexpected eof") == "truncated_output"
        assert classify_compile_error("unexpected end of file") == "truncated_output"

    def test_syntax_error(self):
        assert (
            classify_compile_error(
                '500 Build failed: ERROR: Expected "}" but found ")"'
            )
            == "syntax_error"
        )
        assert classify_compile_error("syntax error at line 3") == "syntax_error"

    def test_other(self):
        assert classify_compile_error("some opaque failure") == "other"

    def test_content_spec_invalid_is_the_sandbox_content_gate(self):
        # Real 400 responses from the sandbox gate (learn-anything-sandbox
        # app/api/compile/route.ts): the parser never rejected the code, what
        # came back is not a slide.
        assert (
            classify_compile_error("400 code must be at least 40 characters")
            == "content_spec_invalid"
        )
        assert (
            classify_compile_error("400 code must declare `export default`")
            == "content_spec_invalid"
        )
        assert (
            classify_compile_error(
                "400 component renders empty or whitespace-only markup"
            )
            == "content_spec_invalid"
        )
        assert (
            classify_compile_error(
                "400 component failed the gate: default export must be a "
                "function (a React component)"
            )
            == "content_spec_invalid"
        )
        # A render that throws on a props-less render is a contract violation.
        assert (
            classify_compile_error(
                "400 component failed the gate: Cannot read properties of "
                "undefined (reading 'page')"
            )
            == "content_spec_invalid"
        )
        assert classify_compile_error("400 invalid JSON body") == (
            "content_spec_invalid"
        )

    def test_gate_rejection_naming_an_unresolved_name_is_unknown_component(self):
        # Same gate, different cause: the throw names a name the module never
        # declared, which is the unknown-component case, not a contract one.
        assert (
            classify_compile_error(
                "400 component failed the gate: arrowRowStyle is not defined"
            )
            == "unknown_component"
        )

    def test_unterminated_string_is_a_syntax_error_not_truncation(self):
        # Checked against persisted attempts: the "Unterminated string literal"
        # rows are complete modules with a quoting bug inside a math string, so
        # only the parser running out of input counts as truncation.
        assert (
            classify_compile_error(
                "500 Build failed with 1 error: compile.tsx:270:56: ERROR: "
                'Unterminated string literal'
            )
            == "syntax_error"
        )

    def test_priority_timeout_wins_over_syntax(self):
        # A timeout mentioning a parse detail still classifies as a timeout.
        assert (
            classify_compile_error("timed out while parsing JSX")
            == "sandbox_timeout"
        )

    def test_every_class_in_the_taxonomy_is_reachable(self):
        """The closed set the C1.2 report prints, and every entry is reachable.

        The sample strings are real persisted errors (see
        tests/fixtures/slide_failure_attempts.json); ``other`` is the residual.
        """
        samples = {
            "syntax_error": '500 Build failed with 1 error: ERROR: Unexpected ">"',
            "unknown_component": "400 component failed the gate: Bin is not defined",
            "truncated_output": (
                '500 Build failed: ERROR: Unexpected end of file before a '
                'closing "MathJaxContext" tag'
            ),
            "sandbox_timeout": "Transport/HTTP error: Read timed out.",
            "content_spec_invalid": "400 code must declare `export default`",
            # The math-syntax gate's own rejection (validate_jsx), as emitted.
            "math_notation": f"{MATHJAX_MATH_RULE} Flagged: \\sqrt",
            "other": "500 Internal Server Error",
        }
        assert set(samples) == set(SLIDE_FAILURE_CLASSES)
        for failure_class, error in samples.items():
            assert classify_compile_error(error) == failure_class


# --- Self-repair bound ---


class TestMaxSelfRepairTurns:
    def test_is_a_non_negative_int(self):
        assert isinstance(MAX_SLIDE_SELF_REPAIR_TURNS, int)
        assert MAX_SLIDE_SELF_REPAIR_TURNS >= 0

    def test_default_is_two(self):
        # The unconfigured default allows two repair turns (three writes).
        assert MAX_SLIDE_SELF_REPAIR_TURNS == 2


class TestSlideOutputCaps:
    """#194: the slide calls were the only model calls left running under
    ``core/llm.py``'s global 65536, which bounds nothing a slide needs."""

    def test_every_write_attempt_caps_what_the_model_may_write(self):
        for attempt in range(1, len(SLIDE_SAMPLING_PROFILES) + 3):
            bound = slide_sampling_bind_kwargs(attempt, WRITE_SLIDE)
            assert bound["max_tokens"] == SLIDE_MAX_OUTPUT_TOKENS

    def test_the_cap_sits_well_above_the_largest_slide_ever_measured(self):
        """Real slides came back at 356-686 completion tokens (#194). The cap is
        a runaway guard, not a budget to fill."""
        assert 4096 <= SLIDE_MAX_OUTPUT_TOKENS < 65536

    def test_a_step_with_many_slide_specs_gets_a_wider_cap(self):
        """How many specs a step asks for is the model's choice, so this cap is
        above a whole plan, not above one slide."""
        assert SLIDE_CONTENTS_MAX_OUTPUT_TOKENS > SLIDE_MAX_OUTPUT_TOKENS

    def test_the_cap_stays_out_of_extra_body(self):
        """It is a standard OpenAI field; nesting it is how a param gets ignored."""
        kwargs = slide_sampling_bind_kwargs(1, WRITE_SLIDE)
        assert "max_tokens" not in kwargs.get("extra_body", {})

    def test_the_output_setting_covers_the_slide_calls_too(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "8192")
        assert slide_sampling_bind_kwargs(1, WRITE_SLIDE)["max_tokens"] == 8192
        assert slide_max_output_tokens(WRITE_SLIDE) == 8192
