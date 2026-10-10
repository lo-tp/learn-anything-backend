"""Tests for graphs/material/sandbox.py."""

from __future__ import annotations

from itertools import pairwise
from unittest.mock import MagicMock, patch

import httpx
import pytest

from graphs.material.sandbox import (
    MAX_BRACE_APPEND,
    MAX_MATERIAL_ATTEMPTS,
    MAX_SLIDE_SELF_REPAIR_TURNS,
    SLIDE_EXTRA_BODY_PARAMS,
    SLIDE_REASONING_EFFORT,
    SLIDE_SAMPLING_PROFILES,
    _compile_slide,
    _placeholder_slide_jsx,
    _strip_markdown_fences,
    _unbalanced_open_braces,
    classify_compile_error,
    compile_error_for_feedback,
    deterministic_repair,
    slide_sampling_bind_kwargs,
    slide_sampling_for_attempt,
    validate_jsx,
)


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
        kwargs = slide_sampling_bind_kwargs(1)
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
        kwargs = slide_sampling_bind_kwargs(1)
        profile = slide_sampling_for_attempt(1)
        top_level = set(kwargs) - {"extra_body"}
        # reasoning_effort is not a profile entry; the bind factory adds it to
        # every slide-writing call (see test_reasoning_effort_is_minimal).
        assert top_level | set(kwargs["extra_body"]) == set(profile) | {
            "reasoning_effort"
        }
        assert not (top_level & set(kwargs["extra_body"]))

    def test_reasoning_effort_is_minimal_on_every_attempt(self):
        """Slide generation always runs with the smallest reasoning budget."""
        for attempt in range(1, len(SLIDE_SAMPLING_PROFILES) + 2):
            kwargs = slide_sampling_bind_kwargs(attempt)
            assert kwargs["reasoning_effort"] == SLIDE_REASONING_EFFORT == "minimal"

    def test_reasoning_effort_stays_out_of_extra_body(self):
        """Nesting it is the vLLM dialect; llama.cpp only reads the top level.

        Measured against the project's llama.cpp backend: a top-level
        `reasoning_effort: "minimal"` cut the completion to 4 tokens with no
        reasoning, while `chat_template_kwargs: {"thinking": false}` was
        ignored and the model kept thinking.
        """
        kwargs = slide_sampling_bind_kwargs(1)
        assert "reasoning_effort" not in kwargs["extra_body"]
        assert "reasoning_effort" in kwargs

    def test_tracks_attempt(self):
        # All profiles are identical, so every attempt binds the same
        # deterministic sampling end to end.
        for i in range(1, len(SLIDE_SAMPLING_PROFILES) + 1):
            assert slide_sampling_bind_kwargs(i) == slide_sampling_bind_kwargs(1)

    def test_returns_fresh_dict(self):
        a = slide_sampling_bind_kwargs(1)
        a["extra_body"]["top_k"] = 999
        assert slide_sampling_bind_kwargs(1)["extra_body"]["top_k"] != 999


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

    def test_priority_timeout_wins_over_syntax(self):
        # A timeout mentioning a parse detail still classifies as a timeout.
        assert (
            classify_compile_error("timed out while parsing JSX")
            == "sandbox_timeout"
        )


# --- Self-repair bound ---


class TestMaxSelfRepairTurns:
    def test_is_a_non_negative_int(self):
        assert isinstance(MAX_SLIDE_SELF_REPAIR_TURNS, int)
        assert MAX_SLIDE_SELF_REPAIR_TURNS >= 0

    def test_default_is_two(self):
        # The unconfigured default allows two repair turns (three writes).
        assert MAX_SLIDE_SELF_REPAIR_TURNS == 2
