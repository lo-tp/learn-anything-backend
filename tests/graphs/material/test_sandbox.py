"""Tests for graphs/material/sandbox.py."""

from __future__ import annotations

from itertools import pairwise
from unittest.mock import MagicMock, patch

import httpx

from graphs.material.sandbox import (
    MAX_MATERIAL_ATTEMPTS,
    SLIDE_EXTRA_BODY_PARAMS,
    SLIDE_SAMPLING_PROFILES,
    _compile_slide,
    _placeholder_slide_jsx,
    slide_sampling_bind_kwargs,
    slide_sampling_for_attempt,
)


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
        assert top_level | set(kwargs["extra_body"]) == set(profile)
        assert not (top_level & set(kwargs["extra_body"]))

    def test_tracks_attempt(self):
        # All profiles are identical, so every attempt binds the same
        # deterministic sampling end to end.
        for i in range(1, len(SLIDE_SAMPLING_PROFILES) + 1):
            assert slide_sampling_bind_kwargs(i) == slide_sampling_bind_kwargs(1)

    def test_returns_fresh_dict(self):
        a = slide_sampling_bind_kwargs(1)
        a["extra_body"]["top_k"] = 999
        assert slide_sampling_bind_kwargs(1)["extra_body"]["top_k"] != 999


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
