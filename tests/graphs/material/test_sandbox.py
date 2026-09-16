"""Tests for graphs/material/sandbox.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from graphs.material.sandbox import (
    MAX_MATERIAL_ATTEMPTS,
    _compile_slide,
    _placeholder_slide_jsx,
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
