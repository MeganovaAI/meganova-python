"""Tests for audio Pydantic models."""

from meganova.models.audio import TranscriptionResponse


class TestTranscriptionResponse:
    def test_basic(self):
        resp = TranscriptionResponse(text="Hello world")
        assert resp.text == "Hello world"

    def test_extra_ignored(self):
        resp = TranscriptionResponse(text="Hi", language="en", duration=3.5)
        assert resp.text == "Hi"
        assert not hasattr(resp, "language")
