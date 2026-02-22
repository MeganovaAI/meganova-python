"""Tests for AudioResource."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest

from meganova.models.audio import TranscriptionResponse
from meganova.resources.audio import AudioResource
from tests.conftest import make_transcription_response


class TestAudioTranscribe:
    def test_transcribe_from_path(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response("Hello world")
        resource = AudioResource(mock_sync_transport)

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        result = resource.transcribe(str(audio_file))
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello world"

    def test_transcribe_from_path_object(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response("Hi")
        resource = AudioResource(mock_sync_transport)

        from pathlib import Path
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        result = resource.transcribe(Path(audio_file))
        assert result.text == "Hi"

    def test_transcribe_from_file_object(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_transcription_response("World")
        resource = AudioResource(mock_sync_transport)

        file_obj = io.BytesIO(b"fake audio data")
        result = resource.transcribe(file_obj, filename="custom.mp3")
        assert result.text == "World"

    def test_sends_correct_endpoint(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response()
        resource = AudioResource(mock_sync_transport)

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")

        resource.transcribe(str(audio_file))
        args = mock_sync_transport.request.call_args
        assert args.args[1] == "/audio/transcriptions"

    def test_sends_model_in_data(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response()
        resource = AudioResource(mock_sync_transport)

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")

        resource.transcribe(str(audio_file), model="whisper-large-v3")
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["data"]["model"] == "whisper-large-v3"

    def test_default_model(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response()
        resource = AudioResource(mock_sync_transport)

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")

        resource.transcribe(str(audio_file))
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["data"]["model"] == "Systran/faster-whisper-large-v3"

    def test_file_object_uses_filename_hint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_transcription_response()
        resource = AudioResource(mock_sync_transport)

        file_obj = io.BytesIO(b"data")
        resource.transcribe(file_obj, filename="recording.wav")
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        files = call_kwargs["files"]
        assert files["file"][0] == "recording.wav"

    def test_path_uses_actual_filename(self, mock_sync_transport, tmp_path):
        mock_sync_transport.request.return_value = make_transcription_response()
        resource = AudioResource(mock_sync_transport)

        audio_file = tmp_path / "my_recording.mp3"
        audio_file.write_bytes(b"data")

        resource.transcribe(str(audio_file))
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        files = call_kwargs["files"]
        assert files["file"][0] == "my_recording.mp3"
