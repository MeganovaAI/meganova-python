"""Tests for VideosResource."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.errors import MeganovaError
from meganova.models.videos import VideoGeneration
from meganova.resources.videos import VideosResource
from tests.conftest import make_video_generation


class TestVideosGenerate:
    def test_basic_generation(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        result = resource.generate(prompt="a cat running", model="veo-3")
        assert isinstance(result, VideoGeneration)
        assert result.status == "processing"

    def test_correct_endpoint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="a cat", model="veo-3")
        args = mock_sync_transport.request.call_args
        assert args.args == ("POST", "/videos/generations")

    def test_payload_includes_required(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="a cat", model="veo-3")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["prompt"] == "a cat"
        assert payload["model"] == "veo-3"
        assert payload["size"] == "1280x720"
        assert payload["n_seconds"] == 5

    def test_image_url_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="animate", model="m", image_url="https://img.com/cat.png")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["image_url"] == "https://img.com/cat.png"

    def test_aspect_ratio_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="x", model="m", aspect_ratio="16:9")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == "16:9"

    def test_seed_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="x", model="m", seed=42)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["seed"] == 42

    def test_none_optionals_excluded(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.generate(prompt="x", model="m")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert "image_url" not in payload
        assert "aspect_ratio" not in payload
        assert "seed" not in payload


class TestVideosGet:
    def test_get_by_id(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation(
            status="completed", vid_id="vid_456"
        )
        resource = VideosResource(mock_sync_transport)

        result = resource.get("vid_456")
        assert result.id == "vid_456"
        assert result.status == "completed"

    def test_get_calls_correct_path(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_video_generation()
        resource = VideosResource(mock_sync_transport)

        resource.get("vid_123")
        mock_sync_transport.request.assert_called_once_with(
            "GET", "/videos/generations/vid_123"
        )


class TestVideoPolling:
    @patch("meganova.resources.videos.time")
    def test_wait_polls_until_completed(self, mock_time, mock_sync_transport):
        mock_time.time.side_effect = [0, 1, 2, 3]
        mock_time.sleep = MagicMock()

        processing = make_video_generation(status="processing")
        completed = make_video_generation(status="completed")
        mock_sync_transport.request.side_effect = [
            processing,        # initial generate
            processing,        # first poll
            completed,         # second poll
        ]

        resource = VideosResource(mock_sync_transport)
        result = resource.generate(prompt="x", model="m", wait=True, poll_interval=1, timeout=600)
        assert result.status == "completed"

    @patch("meganova.resources.videos.time")
    def test_wait_timeout_raises(self, mock_time, mock_sync_transport):
        mock_time.time.side_effect = [0, 700]
        mock_time.sleep = MagicMock()

        processing = make_video_generation(status="processing")
        mock_sync_transport.request.side_effect = [
            processing,  # initial generate
            processing,  # poll (time > deadline)
        ]

        resource = VideosResource(mock_sync_transport)
        with pytest.raises(MeganovaError, match="timed out"):
            resource.generate(prompt="x", model="m", wait=True, timeout=600)

    @patch("meganova.resources.videos.time")
    def test_wait_returns_failed(self, mock_time, mock_sync_transport):
        mock_time.time.side_effect = [0, 1, 2]
        mock_time.sleep = MagicMock()

        processing = make_video_generation(status="processing")
        failed = make_video_generation(status="failed")
        mock_sync_transport.request.side_effect = [processing, failed]

        resource = VideosResource(mock_sync_transport)
        result = resource.generate(prompt="x", model="m", wait=True, timeout=600)
        assert result.status == "failed"
