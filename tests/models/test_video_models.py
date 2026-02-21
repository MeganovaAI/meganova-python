"""Tests for video generation Pydantic models."""

from meganova.models.videos import VideoGeneration


class TestVideoGeneration:
    def test_minimal(self):
        v = VideoGeneration(id="vid_1", created_at=1700000000, status="processing", model="veo-3")
        assert v.id == "vid_1"
        assert v.status == "processing"

    def test_completed_with_url(self):
        v = VideoGeneration(
            id="vid_1", created_at=1700000000, status="completed",
            model="veo-3", url="https://cdn.example.com/video.mp4",
        )
        assert v.url == "https://cdn.example.com/video.mp4"

    def test_failed_with_error(self):
        v = VideoGeneration(
            id="vid_1", created_at=1700000000, status="failed",
            model="veo-3", error={"message": "NSFW content detected"},
        )
        assert v.error["message"] == "NSFW content detected"

    def test_default_object(self):
        v = VideoGeneration(id="v", created_at=0, status="x", model="m")
        assert v.object == "video.generation"

    def test_extra_ignored(self):
        v = VideoGeneration(id="v", created_at=0, status="x", model="m", unknown="val")
        assert not hasattr(v, "unknown")
