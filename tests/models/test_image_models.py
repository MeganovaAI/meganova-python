"""Tests for image generation Pydantic models."""

from meganova.models.images import GeneratedImage, ImageGenerationResponse


class TestGeneratedImage:
    def test_url_only(self):
        img = GeneratedImage(url="https://img.example.com/cat.png")
        assert img.url == "https://img.example.com/cat.png"
        assert img.b64_json is None

    def test_b64_only(self):
        img = GeneratedImage(b64_json="base64data")
        assert img.b64_json == "base64data"
        assert img.url is None

    def test_revised_prompt(self):
        img = GeneratedImage(url="https://x.com/img.png", revised_prompt="a cute cat")
        assert img.revised_prompt == "a cute cat"

    def test_extra_ignored(self):
        img = GeneratedImage(url="https://x.com/img.png", unknown_key="val")
        assert not hasattr(img, "unknown_key")


class TestImageGenerationResponse:
    def test_basic(self):
        resp = ImageGenerationResponse(
            data=[GeneratedImage(url="https://x.com/1.png")],
            created=1700000000,
        )
        assert len(resp.data) == 1
        assert resp.created == 1700000000
