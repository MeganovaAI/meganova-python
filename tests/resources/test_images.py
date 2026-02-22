"""Tests for ImagesResource."""

import pytest

from meganova.models.images import ImageGenerationResponse
from meganova.resources.images import ImagesResource
from tests.conftest import make_image_response


class TestImagesGenerate:
    def test_basic_generation(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        result = resource.generate("a cat", model="flux-1")
        assert isinstance(result, ImageGenerationResponse)
        assert len(result.data) == 1

    def test_correct_endpoint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="flux-1")
        args = mock_sync_transport.request.call_args
        assert args.args == ("POST", "/images/generations")

    def test_payload_includes_prompt_and_model(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a dog", model="dall-e-3")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["prompt"] == "a dog"
        assert payload["model"] == "dall-e-3"
        assert payload["n"] == 1

    def test_size_parameter(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", size="1024x1024")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["size"] == "1024x1024"

    def test_width_height_to_size(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", width=512, height=768)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["size"] == "512x768"

    def test_size_overrides_width_height(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", size="1024x1024", width=512, height=768)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["size"] == "1024x1024"

    def test_quality_parameter(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", quality="hd")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["quality"] == "hd"

    def test_n_parameter(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response(n=3)
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", n=3)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["n"] == 3

    def test_seed_parameter(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", seed=42)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["seed"] == 42

    def test_guidance_scale(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", guidance_scale=7.5)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["guidance_scale"] == 7.5

    def test_num_steps(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", num_steps=50)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["num_steps"] == 50

    def test_none_optionals_excluded(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        for key in ("size", "quality", "response_format", "style", "seed", "num_steps", "guidance_scale"):
            assert key not in payload

    def test_kwargs_passed_through(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_image_response()
        resource = ImagesResource(mock_sync_transport)

        resource.generate("a cat", model="m", custom_param="custom_value")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["custom_param"] == "custom_value"
