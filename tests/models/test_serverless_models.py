"""Tests for serverless Pydantic models."""

from meganova.models.serverless import ServerlessModel, ServerlessModelsResponse


class TestServerlessModel:
    def test_basic(self):
        m = ServerlessModel(model_name="llama-3")
        assert m.model_name == "llama-3"

    def test_full(self):
        m = ServerlessModel(
            model_name="llama-3",
            model_alias="llama",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            modality="text_generation",
        )
        assert m.cost_per_1k_input == 0.001

    def test_tier2_alias(self):
        """tier2+_require field with alias."""
        data = {"model_name": "premium-model", "tier2+_require": True}
        m = ServerlessModel.model_validate(data)
        assert m.tier2_plus_require is True

    def test_extra_ignored(self):
        m = ServerlessModel(model_name="x", extra_field="val")
        assert not hasattr(m, "extra_field")


class TestServerlessModelsResponse:
    def test_basic(self):
        resp = ServerlessModelsResponse(
            models=[ServerlessModel(model_name="llama-3")],
            count=1,
        )
        assert resp.count == 1
        assert len(resp.models) == 1
