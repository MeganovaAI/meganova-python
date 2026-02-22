"""Tests for model listing Pydantic models."""

from meganova.models.models_ import ModelInfo, ModelListResponse


class TestModelInfo:
    def test_minimal(self):
        m = ModelInfo(id="gpt-4")
        assert m.id == "gpt-4"
        assert m.name is None

    def test_full(self):
        m = ModelInfo(
            id="gpt-4",
            object="model",
            owned_by="meganova",
            name="GPT-4",
            description="A powerful model",
            created=1700000000,
            context_length=128000,
            tags=["chat", "vision"],
            capabilities={"chat": True, "vision": True},
        )
        assert m.context_length == 128000
        assert "vision" in m.tags

    def test_extra_fields_ignored(self):
        m = ModelInfo(id="gpt-4", unknown_field="whatever")
        assert m.id == "gpt-4"
        assert not hasattr(m, "unknown_field")


class TestModelListResponse:
    def test_basic(self):
        resp = ModelListResponse(
            data=[ModelInfo(id="gpt-4"), ModelInfo(id="claude-3")]
        )
        assert len(resp.data) == 2
        assert resp.data[0].id == "gpt-4"
