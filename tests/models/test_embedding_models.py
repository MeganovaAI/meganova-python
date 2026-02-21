"""Tests for embedding Pydantic models."""

from meganova.models.embeddings import Embedding, EmbeddingResponse, EmbeddingUsage


class TestEmbedding:
    def test_basic(self):
        e = Embedding(index=0, embedding=[0.1, 0.2, 0.3])
        assert e.index == 0
        assert len(e.embedding) == 3

    def test_default_object(self):
        e = Embedding(index=0, embedding=[0.1])
        assert e.object == "embedding"

    def test_extra_ignored(self):
        e = Embedding(index=0, embedding=[0.1], custom_field="x")
        assert not hasattr(e, "custom_field")


class TestEmbeddingResponse:
    def test_basic(self):
        resp = EmbeddingResponse(
            data=[Embedding(index=0, embedding=[0.1, 0.2])],
            model="ada-002",
        )
        assert resp.model == "ada-002"
        assert resp.object == "list"

    def test_with_usage(self):
        resp = EmbeddingResponse(
            data=[Embedding(index=0, embedding=[0.1])],
            model="ada-002",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        assert resp.usage.total_tokens == 5
