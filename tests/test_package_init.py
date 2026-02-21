"""Tests for meganova package __init__ exports."""

import meganova


class TestPackageExports:
    def test_meganova_client_exported(self):
        assert hasattr(meganova, "MegaNova")

    def test_async_client_exported(self):
        assert hasattr(meganova, "AsyncMegaNova")

    def test_cloud_agent_exported(self):
        assert hasattr(meganova, "CloudAgent")

    def test_meganova_error_exported(self):
        assert hasattr(meganova, "MeganovaError")

    def test_serverless_model_exported(self):
        assert hasattr(meganova, "ServerlessModel")

    def test_serverless_models_response_exported(self):
        assert hasattr(meganova, "ServerlessModelsResponse")

    def test_generated_image_exported(self):
        assert hasattr(meganova, "GeneratedImage")

    def test_image_generation_response_exported(self):
        assert hasattr(meganova, "ImageGenerationResponse")

    def test_transcription_response_exported(self):
        assert hasattr(meganova, "TranscriptionResponse")

    def test_embedding_response_exported(self):
        assert hasattr(meganova, "EmbeddingResponse")

    def test_embedding_exported(self):
        assert hasattr(meganova, "Embedding")

    def test_video_generation_exported(self):
        assert hasattr(meganova, "VideoGeneration")

    def test_all_matches_expected(self):
        expected = {
            "MegaNova", "AsyncMegaNova", "CloudAgent", "MeganovaError",
            "ServerlessModel", "ServerlessModelsResponse",
            "GeneratedImage", "ImageGenerationResponse",
            "TranscriptionResponse", "EmbeddingResponse", "Embedding",
            "VideoGeneration",
        }
        assert set(meganova.__all__) == expected
