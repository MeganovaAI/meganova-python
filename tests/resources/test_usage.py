"""Tests for UsageResource."""

from meganova.resources.usage import UsageResource


class TestUsageResource:
    def test_summary_basic(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {"data": [], "message": "ok", "status": "success"}
        resource = UsageResource(mock_sync_transport)

        result = resource.summary(from_="2024-01-01", to="2024-01-31")
        assert result["status"] == "success"

    def test_summary_correct_params(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {}
        resource = UsageResource(mock_sync_transport)

        resource.summary(from_="2024-01-01", to="2024-01-31", team_id=42)
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["params"]["start_time"] == "2024-01-01"
        assert call_kwargs["params"]["end_time"] == "2024-01-31"
        assert call_kwargs["params"]["team_id"] == 42

    def test_summary_correct_path(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {}
        resource = UsageResource(mock_sync_transport)

        resource.summary(from_="2024-01-01", to="2024-01-31")
        args = mock_sync_transport.request.call_args
        assert args.args == ("GET", "/usage/time_range")

    def test_summary_without_team_id(self, mock_sync_transport):
        mock_sync_transport.request.return_value = {}
        resource = UsageResource(mock_sync_transport)

        resource.summary(from_="2024-01-01", to="2024-01-31")
        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert "team_id" not in call_kwargs["params"]
