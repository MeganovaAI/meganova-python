"""Tests for usage Pydantic models."""

from meganova.models.usage import UsageDetail, UsageSummary


class TestUsageSummary:
    def test_basic(self):
        summary = UsageSummary(
            data=[
                UsageDetail(date="2024-01-01", model="gpt-4", tokens=1000, cost=0.05)
            ],
            message="ok",
            status="success",
        )
        assert len(summary.data) == 1
        assert summary.data[0].tokens == 1000

    def test_multiple_entries(self):
        summary = UsageSummary(
            data=[
                UsageDetail(date="2024-01-01", model="gpt-4", tokens=1000, cost=0.05),
                UsageDetail(date="2024-01-02", model="gpt-4", tokens=2000, cost=0.10),
            ],
            message="ok",
            status="success",
        )
        assert len(summary.data) == 2
