"""Tests for BillingResource."""

from meganova.models.billing import BillingBalance
from meganova.resources.billing import BillingResource
from tests.conftest import make_billing_response


class TestBillingResource:
    def test_get_user_instance_billings(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_billing_response()
        resource = BillingResource(mock_sync_transport)

        result = resource.get_user_instance_billings()
        assert isinstance(result, BillingBalance)
        assert result.total_cost == 5.5

    def test_correct_endpoint(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_billing_response()
        resource = BillingResource(mock_sync_transport)

        resource.get_user_instance_billings()
        mock_sync_transport.request.assert_called_once_with(
            "GET", "/billing/user-instance-billings"
        )
