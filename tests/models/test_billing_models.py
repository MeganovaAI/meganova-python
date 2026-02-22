"""Tests for billing Pydantic models."""

from meganova.models.billing import BillingBalance, InstanceBilling


class TestBillingBalance:
    def test_basic(self):
        billing = BillingBalance(
            data=[
                InstanceBilling(
                    vm_access_info_id=1,
                    running_hours=2.5,
                    fee_accumulated=5.0,
                    tax_rate=0.1,
                    fee_accumulated_taxed=5.5,
                    product_status="running",
                )
            ],
            total_cost=5.5,
            message="ok",
            status="success",
        )
        assert billing.total_cost == 5.5
        assert len(billing.data) == 1

    def test_instance_billing_fields(self):
        ib = InstanceBilling(
            vm_access_info_id=42,
            running_hours=10.0,
            fee_accumulated=20.0,
            tax_rate=0.08,
            fee_accumulated_taxed=21.6,
            product_status="stopped",
        )
        assert ib.vm_access_info_id == 42
        assert ib.product_status == "stopped"
