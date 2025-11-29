from meganova import MegaNova
from datetime import datetime, timedelta

client = MegaNova(api_key="YOUR_API_KEY")

# --- Usage Summary Example ---
print("\n--- Usage Summary ---")
try:
    # Get usage for the last 7 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Format dates as ISO 8601 strings as expected by the API
    # Note: Adjust format if API expects a specific string format (e.g., YYYY-MM-DD)
    from_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    to_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    usage = client.usage.summary(
        from_=from_str,
        to=to_str,
        team_id=123 # Replace with your actual team ID
    )
    print(f"Usage Data: {usage}")
except Exception as e:
    print(f"Error retrieving usage: {e}")

# --- Billing Balance Example ---
print("\n--- Billing Balance ---")
try:
    balance = client.billing.get_user_instance_billings()
    print(f"Total Cost: {balance.total_cost}")
    for item in balance.data:
        print(f"Instance ID: {item.vm_access_info_id}, Fee: {item.fee_accumulated_taxed}")
except Exception as e:
    print(f"Error retrieving billing: {e}")

