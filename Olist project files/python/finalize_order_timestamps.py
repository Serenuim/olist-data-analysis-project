
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "olist_orders_dataset.csv"
)

df = pd.read_csv(FILE)

# Convert timestamp columns.
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

status = df["order_status"].str.lower().str.strip()

purchase = df["order_purchase_timestamp"]
approval = df["order_approved_at"]
carrier = df["order_delivered_carrier_date"]
delivery = df["order_delivered_customer_date"]

# -----------------------------------------
# 1. ANOMALY FLAGS
# -----------------------------------------

df["carrier_before_approval_flag"] = (
    approval.notna()
    & carrier.notna()
    & (carrier < approval)
)

df["delivery_before_carrier_flag"] = (
    carrier.notna()
    & delivery.notna()
    & (delivery < carrier)
)

df["missing_approval_date_flag"] = approval.isna()

df["missing_carrier_date_flag"] = (
    status.eq("delivered") & carrier.isna()
)

df["missing_delivery_date_flag"] = (
    status.eq("delivered") & delivery.isna()
)

# -----------------------------------------
# 2. DURATION ELIGIBILITY
# -----------------------------------------

df["valid_purchase_to_delivery_duration"] = (
    status.eq("delivered")
    & purchase.notna()
    & delivery.notna()
    & (delivery >= purchase)
)

df["valid_approval_to_carrier_duration"] = (
    approval.notna()
    & carrier.notna()
    & (carrier >= approval)
)

df["valid_carrier_to_delivery_duration"] = (
    carrier.notna()
    & delivery.notna()
    & (delivery >= carrier)
)

df["valid_full_logistics_sequence"] = (
    purchase.notna()
    & approval.notna()
    & carrier.notna()
    & delivery.notna()
    & (approval >= purchase)
    & (carrier >= approval)
    & (delivery >= carrier)
)

# -----------------------------------------
# 3. CALCULATE VALID DURATIONS IN HOURS
# -----------------------------------------

def duration_in_hours(start, end):
    return (
        (end - start).dt.total_seconds() / 3600
    )

df["purchase_to_delivery_hours"] = (
    duration_in_hours(purchase, delivery)
    .where(df["valid_purchase_to_delivery_duration"])
)

df["approval_to_carrier_hours"] = (
    duration_in_hours(approval, carrier)
    .where(df["valid_approval_to_carrier_duration"])
)

df["carrier_to_delivery_hours"] = (
    duration_in_hours(carrier, delivery)
    .where(df["valid_carrier_to_delivery_duration"])
)

# -----------------------------------------
# 4. VALIDATION REPORT
# -----------------------------------------

checks = {
    "Carrier before approval":
        df["carrier_before_approval_flag"],

    "Delivery before carrier":
        df["delivery_before_carrier_flag"],

    "Delivered without carrier date":
        df["missing_carrier_date_flag"],

    "Delivered without delivery date":
        df["missing_delivery_date_flag"]
}

print("\n========== ORDER TIMESTAMP VALIDATION ==========")

for name, condition in checks.items():
    print(f"{name}: {condition.sum():,}")

print("\n========== DURATION ELIGIBILITY ==========")

eligibility_columns = [
    "valid_purchase_to_delivery_duration",
    "valid_approval_to_carrier_duration",
    "valid_carrier_to_delivery_duration",
    "valid_full_logistics_sequence"
]

for column in eligibility_columns:
    print(f"{column}: {df[column].sum():,} passed")

# -----------------------------------------
# 5. SAVE PROCESSED DATA
# -----------------------------------------

df.to_csv(FILE, index=False)

print("\nUpdated processed orders dataset saved.")
