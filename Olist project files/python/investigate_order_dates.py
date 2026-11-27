
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

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")


# -----------------------------------------
# 1. CARRIER BEFORE APPROVAL
# -----------------------------------------

carrier_issue = df[
    df["order_delivered_carrier_date"].notna()
    & df["order_approved_at"].notna()
    & (
        df["order_delivered_carrier_date"]
        < df["order_approved_at"]
    )
].copy()

carrier_issue["hours_difference"] = (
    carrier_issue["order_approved_at"]
    - carrier_issue["order_delivered_carrier_date"]
).dt.total_seconds() / 3600

print("\n========== CARRIER BEFORE APPROVAL ==========")

print("\nDifference in hours:")
print(carrier_issue["hours_difference"].describe())

print("\nLargest differences:")
print(
    carrier_issue.nlargest(10, "hours_difference")[
        [
            "order_id",
            "order_status",
            "order_approved_at",
            "order_delivered_carrier_date",
            "hours_difference"
        ]
    ].to_string(index=False)
)

print("\nOrder status distribution:")
print(carrier_issue["order_status"].value_counts())


# -----------------------------------------
# 2. DELIVERY BEFORE CARRIER
# -----------------------------------------

delivery_issue = df[
    df["order_delivered_customer_date"].notna()
    & df["order_delivered_carrier_date"].notna()
    & (
        df["order_delivered_customer_date"]
        < df["order_delivered_carrier_date"]
    )
].copy()

delivery_issue["hours_difference"] = (
    delivery_issue["order_delivered_carrier_date"]
    - delivery_issue["order_delivered_customer_date"]
).dt.total_seconds() / 3600

print("\n========== DELIVERY BEFORE CARRIER ==========")

print(
    delivery_issue[
        [
            "order_id",
            "order_status",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "hours_difference"
        ]
    ].to_string(index=False)
)


# -----------------------------------------
# 3. MONTHLY DISTRIBUTION
# -----------------------------------------

carrier_issue["approval_month"] = (
    carrier_issue["order_approved_at"].dt.to_period("M")
)

print("\n========== MONTHLY CARRIER ANOMALIES ==========")

print(
    carrier_issue["approval_month"]
    .value_counts()
    .sort_index()
)


# -----------------------------------------
# 4. EXPORT INVESTIGATION RESULTS
# -----------------------------------------

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

carrier_issue.to_csv(
    REPORT_DIR / "carrier_before_approval.csv",
    index=False
)

delivery_issue.to_csv(
    REPORT_DIR / "delivery_before_carrier.csv",
    index=False
)

print("\nInvestigation files exported successfully.")
