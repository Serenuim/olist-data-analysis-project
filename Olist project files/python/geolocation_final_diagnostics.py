
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "geolocation"

geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

# Remove exact duplicates for diagnostic purposes only.
geo = geo.drop_duplicates().copy()

# =====================================================
# 1. IDENTIFY GEOGRAPHIC OUTLIERS
# =====================================================

inside_brazil = (
    geo["geolocation_lat"].between(-35, 6)
    & geo["geolocation_lng"].between(-75, -32)
)

geo["outside_brazil_screen_flag"] = ~inside_brazil

geo_clean_screen = geo[inside_brazil].copy()

# =====================================================
# 2. INVESTIGATE CUSTOMER STATE CONFLICTS
# =====================================================

geo_states = (
    geo.groupby("geolocation_zip_code_prefix")
    ["geolocation_state"]
    .agg(lambda x: set(x))
    .to_dict()
)

def classify_state_matches(data, zip_col, state_col):

    result = data[[zip_col, state_col]].copy()

    result["geo_states"] = result[zip_col].map(
        geo_states
    )

    result["match_status"] = result.apply(
        lambda row:
            "unmatched_zip"
            if not isinstance(row["geo_states"], set)
            else (
                "state_match"
                if row[state_col] in row["geo_states"]
                else "state_mismatch"
            ),
        axis=1
    )

    return result


customer_check = classify_state_matches(
    customers,
    "customer_zip_code_prefix",
    "customer_state"
)

seller_check = classify_state_matches(
    sellers,
    "seller_zip_code_prefix",
    "seller_state"
)

print("\n========== CUSTOMER STATE DIAGNOSTICS ==========")
print(customer_check["match_status"].value_counts())

print("\n========== SELLER STATE DIAGNOSTICS ==========")
print(seller_check["match_status"].value_counts())

customer_check[
    customer_check["match_status"] != "state_match"
].to_csv(
    REPORT_DIR / "final_customer_state_anomalies.csv",
    index=False
)

seller_check[
    seller_check["match_status"] != "state_match"
].to_csv(
    REPORT_DIR / "final_seller_state_anomalies.csv",
    index=False
)

# =====================================================
# 3. COORDINATE SPREAD BEFORE AND AFTER SCREENING
# =====================================================

def calculate_spread(data):

    return (
        data.groupby("geolocation_zip_code_prefix")
        .agg(
            coordinate_count=("geolocation_lat", "size"),
            lat_min=("geolocation_lat", "min"),
            lat_max=("geolocation_lat", "max"),
            lng_min=("geolocation_lng", "min"),
            lng_max=("geolocation_lng", "max"),
        )
        .reset_index()
        .assign(
            latitude_spread=lambda x: x["lat_max"] - x["lat_min"],
            longitude_spread=lambda x: x["lng_max"] - x["lng_min"]
        )
        .assign(
            wide_spread_flag=lambda x:
                (x["latitude_spread"] > 0.1)
                | (x["longitude_spread"] > 0.1)
        )
    )


spread_before = calculate_spread(geo)
spread_after = calculate_spread(geo_clean_screen)

print("\n========== COORDINATE SPREAD COMPARISON ==========")

print(
    "Wide spread before geographic screening:",
    spread_before["wide_spread_flag"].sum()
)

print(
    "Wide spread after geographic screening:",
    spread_after["wide_spread_flag"].sum()
)

spread_after.to_csv(
    REPORT_DIR / "coordinate_spread_after_outlier_screen.csv",
    index=False
)

# =====================================================
# 4. FINAL DIAGNOSTIC SUMMARY
# =====================================================

summary = pd.DataFrame({
    "diagnostic": [
        "Unique geographic records",
        "Geographic outlier records",
        "Customer unmatched ZIP records",
        "Customer state mismatches",
        "Seller unmatched ZIP records",
        "Seller state mismatches",
        "Wide coordinate spread before screening",
        "Wide coordinate spread after screening",
    ],
    "result": [
        len(geo),
        (~inside_brazil).sum(),
        (customer_check["match_status"] == "unmatched_zip").sum(),
        (customer_check["match_status"] == "state_mismatch").sum(),
        (seller_check["match_status"] == "unmatched_zip").sum(),
        (seller_check["match_status"] == "state_mismatch").sum(),
        spread_before["wide_spread_flag"].sum(),
        spread_after["wide_spread_flag"].sum(),
    ]
})

summary.to_csv(
    REPORT_DIR / "final_diagnostic_summary.csv",
    index=False
)

print("\n========== FINAL DIAGNOSTIC SUMMARY ==========")
print(summary.to_string(index=False))

print("\nReports saved successfully.")
