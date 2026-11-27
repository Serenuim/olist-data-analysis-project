
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "geolocation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 1. LOAD DATA
# =====================================================

geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

geo_unique = geo.drop_duplicates().copy()

print("\n========== DATASET OVERVIEW ==========")
print(f"Original geolocation rows: {len(geo):,}")
print(f"Unique geolocation rows: {len(geo_unique):,}")
print(f"Exact duplicate rows: {geo.duplicated().sum():,}")

# =====================================================
# 2. ZIP PREFIX STATE AND CITY CONSISTENCY
# =====================================================

zip_summary = (
    geo_unique.groupby("geolocation_zip_code_prefix")
    .agg(
        coordinate_records=("geolocation_lat", "size"),
        unique_coordinates=("geolocation_lat", "nunique"),
        unique_cities=("geolocation_city", "nunique"),
        unique_states=("geolocation_state", "nunique"),
        cities=("geolocation_city",
                lambda x: " | ".join(sorted(x.unique()))),
        states=("geolocation_state",
                lambda x: " | ".join(sorted(x.unique()))),
    )
    .reset_index()
)

zip_summary["state_conflict_flag"] = (
    zip_summary["unique_states"] > 1
)

zip_summary["city_variation_flag"] = (
    zip_summary["unique_cities"] > 1
)

zip_summary.to_csv(
    REPORT_DIR / "complete_zip_consistency.csv",
    index=False
)

print("\n========== ZIP CONSISTENCY ==========")
print("ZIP prefixes:", len(zip_summary))
print("Multiple-state ZIPs:",
      zip_summary["state_conflict_flag"].sum())
print("Multiple-city ZIPs:",
      zip_summary["city_variation_flag"].sum())

# =====================================================
# 3. COORDINATE SPREAD
# =====================================================

coordinate_spread = (
    geo_unique.groupby("geolocation_zip_code_prefix")
    .agg(
        latitude_min=("geolocation_lat", "min"),
        latitude_max=("geolocation_lat", "max"),
        longitude_min=("geolocation_lng", "min"),
        longitude_max=("geolocation_lng", "max"),
        latitude_median=("geolocation_lat", "median"),
        longitude_median=("geolocation_lng", "median"),
        coordinate_count=("geolocation_lng", "size"),
    )
    .reset_index()
)

coordinate_spread["latitude_spread"] = (
    coordinate_spread["latitude_max"]
    - coordinate_spread["latitude_min"]
)

coordinate_spread["longitude_spread"] = (
    coordinate_spread["longitude_max"]
    - coordinate_spread["longitude_min"]
)

# 0.1 degrees is a screening threshold, not proof of bad data.
coordinate_spread["wide_spread_flag"] = (
    (coordinate_spread["latitude_spread"] > 0.1)
    | (coordinate_spread["longitude_spread"] > 0.1)
)

coordinate_spread.to_csv(
    REPORT_DIR / "complete_coordinate_spread.csv",
    index=False
)

print("\n========== COORDINATE SPREAD ==========")
print("ZIPs with wide coordinate spread:",
      coordinate_spread["wide_spread_flag"].sum())

print("\nTop 15 widest coordinate ranges:")
print(
    coordinate_spread.sort_values(
        "latitude_spread",
        ascending=False
    ).head(15).to_string(index=False)
)

# =====================================================
# 4. BRAZIL GEOGRAPHIC OUTLIERS
# =====================================================

brazil_screen = (
    geo_unique["geolocation_lat"].between(-35, 6)
    & geo_unique["geolocation_lng"].between(-75, -32)
)

outliers = geo_unique.loc[~brazil_screen].copy()

outliers.to_csv(
    REPORT_DIR / "final_geographic_outliers.csv",
    index=False
)

print("\n========== GEOGRAPHIC OUTLIERS ==========")
print(f"Unique outlier records: {len(outliers):,}")
print(outliers.to_string(index=False))

# =====================================================
# 5. CUSTOMER AND SELLER ZIP MATCHING
# =====================================================

geo_zip = set(
    geo_unique["geolocation_zip_code_prefix"].dropna()
)

def check_coverage(data, column, label):

    values = data[column].dropna()

    matched = values.isin(geo_zip)

    unique_values = set(values.unique())
    matched_unique = unique_values.intersection(geo_zip)

    unmatched_unique = unique_values.difference(geo_zip)

    row_rate = matched.mean() * 100 if len(values) else 0

    unique_rate = (
        len(matched_unique) / len(unique_values) * 100
        if unique_values else 0
    )

    print(f"\n{label}")
    print("Total rows:", len(data))
    print("Matched rows:", matched.sum())
    print("Unmatched rows:", (~matched).sum())
    print(f"Row match rate: {row_rate:.2f}%")
    print("Unique ZIP prefixes:", len(unique_values))
    print("Matched unique ZIPs:", len(matched_unique))
    print("Unmatched unique ZIPs:", len(unmatched_unique))
    print(f"Unique ZIP match rate: {unique_rate:.2f}%")

    pd.DataFrame({
        "unmatched_zip_prefix": sorted(unmatched_unique)
    }).to_csv(
        REPORT_DIR / f"unmatched_{label.lower()}_zips.csv",
        index=False
    )

    return {
        "dataset": label,
        "total_rows": len(data),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "row_match_rate_percent": row_rate,
        "unique_zip_count": len(unique_values),
        "matched_unique_zips": len(matched_unique),
        "unmatched_unique_zips": len(unmatched_unique),
        "unique_zip_match_rate_percent": unique_rate,
    }


print("\n========== CUSTOMER COVERAGE ==========")

customer_result = check_coverage(
    customers,
    "customer_zip_code_prefix",
    "customers"
)

print("\n========== SELLER COVERAGE ==========")

seller_result = check_coverage(
    sellers,
    "seller_zip_code_prefix",
    "sellers"
)

# =====================================================
# 6. CUSTOMER AND SELLER STATE CONSISTENCY
# =====================================================

# Compare customer/seller state to states found in geolocation
# for the same ZIP prefix.

zip_states = (
    geo_unique.groupby("geolocation_zip_code_prefix")
    ["geolocation_state"]
    .agg(lambda x: set(x))
    .to_dict()
)

def state_check(data, zip_col, state_col, label):

    def matches(row):
        possible_states = zip_states.get(row[zip_col], set())
        return row[state_col] in possible_states

    result = data[[zip_col, state_col]].copy()

    result["state_matches_geolocation"] = result.apply(
        matches, axis=1
    )

    conflicts = result[
        ~result["state_matches_geolocation"]
    ]

    print(f"\n{label} state conflicts:")
    print(f"{len(conflicts):,} rows")

    conflicts.to_csv(
        REPORT_DIR / f"{label.lower()}_state_conflicts.csv",
        index=False
    )

    return len(conflicts)


customer_state_conflicts = state_check(
    customers,
    "customer_zip_code_prefix",
    "customer_state",
    "Customer"
)

seller_state_conflicts = state_check(
    sellers,
    "seller_zip_code_prefix",
    "seller_state",
    "Seller"
)

# =====================================================
# 7. FINAL SUMMARY
# =====================================================

summary = pd.DataFrame([
    {
        "check": "Exact duplicate rows",
        "result": int(geo.duplicated().sum())
    },
    {
        "check": "Unique geolocation rows",
        "result": len(geo_unique)
    },
    {
        "check": "ZIP prefixes with multiple states",
        "result": int(zip_summary["state_conflict_flag"].sum())
    },
    {
        "check": "ZIP prefixes with multiple cities",
        "result": int(zip_summary["city_variation_flag"].sum())
    },
    {
        "check": "ZIP prefixes with wide coordinate spread",
        "result": int(coordinate_spread["wide_spread_flag"].sum())
    },
    {
        "check": "Geographic outlier rows",
        "result": len(outliers)
    },
    {
        "check": "Customer state conflicts",
        "result": customer_state_conflicts
    },
    {
        "check": "Seller state conflicts",
        "result": seller_state_conflicts
    }
])

summary.to_csv(
    REPORT_DIR / "final_geolocation_validation_summary.csv",
    index=False
)

print("\n========== FINAL VALIDATION SUMMARY ==========")
print(summary.to_string(index=False))

print("\nAll reports saved to:", REPORT_DIR)
print("Original datasets were not modified.")
