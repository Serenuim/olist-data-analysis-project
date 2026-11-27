
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "geolocation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

# 1. Remove exact duplicates for geographic analysis.
geo = geo.drop_duplicates().copy()

# 2. Apply broad Brazil screening bounds.
inside_brazil = (
    geo["geolocation_lat"].between(-35, 6)
    & geo["geolocation_lng"].between(-75, -32)
)
geo = geo.loc[inside_brazil].copy()

# 3. Build ZIP-to-state mapping as actual sets.
geo_states = (
    geo.groupby("geolocation_zip_code_prefix")["geolocation_state"]
    .agg(lambda values: set(values.dropna()))
    .to_dict()
)

# 4. Classify seller ZIP/state matches.
seller_check = sellers[
    ["seller_id", "seller_zip_code_prefix", "seller_state"]
].copy()

seller_check["geo_states"] = (
    seller_check["seller_zip_code_prefix"].map(geo_states)
)

def classify_seller(row):
    states = row["geo_states"]

    if not isinstance(states, set) or not states:
        return "unmatched_zip"

    if row["seller_state"] in states:
        return "state_match"

    return "state_mismatch"

seller_check["match_status"] = seller_check.apply(
    classify_seller, axis=1
)

seller_mismatches = seller_check[
    seller_check["match_status"] == "state_mismatch"
].copy()

seller_mismatches["geo_states"] = seller_mismatches[
    "geo_states"
].apply(lambda states: "; ".join(sorted(states)))

seller_mismatches.to_csv(
    REPORT_DIR / "seller_state_mismatches.csv",
    index=False
)

print("\n========== SELLER STATE COUNTS ==========")
print(seller_check["match_status"].value_counts())

print("\n========== SELLER MISMATCHES ==========")
if seller_mismatches.empty:
    print("No seller state mismatches found.")
else:
    print(seller_mismatches.to_string(index=False))

# 5. Calculate distance from each ZIP prefix's median coordinate.
zip_centres = (
    geo.groupby("geolocation_zip_code_prefix")
    .agg(
        median_lat=("geolocation_lat", "median"),
        median_lng=("geolocation_lng", "median"),
        geo_record_count=("geolocation_lat", "size")
    )
    .reset_index()
)

geo = geo.merge(
    zip_centres,
    on="geolocation_zip_code_prefix",
    how="left"
)

lat1 = np.radians(geo["geolocation_lat"])
lon1 = np.radians(geo["geolocation_lng"])
lat2 = np.radians(geo["median_lat"])
lon2 = np.radians(geo["median_lng"])

a = (
    np.sin((lat1 - lat2) / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin((lon1 - lon2) / 2) ** 2
)

geo["distance_from_zip_median_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
)

spread = (
    geo.groupby("geolocation_zip_code_prefix")
    .agg(
        geo_record_count=("geolocation_lat", "size"),
        p95_distance_km=(
            "distance_from_zip_median_km",
            lambda values: values.quantile(0.95)
        ),
        max_distance_km=(
            "distance_from_zip_median_km", "max"
        )
    )
    .reset_index()
    .sort_values("p95_distance_km", ascending=False)
)

spread.to_csv(
    REPORT_DIR / "zip_coordinate_spread_km.csv",
    index=False
)

print("\n========== ZIP SPREAD SUMMARY ==========")
print(spread["p95_distance_km"].describe())

print("\n========== 30 WIDEST ZIP PREFIXES ==========")
print(spread.head(30).to_string(index=False))

print("\n========== SPREAD SCREENING COUNTS ==========")
for threshold in [10, 25, 50]:
    count = (spread["p95_distance_km"] > threshold).sum()
    print(f"P95 distance above {threshold} km: {count}")

print("\nReports saved to:", REPORT_DIR)
