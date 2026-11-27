
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "geolocation"

geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
spread = pd.read_csv(REPORT_DIR / "zip_coordinate_spread_km.csv")

# Remove exact duplicates and apply the same broad screening bounds.
geo = geo.drop_duplicates().copy()

inside_brazil = (
    geo["geolocation_lat"].between(-35, 6)
    & geo["geolocation_lng"].between(-75, -32)
)
geo = geo.loc[inside_brazil].copy()

# 1. Inspect all seller mismatches against geographic records.
seller_mismatch_zips = [
    95076, 21210, 85301, 87360, 86170, 85960, 22783,
    36512, 27277, 31570, 29101, 36010, 91520, 88790,
    37540, 95055, 81560, 88301, 88136, 89803, 83020,
    37795, 86076, 80240, 88075, 44600, 89052, 81020,
    71900, 21320, 28810, 85960, 83321, 31160
]

seller_detail = sellers[
    sellers["seller_zip_code_prefix"].isin(seller_mismatch_zips)
].merge(
    geo.groupby("geolocation_zip_code_prefix").agg(
        geo_states=("geolocation_state",
                    lambda x: ", ".join(sorted(set(x.dropna())))),
        geo_cities=("geolocation_city",
                    lambda x: ", ".join(sorted(set(x.dropna())))),
        geo_rows=("geolocation_lat", "size")
    ),
    left_on="seller_zip_code_prefix",
    right_index=True,
    how="left"
)

seller_detail.to_csv(
    REPORT_DIR / "seller_mismatch_inspection.csv",
    index=False
)

print("\n========== SELLER MISMATCH INSPECTION ==========")
print(
    seller_detail[
        ["seller_id", "seller_zip_code_prefix",
         "seller_state", "geo_states", "geo_cities", "geo_rows"]
    ].to_string(index=False)
)

# 2. Inspect the 15 widest ZIP prefixes.
top_zips = spread.nlargest(15, "p95_distance_km")[
    "geolocation_zip_code_prefix"
].tolist()

geo_top = geo[
    geo["geolocation_zip_code_prefix"].isin(top_zips)
].copy()

geo_top["median_lat"] = geo_top.groupby(
    "geolocation_zip_code_prefix"
)["geolocation_lat"].transform("median")

geo_top["median_lng"] = geo_top.groupby(
    "geolocation_zip_code_prefix"
)["geolocation_lng"].transform("median")

lat1 = np.radians(geo_top["geolocation_lat"])
lon1 = np.radians(geo_top["geolocation_lng"])
lat2 = np.radians(geo_top["median_lat"])
lon2 = np.radians(geo_top["median_lng"])

a = (
    np.sin((lat1 - lat2) / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin((lon1 - lon2) / 2) ** 2
)

geo_top["distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
)

geo_top = geo_top.sort_values(
    ["geolocation_zip_code_prefix", "distance_km"],
    ascending=[True, False]
)

geo_top.to_csv(
    REPORT_DIR / "widest_zip_coordinate_records.csv",
    index=False
)

print("\n========== WIDEST ZIP COORDINATE RECORDS ==========")
print(
    geo_top[
        ["geolocation_zip_code_prefix", "geolocation_city",
         "geolocation_state", "geolocation_lat",
         "geolocation_lng", "distance_km"]
    ].to_string(index=False)
)

print("\nInspection reports saved to:", REPORT_DIR)
