
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "olist_geolocation_dataset.csv"
)

REPORT_DIR = BASE_DIR / "reports" / "geolocation"

df = pd.read_csv(INPUT_FILE)

# =====================================================
# 1. INVESTIGATE COORDINATE OUTLIERS
# =====================================================

print("\n========== COORDINATE OUTLIERS ==========")

brazil_screen = (
    df["geolocation_lat"].between(-35, 6)
    & df["geolocation_lng"].between(-75, -32)
)

outliers = df.loc[~brazil_screen].copy()

print(f"Total outliers: {len(outliers):,}")

print(
    outliers.sort_values(
        by=["geolocation_state", "geolocation_zip_code_prefix"]
    ).to_string(index=False)
)

outliers.to_csv(
    REPORT_DIR / "inspected_coordinate_outliers.csv",
    index=False
)

# =====================================================
# 2. INVESTIGATE ZIP PREFIXES WITH MULTIPLE STATES
# =====================================================

print("\n========== ZIP PREFIX STATE CONFLICTS ==========")

zip_state_counts = (
    df.groupby("geolocation_zip_code_prefix")[
        "geolocation_state"
    ]
    .nunique()
)

conflicting_zip_prefixes = zip_state_counts[
    zip_state_counts > 1
].index

state_conflicts = df[
    df["geolocation_zip_code_prefix"].isin(
        conflicting_zip_prefixes
    )
].copy()

print(
    f"Conflicting ZIP prefixes: "
    f"{len(conflicting_zip_prefixes)}"
)

for zip_prefix in conflicting_zip_prefixes:

    subset = state_conflicts[
        state_conflicts["geolocation_zip_code_prefix"]
        == zip_prefix
    ]

    print(f"\nZIP PREFIX: {zip_prefix}")

    print(
        subset.groupby(
            ["geolocation_state", "geolocation_city"]
        )
        .agg(
            records=("geolocation_zip_code_prefix", "size"),
            unique_coordinates=(
                "geolocation_lat",
                "nunique"
            )
        )
        .to_string()
    )

state_conflicts.to_csv(
    REPORT_DIR / "inspected_zip_state_conflicts.csv",
    index=False
)

# =====================================================
# 3. INVESTIGATE EXACT DUPLICATE DISTRIBUTION
# =====================================================

print("\n========== EXACT DUPLICATE DISTRIBUTION ==========")

duplicate_mask = df.duplicated(keep=False)

duplicate_records = df.loc[duplicate_mask].copy()

duplicate_summary = (
    duplicate_records.groupby(
        "geolocation_zip_code_prefix"
    )
    .size()
    .sort_values(ascending=False)
    .head(15)
)

print("Top 15 ZIP prefixes with repeated records:")
print(duplicate_summary.to_string())

print(
    "\nTotal rows belonging to duplicated groups:",
    len(duplicate_records)
)

print(
    "Exact duplicate rows after first occurrence:",
    df.duplicated().sum()
)

print("\n========== INSPECTION COMPLETED ==========")
