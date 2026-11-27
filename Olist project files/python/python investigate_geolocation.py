
from pathlib import Path
import pandas as pd

# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "olist_geolocation_dataset.csv"
REPORT_DIR = BASE_DIR / "reports" / "geolocation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# =====================================================
# 2. REQUIRED COLUMN VALIDATION
# =====================================================

required_columns = [
    "geolocation_zip_code_prefix",
    "geolocation_lat",
    "geolocation_lng",
    "geolocation_city",
    "geolocation_state",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("\n========== DATASET OVERVIEW ==========")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values:\n{df.isna().sum()}")

# =====================================================
# 3. EXACT DUPLICATE INVESTIGATION
# =====================================================

print("\n========== EXACT DUPLICATES ==========")

duplicate_count = df.duplicated().sum()

print(f"Exact duplicate rows: {duplicate_count:,}")
print(f"Unique rows: {len(df.drop_duplicates()):,}")

# Save duplicate records for inspection
exact_duplicates = df[df.duplicated(keep=False)].copy()

exact_duplicates.to_csv(
    REPORT_DIR / "exact_duplicate_records.csv",
    index=False
)

# =====================================================
# 4. ZIP PREFIX INVESTIGATION
# =====================================================

print("\n========== ZIP PREFIX INVESTIGATION ==========")

zip_stats = (
    df.groupby("geolocation_zip_code_prefix")
    .agg(
        record_count=("geolocation_zip_code_prefix", "size"),
        unique_latitudes=("geolocation_lat", "nunique"),
        unique_longitudes=("geolocation_lng", "nunique"),
        unique_cities=("geolocation_city", "nunique"),
        unique_states=("geolocation_state", "nunique"),
    )
    .reset_index()
)

print(
    "Unique ZIP prefixes:",
    df["geolocation_zip_code_prefix"].nunique()
)

print(
    "ZIP prefixes with multiple coordinate combinations:",
    (
        df.groupby("geolocation_zip_code_prefix")[
            ["geolocation_lat", "geolocation_lng"]
        ].apply(lambda x: x.drop_duplicates().shape[0] > 1)
    ).sum()
)

print(
    "ZIP prefixes associated with multiple states:",
    (zip_stats["unique_states"] > 1).sum()
)

print(
    "ZIP prefixes associated with multiple cities:",
    (zip_stats["unique_cities"] > 1).sum()
)

zip_stats.to_csv(
    REPORT_DIR / "zip_prefix_summary.csv",
    index=False
)

# Save conflicting state assignments
conflicting_states = zip_stats[
    zip_stats["unique_states"] > 1
]

conflicting_states.to_csv(
    REPORT_DIR / "zip_state_conflicts.csv",
    index=False
)

# =====================================================
# 5. COORDINATE VALIDITY
# =====================================================

print("\n========== COORDINATE VALIDITY ==========")

df["geolocation_lat"] = pd.to_numeric(
    df["geolocation_lat"], errors="coerce"
)

df["geolocation_lng"] = pd.to_numeric(
    df["geolocation_lng"], errors="coerce"
)

invalid_latitude = (
    df["geolocation_lat"].isna()
    | ~df["geolocation_lat"].between(-90, 90)
)

invalid_longitude = (
    df["geolocation_lng"].isna()
    | ~df["geolocation_lng"].between(-180, 180)
)

invalid_coordinates = invalid_latitude | invalid_longitude

print(f"Invalid latitude records: {invalid_latitude.sum():,}")
print(f"Invalid longitude records: {invalid_longitude.sum():,}")
print(f"Records with invalid coordinates: {invalid_coordinates.sum():,}")

df.loc[invalid_coordinates].to_csv(
    REPORT_DIR / "invalid_coordinates.csv",
    index=False
)

# =====================================================
# 6. BRAZIL GEOGRAPHIC SCREENING
# =====================================================

print("\n========== BRAZIL GEOGRAPHIC SCREENING ==========")

# Broad screening bounds, NOT an authoritative border.
brazil_screen = (
    df["geolocation_lat"].between(-35, 6)
    & df["geolocation_lng"].between(-75, -32)
)

outside_brazil_screen = ~brazil_screen & ~invalid_coordinates

print(f"Inside broad Brazil screening bounds: {brazil_screen.sum():,}")
print(f"Outside screening bounds: {outside_brazil_screen.sum():,}")

df.loc[outside_brazil_screen].to_csv(
    REPORT_DIR / "outside_brazil_screening.csv",
    index=False
)

# =====================================================
# 7. STATE DISTRIBUTION
# =====================================================

print("\n========== STATE DISTRIBUTION ==========")

state_distribution = (
    df["geolocation_state"]
    .value_counts(dropna=False)
    .rename_axis("state")
    .reset_index(name="record_count")
)

print(state_distribution.to_string(index=False))

state_distribution.to_csv(
    REPORT_DIR / "state_distribution.csv",
    index=False
)

# =====================================================
# 8. COORDINATE SUMMARY
# =====================================================

print("\n========== COORDINATE SUMMARY ==========")

print(
    df[["geolocation_lat", "geolocation_lng"]]
    .describe()
)

# =====================================================
# 9. INVESTIGATION SUMMARY
# =====================================================

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "exact_duplicate_rows",
        "unique_zip_prefixes",
        "invalid_latitude_records",
        "invalid_longitude_records",
        "invalid_coordinate_records",
        "inside_brazil_screening",
        "outside_brazil_screening",
        "zip_prefixes_with_multiple_states",
    ],
    "value": [
        len(df),
        duplicate_count,
        df["geolocation_zip_code_prefix"].nunique(),
        invalid_latitude.sum(),
        invalid_longitude.sum(),
        invalid_coordinates.sum(),
        brazil_screen.sum(),
        outside_brazil_screen.sum(),
        (zip_stats["unique_states"] > 1).sum(),
    ]
})

summary.to_csv(
    REPORT_DIR / "geolocation_investigation_summary.csv",
    index=False
)

print("\n========== INVESTIGATION COMPLETED ==========")
print(f"Reports saved to: {REPORT_DIR}")
print(summary.to_string(index=False))
