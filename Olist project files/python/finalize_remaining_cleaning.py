
from pathlib import Path
import pandas as pd

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "final_cleaning"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

report = []


def log(message):
    print(message)
    report.append(str(message))


def normalize_zip(series):
    """Normalize ZIP prefixes so joins use consistent strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def save(df, filename):
    df.to_csv(DATA_DIR / filename, index=False)


# =========================================================
# 1. GEOLOCATION
# =========================================================

log("\n========== GEOLOCATION ==========")

geo = pd.read_csv(
    DATA_DIR / "olist_geolocation_dataset.csv",
    dtype={"geolocation_zip_code_prefix": "string"}
)

geo["geolocation_zip_code_prefix"] = normalize_zip(
    geo["geolocation_zip_code_prefix"]
)

geo["geolocation_lat"] = pd.to_numeric(
    geo["geolocation_lat"], errors="coerce"
)

geo["geolocation_lng"] = pd.to_numeric(
    geo["geolocation_lng"], errors="coerce"
)

before = len(geo)

# Exact duplicate rows are safe to remove.
geo = geo.drop_duplicates().copy()

log(f"Exact duplicate rows removed: {before - len(geo):,}")

# Approximate Brazil bounding box.
# These are screening boundaries, not proof that a coordinate is correct.
geo["invalid_coordinate_flag"] = (
    geo["geolocation_lat"].isna()
    | geo["geolocation_lng"].isna()
    | ~geo["geolocation_lat"].between(-34, 6)
    | ~geo["geolocation_lng"].between(-74, -34)
)

log(
    "Coordinates outside screening boundaries: "
    f"{geo['invalid_coordinate_flag'].sum():,}"
)

# Keep suspicious records, but exclude them from representative coordinates.
valid_geo = geo.loc[~geo["invalid_coordinate_flag"]].copy()

# Median coordinates reduce the influence of extreme points.
geo_summary = (
    valid_geo.groupby("geolocation_zip_code_prefix")
    .agg(
        representative_lat=("geolocation_lat", "median"),
        representative_lng=("geolocation_lng", "median"),
        coordinate_records=("geolocation_lat", "size"),
        geo_states=("geolocation_state", "nunique")
    )
    .reset_index()
)

geo_summary.to_csv(
    REPORT_DIR / "geolocation_zip_summary.csv",
    index=False
)

save(geo, "olist_geolocation_dataset.csv")

log(f"Geolocation records after exact deduplication: {len(geo):,}")
log(f"ZIP prefixes with representative coordinates: {len(geo_summary):,}")


# =========================================================
# 2. SELLERS
# =========================================================

log("\n========== SELLERS ==========")

sellers = pd.read_csv(
    DATA_DIR / "olist_sellers_dataset.csv",
    dtype={"seller_zip_code_prefix": "string"}
)

sellers["seller_zip_code_prefix"] = normalize_zip(
    sellers["seller_zip_code_prefix"]
)

# Remove helper columns from previous attempts/runs.
helper_columns = [
    "geo_reference_state",
    "geolocation_zip_code_prefix",
    "seller_geo_status"
]

sellers.drop(
    columns=helper_columns,
    inplace=True,
    errors="ignore"
)

# Get the most frequent geolocation state for each ZIP.
zip_state = (
    valid_geo.groupby("geolocation_zip_code_prefix")["geolocation_state"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    .rename("geo_reference_state")
    .reset_index()
)

sellers = sellers.merge(
    zip_state,
    left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left",
    validate="many_to_one"
)

sellers["seller_geo_status"] = "state_match"

sellers.loc[
    sellers["geo_reference_state"].isna(),
    "seller_geo_status"
] = "unmatched_zip"

sellers.loc[
    sellers["geo_reference_state"].notna()
    & sellers["seller_state"].ne(sellers["geo_reference_state"]),
    "seller_geo_status"
] = "state_mismatch"

sellers.drop(
    columns=["geolocation_zip_code_prefix"],
    inplace=True
)

log("Seller state comparison:")
log(sellers["seller_geo_status"].value_counts(dropna=False).to_string())

sellers.to_csv(
    REPORT_DIR / "seller_state_validation.csv",
    index=False
)

save(sellers, "olist_sellers_dataset.csv")

# =========================================================
# 3. PAYMENTS
# =========================================================

log("\n========== PAYMENTS ==========")

payments = pd.read_csv(
    DATA_DIR / "olist_order_payments_dataset.csv"
)

payments["payment_value"] = pd.to_numeric(
    payments["payment_value"], errors="coerce"
)

payments["payment_installments"] = pd.to_numeric(
    payments["payment_installments"], errors="coerce"
)

payments["zero_payment_value_flag"] = (
    payments["payment_value"].eq(0)
)

payments["negative_payment_value_flag"] = (
    payments["payment_value"] < 0
)

payments["zero_installments_flag"] = (
    payments["payment_installments"].eq(0)
)

payments["invalid_installments_flag"] = (
    payments["payment_installments"].isna()
    | (payments["payment_installments"] < 0)
)

payments["undefined_payment_type_flag"] = (
    payments["payment_type"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("not_defined")
)

log(f"Zero payment values: {payments['zero_payment_value_flag'].sum():,}")
log(f"Negative payment values: {payments['negative_payment_value_flag'].sum():,}")
log(f"Zero installments: {payments['zero_installments_flag'].sum():,}")
log(f"Invalid installments: {payments['invalid_installments_flag'].sum():,}")
log(f"Undefined payment types: {payments['undefined_payment_type_flag'].sum():,}")

# Do not delete or replace suspicious payment records automatically.
payments.to_csv(
    REPORT_DIR / "payment_anomalies.csv",
    index=False
)

save(payments, "olist_order_payments_dataset.csv")


# =========================================================
# 4. REVIEWS
# =========================================================

log("\n========== REVIEWS ==========")

reviews = pd.read_csv(
    DATA_DIR / "olist_order_reviews_dataset.csv"
)

# Exact duplicate rows are safe to remove.
before = len(reviews)
reviews = reviews.drop_duplicates().copy()

log(f"Exact duplicate review rows removed: {before - len(reviews):,}")

# Repeated review IDs are investigated, not automatically deleted.
reviews["duplicate_review_id_flag"] = (
    reviews["review_id"].duplicated(keep=False)
)

# Multiple reviews per order are retained and flagged.
reviews["multiple_reviews_for_order_flag"] = (
    reviews["order_id"].duplicated(keep=False)
)

review_id_report = (
    reviews.loc[reviews["duplicate_review_id_flag"]]
    .sort_values("review_id")
)

order_review_report = (
    reviews.loc[reviews["multiple_reviews_for_order_flag"]]
    .sort_values("order_id")
)

review_id_report.to_csv(
    REPORT_DIR / "duplicate_review_ids.csv",
    index=False
)

order_review_report.to_csv(
    REPORT_DIR / "multiple_reviews_per_order.csv",
    index=False
)

log(
    "Rows with repeated review IDs: "
    f"{reviews['duplicate_review_id_flag'].sum():,}"
)

log(
    "Rows belonging to orders with multiple reviews: "
    f"{reviews['multiple_reviews_for_order_flag'].sum():,}"
)

save(reviews, "olist_order_reviews_dataset.csv")


# =========================================================
# 5. PRODUCTS + CATEGORY TRANSLATION
# =========================================================

log("\n========== PRODUCTS ==========")

products = pd.read_csv(
    DATA_DIR / "olist_products_dataset.csv"
)

translation = pd.read_csv(
    DATA_DIR / "product_category_name_translation.csv"
)

# Validate translation key uniqueness before joining.
duplicate_translation_keys = translation[
    translation["product_category_name"].duplicated(keep=False)
]

if not duplicate_translation_keys.empty:
    log(
        "WARNING: Duplicate category keys found in translation table. "
        "Translation join skipped."
    )
else:
    products = products.merge(
        translation,
        left_on="product_category_name_clean",
        right_on="product_category_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_translation")
    )

    products["category_translation_missing_flag"] = (
        products["product_category_name_clean"].notna()
        & products["product_category_name_clean"].ne("unknown")
        & products["product_category_name_english"].isna()
    )

    log(
        "Products with missing English category translation: "
        f"{products['category_translation_missing_flag'].sum():,}"
    )

    products.drop(
        columns=["product_category_name_translation"],
        inplace=True,
        errors="ignore"
    )

    products.to_csv(
        REPORT_DIR / "product_category_validation.csv",
        index=False
    )

# Validate measurements after the earlier imputation.
measurement_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for column in measurement_columns:
    if column not in products.columns:
        continue

    products[column] = pd.to_numeric(
        products[column], errors="coerce"
    )

    invalid = products[column].isna() | (products[column] <= 0)

    log(
        f"{column}: remaining missing/zero/negative values = "
        f"{invalid.sum():,}"
    )

products.to_csv(
    REPORT_DIR / "product_validation.csv",
    index=False
)

save(products, "olist_products_dataset.csv")


# =========================================================
# FINAL REPORT
# =========================================================

log("\n========== FINAL CLEANING COMPLETE ==========")

with open(
    REPORT_DIR / "final_cleaning_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report))

log(f"\nReports saved to: {REPORT_DIR}")
log("Raw CSV files were not modified.")
