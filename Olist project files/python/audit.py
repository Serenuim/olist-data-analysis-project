
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = []


def log(message=""):
    REPORT.append(str(message))
    print(message)


def load_csv(filename):
    """Load a CSV while supporting common file encodings."""
    path = BASE_DIR / filename

    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read {filename}")


def save_csv(df, filename):
    """Save the cleaned dataset without overwriting raw data."""
    df.to_csv(PROCESSED_DIR / filename, index=False)


# --------------------------------------------------
# 2. ORDERS: INVESTIGATE TIMESTAMP COMPLETENESS
# --------------------------------------------------

def clean_orders():
    filename = "olist_orders_dataset.csv"
    df = load_csv(filename)

    log("\n========== ORDERS ==========")

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    # Convert date strings into actual datetime values.
    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    status = df["order_status"].str.lower().str.strip()

    # Missing timestamps are not automatically errors.
    df["is_approved"] = df["order_approved_at"].notna()

    df["was_sent_to_carrier"] = (
        df["order_delivered_carrier_date"].notna()
    )

    df["was_delivered"] = (
        df["order_delivered_customer_date"].notna()
    )

    # A delivered order should have a delivery timestamp.
    df["missing_delivery_date_flag"] = (
        status.eq("delivered")
        & df["order_delivered_customer_date"].isna()
    )

    # Delivered orders should normally have carrier dates too.
    df["missing_carrier_date_flag"] = (
        status.eq("delivered")
        & df["order_delivered_carrier_date"].isna()
    )

    # Check timestamp order.
    df["invalid_timestamp_sequence_flag"] = (
        (
            df["order_approved_at"].notna()
            & (
                df["order_approved_at"]
                < df["order_purchase_timestamp"]
            )
        )
        |
        (
            df["order_delivered_carrier_date"].notna()
            & df["order_approved_at"].notna()
            & (
                df["order_delivered_carrier_date"]
                < df["order_approved_at"]
            )
        )
        |
        (
            df["order_delivered_customer_date"].notna()
            & df["order_delivered_carrier_date"].notna()
            & (
                df["order_delivered_customer_date"]
                < df["order_delivered_carrier_date"]
            )
        )
    )

    log("Missing timestamps:")
    log(df[date_columns].isna().sum())

    log(
        "Delivered orders missing delivery dates: "
        f"{df['missing_delivery_date_flag'].sum()}"
    )

    log(
        "Delivered orders missing carrier dates: "
        f"{df['missing_carrier_date_flag'].sum()}"
    )

    log(
        "Invalid timestamp sequences: "
        f"{df['invalid_timestamp_sequence_flag'].sum()}"
    )

    # Do not fabricate missing order dates.
    save_csv(df, filename)


# --------------------------------------------------
# 3. REVIEWS: PRESERVE GENUINELY MISSING COMMENTS
# --------------------------------------------------

def clean_reviews():
    filename = "olist_order_reviews_dataset.csv"
    df = load_csv(filename)

    log("\n========== REVIEWS ==========")

    text_columns = [
        "review_comment_title",
        "review_comment_message"
    ]

    # Empty strings and whitespace are also missing text.
    for column in text_columns:
        df[column] = df[column].replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

    # Preserve the original missing values.
    df["has_review_title"] = (
        df["review_comment_title"].notna()
    )

    df["has_review_message"] = (
        df["review_comment_message"].notna()
    )

    # Review score must be between 1 and 5.
    df["review_score"] = pd.to_numeric(
        df["review_score"],
        errors="coerce"
    )

    df["invalid_review_score_flag"] = (
        ~df["review_score"].between(1, 5)
    )

    log("Missing review values:")
    log(df[text_columns].isna().sum())

    log(
        "Invalid review scores: "
        f"{df['invalid_review_score_flag'].sum()}"
    )

    log(
        "Reviews without written messages: "
        f"{(~df['has_review_message']).sum()}"
    )

    save_csv(df, filename)


# --------------------------------------------------
# 4. PRODUCTS: INVESTIGATE AND CLEAN MEASUREMENTS
# --------------------------------------------------

def clean_products():
    filename = "olist_products_dataset.csv"
    df = load_csv(filename)

    log("\n========== PRODUCTS ==========")

    # Clean category text without inventing a real category.
    df["product_category_name"] = (
        df["product_category_name"]
        .replace(r"^\s*$", pd.NA, regex=True)
    )

    df["product_category_name_clean"] = (
        df["product_category_name"].fillna("unknown")
    )

    df["product_category_missing_flag"] = (
        df["product_category_name"].isna()
    )

    # Check product name and description availability.
    df["has_product_name_length"] = (
        df["product_name_lenght"].notna()
    )

    df["has_product_description_length"] = (
        df["product_description_lenght"].notna()
    )

    # Missing photo count must not automatically become zero.
    df["has_product_photo_count"] = (
        df["product_photos_qty"].notna()
    )

    measurement_columns = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    # Convert measurement columns to numeric.
    for column in measurement_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Zero and negative physical measurements are invalid.
        df[f"{column}_invalid_flag"] = (
            df[column].notna() & (df[column] <= 0)
        )

        # Record missing or invalid measurements before filling.
        df[f"{column}_imputed_flag"] = (
            df[column].isna() | (df[column] <= 0)
        )

        df.loc[
            df[f"{column}_imputed_flag"],
            column
        ] = pd.NA

    # Investigate original missingness.
    log("Missing product values before imputation:")
    log(df.isna().sum()[df.isna().sum() > 0])

    # Use category median, then global median if necessary.
    for column in measurement_columns:

        category_medians = df.groupby(
            "product_category_name_clean"
        )[column].transform("median")

        global_median = df[column].median()

        # First attempt: median from the same category.
        df[column] = df[column].fillna(category_medians)

        # Fallback: overall median.
        if pd.notna(global_median):
            df[column] = df[column].fillna(global_median)

        log(
            f"{column}: "
            f"{df[f'{column}_imputed_flag'].sum()} "
            "records requiring imputation"
        )

    # Validate the measurements after imputation.
    for column in measurement_columns:
        invalid_remaining = (
            df[column].isna() | (df[column] <= 0)
        )

        log(
            f"{column} remaining invalid values: "
            f"{invalid_remaining.sum()}"
        )

    # Check text-length and photo-count values.
    count_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty"
    ]

    for column in count_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[f"{column}_invalid_flag"] = (
            df[column].notna() & (df[column] < 0)
        )

        log(
            f"{column} negative values: "
            f"{df[f'{column}_invalid_flag'].sum()}"
        )

    save_csv(df, filename)


# --------------------------------------------------
# 5. COPY AND VALIDATE DATASETS WITHOUT MISSING VALUES
# --------------------------------------------------

def preserve_other_datasets():
    filenames = [
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv"
    ]

    log("\n========== OTHER DATASETS ==========")

    for filename in filenames:
        df = load_csv(filename)

        log(
            f"{filename}: "
            f"{len(df):,} rows, "
            f"{df.shape[1]} columns, "
            f"{df.isna().sum().sum():,} missing cells"
        )

        # These datasets have no missing values in the audit.
        # Preserve their original values.
        save_csv(df, filename)


# --------------------------------------------------
# 6. FINAL VALIDATION AND REPORT
# --------------------------------------------------

def final_validation():
    log("\n========== FINAL VALIDATION ==========")

    filenames = list(PROCESSED_DIR.glob("*.csv"))

    for path in filenames:
        df = pd.read_csv(path)

        log(f"\nDataset: {path.name}")
        log(f"Rows: {len(df):,}")
        log(f"Columns: {len(df.columns)}")
        log(f"Duplicate rows: {df.duplicated().sum()}")

        missing = df.isna().sum()
        missing = missing[missing > 0]

        if missing.empty:
            log("Missing values: None")
        else:
            log("Remaining missing values:")
            log(missing)

    log("\nCleaning outputs saved successfully.")


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():
    log("STARTING OLIST DATA CLEANING")

    clean_orders()
    clean_reviews()
    clean_products()
    preserve_other_datasets()

    final_validation()

    report_path = REPORT_DIR / "cleaning_report.txt"

    report_path.write_text(
        "\n".join(REPORT),
        encoding="utf-8"
    )

    print(f"\nReport saved to: {report_path}")
    print(f"Processed files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
