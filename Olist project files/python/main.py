
import io
from pathlib import Path

import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================

FOLDER = Path(".")
OUTPUT = Path("data_quality_report.txt")

RECURSIVE = False

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")

TOP_N = 5

# ZIP prefixes are identifiers, not quantities.
# Preserve leading zeros when loading them.
DTYPE_OVERRIDES = {
    "customer_zip_code_prefix": "string",
    "seller_zip_code_prefix": "string",
    "geolocation_zip_code_prefix": "string",
}

# Candidate keys for duplicate-key investigation.
# These are validation expectations, not assumptions
# that every dataset must have a unique individual ID.

KEYS = {
    "olist_customers_dataset.csv": ["customer_id"],
    "olist_geolocation_dataset.csv": None,
    "olist_order_items_dataset.csv": ["order_id", "order_item_id"],
    "olist_order_payments_dataset.csv": ["order_id", "payment_sequential"],
    "olist_order_reviews_dataset.csv": ["review_id"],
    "olist_orders_dataset.csv": ["order_id"],
    "olist_products_dataset.csv": ["product_id"],
    "olist_sellers_dataset.csv": ["seller_id"],
    "product_category_name_translation.csv": [
        "product_category_name"
    ],
}

DATE_KEYWORDS = ("date", "timestamp")


# ==================================================
# CSV LOADING
# ==================================================

def read_csv_any(path):
    """Load CSV with encoding fallbacks."""

    last_error = None

    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                dtype=DTYPE_OVERRIDES
            )
            return df, encoding

        except Exception as error:
            last_error = error

    raise last_error


# ==================================================
# REPORT HELPERS
# ==================================================

def heading(title):
    return "\n" + "=" * 75 + "\n" + title + "\n" + "=" * 75


def section(title):
    return "\n" + "-" * 75 + "\n" + title + "\n" + "-" * 75


def missing_report(df):
    """Missing values and percentages."""

    missing = df.isna().sum()
    result = []

    for col in df.columns:
        count = int(missing[col])

        if count > 0:
            percentage = count / len(df) * 100

            result.append(
                f"{col:<40} "
                f"Missing: {count:>10,} "
                f"({percentage:>7.2f}%)"
            )

    if not result:
        return ["No missing values detected."]

    return result


def duplicate_report(df, filename):
    """Full-row and candidate-key duplicate checks."""

    lines = []

    full_duplicates = int(df.duplicated().sum())

    lines.append(
        f"Full duplicate rows: {full_duplicates:,}"
    )

    keys = KEYS.get(filename)

    if keys:
        missing_keys = int(df[keys].isna().any(axis=1).sum())

        duplicate_keys = int(
            df.duplicated(subset=keys, keep=False).sum()
        )

        lines.append(f"Candidate key: {keys}")
        lines.append(f"Rows with missing key values: {missing_keys:,}")
        lines.append(f"Rows involved in duplicate keys: {duplicate_keys:,}")

    else:
        lines.append(
            "No single-row unique key expected; "
            "geolocation records can repeat ZIP prefixes."
        )

    return lines


def numerical_report(df):
    """Statistical summary of numerical columns."""

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return ["No numerical columns found."]

    summary = numeric.describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T

    return summary.to_string(
        float_format=lambda x: f"{x:,.2f}"
    ).splitlines()


def categorical_report(df):
    """Categorical distributions and cardinality."""

    lines = []

    categorical = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    )

    if categorical.empty:
        return ["No categorical columns found."]

    for col in categorical.columns:

        unique_count = df[col].nunique(dropna=True)

        lines.append(
            f"\n{col}: {unique_count:,} unique values"
        )

        # Avoid printing thousands of identifier values.
        if unique_count <= 50:

            frequencies = (
                df[col]
                .value_counts(dropna=False)
                .head(TOP_N)
            )

            for value, count in frequencies.items():
                lines.append(f"    {str(value)[:50]}: {count:,}")

        else:
            lines.append(
                "    High-cardinality column; "
                "frequency listing skipped."
            )

    return lines


def date_report(df):
    """Inspect date columns without changing the original data."""

    lines = []

    date_columns = [
        col for col in df.columns
        if any(word in col.lower() for word in DATE_KEYWORDS)
    ]

    if not date_columns:
        return ["No date columns detected."]

    for col in date_columns:

        original_non_null = df[col].notna().sum()

        parsed = pd.to_datetime(
            df[col],
            errors="coerce",
            format="mixed"
        )

        invalid = int(
            (df[col].notna() & parsed.isna()).sum()
        )

        lines.append(f"\n{col}")
        lines.append(f"    Original non-null: {original_non_null:,}")
        lines.append(f"    Invalid/unparsed: {invalid:,}")

        if parsed.notna().any():
            lines.append(f"    Earliest: {parsed.min()}")
            lines.append(f"    Latest:   {parsed.max()}")

    return lines


# ==================================================
# DATA ANOMALIES
# ==================================================

def anomaly_report(df, filename):
    """Check obvious domain-specific data problems."""

    issues = []

    checks = {
        "olist_order_items_dataset.csv": {
            "price": lambda s: s < 0,
            "freight_value": lambda s: s < 0,
            "order_item_id": lambda s: s <= 0,
        },

        "olist_order_payments_dataset.csv": {
            "payment_value": lambda s: s < 0,
            "payment_installments": lambda s: s < 0,
            "payment_sequential": lambda s: s <= 0,
        },

        "olist_order_reviews_dataset.csv": {
            "review_score": lambda s: ~s.between(1, 5),
        },

        "olist_products_dataset.csv": {
            "product_weight_g": lambda s: s <= 0,
            "product_length_cm": lambda s: s <= 0,
            "product_height_cm": lambda s: s <= 0,
            "product_width_cm": lambda s: s <= 0,
        },

        "olist_geolocation_dataset.csv": {
            "geolocation_lat": lambda s: ~s.between(-90, 90),
            "geolocation_lng": lambda s: ~s.between(-180, 180),
        },
    }

    filename_checks = checks.get(filename, {})

    for col, condition in filename_checks.items():

        if col not in df.columns:
            continue

        invalid = int(condition(df[col].dropna()).sum())

        issues.append(
            f"{col}: {invalid:,} suspicious values"
        )

    if not issues:
        return ["No configured anomaly checks for this file."]

    return issues


# ==================================================
# DATA TYPE INSPECTION
# ==================================================

def datatype_report(df):
    """Display data types and potential conversion candidates."""

    lines = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        note = ""

        if (
            dtype == "object"
            and any(word in col.lower() for word in DATE_KEYWORDS)
        ):
            note = "  [DATE COLUMN: parsing inspection required]"

        elif (
            dtype == "object"
            and col.lower().endswith("_id")
        ):
            note = "  [IDENTIFIER: keep as string]"

        lines.append(f"{col:<40} {dtype:<15}{note}")

    return lines


# ==================================================
# MAIN AUDIT
# ==================================================

def main():

    pattern = "**/*.csv" if RECURSIVE else "*.csv"

    files = sorted(FOLDER.glob(pattern))

    if not files:
        print(f"No CSV files found in {FOLDER.resolve()}")
        return

    report = []

    report.append(heading("OLIST E-COMMERCE - DATA QUALITY AUDIT"))
    report.append(f"Folder: {FOLDER.resolve()}")
    report.append(f"Files discovered: {len(files)}")

    successful = 0
    failed = []

    overview = []

    for path in files:

        print(f"[AUDITING] {path.name}")

        try:

            df, encoding = read_csv_any(path)

            rows, columns = df.shape

            overview.append({
                "File": path.name,
                "Rows": rows,
                "Columns": columns,
                "Memory_MB": round(
                    df.memory_usage(deep=True).sum() / 1024**2, 2
                ),
                "Encoding": encoding,
                "Full_Duplicates": int(df.duplicated().sum()),
                "Missing_Cells": int(df.isna().sum().sum()),
            })

            report.append(heading(f"FILE: {path.name}"))

            report.append(
                f"Rows: {rows:,} | Columns: {columns:,}"
            )

            report.append(
                f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

            report.append(f"Encoding: {encoding}")

            report.append(section("1. MISSING VALUE ANALYSIS"))
            report.extend(missing_report(df))

            report.append(section("2. DUPLICATE ANALYSIS"))
            report.extend(duplicate_report(df, path.name))

            report.append(section("3. DATA TYPE ANALYSIS"))
            report.extend(datatype_report(df))

            report.append(section("4. NUMERICAL STATISTICS"))
            report.extend(numerical_report(df))

            report.append(section("5. CATEGORICAL ANALYSIS"))
            report.extend(categorical_report(df))

            report.append(section("6. DATE ANALYSIS"))
            report.extend(date_report(df))

            report.append(section("7. DATA ANOMALIES"))
            report.extend(anomaly_report(df, path.name))

            successful += 1

            print(f"[SUCCESS] {path.name}")

        except Exception as error:

            failed.append((path.name, str(error)))

            report.append(
                f"\nFAILED: {path.name}\nERROR: {error}"
            )

            print(f"[FAILED] {path.name}: {error}")

        finally:
            # Release this dataset before loading the next.
            if "df" in locals():
                del df

    # Consolidated dataset overview
    report.append(heading("CONSOLIDATED DATASET OVERVIEW"))

    overview_df = pd.DataFrame(overview)

    if not overview_df.empty:
        report.append(
            overview_df.to_string(index=False)
        )

    report.append(heading("FINAL AUDIT SUMMARY"))
    report.append(f"Files discovered: {len(files)}")
    report.append(f"Successfully audited: {successful}")
    report.append(f"Failed: {len(failed)}")

    for filename, error in failed:
        report.append(f"{filename}: {error}")

    OUTPUT.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Report: {OUTPUT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
