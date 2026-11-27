from pathlib import Path
import pandas as pd
import numpy as np

# Reads the ORIGINAL nine CSVs from the project root.
# Writes cleaned copies to data/cleaned and exceptions/reports to reports/final_cleaning.
# Never edits the original files or reads data/processed.

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "cleaned"
REPORT = BASE / "reports" / "final_cleaning"
OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
status, notes = [], []

def read(key, **kwargs):
    p = BASE / FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"Missing original CSV: {p}")
    return pd.read_csv(p, **kwargs)

def save(key, df):
    df.to_csv(OUT / FILES[key], index=False)

def exceptions(df, name):
    if df is not None and not df.empty:
        df.to_csv(REPORT / name, index=False)

def zip_clean(s):
    x = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    x = x.str.replace(r"\D", "", regex=True)
    return x.str.zfill(5).where(x.notna(), pd.NA)

def numeric(s):
    return pd.to_numeric(s, errors="coerce")

def finish(key, df, issue_mask, reason):
    save(key, df)
    n = int(issue_mask.sum()) if hasattr(issue_mask, "sum") else int(bool(issue_mask))
    status.append({
        "dataset": key,
        "rows": len(df),
        "status": "REVIEW REQUIRED" if n else "CLEAN",
        "flagged_rows": n,
        "reason": reason if n else "No outstanding validation flags."
    })

def log(s):
    print(s)
    notes.append(str(s))

# 1. CUSTOMERS
log("\\n========== CUSTOMERS ==========")
d = read("customers", dtype={
    "customer_id": "string", "customer_unique_id": "string",
    "customer_zip_code_prefix": "string"
})
d["customer_zip_code_prefix"] = zip_clean(d["customer_zip_code_prefix"])
d["duplicate_customer_id_flag"] = d.customer_id.duplicated(keep=False)
d["missing_customer_unique_id_flag"] = d.customer_unique_id.isna()
d["missing_customer_zip_flag"] = d.customer_zip_code_prefix.isna()
mask = d[["duplicate_customer_id_flag", "missing_customer_unique_id_flag",
          "missing_customer_zip_flag"]].any(axis=1)
exceptions(d[mask], "customers_review_required.csv")
log(f"Rows={len(d):,}; duplicate customer IDs={d.duplicate_customer_id_flag.sum():,}; "
    f"missing unique IDs={d.missing_customer_unique_id_flag.sum():,}; "
    f"missing ZIPs={d.missing_customer_zip_flag.sum():,}")
finish("customers", d, mask, "Inspect duplicate customer IDs and missing unique IDs/ZIPs.")

# 2. GEOLOCATION
log("\\n========== GEOLOCATION ==========")
geo = read("geolocation", dtype={"geolocation_zip_code_prefix": "string"})
geo["geolocation_zip_code_prefix"] = zip_clean(geo.geolocation_zip_code_prefix)
for c in ["geolocation_lat", "geolocation_lng"]:
    geo[c] = numeric(geo[c])
before = len(geo)
geo = geo.drop_duplicates().copy()
removed = before - len(geo)
geo["invalid_coordinate_flag"] = (
    geo.geolocation_lat.isna() | geo.geolocation_lng.isna()
    | ~geo.geolocation_lat.between(-34, 6)
    | ~geo.geolocation_lng.between(-74, -34)
)
geo["missing_zip_flag"] = geo.geolocation_zip_code_prefix.isna()
gmask = geo.invalid_coordinate_flag | geo.missing_zip_flag
exceptions(geo[gmask], "geolocation_review_required.csv")
valid_geo = geo[~gmask].copy()
geo_summary = valid_geo.groupby("geolocation_zip_code_prefix").agg(
    representative_lat=("geolocation_lat", "median"),
    representative_lng=("geolocation_lng", "median"),
    coordinate_records=("geolocation_lat", "size"),
    distinct_states=("geolocation_state", "nunique")
).reset_index()
geo_summary.to_csv(REPORT / "geolocation_zip_summary.csv", index=False)
log(f"Exact duplicate rows removed={removed:,}; remaining={len(geo):,}; "
    f"invalid coordinates={geo.invalid_coordinate_flag.sum():,}; "
    f"missing ZIP={geo.missing_zip_flag.sum():,}; "
    f"ZIP coordinate summaries={len(geo_summary):,}")
finish("geolocation", geo, gmask,
       "Suspicious coordinates/missing ZIPs retained and flagged; use ZIP medians for mapping.")

# 3. ORDER ITEMS
log("\\n========== ORDER ITEMS ==========")
d = read("order_items", dtype={"order_id": "string", "product_id": "string", "seller_id": "string"})
for c in ["price", "freight_value", "order_item_id"]:
    d[c] = numeric(d[c])
d["negative_price_flag"] = d.price < 0
d["zero_or_missing_price_flag"] = d.price.isna() | d.price.eq(0)
d["negative_freight_flag"] = d.freight_value < 0
d["missing_freight_flag"] = d.freight_value.isna()
d["invalid_item_number_flag"] = d.order_item_id.isna() | (d.order_item_id <= 0)
d["duplicate_order_item_key_flag"] = d.duplicated(["order_id", "order_item_id"], keep=False)
mask = d[[c for c in d.columns if c.endswith("_flag")]].any(axis=1)
exceptions(d[mask], "order_items_review_required.csv")
log(f"Rows={len(d):,}; flagged rows={mask.sum():,}; "
    f"negative prices={d.negative_price_flag.sum():,}; "
    f"zero/missing prices={d.zero_or_missing_price_flag.sum():,}; "
    f"negative freight={d.negative_freight_flag.sum():,}; "
    f"duplicate order-item keys={d.duplicate_order_item_key_flag.sum():,}")
finish("order_items", d, mask, "Inspect invalid prices/freight/item numbers and duplicate composite keys.")

# 4. PAYMENTS
log("\\n========== PAYMENTS ==========")
d = read("payments", dtype={"order_id": "string"})
for c in ["payment_value", "payment_installments", "payment_sequential"]:
    d[c] = numeric(d[c])
d["zero_payment_value_flag"] = d.payment_value.eq(0)
d["negative_payment_value_flag"] = d.payment_value < 0
d["missing_payment_value_flag"] = d.payment_value.isna()
d["zero_installments_flag"] = d.payment_installments.eq(0)
d["invalid_installments_flag"] = d.payment_installments.isna() | (d.payment_installments < 0)
d["invalid_payment_sequential_flag"] = d.payment_sequential.isna() | (d.payment_sequential <= 0)
d["undefined_payment_type_flag"] = (
    d.payment_type.astype("string").str.strip().str.lower().isin(["not_defined", ""])
    | d.payment_type.isna()
)
d["duplicate_payment_key_flag"] = d.duplicated(["order_id", "payment_sequential"], keep=False)
mask = d[[c for c in d.columns if c.endswith("_flag")]].any(axis=1)
exceptions(d[mask], "payments_review_required.csv")
log(f"Rows={len(d):,}; flagged rows={mask.sum():,}; "
    f"zero values={d.zero_payment_value_flag.sum():,}; "
    f"negative values={d.negative_payment_value_flag.sum():,}; "
    f"zero installments={d.zero_installments_flag.sum():,}; "
    f"undefined types={d.undefined_payment_type_flag.sum():,}; "
    f"duplicate payment keys={d.duplicate_payment_key_flag.sum():,}")
finish("payments", d, mask, "Keep and investigate anomalous payment values, types, installments and keys.")

# 5. REVIEWS
log("\\n========== REVIEWS ==========")
d = read("reviews", dtype={"review_id": "string", "order_id": "string"})
for c in ["review_comment_title", "review_comment_message"]:
    d[c] = d[c].astype("string").str.strip().replace("", pd.NA)
d["review_score"] = numeric(d.review_score)
for c in ["review_creation_date", "review_answer_timestamp"]:
    if c in d:
        d[c] = pd.to_datetime(d[c], errors="coerce", format="mixed")
before = len(d)
d = d.drop_duplicates().copy()
removed = before - len(d)
d["invalid_review_score_flag"] = d.review_score.isna() | ~d.review_score.between(1, 5)
d["missing_review_id_flag"] = d.review_id.isna()
d["missing_order_id_flag"] = d.order_id.isna()
d["duplicate_review_id_flag"] = d.review_id.duplicated(keep=False)
d["multiple_reviews_for_order_flag"] = d.order_id.duplicated(keep=False)
d["has_review_title"] = d.review_comment_title.notna()
d["has_review_message"] = d.review_comment_message.notna()
exceptions(d[d.duplicate_review_id_flag], "reviews_duplicate_ids.csv")
exceptions(d[d.multiple_reviews_for_order_flag], "reviews_multiple_per_order.csv")
mask = d[["invalid_review_score_flag", "missing_review_id_flag", "missing_order_id_flag",
          "duplicate_review_id_flag", "multiple_reviews_for_order_flag"]].any(axis=1)
exceptions(d[mask], "reviews_review_required.csv")
log(f"Exact duplicates removed={removed:,}; invalid scores={d.invalid_review_score_flag.sum():,}; "
    f"repeated review-ID rows={d.duplicate_review_id_flag.sum():,}; "
    f"multiple-review order rows={d.multiple_reviews_for_order_flag.sum():,}")
finish("reviews", d, mask, "Repeated IDs/multiple reviews are retained pending investigation.")

# 6. ORDERS
log("\\n========== ORDERS ==========")
d = read("orders", dtype={"order_id": "string", "customer_id": "string"})
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
for c in date_cols:
    d[c] = pd.to_datetime(d[c], errors="coerce", format="mixed")
p, a, c, delivery = (d["order_purchase_timestamp"], d["order_approved_at"],
                     d["order_delivered_carrier_date"], d["order_delivered_customer_date"])
d["is_approved"] = a.notna()
d["was_sent_to_carrier"] = c.notna()
d["was_delivered"] = d.order_status.eq("delivered")
d["missing_approval_date_flag"] = a.isna()
d["missing_carrier_date_flag"] = c.isna()
d["missing_delivery_date_flag"] = delivery.isna()
d["approval_before_purchase_flag"] = a.notna() & p.notna() & (a < p)
d["carrier_before_approval_flag"] = c.notna() & a.notna() & (c < a)
d["delivery_before_carrier_flag"] = delivery.notna() & c.notna() & (delivery < c)
d["invalid_timestamp_sequence_flag"] = d[
    ["approval_before_purchase_flag", "carrier_before_approval_flag",
     "delivery_before_carrier_flag"]].any(axis=1)
d["valid_purchase_to_delivery_duration"] = p.notna() & delivery.notna() & (delivery >= p)
d["valid_approval_to_carrier_duration"] = a.notna() & c.notna() & (c >= a)
d["valid_carrier_to_delivery_duration"] = c.notna() & delivery.notna() & (delivery >= c)
d["valid_full_logistics_sequence"] = (
    p.notna() & a.notna() & c.notna() & delivery.notna()
    & (a >= p) & (c >= a) & (delivery >= c)
)
d["purchase_to_delivery_hours"] = ((delivery - p).dt.total_seconds() / 3600).where(d.valid_purchase_to_delivery_duration)
d["approval_to_carrier_hours"] = ((c - a).dt.total_seconds() / 3600).where(d.valid_approval_to_carrier_duration)
d["carrier_to_delivery_hours"] = ((delivery - c).dt.total_seconds() / 3600).where(d.valid_carrier_to_delivery_duration)
d["duplicate_order_id_flag"] = d.order_id.duplicated(keep=False)
mask = d.invalid_timestamp_sequence_flag | d.duplicate_order_id_flag
exceptions(d[mask], "orders_review_required.csv")
log(f"Rows={len(d):,}; approval-before-purchase={d.approval_before_purchase_flag.sum():,}; "
    f"carrier-before-approval={d.carrier_before_approval_flag.sum():,}; "
    f"delivery-before-carrier={d.delivery_before_carrier_flag.sum():,}; "
    f"valid full sequences={d.valid_full_logistics_sequence.sum():,}")
finish("orders", d, mask, "Timestamp anomalies retained; invalid durations are excluded through eligibility flags.")

# 7. PRODUCTS
log("\\n========== PRODUCTS ==========")
d = read("products", dtype={"product_id": "string"})
translation = read("translation", dtype={
    "product_category_name": "string",
    "product_category_name_english": "string"
})
d["product_category_name_clean"] = d.product_category_name.astype("string").str.strip()
d["missing_category_flag"] = d.product_category_name_clean.isna() | d.product_category_name_clean.eq("")
d.loc[d.missing_category_flag, "product_category_name_clean"] = "unknown"
d["duplicate_product_id_flag"] = d.product_id.duplicated(keep=False)
measurements = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
for col in measurements:
    d[col] = numeric(d[col])
    invalid = d[col].isna() | (d[col] <= 0)
    d[f"{col}_invalid_flag"] = invalid
    d[f"{col}_imputed_flag"] = False
    valid = d.loc[~invalid]
    global_median = valid[col].median()
    cat_medians = valid.groupby("product_category_name_clean")[col].median()
    for idx in d.index[invalid]:
        fill = cat_medians.get(d.at[idx, "product_category_name_clean"], np.nan)
        if pd.isna(fill):
            fill = global_median
        if pd.notna(fill):
            d.at[idx, col] = fill
            d.at[idx, f"{col}_imputed_flag"] = True
for col in ["product_name_lenght", "product_description_lenght", "product_photos_qty"]:
    if col in d:
        d[col] = numeric(d[col])
        d[f"{col}_invalid_flag"] = d[col].isna() | (d[col] < 0)
translation["product_category_name"] = translation.product_category_name.astype("string").str.strip()
translation["product_category_name_english"] = translation.product_category_name_english.astype("string").str.strip()
translation_dup = translation.product_category_name.duplicated(keep=False)
if translation_dup.any():
    exceptions(translation[translation_dup], "translation_duplicate_keys.csv")
    d["category_translation_missing_flag"] = True
else:
    d = d.merge(translation, left_on="product_category_name_clean",
                right_on="product_category_name", how="left",
                validate="many_to_one", suffixes=("", "_translation"))
    d["category_translation_missing_flag"] = (
        d.product_category_name_clean.ne("unknown")
        & d.product_category_name_english.isna()
    )
d["unresolved_measurement_flag"] = False
for col in measurements:
    d["unresolved_measurement_flag"] |= d[col].isna() | (d[col] <= 0)
mask_cols = ["duplicate_product_id_flag", "category_translation_missing_flag", "unresolved_measurement_flag"]
mask = d[mask_cols].any(axis=1)
for col in ["product_name_lenght", "product_description_lenght", "product_photos_qty"]:
    flag = f"{col}_invalid_flag"
    if flag in d:
        mask |= d[flag]
exceptions(d[mask], "products_review_required.csv")
log(f"Rows={len(d):,}; missing source categories={d.missing_category_flag.sum():,}; "
    f"untranslated categories in product rows={d.category_translation_missing_flag.sum():,}; "
    f"duplicate product IDs={d.duplicate_product_id_flag.sum():,}")
for col in measurements:
    log(f"{col}: imputed={d[f'{col}_imputed_flag'].sum():,}; remaining invalid={(d[col].isna() | (d[col] <= 0)).sum():,}")
finish("products", d, mask, "Review untranslated categories, duplicate IDs, invalid metadata and unresolved measurements.")

# 8. SELLERS
log("\\n========== SELLERS ==========")
d = read("sellers", dtype={"seller_id": "string", "seller_zip_code_prefix": "string"})
d["seller_zip_code_prefix"] = zip_clean(d.seller_zip_code_prefix)
d["duplicate_seller_id_flag"] = d.seller_id.duplicated(keep=False)
d["missing_seller_zip_flag"] = d.seller_zip_code_prefix.isna()
zip_state = valid_geo.groupby("geolocation_zip_code_prefix")["geolocation_state"].agg(
    lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
).rename("geo_reference_state").reset_index()
d = d.merge(zip_state, left_on="seller_zip_code_prefix",
            right_on="geolocation_zip_code_prefix", how="left", validate="many_to_one")
d["seller_geo_status"] = "state_match"
d.loc[d.geo_reference_state.isna(), "seller_geo_status"] = "unmatched_zip"
d.loc[d.geo_reference_state.notna() & d.seller_state.ne(d.geo_reference_state),
      "seller_geo_status"] = "state_mismatch"
d.drop(columns=["geolocation_zip_code_prefix"], inplace=True)
mask = (d.duplicate_seller_id_flag | d.missing_seller_zip_flag
        | d.seller_geo_status.ne("state_match"))
exceptions(d[mask], "sellers_review_required.csv")
log("Seller ZIP/state comparison:")
log(d.seller_geo_status.value_counts(dropna=False).to_string())
finish("sellers", d, mask, "Do not overwrite seller states; investigate mismatches and unmatched ZIPs.")

# 9. CATEGORY TRANSLATION
log("\\n========== CATEGORY TRANSLATION ==========")
d = read("translation", dtype={
    "product_category_name": "string",
    "product_category_name_english": "string"
})
for col in ["product_category_name", "product_category_name_english"]:
    d[col] = d[col].astype("string").str.strip().replace("", pd.NA)
d["missing_source_category_flag"] = d.product_category_name.isna()
d["missing_english_translation_flag"] = d.product_category_name_english.isna()
d["duplicate_category_key_flag"] = d.product_category_name.duplicated(keep=False)
mask = d[["missing_source_category_flag", "missing_english_translation_flag",
          "duplicate_category_key_flag"]].any(axis=1)
exceptions(d[mask], "translation_review_required.csv")
log(f"Rows={len(d):,}; duplicate keys={d.duplicate_category_key_flag.sum():,}; "
    f"missing source labels={d.missing_source_category_flag.sum():,}; "
    f"missing English labels={d.missing_english_translation_flag.sum():,}")
finish("translation", d, mask, "Review duplicate keys or missing category labels.")

# Final summary
summary = pd.DataFrame(status)
summary.to_csv(REPORT / "dataset_status.csv", index=False)
with open(REPORT / "final_cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write("\\n".join(notes))
    f.write("\\n\\n========== DATASET STATUS ==========\\n")
    f.write(summary.to_string(index=False))

print("\\n========== FINAL DATASET STATUS ==========")
print(summary.to_string(index=False))
print(f"\\nCleaned CSVs: {OUT}")
print(f"Exceptions and reports: {REPORT}")
print("Original root CSV files were NOT modified.")
