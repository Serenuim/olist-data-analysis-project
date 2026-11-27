import pandas as pd, numpy as np
from math import radians, sin, cos, asin, sqrt

d= "path/to/folder/"
orders   = pd.read_csv("olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp","order_approved_at","order_delivered_carrier_date",
    "order_delivered_customer_date","order_estimated_delivery_date"])
items    = pd.read_csv("olist_order_items_dataset.csv")
pays     = pd.read_csv("olist_order_payments_dataset.csv")
revs     = pd.read_csv("olist_order_reviews_dataset.csv")
prods    = pd.read_csv("olist_products_dataset.csv")
sellers  = pd.read_csv("olist_sellers_dataset.csv")
custs    = pd.read_csv("olist_customers_dataset.csv")
trans    = pd.read_csv("product_category_name_translation.csv")

# Pre-aggregate to avoid fan-out
pays_agg = pays.groupby("order_id").agg(
    payment_value=("payment_value","sum"),
    n_payments=("payment_sequential","count"),
    max_installments=("payment_installments","max"),
    payment_type=("payment_type","first")).reset_index()
revs_agg = revs.groupby("order_id").agg(review_score=("review_score","mean")).reset_index()

master = (items
    .merge(orders, on="order_id", how="left")
    .merge(custs, on="customer_id", how="left")
    .merge(sellers, on="seller_id", how="left")
    .merge(prods, on="product_id", how="left")
    .merge(trans, on="product_category_name", how="left")
    .merge(pays_agg, on="order_id", how="left")
    .merge(revs_agg, on="order_id", how="left"))

master["delivery_days"] = (master.order_delivered_customer_date - master.order_purchase_timestamp).dt.days
master["late"] = (master.order_delivered_customer_date > master.order_estimated_delivery_date).astype(int)
master["revenue"] = master.price + master.freight_value
master.to_parquet("olist_master.csv")   # ~112k line-item rows, one clean table