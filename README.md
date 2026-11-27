# olist-data-analysis-project
# Olist E-Commerce Business Analysis

## Project Overview

This project explores the Brazilian Olist e-commerce dataset to evaluate business performance across orders, revenue, customers, products, payments, delivery operations, and customer reviews.

The objective is to transform raw transactional data into meaningful business insights using Python and Power BI, while identifying patterns that can support better business decisions.

## Business Objectives

The analysis focuses on answering key business questions:

- How much revenue does the business generate?
- How does product revenue change over time?
- Which product categories contribute most to revenue?
- How are customers distributed across Brazilian states?
- Which payment methods contribute the most payment value?
- What is the distribution of order statuses?
- How efficiently are orders delivered?
- What do customer reviews reveal about satisfaction?

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Data cleaning, preprocessing and validation |
| Pandas | Data manipulation and analysis |
| Power BI | Interactive business intelligence dashboard |
| DAX | Business metrics and KPI calculations |
| Git & GitHub | Project documentation and version control |

## Dataset

The project uses the Olist Brazilian E-Commerce Public Dataset, containing nine related datasets covering customers, orders, products, sellers, payments, reviews, order items, geolocation and product category translations.

**Dataset source:** [Olist Brazilian E-Commerce Dataset – Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Data Preparation and Quality Assessment

The original datasets were inspected and cleaned before analysis.

Key preparation activities included:

- Checking missing values and duplicate records.
- Validating unique identifiers and dataset relationships.
- Removing exact duplicate geolocation records.
- Screening invalid geographical coordinates.
- Validating order and logistics timestamps.
- Checking payment values and installment records.
- Reviewing repeated customer reviews and multiple reviews per order.
- Validating product dimensions and handling selected missing values.
- Checking seller state information against geolocation records.
- Preserving questionable records with appropriate flags instead of automatically deleting potentially valid business data.

Original source files were preserved. Cleaned datasets and validation reports were stored separately.

## Data Model

The nine datasets were connected in Power BI using their relevant identifiers.

The model supports analysis across:

- Orders and customers
- Orders and order items
- Order items and products
- Order items and sellers
- Orders and payments
- Orders and reviews
- Products and category translations

The relationships were designed around the different levels of detail in the datasets to reduce incorrect aggregation.

## Key Performance Indicators

The analysis includes the following measures:

| KPI | Description |
|---|---|
| Total Orders | Distinct number of orders |
| Total Product Revenue | Sum of order-item product prices |
| Total Freight | Sum of shipping charges |
| Total Customers | Distinct number of unique customers |
| Total Products Sold | Number of order-item records |
| Total Payments | Sum of recorded payment values |
| Average Order Value | Product revenue divided by distinct orders |
| Average Review Score | Average customer review score |
| Delivered Orders | Number of delivered orders |
| Cancelled Orders | Number of cancelled orders |
| Cancellation Rate | Cancelled orders as a proportion of all orders |
| Average Delivery Days | Average eligible purchase-to-delivery duration |

## Power BI Dashboard

The report explores business performance through the following analysis areas:

- Executive Dashboard
- Revenue and Product Performance
- Customer Analysis
- Payment Analysis
- Delivery Analysis
- Customer Reviews

The report includes KPI cards, charts and business-focused comparisons.

## Project Deliverables

**Final Power BI Report (PDF)**

The PDF version of the report is available in this repository:

`reports/Olist_Ecommerce_Analysis.pdf`

## Business Value

This project demonstrates how raw e-commerce data can be transformed into structured business intelligence.

The analysis provides a foundation for understanding revenue performance, customer distribution, product contribution, payment behaviour, delivery operations and customer satisfaction.

It also demonstrates practical experience in data quality assessment, relational data modelling, KPI development and dashboard reporting.

## Project Status

- [x] Dataset inspection
- [x] Data cleaning and validation
- [x] Power BI data modelling
- [x] KPI development
- [x] Dashboard development
- [x] PDF report preparation
- [x] Final Python code cleanup


Data Analytics Portfolio Project

**Tools:** Python | Pandas | Power BI | DAX | GitHub
