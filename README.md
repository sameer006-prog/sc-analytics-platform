# 🏭 Supply Chain Process Analytics Platform

An end-to-end data pipeline and analytics solution designed to monitor, evaluate, and optimize the three core pillars of supply chain operations. This project generates synthetic operational data, processes it through machine learning and statistical models, and outputs business-ready insights for Power BI dashboards.

## 🚀 Core Modules

* **Source-to-Pay (S2P):** Evaluates supplier performance using custom scoring models, tracking defect rates, spend analysis, and on-time delivery metrics.
* **Forecast-to-Produce (F2P):** Analyzes demand forecasting accuracy using MAPE (Mean Absolute Percentage Error) and detects macro trends across product categories.
* **Plan-to-Repair (P2R):** Utilizes Machine Learning to predict equipment failures and prioritize maintenance schedules based on live sensor and historical age data.

## 🛠️ Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Data Processing** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest, Gradient Boosting, Linear Regression) |
| **Data Visualization** | Excel (openpyxl) → Power BI |
| **Version Control** | Git & GitHub |

## 📁 Project Structure

```text
sc_analytics_platform/
├── main.py              # Entry point — runs the full end-to-end pipeline
├── generate_data.py     # Generates synthetic, realistic SC operational data
├── analytics.py         # Houses the 3 analytics modules + Excel export logic
├── data/                # Directory for generated raw CSVs (auto-created)
└── output/              # Directory for processed CSVs + Power BI workbook (auto-created)