"""
generate_data.py
Supply Chain Process Analytics Platform
Generates synthetic but realistic data for all three SC modules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)


# --- 1. SOURCE-TO-PAY: Supplier & Order Data ---

SUPPLIERS = {
    "SUP-001": {"name": "Schmidt Metallteile GmbH",   "risk": "low",    "lead_days": 5},
    "SUP-002": {"name": "Fischer Kunststoff AG",       "risk": "medium", "lead_days": 8},
    "SUP-003": {"name": "Müller Rohre & Armaturen",   "risk": "low",    "lead_days": 6},
    "SUP-004": {"name": "Weber International Ltd.",   "risk": "high",   "lead_days": 12},
    "SUP-005": {"name": "Krause Logistik KG",         "risk": "medium", "lead_days": 7},
    "SUP-006": {"name": "Bauer Präzisionsteile",      "risk": "low",    "lead_days": 4},
    "SUP-007": {"name": "Hoffmann Global Supply",     "risk": "high",   "lead_days": 15},
    "SUP-008": {"name": "Schneider Dichtungen GmbH",  "risk": "medium", "lead_days": 9},
}

PRODUCTS = [
    ("PIPE-001", "Copper Pipe 15mm",      "Pipes"),
    ("PIPE-002", "Steel Pipe 22mm",       "Pipes"),
    ("VALV-001", "Ball Valve 1/2\"",      "Valves"),
    ("VALV-002", "Pressure Relief Valve", "Valves"),
    ("FIXT-001", "Push-Fit Connector",    "Fittings"),
    ("FIXT-002", "Elbow 90deg 15mm",      "Fittings"),
    ("SEAL-001", "EPDM O-Ring Pack",      "Seals"),
    ("SEAL-002", "Press Fitting Seal",    "Seals"),
]

def generate_s2p_data(n=1200):
    rows = []
    sup_ids = list(SUPPLIERS.keys())
    for i in range(n):
        sup_id = np.random.choice(sup_ids)
        sup = SUPPLIERS[sup_id]
        product = PRODUCTS[np.random.randint(len(PRODUCTS))]

        order_date = START_DATE + timedelta(days=np.random.randint(0, 730))
        planned_days = sup["lead_days"] + np.random.randint(-1, 2)

        # Risk profile drives delay probability
        delay_prob = {"low": 0.08, "medium": 0.20, "high": 0.38}[sup["risk"]]
        delayed = np.random.random() < delay_prob
        actual_days = planned_days + (np.random.randint(2, 8) if delayed else 0)

        delivery_date = order_date + timedelta(days=actual_days)
        on_time = actual_days <= planned_days + 1

        qty = np.random.randint(50, 500)
        unit_price = round(np.random.uniform(2.5, 85.0), 2)
        defect_prob = {"low": 0.02, "medium": 0.05, "high": 0.10}[sup["risk"]]
        defect_qty = int(qty * np.random.uniform(0, defect_prob * 2))
        invoice_match = np.random.random() > 0.06   # 6% invoice mismatch rate

        rows.append({
            "order_id":        f"PO-{10000 + i}",
            "supplier_id":     sup_id,
            "supplier_name":   sup["name"],
            "supplier_risk":   sup["risk"],
            "product_id":      product[0],
            "product_name":    product[1],
            "category":        product[2],
            "order_date":      order_date.date(),
            "planned_delivery":order_date + timedelta(days=planned_days),
            "actual_delivery": delivery_date.date(),
            "planned_lead_days":planned_days,
            "actual_lead_days": actual_days,
            "on_time_delivery": on_time,
            "quantity_ordered": qty,
            "defect_quantity":  defect_qty,
            "defect_rate":      round(defect_qty / qty, 4),
            "unit_price_eur":   unit_price,
            "total_value_eur":  round(qty * unit_price, 2),
            "invoice_match":    invoice_match,
        })
    return pd.DataFrame(rows)


# --- 2. FORECAST-TO-PRODUCE: Demand Forecasting ---

def generate_f2p_data():
    rows = []
    date_range = pd.date_range(START_DATE, END_DATE, freq="W-MON")

    for product in PRODUCTS:
        # Base demand with seasonality + trend
        base = np.random.randint(80, 300)
        trend = np.random.uniform(-0.3, 0.8)

        for i, week in enumerate(date_range):
            seasonality = 1 + 0.25 * np.sin(2 * np.pi * (week.month - 1) / 12)
            noise = np.random.normal(0, 0.10)
            actual = max(0, int(base * seasonality * (1 + trend * i / 100) * (1 + noise)))

            # Simulated forecast (slightly off)
            forecast = max(0, int(actual * np.random.uniform(0.85, 1.18)))

            rows.append({
                "week_start":    week.date(),
                "year_week":     week.strftime("%Y-W%U"),
                "product_id":    product[0],
                "product_name":  product[1],
                "category":      product[2],
                "actual_demand": actual,
                "forecast":      forecast,
                "forecast_error":forecast - actual,
                "ape":           abs(forecast - actual) / max(actual, 1),
            })
    return pd.DataFrame(rows)


# --- 3. PLAN-TO-REPAIR: Maintenance Data ---

MACHINES = [
    ("MCH-001", "Pipe Bending Line A",    "Production",  2018),
    ("MCH-002", "Hydraulic Press #1",     "Production",  2015),
    ("MCH-003", "CNC Lathe Station 3",    "Machining",   2019),
    ("MCH-004", "Valve Assembly Robot",   "Assembly",    2020),
    ("MCH-005", "Quality Test Bench",     "QA",          2017),
    ("MCH-006", "Packaging Unit B",       "Logistics",   2016),
    ("MCH-007", "Hydraulic Press #2",     "Production",  2014),
    ("MCH-008", "Welding Station Alpha",  "Production",  2021),
    ("MCH-009", "Cooling Circuit Pump",   "Utilities",   2013),
    ("MCH-010", "Conveyor System Main",   "Logistics",   2016),
]

SENSOR_PARAMS = {
    "pressure_bar":    (120, 15),
    "temperature_c":   (75,  12),
    "vibration_mm_s":  (3.5, 1.2),
    "oil_viscosity":   (46,  5),
    "cycle_count":     None,
}

def generate_p2r_data(n_readings=3000):
    rows = []
    for i in range(n_readings):
        mch_id, mch_name, dept, install_year = MACHINES[i % len(MACHINES)]
        age_years = 2024 - install_year
        read_date = START_DATE + timedelta(days=np.random.randint(0, 730))

        # Older machines degrade more
        degradation = min(age_years / 10, 0.9)

        pressure = np.random.normal(120, 15 + degradation * 10)
        temperature = np.random.normal(75,  12 + degradation * 8)
        vibration = abs(np.random.normal(3.5, 1.2 + degradation * 2))
        oil_visc = np.random.normal(46, 5 + degradation * 4)
        cycles = np.random.randint(5000, 50000) * (1 + int(age_years / 3))

        # Failure logic: probability rises with degradation
        failure_prob = 0.03 + degradation * 0.18
        if vibration > 7.0 or temperature > 95 or pressure < 90:
            failure_prob += 0.15
        failure = np.random.random() < failure_prob

        # Maintenance score 0-100 (higher = more urgent)
        m_score = min(100, int(
            degradation * 40
            + (max(0, vibration - 5) * 8)
            + (max(0, temperature - 85) * 1.2)
            + (max(0, 100 - pressure) * 0.5)
        ))

        rows.append({
            "reading_id":       f"RD-{20000 + i}",
            "machine_id":       mch_id,
            "machine_name":     mch_name,
            "department":       dept,
            "install_year":     install_year,
            "age_years":        age_years,
            "reading_date":     read_date.date(),
            "pressure_bar":     round(pressure, 2),
            "temperature_c":    round(temperature, 2),
            "vibration_mm_s":   round(vibration, 3),
            "oil_viscosity":    round(oil_visc, 2),
            "cycle_count":      cycles,
            "maintenance_score":m_score,
            "failure_occurred": failure,
            "priority":         "HIGH" if m_score >= 70 else ("MEDIUM" if m_score >= 40 else "LOW"),
        })
    return pd.DataFrame(rows)


# --- MAIN: Generate & Save All ---

if __name__ == "__main__":
    print("Generating Source-to-Pay data...")
    s2p = generate_s2p_data(1200)
    s2p.to_csv(f"{OUTPUT_DIR}/s2p_orders.csv", index=False)
    print(f" -> {len(s2p)} purchase orders saved.")

    print("Generating Forecast-to-Produce data...")
    f2p = generate_f2p_data()
    f2p.to_csv(f"{OUTPUT_DIR}/f2p_demand.csv", index=False)
    print(f" -> {len(f2p)} weekly demand records saved.")

    print("Generating Plan-to-Repair data...")
    p2r = generate_p2r_data(3000)
    p2r.to_csv(f"{OUTPUT_DIR}/p2r_sensors.csv", index=False)
    print(f" -> {len(p2r)} sensor readings saved.")

    print("\nAll raw data successfully generated in /data/ directory.")