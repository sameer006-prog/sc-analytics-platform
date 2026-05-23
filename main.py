"""
main.py
Supply Chain Process Analytics Platform
"""

import os
from generate_data import generate_s2p_data, generate_f2p_data, generate_p2r_data
from analytics import run_s2p_module, run_f2p_module, run_p2r_module, export_excel_workbook, print_impact_summary

DATA_DIR = "data"
OUTPUT_DIR = "output"

# setup directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nInitializing Supply Chain Process Analytics...")
print("Modules: Source-to-Pay | Forecast-to-Produce | Plan-to-Repair\n")

# generate synthetic data
print("Generating synthetic data...")
s2p_raw = generate_s2p_data(1200)
s2p_raw.to_csv(f"{DATA_DIR}/s2p_orders.csv", index=False)
print(f" -> Created {len(s2p_raw)} purchase orders")

f2p_raw = generate_f2p_data()
f2p_raw.to_csv(f"{DATA_DIR}/f2p_demand.csv", index=False)
print(f" -> Created {len(f2p_raw)} weekly demand records")

p2r_raw = generate_p2r_data(3000)
p2r_raw.to_csv(f"{DATA_DIR}/p2r_sensors.csv", index=False)
print(f" -> Created {len(p2r_raw)} sensor readings")

# run the main analytics modules
print("\nRunning analytics modules...")
run_s2p_module()
run_f2p_module()
run_p2r_module()

# export results for dashboarding
print("\nExporting Power BI workbook...")
export_excel_workbook()

# display final summary
print_impact_summary()

print("\nPipeline complete.")
print("Ready for Power BI: Import 'output/SC_Analytics_Platform_PowerBI.xlsx' and connect the sheets.")