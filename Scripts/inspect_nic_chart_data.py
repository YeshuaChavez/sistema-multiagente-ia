import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"

# Let's inspect 'Chart Data' sheet
try:
    df_chart = pd.read_excel(file_path, sheet_name='Chart Data', header=None)
    print("Chart Data sheet rows with data:")
    # Print rows that have values and are not fully NaN
    non_null_rows = 0
    for idx, row in df_chart.iterrows():
        if row.notnull().sum() > 3:
            non_null_rows += 1
            if non_null_rows <= 30:
                print(f"Row {idx}: {row.dropna().tolist()[:8]}")
    print(f"Total non-null rows in Chart Data: {non_null_rows}")
except Exception as e:
    print("Error reading Chart Data:", e)

# Let's inspect 'Data Summary' sheet
try:
    df_summary = pd.read_excel(file_path, sheet_name='Data Summary', header=None)
    print("\nData Summary sheet rows with data:")
    non_null_rows = 0
    for idx, row in df_summary.iterrows():
        if row.notnull().sum() > 3:
            non_null_rows += 1
            if non_null_rows <= 30:
                print(f"Row {idx}: {row.dropna().tolist()[:8]}")
    print(f"Total non-null rows in Data Summary: {non_null_rows}")
except Exception as e:
    print("Error reading Data Summary:", e)
