import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"

try:
    df_ladders = pd.read_excel(file_path, sheet_name='Ladders', header=None)
    print("--- Ladders sheet rows ---")
    non_null_rows = 0
    for idx, row in df_ladders.iterrows():
        non_null_count = row.notnull().sum()
        if non_null_count > 1:
            non_null_rows += 1
            if non_null_rows <= 40:
                print(f"Row {idx} ({non_null_count} cols): {row.dropna().tolist()[:10]}")
    print(f"Total non-null rows in Ladders: {non_null_rows}")
except Exception as e:
    print("Error reading Ladders:", e)
