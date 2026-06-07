import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"

try:
    df_sum = pd.read_excel(file_path, sheet_name='Data Summary', header=None)
    print("--- Checking all rows in Data Summary for year numbers ---")
    
    found_rows = 0
    for idx, row in df_sum.iterrows():
        year_val = row[2]
        # Check if year_val is a number between 1990 and 2025
        if isinstance(year_val, (int, float)) and 1990 <= year_val <= 2025:
            found_rows += 1
            if found_rows <= 30:
                print(f"Row {idx:03d} | Source: {str(row[0])[:25]:<25} | Year: {year_val} | Non-nulls: {row.notnull().sum()}")
                # Print some values: Improved (col 3), Piped (col 4), Surface Water (col 5)
                print(f"  Improved: {row[3]} | Piped: {row[4]} | Surface: {row[5]} | 30min: {row[6]} | Premises: {row[7]}")
    print(f"Total rows found with valid years: {found_rows}")
except Exception as e:
    print("Error reading Data Summary:", e)
