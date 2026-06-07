import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"
df = pd.read_excel(file_path, sheet_name='Estimates', header=None)

print(f"Total rows in Estimates: {len(df)}")

# Let's find rows where the row is not entirely NaN and print their first few values
non_empty_rows = 0
for idx, row in df.iterrows():
    # count non-null values
    non_null_count = row.notnull().sum()
    if non_null_count > 3:
        non_empty_rows += 1
        if non_empty_rows <= 50:
            print(f"Row {idx} (non-null cols {non_null_count}): {row.iloc[:6].tolist()}")

print(f"Total non-empty rows: {non_empty_rows}")
