import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"
df = pd.read_excel(file_path, sheet_name='Estimates', header=None)

print("First 10 rows of Estimates sheet (raw):")
print(df.iloc[:10, :15].to_string())

# Search for rows where row index is greater than 3, and column 1 (Año) is a number
# Let's inspect rows from row 2 onwards
for idx, row in df.iterrows():
    if idx < 2:
        continue
    # If the row has a year in column 1
    val_1 = row[1]
    if isinstance(val_1, (int, float)) and not pd.isna(val_1):
        print(f"Row {idx}: {row[0]}, Year={row[1]}, Env={row[2]}, Pop={row[3]}, Val4={row[4]}, Val5={row[5]}, Val6={row[6]}, Val7={row[7]}, Val8={row[8]}, Val9={row[9]}")
