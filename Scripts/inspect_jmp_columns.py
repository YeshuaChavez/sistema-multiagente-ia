import openpyxl
import pandas as pd

file_path = "Scripts/jmp_global_household.xlsx"
print("Reading JMP Excel file...")

# Load only column names and a few rows of the 'wat' sheet to save memory
# First read sheet headers
wb = openpyxl.load_workbook(file_path, read_only=True)
sheet = wb['wat']

# Read the first few rows to see the column names and structure
rows = []
for i, row in enumerate(sheet.iter_rows(values_only=True)):
    rows.append(row)
    if i >= 5: # just get headers and first few rows
        break

# Display the headers
print("Row 0 (potential headers):", rows[0][:30])
print("Row 1 (potential sub-headers):", rows[1][:30])
print("Row 2 (potential sub-headers):", rows[2][:30])

# Let's inspect columns using pandas if possible, reading just the first 10 rows
df = pd.read_excel(file_path, sheet_name='wat', nrows=10)
print("\nPandas columns preview:")
print(df.columns.tolist()[:30])
