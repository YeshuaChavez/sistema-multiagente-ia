import pandas as pd
import openpyxl
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"
wb = openpyxl.load_workbook(file_path, read_only=True)
print("Sheets in Nicaragua workbook:", wb.sheetnames)

# Let's inspect the sheet 'Estimates' if it exists.
# We know from the logs that it does exist. Let's read the first 20 rows of 'Estimates' to see the layout.
df_est = pd.read_excel(file_path, sheet_name='Estimates', nrows=25)
print("\nEstimates sheet head (first 10 columns):")
print(df_est.iloc[:15, :10].to_string())

# Let's also inspect 'Water Data' sheet
df_wat = pd.read_excel(file_path, sheet_name='Water Data', nrows=25)
print("\nWater Data sheet head (first 10 columns):")
print(df_wat.iloc[:15, :10].to_string())
