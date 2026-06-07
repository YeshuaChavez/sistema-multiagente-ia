import pandas as pd
import openpyxl
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"
wb = openpyxl.load_workbook(file_path, read_only=True)

for sheet_name in wb.sheetnames:
    print(f"\n--- Checking sheet: {sheet_name} ---")
    try:
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=30, header=None)
        # Check if there are years and numbers
        # Let's search if the sheet has the word 'Nicaragua' and numeric values in any row
        has_numbers = False
        for idx, row in df.iterrows():
            row_str = row.astype(str).str.cat(sep=" ")
            if "nicaragua" in row_str.lower():
                # check if there are numbers like 2000, 2005, etc.
                numeric_count = row.apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).sum()
                if numeric_count > 2:
                    print(f"Row {idx} contains 'Nicaragua' and {numeric_count} numeric values:")
                    print(f"  {row.dropna().tolist()[:10]}")
                    has_numbers = True
        
        # Also print a small preview of the first 5 rows and 5 columns
        print("Preview:")
        print(df.iloc[:5, :5].to_string())
    except Exception as e:
        print(f"Error checking sheet {sheet_name}: {e}")
