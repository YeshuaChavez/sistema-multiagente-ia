import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"

try:
    df_sum = pd.read_excel(file_path, sheet_name='Data Summary', header=None)
    print("--- Data Summary sheet shape:", df_sum.shape)
    
    # We want to find which columns contain the year and the water ladder indicators
    # Let's inspect the first 10 rows to see the column structure
    print("\nHeader rows (first 10 rows):")
    print(df_sum.iloc[:10, :15].to_string())
    
    # Let's search for rows that correspond to survey records
    # Typically, the rows from index 5 or 10 onwards have the survey name in column 0, and the year in column 2.
    print("\nSurvey data rows:")
    survey_count = 0
    for idx, row in df_sum.iterrows():
        if idx < 5:
            continue
        # Check if column 0 (Source) is a string and not NaN
        source = row[0]
        year = row[2]
        if isinstance(source, str) and pd.notna(source) and source.strip() != "":
            survey_count += 1
            if survey_count <= 40:
                print(f"Row {idx:03d} | Source: {source[:25]:<25} | Year: {year} | Non-null values count: {row.notnull().sum()}")
                # Print some values from this row
                # Let's see what is in columns 3, 4, 5, 6, 7 (Improved, Piped, etc.)
                # Typically, JMP Data Summary sheets have:
                # col 3: w_imp (improved), col 4: w_pip (piped), col 5: w_imp_npip (improved non-piped), etc.
                vals = [f"Col{c}:{val}" for c, val in enumerate(row.iloc[3:12], 3) if pd.notna(val)]
                print("  Values:", vals)
    print(f"\nTotal survey records found: {survey_count}")
except Exception as e:
    print("Error reading Data Summary:", e)
