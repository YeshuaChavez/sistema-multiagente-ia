import pandas as pd

file_path = "Scripts/jmp_global_household.xlsx"
df = pd.read_excel(file_path, sheet_name='wat')

# Look at rows where name is Nicaragua or iso3 is NIC
# Wait, let's see which column contains the ISO3 code. In the previous logs, we saw 'iso3' is in the columns.
# Let's see the column names of df
cols = df.columns.tolist()
print("All columns:", cols)

# Find Nicaragua rows
nic_rows = df[df['name'].str.lower() == 'nicaragua']
if not nic_rows.empty:
    print(f"\nFound {len(nic_rows)} rows for Nicaragua.")
    # Show columns for year 2020
    row_2020 = nic_rows[nic_rows['year'] == 2020]
    if not row_2020.empty:
        print("\nNicaragua 2020 values:")
        for col in cols:
            val = row_2020[col].values[0]
            print(f"  {col}: {val}")
    else:
        print("\nNo year 2020 for Nicaragua in this sheet.")
else:
    # Let's search by name2 or look for rows with 'NIC' in any column
    print("Nicaragua not found by name, searching for NIC...")
    # Let's search if any column contains 'NIC'
    for col in cols:
        mask = df[col].astype(str) == 'NIC'
        if mask.any():
            print(f"Found NIC in column: {col}")
            print(df[mask][['name', 'year']].head())
            break
