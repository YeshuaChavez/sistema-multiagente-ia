import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_nicaragua_household.xlsx"
df = pd.read_excel(file_path, sheet_name='Estimates', skiprows=2)

print("Columns in Estimates sheet:")
print(df.columns.tolist()[:15])

# Print rows that have a non-null year and where Entorno is 'Total' or 'Nacional' or 'National'
df_filtered = df[df['Año'].notnull()]
print(f"\nFound {len(df_filtered)} rows with non-null year.")

# Let's inspect the unique values of 'Entorno'
print("\nUnique environments:", df_filtered['Entorno'].unique())

# Filter for Entorno == 'Nacional' or 'Total' or 'National' or whatever corresponds to total
# Let's see the first 30 rows of df_filtered where Entorno == 'Total'
df_total = df_filtered[df_filtered['Entorno'] == 'Total']
print(f"\nFound {len(df_total)} rows for Entorno == 'Total'.")
if not df_total.empty:
    cols_to_show = ['País', 'Año', 'Entorno', 'Población (1000s)', 'Agua mejorada', 'Al menos básico (mejorado en 30 minutos)', 'Limitado (mejorado\n > 30 minutos)', 'Agua no mejorada', 'Agua superficial', 'Entubada']
    # Let's clean the columns since they might have slightly different names
    existing_cols = [c for c in cols_to_show if c in df_total.columns]
    print(df_total[existing_cols].head(30).to_string())
else:
    print("No rows with Entorno == 'Total'. Let's show first 10 rows of df_filtered:")
    print(df_filtered.iloc[:10, :10].to_string())
