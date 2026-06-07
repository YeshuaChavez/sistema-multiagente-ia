import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_global_household.xlsx"
df = pd.read_excel(file_path, sheet_name='wat')

PAISES_ISO = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
    "HTI", "HND", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY", "VEN",
]

# Filter rows by the 20 ISO countries
df_latam = df[df['iso3'].isin(PAISES_ISO)].copy()
print(f"Total rows for the 20 Latam countries: {len(df_latam)}")

# Let's see non-null counts for key columns in our Latam subset
key_cols = [
    'name', 'year', 'iso3',
    'wat_basal_t', 'wat_lim_t', 'wat_unimp_t', 'wat_ns_t',
    'wat_sm_t', 'wat_pip_t', 'wat_imp_npip_t',
    'wat_imp_prem_t', 'wat_imp_av_t', 'wat_imp_qual_t'
]

print("\nNon-null count in Latin America subset:")
print(df_latam[key_cols].notnull().sum())

print("\nExample data for a few countries in 2020:")
sample_countries = ['PER', 'BRA', 'MEX', 'COL', 'HTI']
df_2020 = df_latam[(df_latam['year'] == 2020) & (df_latam['iso3'].isin(sample_countries))]
print(df_2020[key_cols].to_string(index=False))
