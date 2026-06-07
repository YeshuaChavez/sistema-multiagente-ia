import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = "Scripts/jmp_global_household.xlsx"
df = pd.read_excel(file_path, sheet_name='wat')

ca_countries = ["HND", "SLV", "GTM", "CRI"]
df_ca = df[(df['iso3'].isin(ca_countries)) & (df['year'] == 2020)]

cols_to_show = ['name', 'iso3', 'wat_basal_t', 'wat_lim_t', 'wat_unimp_t', 'wat_ns_t', 'wat_pip_t', 'wat_imp_npip_t']
print(df_ca[cols_to_show].to_string(index=False))
