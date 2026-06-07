import pandas as pd

csv_path = "Scripts/agua_latam_2000_2024.csv"
df = pd.read_csv(csv_path)

# Nicaragua data from SDG 6 Portal
nicaragua_basica = {
    2000: 81.25,
    2001: 81.34,
    2002: 81.43,
    2003: 81.51,
    2004: 81.60,
    2005: 81.69,
    2006: 81.79,
    2007: 81.90,
    2008: 82.01,
    2009: 82.11,
    2010: 82.22,
    2011: 82.33,
    2012: 82.44,
    2013: 82.55,
    2014: 82.67,
    2015: 82.78,
    2016: 82.89,
    2017: 82.97,
    2018: 83.04,
    2019: 83.13,
    2020: 83.22,
    2021: 83.22,
    2022: 83.22,
    2023: 83.22,
    2024: 83.22,
}

nicaragua_segura = {
    2000: 47.85,
    2001: 47.95,
    2002: 48.05,
    2003: 48.15,
    2004: 48.25,
    2005: 48.36,
    2006: 48.48,
    2007: 48.60,
    2008: 49.64,
    2009: 50.66,
    2010: 51.67,
    2011: 52.67,
    2012: 53.65,
    2013: 54.63,
    2014: 55.08,
    2015: 55.17,
    2016: 55.26,
    2017: 55.31,
    2018: 55.37,
    2019: 55.44,
    2020: 55.52,
    2021: 55.52,
    2022: 55.52,
    2023: 55.52,
    2024: 55.52,
}

# Update values for NIC
for year in range(2000, 2025):
    mask = (df["iso_a0"] == "NIC") & (df["ano"] == year)
    df.loc[mask, "agua_basica"] = nicaragua_basica[year]
    df.loc[mask, "agua_segura"] = nicaragua_segura[year]

# Save back to CSV
df.to_csv(csv_path, index=False)
print("Updated Nicaragua water access values in CSV successfully.")

# Print summary for verification
nic_rows = df[df["iso_a0"] == "NIC"]
print(nic_rows)
