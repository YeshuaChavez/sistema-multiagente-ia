import pandas as pd, sys
sys.path.insert(0, 'agentes')
from dotenv import load_dotenv; load_dotenv('.env')
import s3_client as s3

path = r'Base de Datos\datos_crudos\departamentos_coordenadas.csv'
df   = pd.read_csv(path)
antes = len(df)
df = df[df['iso_a0'].str.upper() != 'NIC'].reset_index(drop=True)
df.to_csv(path, index=False)
s3.upload(path, s3.PREFIX_CRUDOS + 'departamentos_coordenadas.csv')
print(f'Coordenadas: {antes} -> {len(df)} filas')
print(f'Paises: {sorted(df["iso_a0"].unique())}')
