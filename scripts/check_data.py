import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_data.csv")
print('Shape:', df.shape)
print('Missing storage_gb:', df['storage_gb'].isnull().sum(), f'({df["storage_gb"].isnull().mean()*100:.1f}%)')
print()
print('Top 20 series:')
print(df['series'].value_counts().head(20))
print()
print('Top 10 brands:')
print(df['brand'].value_counts().head(10))
print()
print('Condition counts:')
print(df['condition'].value_counts())
