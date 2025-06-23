import pandas as pd

df = pd.read_excel("data_cleaned.xlsx")
print(df.columns)
print("Nombre de lignes :", len(df))
print("Types de données :")
print(df.dtypes)