from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("mental_health_risk_dataset_oficial.csv", sep=';')

# quick check: confirmar colunas
print("Colunas lidas:", df.columns.tolist())

# Gerar subamostra estratificada de 10%
subamostra, _ = train_test_split(
    df,
    train_size=0.10,
    stratify=df["mental_health_risk"],
    random_state=42
)

# Verificar tamanho das bases
print("Tamanho da base original:", len(df))
print("Tamanho da subamostra:", len(subamostra))

# Distribuição absoluta das classes
print("\nDistribuição na base original:")
print(df["mental_health_risk"].value_counts())

print("\nDistribuição na subamostra:")
print(subamostra["mental_health_risk"].value_counts())

# Distribuição percentual das classes
print("\nPercentuais na base original:")
print(df["mental_health_risk"].value_counts(normalize=True) * 100)

print("\nPercentuais na subamostra:")
print(subamostra["mental_health_risk"].value_counts(normalize=True) * 100)

# Salvar a subamostra
subamostra.to_csv("subamostra_10_porcento.csv", index=False)

print("\nSubamostra salva com sucesso!")