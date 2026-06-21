import pandas as pd

# Ler dataset original e a subamostra de 10%
df_original = pd.read_csv("mental_health_risk_dataset_oficial.csv", sep=';')
df_10porcento = pd.read_csv("subamostra_10_porcento.csv")

# Gerar os 90% restantes removendo os índices que estão na subamostra
df_90porcento = df_original.drop(df_10porcento.index)

# Verificar tamanho das bases
print("Tamanho da base original:", len(df_original))
print("Tamanho dos 10%:", len(df_10porcento))
print("Tamanho dos 90%:", len(df_90porcento))
print("Total (10% + 90%):", len(df_10porcento) + len(df_90porcento))

# Verificar distribuição de classes nos 90%
print("\nDistribuição nos 90%:")
print(df_90porcento["mental_health_risk"].value_counts())

print("\nPercentuais nos 90%:")
print(df_90porcento["mental_health_risk"].value_counts(normalize=True) * 100)

# Salvar os 90%
df_90porcento.to_csv("subamostra_90_porcento.csv", index=False)

print("\nArquivo dos 90% salvo com sucesso!")
