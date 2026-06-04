import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler a subamostra (gerada previamente)
df = pd.read_csv("subamostra_10_porcento.csv")

print("Colunas disponíveis:", df.columns.tolist())

vars_hist = [
	"work_stress_level",
	"screen_time_hours_per_day",
	"physical_activity_hours_per_week",
]

sns.set(style="whitegrid")

for var in vars_hist:
	if var not in df.columns:
		print(f"Coluna '{var}' não encontrada. Pulando.")
		continue

	series = df[var].dropna()
	print(f"\nVar: {var} — n={len(series)}, missing={df[var].isna().sum()}")
	print(series.describe())

	plt.figure(figsize=(8, 5))
	sns.histplot(series, kde=True, bins=30)
	plt.title(f"Histograma: {var}")
	plt.xlabel(var)
	plt.ylabel("Frequência")
	plt.tight_layout()

	out_fname = f"hist_{var}.png"
	plt.savefig(out_fname)
	plt.close()
	print(f"Salvo: {out_fname}")

print("\nAnálise de histogramas completa.")
