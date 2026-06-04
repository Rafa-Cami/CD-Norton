import pandas as pd


def load_data(path="subamostra_10_porcento.csv"):
	return pd.read_csv(path)


def summarize_check(name, mask, df):
	n_total = len(df)
	n = int(mask.sum())
	pct = 100 * n / n_total if n_total > 0 else 0
	status = "OK" if n == 0 else "Atenção"
	return {"sanity_check": name, "n_records": n, "percent_sample": round(pct, 4), "status": status}


def show_examples(df, mask, n=5):
	if mask.sum() == 0:
		print("  Nenhuma ocorrência encontrada.")
	else:
		print(df[mask].head(n).to_string(index=False))


def main():
	df = load_data()
	print(f"Amostra carregada: {len(df)} registros")

	checks = []

	# 1) Pessoas entre 18 e 20 anos trabalhando em tempo full OR casados
	age_mask = df["age"].between(18, 20)
	# definir full-time: employment_status indicando 'full' OU hours >=35
	emp_col = df.get("employment_status")
	wh_col = df.get("working_hours_per_week")
	married_col = df.get("marital_status")

	full_time_mask = pd.Series(False, index=df.index)
	if emp_col is not None:
		full_time_mask = emp_col.fillna("").str.contains("full", case=False, na=False)
	if wh_col is not None:
		full_time_mask = full_time_mask | (wh_col >= 35)

	married_mask = pd.Series(False, index=df.index)
	if married_col is not None:
		married_mask = married_col.fillna("").str.contains("married", case=False, na=False)

	mask1 = age_mask & (full_time_mask | married_mask)
	print("\n1) Pessoas entre 18 e 20 anos trabalhando full-time ou casados:")
	checks.append(summarize_check("18-20 & full-time/married", mask1, df))
	show_examples(df, mask1)

	# 2) Desempregados com carga horária de trabalho > 0
	unemployed_mask = pd.Series(False, index=df.index)
	if emp_col is not None:
		unemployed_mask = emp_col.fillna("").str.contains("unemploy|not working|not employed|jobless", case=False, na=False)
	mask2 = unemployed_mask & (wh_col.notna() & (wh_col > 0)) if wh_col is not None else pd.Series(False, index=df.index)
	print("\n2) Desempregados com carga horária > 0:")
	checks.append(summarize_check("Unemployed with >0 hours", mask2, df))
	show_examples(df, mask2)

	# 3) Pessoas empregadas com carga horária == 0
	employed_mask = pd.Series(False, index=df.index)
	if emp_col is not None:
		employed_mask = emp_col.fillna("").str.contains("employ|work|employee", case=False, na=False)
	mask3 = employed_mask & (wh_col.notna() & (wh_col == 0)) if wh_col is not None else pd.Series(False, index=df.index)
	print("\n3) Pessoas empregadas com carga horária == 0:")
	checks.append(summarize_check("Employed with 0 hours", mask3, df))
	show_examples(df, mask3)

	# 4) Valores fora dos intervalos esperados
	expected_ranges = {
		"age": (0, 120),
		"sleep_hours": (0, 24),
		"physical_activity_hours_per_week": (0, 168),
		"screen_time_hours_per_day": (0, 24),
		"working_hours_per_week": (0, 168),
	}

	for var, (mn, mx) in expected_ranges.items():
		if var not in df.columns:
			print(f"\nVariável '{var}' não presente. Pulando verificação de intervalo.")
			continue
		col = df[var]
		mask_var = col.notna() & ((col < mn) | (col > mx))
		print(f"\n4) Valores fora do intervalo esperado para '{var}' (esperado: {mn} a {mx}):")
		checks.append(summarize_check(f"OutOfRange: {var}", mask_var, df))
		show_examples(df, mask_var)

	# Montar tabela resumo
	summary_df = pd.DataFrame(checks)
	summary_df = summary_df[["sanity_check", "n_records", "percent_sample", "status"]]
	print("\nTabela-resumo:")
	print(summary_df.to_string(index=False))

	out_summary = "sanity_summary.csv"
	summary_df.to_csv(out_summary, index=False)
	print(f"\nResumo salvo em: {out_summary}")


if __name__ == "__main__":
	main()
