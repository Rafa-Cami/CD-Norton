import pandas as pd
import numpy as np


def load(path="subamostra_10_porcento.csv"):
    return pd.read_csv(path)


def numeric_summary(df, vars_list):
    s = df[vars_list].describe().transpose()
    s = s.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    s["missing"] = df[vars_list].isna().sum()
    return s[["count", "missing", "mean", "std", "min", "q1", "median", "q3", "max"]]


def categorical_summary(df, vars_list, top_n=5):
    rows = []
    details = {}
    for v in vars_list:
        series = df[v].astype(object)
        n_missing = series.isna().sum()
        n_unique = series.nunique(dropna=True)
        top = series.value_counts(dropna=True).head(top_n)
        rows.append({"variable": v, "missing": int(n_missing), "n_unique": int(n_unique)})
        details[v] = top
    return pd.DataFrame(rows), details


def save_details(details):
    for var, counts in details.items():
        out = f"cat_counts_{var}.csv"
        counts.rename_axis(var).reset_index(name="count").to_csv(out, index=False)


def main():
    df = load()

    groups = {
        "Estilo_de_vida": [
            "sleep_hours",
            "physical_activity_hours_per_week",
            "screen_time_hours_per_day",
            "working_hours_per_week",
        ],
        "Psicologicas_emocionais": [
            "anxiety_score",
            "depression_score",
            "stress_level",
            "mood_swings_frequency",
            "concentration_difficulty_level",
        ],
        "Socioeconomicas": [
            "employment_status",
            "job_satisfaction_score",
            "financial_stress_level",
            "education_level",
        ],
        "Suporte_social": [
            "social_support_score",
            "marital_status",
        ],
        "Historico_clinico": [
            "panic_attack_history",
            "family_history_mental_illness",
            "previous_mental_health_diagnosis",
            "therapy_history",
            "substance_use",
        ],
    }

    summary_frames = []
    cat_details_all = {}

    for group_name, vars_list in groups.items():
        present = [v for v in vars_list if v in df.columns]
        missing_vars = [v for v in vars_list if v not in df.columns]
        print(f"\nGroup: {group_name}")
        if missing_vars:
            print(f"  Variáveis ausentes (serão ignoradas): {missing_vars}")

        # numeric vs categorical split
        numeric = [v for v in present if pd.api.types.is_numeric_dtype(df[v])]
        categorical = [v for v in present if not pd.api.types.is_numeric_dtype(df[v])]

        if numeric:
            print("  Numeric vars:", numeric)
            nsum = numeric_summary(df, numeric)
            out_csv = f"desc_numeric_{group_name}.csv"
            nsum.to_csv(out_csv)
            print(f"  Saved numeric summary to {out_csv}")
            # Print numeric summary to terminal
            print("\n  Numeric summary (first rows):")
            print(nsum.to_string())
            summary_frames.append((group_name, nsum))

        if categorical:
            print("  Categorical vars:", categorical)
            csum, details = categorical_summary(df, categorical)
            out_csv = f"desc_categorical_{group_name}.csv"
            csum.to_csv(out_csv, index=False)
            print(f"  Saved categorical summary to {out_csv}")
            # Print categorical summary and top counts
            print("\n  Categorical summary:")
            print(csum.to_string(index=False))
            for v, top in details.items():
                print(f"\n    Top counts for {v}:")
                print(top.to_string())
            save_details(details)
            cat_details_all.update(details)

    # combined summary file
    # For convenience, save a combined numeric summary
    all_numeric = [v for g in groups.values() for v in g if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    if all_numeric:
        combined = numeric_summary(df, all_numeric)
        combined.to_csv("desc_numeric_all_groups.csv")
        print("\nSaved combined numeric summary to desc_numeric_all_groups.csv")
        print("\nCombined numeric summary:")
        print(combined.to_string())

    print("\nEstatísticas descritivas geradas. Arquivos CSV salvos no diretório.")


if __name__ == "__main__":
    main()
