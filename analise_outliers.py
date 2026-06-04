import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Carregar dados
df = pd.read_csv("mental_health_risk_dataset_oficial.csv", sep=';')

# Variáveis quantitativas a analisar
variaveis_quantitativas = ['age', 'sleep_hours', 'working_hours_per_week']

print("="*70)
print("ANÁLISE DE OUTLIERS (DISCREPANTES)")
print("="*70)

# ============================================================================
# 1. ANÁLISE DESCRITIVA DAS VARIÁVEIS
# ============================================================================
print("\n1. ESTATÍSTICAS DESCRITIVAS:\n")
print(df[variaveis_quantitativas].describe())

# ============================================================================
# 2. MÉTODO IQR (INTERVALO INTERQUARTIL)
# ============================================================================
print("\n" + "="*70)
print("2. DETECÇÃO DE OUTLIERS PELO MÉTODO IQR:")
print("="*70)

outliers_info = {}

for var in variaveis_quantitativas:
    print(f"\n--- {var.upper()} ---")
    
    # Calcular quartis
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    
    # Limites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers = (df[var] < limite_inferior) | (df[var] > limite_superior)
    quantidade_outliers = outliers.sum()
    percentual_outliers = (quantidade_outliers / len(df)) * 100
    
    # Armazenar informações
    outliers_info[var] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'quantidade': quantidade_outliers,
        'percentual': percentual_outliers
    }
    
    # Imprimir resultados
    print(f"Q1 (25º percentil): {Q1:.2f}")
    print(f"Q3 (75º percentil): {Q3:.2f}")
    print(f"IQR (Q3 - Q1): {IQR:.2f}")
    print(f"Limite inferior: {limite_inferior:.2f}")
    print(f"Limite superior: {limite_superior:.2f}")
    print(f"Quantidade de outliers: {quantidade_outliers} ({percentual_outliers:.2f}%)")
    
    if quantidade_outliers > 0:
        print(f"Valores identificados como outliers:")
        outliers_valores = df[outliers][var].sort_values()
        print(f"  Mínimo: {outliers_valores.min():.2f}")
        print(f"  Máximo: {outliers_valores.max():.2f}")

# ============================================================================
# 3. GRÁFICOS DE BOXPLOT
# ============================================================================
print("\n" + "="*70)
print("3. GERANDO GRÁFICOS DE BOXPLOT...")
print("="*70)

# Criar subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, var in enumerate(variaveis_quantitativas):
    # Boxplot
    sns.boxplot(y=df[var], ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Boxplot - {var}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(var, fontsize=10)
    
    # Adicionar linhas dos limites IQR
    info = outliers_info[var]
    axes[idx].axhline(info['limite_inferior'], color='red', linestyle='--', 
                      linewidth=1.5, label=f"Limite inf: {info['limite_inferior']:.2f}")
    axes[idx].axhline(info['limite_superior'], color='green', linestyle='--', 
                      linewidth=1.5, label=f"Limite sup: {info['limite_superior']:.2f}")
    axes[idx].legend(fontsize=8)
    
    # Adicionar anotação com quantidade de outliers
    axes[idx].text(0.5, 0.95, f"Outliers: {info['quantidade']} ({info['percentual']:.2f}%)",
                   transform=axes[idx].transAxes, verticalalignment='top',
                   horizontalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=9)

plt.tight_layout()
plt.savefig('analise_outliers_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo como 'analise_outliers_boxplot.png'")
plt.show()

# ============================================================================
# 4. GRÁFICO COM DISTRIBUIÇÃO E OUTLIERS MARCADOS
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, var in enumerate(variaveis_quantitativas):
    # Criar scatter plot
    outliers = (df[var] < outliers_info[var]['limite_inferior']) | \
               (df[var] > outliers_info[var]['limite_superior'])
    
    axes[idx].scatter(np.random.normal(0, 0.04, len(df[~outliers])), 
                      df[~outliers][var], alpha=0.5, s=30, label='Normal', color='blue')
    axes[idx].scatter(np.random.normal(0, 0.04, len(df[outliers])), 
                      df[outliers][var], alpha=0.8, s=50, label='Outliers', color='red', marker='x')
    
    axes[idx].axhline(outliers_info[var]['limite_inferior'], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
    axes[idx].axhline(outliers_info[var]['limite_superior'], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
    
    axes[idx].set_title(f'Distribuição - {var}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(var, fontsize=10)
    axes[idx].set_xlabel('Densidade')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_outliers_distribuicao.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo como 'analise_outliers_distribuicao.png'")
plt.show()

# ============================================================================
# 5. RESUMO EXECUTIVO
# ============================================================================
print("\n" + "="*70)
print("RESUMO EXECUTIVO:")
print("="*70)
total_outliers = sum([outliers_info[var]['quantidade'] for var in variaveis_quantitativas])
print(f"\nTotal de outliers encontrados: {total_outliers}")
print("\nVariável com mais outliers:")
var_max = max(variaveis_quantitativas, key=lambda v: outliers_info[v]['quantidade'])
print(f"  {var_max}: {outliers_info[var_max]['quantidade']} ({outliers_info[var_max]['percentual']:.2f}%)")

print("\n" + "="*70)
