"""
src/eda.py — Análise Exploratória dos Dados
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.preprocess import (
    load_data, TARGET, NUMERIC_COLS, BINARY_COLS, ORDINAL_COLS, TARGET_ORDER
)


def run_eda(args):
    print("\n🔍 Iniciando Análise Exploratória dos Dados...\n")

    df = load_data(args.input)

    # ── Resumo geral ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"📊 Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"🎯 Target: {TARGET}")
    print()

    print("📋 Distribuição do Target:")
    vc = df[TARGET].value_counts()
    for label in TARGET_ORDER:
        count = vc.get(label, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<12} {count:>6,}  ({pct:5.1f}%)  {bar}")

    print()
    print("🔍 Valores ausentes:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  Nenhum valor ausente! ✅")
    else:
        for col, n in missing.items():
            print(f"  {col:<35} {n:>5} ({n/len(df)*100:.1f}%)")

    print()
    print("📈 Estatísticas das variáveis numéricas:")
    print(df[NUMERIC_COLS].describe().round(2).to_string())

    # ── Plots ──────────────────────────────────────────────────────────────
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    _plot_target_distribution(df, output_dir)
    _plot_numeric_distributions(df, output_dir)
    _plot_correlation_matrix(df, output_dir)
    _plot_categorical_vs_target(df, output_dir)

    if not args.no_report:
        _generate_html_report(df, args.output)

    print("\n✅ EDA concluída! Plots salvos em reports/")


def _plot_target_distribution(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET].value_counts().reindex(TARGET_ORDER)
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
    bars = ax.bar(TARGET_ORDER, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Distribuição do Risco de Doença", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risco")
    ax.set_ylabel("Quantidade de Pacientes")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "01_target_distribution.png", dpi=150)
    plt.close()
    print("  📊 Plot 1/4 salvo: target_distribution")


def _plot_numeric_distributions(df, output_dir):
    n = len(NUMERIC_COLS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    palette = {"Baixo": "#2ecc71", "Moderado": "#f39c12", "Alto": "#e74c3c", "Muito Alto": "#8e44ad"}
    for i, col in enumerate(NUMERIC_COLS):
        for risk in TARGET_ORDER:
            subset = df[df[TARGET] == risk][col].dropna()
            axes[i].hist(subset, alpha=0.5, label=risk, bins=30, color=palette[risk])
        axes[i].set_title(col, fontweight="bold")
        axes[i].spines[["top", "right"]].set_visible(False)
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribuição de Variáveis Numéricas por Risco", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "02_numeric_distributions.png", dpi=150)
    plt.close()
    print("  📊 Plot 2/4 salvo: numeric_distributions")


def _plot_correlation_matrix(df, output_dir):
    numeric_df = df[NUMERIC_COLS].copy()
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Matriz de Correlação", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "03_correlation_matrix.png", dpi=150)
    plt.close()
    print("  📊 Plot 3/4 salvo: correlation_matrix")


def _plot_categorical_vs_target(df, output_dir):
    cat_cols = BINARY_COLS + ORDINAL_COLS
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(16, 5))
    for ax, col in zip(axes, cat_cols):
        ct = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
        ct = ct.reindex(columns=TARGET_ORDER)
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
        ct.plot(kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% dentro do grupo")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Risco", fontsize=7, title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Variáveis Categóricas vs Risco de Doença (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "04_categorical_vs_target.png", dpi=150)
    plt.close()
    print("  📊 Plot 4/4 salvo: categorical_vs_target")


def _generate_html_report(df, output_path):
    """Gera relatório HTML simples (sem ydata-profiling para manter deps leves)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = df[NUMERIC_COLS].describe().round(2)
    missing = df.isnull().sum()
    target_dist = df[TARGET].value_counts().reindex(TARGET_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>EDA — Saúde Brasil</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem; background: #f8f9fa; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: .5rem; }}
    h2 {{ color: #34495e; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    th {{ background: #3498db; color: white; padding: .6rem 1rem; text-align: left; font-size: .85rem; }}
    td {{ padding: .5rem 1rem; border-bottom: 1px solid #ecf0f1; font-size: .85rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 12px; font-size: .8rem; font-weight: bold; color: white; }}
    .low {{ background: #2ecc71; }} .mod {{ background: #f39c12; }} .high {{ background: #e74c3c; }} .vhigh {{ background: #8e44ad; }}
    .disclaimer {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 4px; margin: 1rem 0; }}
    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: .5rem 0; }}
  </style>
</head>
<body>
  <h1>📊 Análise Exploratória — Dataset Saúde Brasil</h1>
  <div class="disclaimer">⚠️ <strong>Este conteúdo é destinado apenas para fins educacionais.</strong> Os dados exibidos são ilustrativos e podem não corresponder a situações reais.</div>

  <h2>📋 Visão Geral</h2>
  <table>
    <tr><th>Métrica</th><th>Valor</th></tr>
    <tr><td>Total de registros</td><td>{len(df):,}</td></tr>
    <tr><td>Número de features</td><td>{df.shape[1] - 1}</td></tr>
    <tr><td>Valores ausentes (total)</td><td>{missing.sum():,}</td></tr>
  </table>

  <h2>🎯 Distribuição do Target</h2>
  <table>
    <tr><th>Risco</th><th>Contagem</th><th>%</th></tr>
    {"".join(f'<tr><td>{r}</td><td>{target_dist[r]:,}</td><td>{target_dist[r]/len(df)*100:.1f}%</td></tr>' for r in TARGET_ORDER)}
  </table>

  <h2>📈 Estatísticas Descritivas</h2>
  {stats.to_html(classes="", border=0)}

  <h2>🖼️ Visualizações</h2>
  <img src="01_target_distribution.png" alt="Distribuição do Target">
  <img src="02_numeric_distributions.png" alt="Distribuições Numéricas">
  <img src="03_correlation_matrix.png" alt="Correlação">
  <img src="04_categorical_vs_target.png" alt="Categóricas vs Target">
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 Relatório HTML salvo em: {output_path}")
