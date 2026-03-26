"""
src/evaluate.py — Avaliação do modelo salvo com relatório HTML
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score,
)

from src.preprocess import load_data, encode_target, get_feature_names, TARGET, TARGET_ORDER


def run_evaluation(args):
    print("\n📊 Avaliando modelo...\n")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]
    target_order = bundle["target_order"]

    df = load_data(args.input)
    df, le = encode_target(df)
    feature_cols = get_feature_names(df)
    X = df[feature_cols]
    y = df[TARGET]

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Macro:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print()
    print("  Relatório por Classe:")
    print(classification_report(y, y_pred, target_names=target_order))

    # Confusion matrix plot
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_order, yticklabels=target_order, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  📊 Matriz de confusão salva em reports/confusion_matrix.png")
