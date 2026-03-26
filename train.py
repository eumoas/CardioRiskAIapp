"""
src/train.py — Treinamento, comparação de modelos e registro no MLflow
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - dependência opcional
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - dependência opcional
    lgb = None

from src.preprocess import (
    load_data, encode_target, build_preprocessor,
    get_feature_names, TARGET, TARGET_ORDER,
)

MODEL_REGISTRY = {
    "rf": ("Random Forest", RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
    "logreg": ("Logistic Regression", LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=42,
    )),
    "svm": ("SVM", SVC(
        C=1.0, kernel="rbf", class_weight="balanced",
        probability=True, random_state=42,
    )),
}

if xgb is not None:
    MODEL_REGISTRY["xgb"] = ("XGBoost", xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    ))

if lgb is not None:
    MODEL_REGISTRY["lgbm"] = ("LightGBM", lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        num_leaves=63, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    ))


def run_training(args):
    print("\n🚀 Iniciando Pipeline de Treinamento...\n")

    # ── Dados ─────────────────────────────────────────────────────────────
    df = load_data(args.input)
    df, le = encode_target(df)

    feature_cols = get_feature_names(df)
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"✂️  Treino: {len(X_train):,} | Teste: {len(X_test):,}\n")

    preprocessor = build_preprocessor()

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment)
    print(f"🔬 Experimento MLflow: '{args.experiment}'\n")

    results = []
    best_score = -1
    best_pipeline = None
    best_name = ""

    for model_key in args.models:
        model_name, estimator = MODEL_REGISTRY[model_key]
        print(f"  ⏳ Treinando {model_name}...", end=" ", flush=True)
        t0 = time.time()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        with mlflow.start_run(run_name=model_name):
            # Tags
            mlflow.set_tag("model_type", model_key)
            mlflow.set_tag("dataset", args.input)
            mlflow.set_tag("target", TARGET)
            mlflow.set_tag("target_order", ",".join(TARGET_ORDER))

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())

            # Treino completo no train set
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t0

            # Métricas no test set
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1_macro", f1_macro)
            mlflow.log_metric("test_f1_weighted", f1_weighted)
            mlflow.log_metric("test_roc_auc", auc)
            mlflow.log_metric("train_time_sec", elapsed)

            # Log do modelo
            mlflow.sklearn.log_model(pipeline, "model")

            # Parâmetros
            mlflow.log_param("model", model_key)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("n_features", X.shape[1])

        results.append({
            "Modelo": model_name,
            "CV Acc (mean)": f"{cv_scores.mean():.4f}",
            "CV Acc (std)": f"±{cv_scores.std():.4f}",
            "Test Acc": f"{acc:.4f}",
            "F1 Macro": f"{f1_macro:.4f}",
            "ROC-AUC": f"{auc:.4f}",
            "Tempo (s)": f"{elapsed:.1f}",
        })

        print(f"✅ Acc={acc:.4f} | F1={f1_macro:.4f} | AUC={auc:.4f} | {elapsed:.1f}s")

        if acc > best_score:
            best_score = acc
            best_pipeline = pipeline
            best_name = model_name

    # ── Tabela comparativa ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # ── Salva melhor modelo ───────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": best_pipeline, "label_encoder": le, "target_order": TARGET_ORDER}, f)

    print(f"\n🏆 Melhor modelo: {best_name} (Acc={best_score:.4f})")
    print(f"💾 Salvo em: {model_path}")
    print(f"\n💡 Para ver o MLflow UI: python cli.py mlflow-ui")
    print(f"💡 Para predição: python cli.py predict --model {model_path} --idade 45 --imc 28.5 --sexo Masculino")
