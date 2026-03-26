"""
Streamlit app para predição de risco cardíaco com suporte a modelos locais e MLflow.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import TARGET, TARGET_ORDER, build_preprocessor, encode_target, load_data

DISCLAIMER = (
    "Este conteúdo é destinado apenas para fins educacionais. "
    "Os dados exibidos são ilustrativos e não substituem avaliação médica."
)

DEFAULT_MODEL_PATH = Path("models/best_model.pkl")
DEFAULT_DATASET_PATH = Path("dataset_saude_brasil.csv")
DEFAULT_TRACKING_URI = f"file://{Path('mlruns').resolve()}"
TARGET_ORDER_FALLBACK = ["Baixo", "Moderado", "Alto", "Muito Alto"]


def build_patient_frame(
    idade: float,
    sexo: str,
    imc: float,
    passos_diarios: float,
    horas_sono: float,
    agua_litros: float,
    calorias: float,
    fumante: str,
    alcool: str,
    horas_trabalho: int,
    freq_cardiaca: int,
    pressao_sistolica: int,
    pressao_diastolica: int,
    colesterol: float,
    historico_familiar: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Idade": idade,
                "Sexo": sexo,
                "IMC": imc,
                "Passos_Diarios": passos_diarios,
                "Horas_Sono": horas_sono,
                "Agua_Litros": agua_litros,
                "Calorias": calorias,
                "Fumante": fumante,
                "Alcool": alcool,
                "Horas_Trabalho": horas_trabalho,
                "Frequencia_Cardiaca_Repouso": freq_cardiaca,
                "Pressao_Sistolica": pressao_sistolica,
                "Pressao_Diastolica": pressao_diastolica,
                "Colesterol": colesterol,
                "Historico_Familiar": historico_familiar,
            }
        ]
    )


@st.cache_resource(show_spinner=False)
def load_local_bundle(model_path: str):
    with open(model_path, "rb") as file:
        bundle = pickle.load(file)
    return bundle


@st.cache_data(show_spinner=False)
def list_mlflow_runs(experiment_name: str, tracking_uri: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
    )
    if runs.empty:
        return runs

    preferred_columns = [
        "run_id",
        "tags.mlflow.runName",
        "metrics.test_accuracy",
        "metrics.test_f1_macro",
        "metrics.test_roc_auc",
        "params.model",
        "start_time",
    ]
    available_columns = [column for column in preferred_columns if column in runs.columns]
    result = runs[available_columns].copy()
    if "tags.mlflow.runName" not in result.columns:
        result["tags.mlflow.runName"] = "run_sem_nome"
    return result


@st.cache_resource(show_spinner=False)
def load_mlflow_model(run_id: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    run = mlflow.get_run(run_id)
    target_order = run.data.tags.get("target_order", ",".join(TARGET_ORDER_FALLBACK)).split(",")
    return model, target_order, run


@st.cache_resource(show_spinner=False)
def train_demo_model(dataset_path: str):
    df = load_data(dataset_path)
    df, _ = encode_target(df)
    feature_cols = [column for column in df.columns if column not in ["ID", TARGET]]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, TARGET_ORDER


def render_prediction(model, target_order: list[str], patient_df: pd.DataFrame):
    pred_idx = int(model.predict(patient_df)[0])
    pred_label = target_order[pred_idx]
    probabilities = model.predict_proba(patient_df)[0]

    st.success(f"Risco predito: {pred_label}")

    prob_df = pd.DataFrame(
        {
            "Risco": target_order,
            "Probabilidade": probabilities,
        }
    ).sort_values("Probabilidade", ascending=False)

    st.dataframe(
        prob_df.assign(
            Probabilidade=prob_df["Probabilidade"].map(lambda value: f"{value * 100:.2f}%")
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(prob_df.set_index("Risco"))


def main():
    has_default_model = DEFAULT_MODEL_PATH.exists()
    st.set_page_config(
        page_title="CardioRisk",
        page_icon="❤️",
        layout="wide",
    )

    st.title("CardioRisk")
    st.caption("Predição de risco cardíaco com Streamlit + MLflow")
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Fonte do modelo")
        source_options = ["Arquivo local", "MLflow", "Modelo demo"] if has_default_model else ["Modelo demo", "Arquivo local", "MLflow"]
        source = st.radio(
            "Escolha como carregar o modelo",
            options=source_options,
            index=0,
        )

        experiment_name = "risco_saude"
        tracking_uri = DEFAULT_TRACKING_URI
        selected_run_id = None
        model = None
        target_order = TARGET_ORDER_FALLBACK
        model_summary = {}

        if source == "MLflow":
            tracking_uri = st.text_input("MLflow Tracking URI", value=tracking_uri)
            experiment_name = st.text_input("Experimento", value=experiment_name)
            runs_df = list_mlflow_runs(experiment_name, tracking_uri)

            if runs_df.empty:
                st.info("Nenhuma run encontrada nesse experimento. Treine o modelo antes de usar essa opção.")
            else:
                options = runs_df["run_id"].tolist()
                selected_run_id = st.selectbox(
                    "Run",
                    options=options,
                    format_func=lambda run_id: (
                        f"{run_id[:8]} - "
                        f"{runs_df.loc[runs_df['run_id'] == run_id, 'tags.mlflow.runName'].iloc[0]}"
                    ),
                )
                model, target_order, run = load_mlflow_model(selected_run_id, tracking_uri)
                model_summary = {
                    "Run ID": selected_run_id,
                    "Nome": run.data.tags.get("mlflow.runName", "-"),
                    "Accuracy": run.data.metrics.get("test_accuracy", 0.0),
                    "F1 Macro": run.data.metrics.get("test_f1_macro", 0.0),
                    "ROC-AUC": run.data.metrics.get("test_roc_auc", 0.0),
                }

        if source == "Arquivo local":
            model_path = st.text_input("Caminho do modelo", value=str(DEFAULT_MODEL_PATH))
            if Path(model_path).exists():
                bundle = load_local_bundle(model_path)
                model = bundle["pipeline"]
                target_order = bundle.get("target_order", TARGET_ORDER_FALLBACK)
                model_summary = {
                    "Arquivo": model_path,
                    "Tipo": type(model.named_steps["model"]).__name__,
                }
            else:
                st.error("Modelo local não encontrado.")
                st.caption("Envie `models/best_model.pkl` para o repositório ou use `Modelo demo`.")

        if source == "Modelo demo":
            dataset_path = st.text_input("Dataset para treino rápido", value=str(DEFAULT_DATASET_PATH))
            if Path(dataset_path).exists():
                model, target_order = train_demo_model(dataset_path)
                model_summary = {
                    "Fonte": "Treino rápido no deploy",
                    "Dataset": dataset_path,
                    "Algoritmo": "RandomForestClassifier",
                }
            else:
                st.error("Dataset de treino não encontrado para montar o modelo demo.")

        if model_summary:
            st.divider()
            st.subheader("Resumo do modelo")
            for key, value in model_summary.items():
                if isinstance(value, float):
                    st.write(f"**{key}:** {value:.4f}")
                else:
                    st.write(f"**{key}:** {value}")

    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.slider("Idade", 1, 100, 45)
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
        imc = st.slider("IMC", 10.0, 60.0, 25.0, 0.1)
        historico_familiar = st.selectbox("Histórico familiar", ["Não", "Sim"])
        fumante = st.selectbox("Fumante", ["Não", "Sim"])

    with col2:
        passos_diarios = st.slider("Passos diários", 0, 30000, 8000, 500)
        horas_sono = st.slider("Horas de sono", 3.0, 12.0, 7.0, 0.5)
        agua_litros = st.slider("Água por dia (L)", 0.0, 5.0, 2.0, 0.1)
        calorias = st.slider("Calorias", 1000, 5000, 2000, 100)
        alcool = st.selectbox("Consumo de álcool", ["Baixo", "Moderado", "Alto"])

    with col3:
        horas_trabalho = st.slider("Horas de trabalho", 0, 16, 8)
        freq_cardiaca = st.slider("Freq. cardíaca em repouso", 40, 150, 72)
        pressao_sistolica = st.slider("Pressão sistólica", 80, 220, 120)
        pressao_diastolica = st.slider("Pressão diastólica", 50, 140, 80)
        colesterol = st.slider("Colesterol", 100, 400, 180, 5)

    patient_df = build_patient_frame(
        idade=idade,
        sexo=sexo,
        imc=imc,
        passos_diarios=passos_diarios,
        horas_sono=horas_sono,
        agua_litros=agua_litros,
        calorias=calorias,
        fumante=fumante,
        alcool=alcool,
        horas_trabalho=horas_trabalho,
        freq_cardiaca=freq_cardiaca,
        pressao_sistolica=pressao_sistolica,
        pressao_diastolica=pressao_diastolica,
        colesterol=colesterol,
        historico_familiar=historico_familiar,
    )

    if st.button("Calcular risco", type="primary", use_container_width=True):
        if model is None:
            st.error("Carregue um modelo válido antes de realizar a predição.")
        else:
            render_prediction(model, target_order, patient_df)


if __name__ == "__main__":
    main()
