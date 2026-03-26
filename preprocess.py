"""
src/preprocess.py — Preparação e transformação dos dados
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


TARGET = "Risco_Doenca"
TARGET_ORDER = ["Baixo", "Moderado", "Alto", "Muito Alto"]

NUMERIC_COLS = [
    "Idade", "IMC", "Passos_Diarios", "Horas_Sono", "Agua_Litros",
    "Calorias", "Horas_Trabalho", "Frequencia_Cardiaca_Repouso",
    "Pressao_Sistolica", "Pressao_Diastolica", "Colesterol",
]

BINARY_COLS = ["Sexo", "Fumante", "Historico_Familiar"]  # 2 categorias
ORDINAL_COLS = ["Alcool"]  # Baixo < Moderado < Alto
ORDINAL_CATEGORIES = [["Baixo", "Moderado", "Alto"]]

DROP_COLS = ["ID"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    for column in NUMERIC_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    print(f"✅ Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    le.classes_ = np.array(TARGET_ORDER)
    df = df.copy()
    df[TARGET] = le.transform(df[TARGET])
    return df, le


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Retorna as colunas de features (sem target e ID)."""
    return [c for c in df.columns if c not in DROP_COLS + [TARGET]]


def build_preprocessor() -> ColumnTransformer:
    """Constrói o ColumnTransformer de pré-processamento."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=ORDINAL_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("bin", binary_pipeline, BINARY_COLS),
            ("ord", ordinal_pipeline, ORDINAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_single_patient(args) -> pd.DataFrame:
    """Converte args do CLI em DataFrame de 1 linha para predição."""
    data = {
        "Idade": [args.idade],
        "Sexo": [args.sexo],
        "IMC": [args.imc],
        "Passos_Diarios": [args.passos_diarios],
        "Horas_Sono": [args.horas_sono],
        "Agua_Litros": [args.agua_litros],
        "Calorias": [args.calorias],
        "Fumante": [args.fumante],
        "Alcool": [args.alcool],
        "Horas_Trabalho": [args.horas_trabalho],
        "Frequencia_Cardiaca_Repouso": [args.freq_cardiaca],
        "Pressao_Sistolica": [args.pressao_sistolica],
        "Pressao_Diastolica": [args.pressao_diastolica],
        "Colesterol": [args.colesterol],
        "Historico_Familiar": [args.historico_familiar],
    }
    return pd.DataFrame(data)
