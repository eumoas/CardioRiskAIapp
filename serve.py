"""
src/serve.py — API REST com FastAPI para deploy
"""

import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd


DISCLAIMER = "Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais."


class PatientInput(BaseModel):
    idade: float = Field(..., ge=0, le=120, example=45)
    sexo: str = Field(..., example="Masculino")
    imc: float = Field(..., ge=10, le=70, example=28.5)
    passos_diarios: float = Field(default=8000, ge=0)
    horas_sono: float = Field(default=7.0, ge=0, le=24)
    agua_litros: float = Field(default=2.0, ge=0)
    calorias: float = Field(default=2000, ge=0)
    fumante: str = Field(default="Não", example="Não")
    alcool: str = Field(default="Baixo", example="Baixo")
    horas_trabalho: int = Field(default=8, ge=0, le=24)
    frequencia_cardiaca_repouso: int = Field(default=72, ge=30, le=200)
    pressao_sistolica: int = Field(default=120, ge=60, le=250)
    pressao_diastolica: int = Field(default=80, ge=40, le=150)
    colesterol: float = Field(default=180, ge=50, le=500)
    historico_familiar: str = Field(default="Não", example="Não")


def run_server(args):
    model_path = args.model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]
    target_order = bundle["target_order"]

    app = FastAPI(
        title="🏥 Predição de Risco de Doenças",
        description=f"{DISCLAIMER}\n\nAPI para predição de risco de doenças cardíacas baseada em dados de saúde.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "status": "online",
            "disclaimer": DISCLAIMER,
            "docs": "/docs",
            "predict": "/predict",
        }

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/predict")
    def predict(patient: PatientInput):
        X = pd.DataFrame([{
            "Idade": patient.idade,
            "Sexo": patient.sexo,
            "IMC": patient.imc,
            "Passos_Diarios": patient.passos_diarios,
            "Horas_Sono": patient.horas_sono,
            "Agua_Litros": patient.agua_litros,
            "Calorias": patient.calorias,
            "Fumante": patient.fumante,
            "Alcool": patient.alcool,
            "Horas_Trabalho": patient.horas_trabalho,
            "Frequencia_Cardiaca_Repouso": patient.frequencia_cardiaca_repouso,
            "Pressao_Sistolica": patient.pressao_sistolica,
            "Pressao_Diastolica": patient.pressao_diastolica,
            "Colesterol": patient.colesterol,
            "Historico_Familiar": patient.historico_familiar,
        }])

        pred_idx = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        pred_label = target_order[pred_idx]

        return {
            "risco_predito": pred_label,
            "probabilidades": {target_order[i]: round(float(p), 4) for i, p in enumerate(proba)},
            "disclaimer": DISCLAIMER,
        }

    print(f"\n🌐 Servidor rodando em http://{args.host}:{args.port}")
    print(f"📖 Documentação: http://{args.host}:{args.port}/docs")
    print(f"⚠️  {DISCLAIMER}\n")

    uvicorn.run(app, host=args.host, port=args.port)
