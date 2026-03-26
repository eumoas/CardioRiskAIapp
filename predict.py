"""
src/predict.py — Predição para um único paciente via CLI
"""

import pickle
import json
import sys


RISK_COLORS = {
    "Baixo": "\033[92m",      # Verde
    "Moderado": "\033[93m",   # Amarelo
    "Alto": "\033[91m",       # Vermelho
    "Muito Alto": "\033[35m", # Magenta
}
RESET = "\033[0m"


def run_prediction(args):
    from src.preprocess import prepare_single_patient

    # Carrega modelo
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]
    le = bundle["label_encoder"]
    target_order = bundle["target_order"]

    # Prepara dados
    X = prepare_single_patient(args)

    # Predição
    pred_idx = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    pred_label = target_order[pred_idx]

    if args.output_json:
        result = {
            "risco_predito": pred_label,
            "probabilidades": {target_order[i]: round(float(p), 4) for i, p in enumerate(proba)},
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Output bonito
    color = RISK_COLORS.get(pred_label, "")
    print()
    print("=" * 50)
    print("  🏥 PREDIÇÃO DE RISCO DE DOENÇA")
    print("=" * 50)
    print(f"  Risco Predito: {color}{pred_label}{RESET}")
    print()
    print("  Probabilidades por classe:")
    for risk, prob in zip(target_order, proba):
        bar = "█" * int(prob * 30)
        c = RISK_COLORS.get(risk, "")
        print(f"    {c}{risk:<12}{RESET}  {prob*100:5.1f}%  {bar}")
    print()
    print("  ⚠️  Este resultado é apenas para fins educacionais.")
    print("      Não substitui avaliação médica profissional.")
    print("=" * 50)
    print()
