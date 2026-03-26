#!/usr/bin/env python3
"""
CLI principal do Pipeline de ML - Risco de Doenças (Saúde Brasil)
Uso: python cli.py [COMANDO] [OPÇÕES]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="ml-saude",
        description="🏥 Pipeline de ML para Predição de Risco de Doenças",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python cli.py eda --input data/dataset_saude_brasil.csv
  python cli.py train --input data/dataset_saude_brasil.csv --experiment meu_experimento
  python cli.py evaluate --model models/best_model.pkl
  python cli.py predict --model models/best_model.pkl --idade 45 --imc 28.5 --sexo Masculino
  python cli.py serve --model models/best_model.pkl --port 8000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # ── EDA ──────────────────────────────────────────────────────────────────
    eda_parser = subparsers.add_parser("eda", help="Análise Exploratória dos Dados")
    eda_parser.add_argument("--input", required=True, help="Caminho para o CSV")
    eda_parser.add_argument(
        "--output", default="reports/eda_report.html", help="Relatório HTML de saída"
    )
    eda_parser.add_argument(
        "--no-report", action="store_true", help="Pula geração do relatório HTML"
    )

    # ── TRAIN ────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Treina e compara modelos")
    train_parser.add_argument("--input", required=True, help="Caminho para o CSV")
    train_parser.add_argument(
        "--experiment", default="risco_saude", help="Nome do experimento MLflow"
    )
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgb", "lgbm", "logreg"],
        choices=["rf", "xgb", "lgbm", "logreg", "svm"],
        help="Modelos a treinar",
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proporção do conjunto de teste"
    )
    train_parser.add_argument(
        "--output-dir", default="models", help="Diretório para salvar modelos"
    )
    train_parser.add_argument(
        "--tune", action="store_true", help="Executa tuning de hiperparâmetros"
    )

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Avalia o modelo salvo")
    eval_parser.add_argument("--model", required=True, help="Caminho para o modelo .pkl")
    eval_parser.add_argument("--input", required=True, help="Dados de teste (CSV)")
    eval_parser.add_argument(
        "--output", default="reports/evaluation.html", help="Relatório de avaliação"
    )

    # ── PREDICT ──────────────────────────────────────────────────────────────
    pred_parser = subparsers.add_parser("predict", help="Predição para um paciente")
    pred_parser.add_argument("--model", required=True, help="Caminho para o modelo .pkl")
    pred_parser.add_argument("--idade", type=float, required=True)
    pred_parser.add_argument("--sexo", required=True, choices=["Masculino", "Feminino"])
    pred_parser.add_argument("--imc", type=float, required=True)
    pred_parser.add_argument("--passos-diarios", type=float, default=8000)
    pred_parser.add_argument("--horas-sono", type=float, default=7.0)
    pred_parser.add_argument("--agua-litros", type=float, default=2.0)
    pred_parser.add_argument("--calorias", type=float, default=2000)
    pred_parser.add_argument("--fumante", choices=["Sim", "Não"], default="Não")
    pred_parser.add_argument(
        "--alcool", choices=["Baixo", "Moderado", "Alto"], default="Baixo"
    )
    pred_parser.add_argument("--horas-trabalho", type=int, default=8)
    pred_parser.add_argument("--freq-cardiaca", type=int, default=72)
    pred_parser.add_argument("--pressao-sistolica", type=int, default=120)
    pred_parser.add_argument("--pressao-diastolica", type=int, default=80)
    pred_parser.add_argument("--colesterol", type=float, default=180)
    pred_parser.add_argument(
        "--historico-familiar", choices=["Sim", "Não"], default="Não"
    )
    pred_parser.add_argument(
        "--json", action="store_true", dest="output_json", help="Saída em JSON"
    )

    # ── SERVE ────────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Sobe API REST com FastAPI")
    serve_parser.add_argument("--model", required=True, help="Caminho para o modelo .pkl")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", default="0.0.0.0")

    # ── MLFLOW UI ────────────────────────────────────────────────────────────
    mlflow_parser = subparsers.add_parser("mlflow-ui", help="Abre o MLflow UI")
    mlflow_parser.add_argument("--port", type=int, default=5000)

    # ── Parse ─────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "eda":
        from src.eda import run_eda
        run_eda(args)

    elif args.command == "train":
        from src.train import run_training
        run_training(args)

    elif args.command == "evaluate":
        from src.evaluate import run_evaluation
        run_evaluation(args)

    elif args.command == "predict":
        from src.predict import run_prediction
        run_prediction(args)

    elif args.command == "serve":
        from src.serve import run_server
        run_server(args)

    elif args.command == "mlflow-ui":
        import subprocess
        subprocess.run(["mlflow", "ui", "--port", str(args.port)])


if __name__ == "__main__":
    main()
