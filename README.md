# 🏥 CardioRisk — Predição de Risco de Doenças com MLflow e Streamlit

> **Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.**

Projeto de Machine Learning para predição de risco de doenças cardíacas com base em dados de saúde de pacientes brasileiros. Desenvolvido como atividade final da UC Aprendizado de Máquina 2026/1, seguindo a metodologia CRISP-DM.

---

## 📂 Estrutura do Projeto

```
Cardiorisk/
├── app.py                  # App Streamlit
├── cli.py                  # CLI principal
├── preprocess.py           # Pré-processamento e transformação
├── eda.py                  # Análise exploratória
├── train.py                # Treinamento + MLflow
├── evaluate.py             # Avaliação do modelo
├── predict.py              # Predição via CLI
├── serve.py                # API REST (FastAPI)
├── src/                    # Compatibilidade para imports src.*
├── models/                 # Modelos salvos (.pkl)
├── reports/                # Plots e relatórios HTML
├── mlruns/                 # Experimentos MLflow
├── dataset_saude_brasil.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Como Rodar

### 1. Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Use o dataset do projeto

```bash
O arquivo `dataset_saude_brasil.csv` já está na raiz do projeto.
```

---

## 🖥️ CLI — Comandos

### 🔍 Análise Exploratória

```bash
python3 cli.py eda --input dataset_saude_brasil.csv
```
Gera plots e relatório HTML em `reports/`.

---

### 🤖 Treinar Modelos

```bash
# Treinar todos os modelos padrão
python3 cli.py train --input dataset_saude_brasil.csv

# Escolher modelos específicos
python3 cli.py train --input dataset_saude_brasil.csv --models rf xgb lgbm

# Nome do experimento no MLflow
python3 cli.py train --input dataset_saude_brasil.csv --experiment meu_experimento_v1
```

Modelos disponíveis: `rf` (Random Forest), `xgb` (XGBoost), `lgbm` (LightGBM), `logreg` (Regressão Logística), `svm` (SVM)

---

### 📊 Avaliar Modelo

```bash
python3 cli.py evaluate --model models/best_model.pkl --input dataset_saude_brasil.csv
```

---

### 🔮 Predição para um Paciente

```bash
python3 cli.py predict \
  --model models/best_model.pkl \
  --idade 55 \
  --sexo Masculino \
  --imc 31.2 \
  --passos-diarios 4000 \
  --horas-sono 5.5 \
  --fumante Sim \
  --alcool Alto \
  --pressao-sistolica 145 \
  --colesterol 240 \
  --historico-familiar Sim

# Saída em JSON (útil para integração)
python3 cli.py predict --model models/best_model.pkl --idade 45 --imc 28 --sexo Feminino --json
```

---

### 🌐 API REST (FastAPI)

```bash
python3 cli.py serve --model models/best_model.pkl --port 8000
```

Acesse: http://localhost:8000/docs (Swagger UI interativo)

Exemplo de request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"idade": 45, "sexo": "Masculino", "imc": 28.5}'
```

---

### 📈 MLflow UI

```bash
python3 cli.py mlflow-ui
```

Acesse: http://localhost:5000

---

## 📺 App Streamlit

Depois de treinar ao menos uma vez:

```bash
streamlit run app.py
```

O app permite:

- carregar `models/best_model.pkl`
- ou carregar diretamente uma `run` do `MLflow`
- visualizar métricas do experimento e probabilidades por classe

Para usar a opção de MLflow local, mantenha o diretório `mlruns/` no projeto.

---

## ☁️ Deploy no Streamlit Community Cloud

1. Envie o projeto para um repositório GitHub
2. Garanta que estes arquivos estejam no repositório: `app.py`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml` e `dataset_saude_brasil.csv`
3. Garanta que `requirements.txt` contenha `streamlit`, `mlflow`, `scikit-learn`, `xgboost` e `lightgbm`
4. No Streamlit Cloud, selecione:
   - repositório
   - branch
   - arquivo principal: `app.py`
5. O app funciona de três formas:
   - `Modelo demo`: treina automaticamente um modelo leve com o dataset versionado
   - `Arquivo local`: usa `models/best_model.pkl` se você versionar esse artefato
   - `MLflow`: usa `mlruns/` se você também publicar as runs no repositório
6. Se quiser usar modelos treinados via MLflow no deploy, publique também a pasta `mlruns/` ou deixe `models/best_model.pkl` versionado no repositório

O `Modelo demo` existe para o deploy subir funcional mesmo sem artefatos pré-treinados.

---

## 🎯 Dataset

| Feature | Tipo | Descrição |
|---|---|---|
| Idade | Numérico | Idade em anos |
| Sexo | Categórico | Masculino / Feminino |
| IMC | Numérico | Índice de Massa Corporal |
| Passos_Diarios | Numérico | Média de passos por dia |
| Horas_Sono | Numérico | Horas de sono por noite |
| Agua_Litros | Numérico | Consumo diário de água |
| Fumante | Binário | Sim / Não |
| Alcool | Ordinal | Baixo / Moderado / Alto |
| Pressao_Sistolica | Numérico | Pressão sistólica (mmHg) |
| Colesterol | Numérico | Colesterol total (mg/dL) |
| Historico_Familiar | Binário | Histórico familiar de doenças |
| **Risco_Doenca** | **Target** | **Baixo / Moderado / Alto / Muito Alto** |

---

## 📊 Resultados (exemplo)

| Modelo | CV Acc | Test Acc | F1 Macro | ROC-AUC |
|---|---|---|---|---|
| Random Forest | ~0.85 | ~0.84 | ~0.82 | ~0.96 |
| XGBoost | ~0.86 | ~0.85 | ~0.83 | ~0.97 |
| LightGBM | ~0.86 | ~0.85 | ~0.83 | ~0.97 |
| Logistic Regression | ~0.75 | ~0.74 | ~0.72 | ~0.92 |

---

## 🔧 MLOps

- **Rastreamento de experimentos**: MLflow (métricas, parâmetros, modelos)
- **Pipeline scikit-learn**: pré-processamento + modelo em um único objeto serializável
- **Deploy Web**: Streamlit
- **API opcional**: FastAPI + Uvicorn

---

## ⚠️ Disclaimer

> Este projeto é desenvolvido para fins educacionais. Os dados utilizados são ilustrativos e simulados. Não use este sistema para diagnósticos médicos reais.
