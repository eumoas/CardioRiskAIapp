# CODEX_PROMPT.md — Guia para usar o Codex neste projeto

## Contexto do Projeto

**Dataset:** `dataset_saude_brasil.csv`  
- 20.000 linhas × 17 colunas  
- Target: `Risco_Doenca` (Baixo / Moderado / Alto / Muito Alto) — classificação multiclasse  
- Features: Idade, IMC, Passos_Diarios, Horas_Sono, Fumante, Alcool, Pressao_Sistolica, Colesterol, etc.  
- Valores ausentes em: Idade (~350), IMC (~350), Passos_Diarios (~200), Calorias (~200), Colesterol (~200)

**Stack:** Python 3.11+, scikit-learn, XGBoost, LightGBM, MLflow, FastAPI, pandas

---

## Prompts Eficientes para o Codex por Etapa

### 1️⃣ EDA
```
Gere uma análise exploratória completa para o arquivo src/eda.py.
O dataset tem 20k linhas, target multiclasse Risco_Doenca (4 classes).
Adicione: distribuição do target, boxplots por classe, heatmap de correlação,
análise de outliers com IQR, e gere um relatório HTML em reports/eda_report.html.
Salve todos os plots em reports/.
```

### 2️⃣ Pré-processamento
```
No arquivo src/preprocess.py, crie um ColumnTransformer scikit-learn que:
- Imputa medianas nas colunas numéricas: Idade, IMC, Passos_Diarios, Calorias, Colesterol
- StandardScaler nas numéricas
- OrdinalEncoder para Alcool (Baixo < Moderado < Alto)
- OrdinalEncoder para Sexo, Fumante, Historico_Familiar
- Encode ordinal do target Risco_Doenca (Baixo=0, Moderado=1, Alto=2, Muito Alto=3)
```

### 3️⃣ Treino + MLflow
```
No arquivo src/train.py, implemente:
- Train/test split estratificado (80/20)
- Pipeline(preprocessor + model) para cada modelo
- StratifiedKFold 5-fold para cada modelo
- MLflow tracking: log accuracy, f1_macro, roc_auc, tempo de treino
- Salva o melhor modelo (maior accuracy) em models/best_model.pkl como dict
  {'pipeline': pipeline, 'label_encoder': le, 'target_order': TARGET_ORDER}
Modelos: RandomForest(200 trees), XGBoost, LightGBM, LogisticRegression
```

### 4️⃣ API FastAPI
```
No arquivo src/serve.py, crie uma API FastAPI com:
- POST /predict recebendo JSON com as 15 features do paciente
- Retorna {'risco_predito': str, 'probabilidades': dict, 'disclaimer': str}
- Disclaimer obrigatório: "Este conteúdo é destinado apenas para fins educacionais..."
- GET /health para healthcheck do deploy
- CORS habilitado para permitir frontend
- Documentação Swagger automática em /docs
```

### 5️⃣ Deploy Hugging Face Spaces
```
Crie um arquivo app.py para Hugging Face Spaces com Gradio:
- Interface com sliders/dropdowns para todas as 15 features
- Botão "Calcular Risco"
- Output: risco predito + barras de probabilidade coloridas
- Texto de disclaimer visível: "Este conteúdo é destinado apenas para fins educacionais..."
- Carrega o modelo de models/best_model.pkl
```

---

## Estrutura para o Codex entender o projeto

Quando usar o Codex, passe sempre:
1. O arquivo relevante completo como contexto
2. O erro exato (se for correção)
3. A saída esperada

### Exemplo de prompt bem estruturado:
```
Contexto: arquivo src/train.py (cole o conteúdo)
Tarefa: adicione feature importance plot para o Random Forest após treino.
Salve em reports/feature_importance.png.
Use matplotlib, top 15 features, barras horizontais ordenadas.
```

---

## Fluxo recomendado com Codex CLI

```bash
# Passo 1: EDA
codex "analise o arquivo src/eda.py e adicione análise de outliers com boxplots por classe de risco"

# Passo 2: Treino
codex "no src/train.py, adicione GridSearchCV para o Random Forest com params: n_estimators=[100,200], max_depth=[10,15,None]"

# Passo 3: Deploy
codex "crie app.py com Gradio para deploy no Hugging Face Spaces carregando models/best_model.pkl"

# Passo 4: Bugfix
codex "o erro abaixo ocorre ao rodar python cli.py train: [cole o traceback]"
```
