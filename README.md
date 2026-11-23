# Telecom QoS 3G/4G — AI-Assisted Network Quality Assessment

[![Validate Telecom QoS AI](https://github.com/MahdiBenHalima/Telecom-QoS-3G-4G-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/MahdiBenHalima/Telecom-QoS-3G-4G-AI/actions/workflows/tests.yml)

Academic telecommunications engineering project: evaluate radio quality using measured network KPIs, a supervised 3G model and a rule-based 4G scoring engine.

## Highlights

- **3G**: Random Forest pipeline trained to predict KPI-derived quality categories using RSCP, Ec/N0 and cellular parameters (requires original model artifacts).
- **4G**: deterministic KPI scoring using RSRP (30%), RSRQ (30%) and downlink throughput (40%).
- **FastAPI**: /predict and /chat endpoints.
- **Explainability**: interpretable KPI categories and diagnostic descriptions.
- **Assistant**: rule-based technical responses; **not** an LLM, Agentic AI or RAG implementation.

## Architecture

```text
3G / 4G radio KPIs
        |
        +--> 3G: Random Forest model (optional artifact)
        |         |
        +--> 4G: weighted KPI scoring
                  |
             FastAPI /predict
                  |
         explanations + rules-based /chat
```

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/docs**.

4G example:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"network_type":"4G","rsrp":-86,"rsrq":-9,"throughput_dl":22}'
```

The example produces a **global score of 3.0**.

## 3G trained model

The original 3G model is **not bundled with this initial public release**. The API reports model availability at GET / and returns HTTP 503 for 3G predictions when the model is absent. After verifying rights to redistribute the trained artifacts, place trusted `rf_3g_pipeline.joblib` and `label_encoder_3g.joblib` in `backend/models/`. Never deserialize untrusted joblib files.

## Tests

```bash
python -m pytest -q
```

CI is configured under `.github/workflows/tests.yml`.

## Data and methodology

The original engineering project contains 3G and 4G measurement datasets, a training notebook and a Flutter mobile frontend. Raw datasets and the Flutter project are excluded from this recruiter-facing AI-only repository pending a publication review. Quality categories may be derived from KPI thresholds, so a model's predictive accuracy on these labels must not be confused with independent ground-truth network quality.

## Stack

Python · scikit-learn · Random Forest · pandas · FastAPI · pytest · GitHub Actions

## Author

Mahdi Ben Halima — ENET'Com, Telecommunications Engineering.
