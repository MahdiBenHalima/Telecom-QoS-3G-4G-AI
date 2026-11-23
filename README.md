# Telecom QoS 3G/4G — Machine Learning & Network Quality Assessment

**Telecommunications Engineering · Radio KPI Analysis · Machine Learning · FastAPI**

[![Validate Telecom QoS AI](https://github.com/MahdiBenHalima/Telecom-QoS-3G-4G-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/MahdiBenHalima/Telecom-QoS-3G-4G-AI/actions/workflows/tests.yml)

This repository documents the **AI and network-quality analysis components** of an academic telecommunications engineering project developed to assess 3G/4G radio quality from measured network performance indicators (KPIs).

The original project includes network measurement preprocessing, a 3G supervised-learning experiment, a 4G KPI-based quality scoring system, a FastAPI service, a rule-based diagnostic assistant, and a Flutter mobile application. This **public AI-focused repository** preserves the key prediction logic, provides a source-derived 3G experiment reconstruction, and adds privacy-safe validation tests. Original measurement files, serialized fitted models, and the full mobile application are not distributed here.

> **Scope:** Research/engineering demonstration. Scores are not verified operator service-level assessments. The 3G prediction target is *derived from input radio KPIs*, rather than independently measured quality-of-experience labels. The 4G component is a deterministic scoring system—not an ML-trained 4G predictor. The assistant uses predefined rules, not an LLM or RAG.

## Key components

| Component | Public implementation | What it demonstrates |
| --- | --- | --- |
| 3G original feature schema | [`training/train_3g.py`](training/train_3g.py) | Seven named input features from the original training notebook |
| 3G data preprocessing and labeling | [`training/train_3g.py`](training/train_3g.py) | Parsing, normalization, missing-value handling, RSCP/Ec/N0 category construction |
| 3G ML training and evaluation | [`training/train_3g.py`](training/train_3g.py) | Stratified 80/20 split; Logistic Regression, Decision Tree, Random Forest; accuracy and weighted F1 |
| Reproducible notebook | [`notebooks/reproduce_3g_training.ipynb`](notebooks/reproduce_3g_training.ipynb) | Rebuild the experiment locally using authorized data; no bundled operator records |
| 3G prediction interface | [`backend/services/predictor_3g.py`](backend/services/predictor_3g.py) | Construct pandas feature input and call locally loaded Random Forest pipeline |
| 4G QoS scoring | [`backend/services/predictor_4g.py`](backend/services/predictor_4g.py) | RSRP, RSRQ and downlink throughput classification and weighted decision |
| FastAPI inference service | [`backend/main.py`](backend/main.py) | HTTP endpoints for network prediction, health and technical questions |
| Rule-based technical assistant | [`backend/services/chat_service.py`](backend/services/chat_service.py) | Identify degraded KPI and explain a decision with deterministic rules |
| Tests and CI | [`tests/`](tests/), [GitHub Actions](.github/workflows/tests.yml) | Source-level behavior, API input contracts, and training on fabricated records |
| Traceability and limitations | [`TRACEABILITY.md`](TRACEABILITY.md), [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Clearly distinguish historical source, reconstructed methods and synthetic tests |

## Project architecture

```text
      3G/4G radio measurements — private source datasets
                            |
                    KPI preprocessing
                            |
             +--------------+---------------+
             |                              |
             v                              v
  3G: RSCP / Ec/N0 + parameters    4G: RSRP / RSRQ / throughput
             |                              |
  Derived quality labels                     |
             |                              |
  Supervised model experiments       KPI class thresholds
  LR / Decision Tree / RF                      |
             |                      Weighted 30% / 30% / 40%
  Trained 3G Random Forest                     |
             |                              |
             +--------------+---------------+
                            |
                    FastAPI /predict
                            |
                  KPI explanation module
                            |
             Rule-based /chat assistant
                            |
              Original Flutter application
                 (not in public release)
```

The public API can serve 4G quality scoring without any model weights. For 3G inference, it requires an authorized, locally reconstructed/trusted trained pipeline and target-label encoder.

## Network KPIs and assessment logic

### 3G — KPI-derived classification with Random Forest

The original notebook engineers a target class from **RSCP** (signal strength) and **Ec/N0** (signal quality), then trains classifiers to approximate that labeling scheme. Seven inputs are used for the ML model:

| Ordered feature | Meaning |
| --- | --- |
| `band_active` | Active 3G frequency band |
| `channel_active` | Active channel number |
| `scrambling_active` | Active scrambling code |
| `rscp_active` | Active RSCP signal power |
| `ecno_active` | Active Ec/N0 signal quality |
| `rscp_detected` | Detected-cell RSCP |
| `scrambling_detected` | Detected scrambling code |

The quality policy scores RSCP and Ec/N0 separately from 1 to 4, then calculates:

```text
3G global score = 0.50 × RSCP score + 0.50 × Ec/N0 score
```

The original decision bands are:

| 3G score | Engineered decision |
| --- | --- |
| 3.2 or above | Connexion recommandée |
| 2.2 to below 3.2 | Connexion possible mais instable |
| Below 2.2 | Connexion faible / non recommandée |

**Important:** The ML model is trained on these engineered labels. Its holdout accuracy measures reproduction of the chosen scoring policy, **not independent real-world quality or user experience**.

### 4G — Transparent weighted KPI scoring

The 4G module classifies three radio indicators, converts classes into ordinal scores, and produces:

```text
4G global score =
    0.30 × RSRP class score
  + 0.30 × RSRQ class score
  + 0.40 × downlink throughput class score
```

| KPI | Weight | Role |
| --- | ---: | --- |
| RSRP | 30% | Received reference-signal power |
| RSRQ | 30% | Reference-signal quality |
| Download throughput | 40% | Downlink data rate |

4G decision thresholds:

| 4G score | Decision |
| --- | --- |
| 3.5 or above | Connexion recommandée |
| 2.3 to below 3.5 | Connexion possible mais instable |
| Below 2.3 | Connexion faible / non recommandée |

For the demonstration input **RSRP = −86, RSRQ = −9, throughput = 22**, the implemented scoring rules return **3.0**, resulting in *Connexion possible mais instable*. This is a **calculated example**, not an operator measurement or statistical performance claim.

## Source experiment vs public reconstruction

The original private notebook `ai/sample.ipynb` includes preprocessing, KPI scoring, labeling, a stratified holdout experiment and model export. The public [`training/train_3g.py`](training/train_3g.py) reconstructs the core training flow:

1. Read original-style semicolon-delimited `DT2.csv` data from an authorized local path.
2. Normalize raw column names and keep the first value of comma-separated fields.
3. Convert numeric parameters; remove records missing valid time, RSCP or Ec/N0.
4. Build the three 3G decision categories from fixed RSCP/Ec/N0 score thresholds.
5. Encode the target using `LabelEncoder`; split 80/20 with stratification and seed 42.
6. Use `ColumnTransformer` and `Pipeline` for numeric imputation/scaling and categorical one-hot encoding.
7. Compare Logistic Regression, Decision Tree and Random Forest on the held-out split.
8. Fit a final Random Forest (**200 trees**, balanced class weighting, seed 42) on all filtered samples and save model artifacts.

The original notebook optionally includes **XGBoost when installed**. It is intentionally not asserted that the portable public baseline fully reproduces all historical experiment runs, all versions of preprocessing, or the actual private model's numeric scores. The public model reconstruction is designed to make the **software workflow reviewable and repeatable**, given permission to use the source data.

For the provenance mapping and exact scope, see [TRACEABILITY.md](TRACEABILITY.md). For methodological details see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Run the public API locally

Python **3.11** is used in GitHub Actions.

```bash
git clone https://github.com/MahdiBenHalima/Telecom-QoS-3G-4G-AI.git
cd Telecom-QoS-3G-4G-AI
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and start the API:

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive FastAPI interface.

### Example: 4G prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"network_type":"4G","rsrp":-86,"rsrq":-9,"throughput_dl":22}'
```

Expected decision and example global score:

```json
{
  "network_type": "4G",
  "classes": {
    "rsrp_class": "Bonne",
    "rsrq_class": "Acceptable",
    "throughput_class": "Assez bien"
  },
  "scores": {
    "rsrp_score": 3,
    "rsrq_score": 3,
    "throughput_score": 3,
    "global_score": 3.0
  },
  "decision_finale": "Connexion possible mais instable"
}
```

The actual API response includes additional fields such as `input`, `zone`, `band` and `explication`.

### Example: rule-based diagnostic assistant

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"network_type":"4G","question":"Quel KPI est critique ?","decision_finale":"Connexion faible / non recommandée","classes":{"rsrp_class":"Mauvaise","rsrq_class":"Acceptable","throughput_class":"Très faible"}}'
```

This module uses deterministic logic to identify a weak KPI. It is **not** a large language model, multi-agent pipeline, or retrieval-augmented system.

## Reproduce the 3G training experiment

The original 3G measurement file is **not included** in the public repository. With a legitimate, locally accessible `DT2.csv` dataset in the expected original column format:

```bash
python -m training.train_3g --data "/path/to/authorized/DT2.csv" --output backend/models
```

This locally creates:

```text
backend/models/
├── rf_3g_pipeline.joblib
├── label_encoder_3g.joblib
└── evaluation_3g.json
```

These outputs are ignored by Git to help prevent accidental release. Restart the API after fitting so it can load the trusted artifacts at startup. The `/predict` 3G route returns **HTTP 503** if the 3G model is not available.

The [Jupyter notebook](notebooks/reproduce_3g_training.ipynb) provides the same local entry point. The repository does not distribute precomputed private-data metrics.

**Security:** `joblib`/pickle deserialization can execute code; load only files you created yourself or otherwise trust.

## Tests and validation

```bash
python -m pytest -q
```

| Test scope | What is checked |
| --- | --- |
| 4G deterministic scoring | KPI class maps, weighted score and connection decision on fabricated inputs |
| 3G training logic | Derived-label boundaries, private-source-format parsing and synthetic training/export |
| HTTP contract | FastAPI health, 4G /predict, missing-field rejection and /chat behavior |
| Workflow | GitHub Actions installs dependencies and runs pytest on commits and pull requests |

The test dataset is **fabricated within the tests**; it contains no samples copied from the original measurement datasets. The GitHub Actions badge reflects the workflow state as reported by GitHub. CI passing validates these engineering checks—not real-operator generalization or the entire private Flutter app.

**Not validated by the public suite:** real-world QoE, independent ground-truth labels, calibration, cross-cell transfer, chronological robustness, deployed API security, mobile end-to-end integration and operator SLA compliance.

## Repositories: public AI portfolio vs full project

| Public AI repository | Original full project (private) |
| --- | --- |
| 3G training reconstruction and reproducible notebook | Source measurement files and original research notebook |
| True 3G inference input logic, model loading interface | Historical serialized model artifacts and original application configuration |
| 4G KPI scoring engine | Original 4G measurements and full workflows |
| FastAPI inference and rule-based chat | Flutter application, local project configuration and integration |
| Source traceability and fabricated-data tests | Data requiring provenance and sharing review |

These projects are intentionally separated to make AI engineering reviewable without distributing the private measurements or application.

## Data provenance, limitations and responsible use

- No original 3G/4G raw or processed measurement spreadsheets are included.
- No private original fitted model weights or mobile application code are distributed.
- No confidential credentials, external service tokens or logs should be committed.
- The 3G labels come from RSCP and Ec/N0 thresholds. **Holdout classification metrics are not independent network-quality measurements.**
- A random split of related network measurements may overestimate generalization if nearby measurements or cell identifiers occur in train and test; grouped/time-aware validation should be performed for future work.
- The 4G score is a hand-designed weighted rubric, not a learned predictor, calibrated probability or physical model.
- The API is a local engineering demo; public hosting would require authentication, rate limiting and additional security controls.
- No measured test-set accuracy, F1 or live-network effectiveness is claimed in this README.

See [Data and security notes](docs/SECURITY_AND_DATA.md) for further precautions.

## Technology stack

**Machine learning:** Python · pandas · NumPy · scikit-learn · Random Forest · Logistic Regression · Decision Tree

**Software engineering:** FastAPI · Pydantic · joblib · pytest · GitHub Actions · Jupyter Notebook

**Domain:** 3G/4G radio KPIs · RSCP · Ec/N0 · RSRP · RSRQ · throughput · network quality scoring

**Original full-project integration (private):** Flutter mobile application.

---

**Author:** Mahdi Ben Halima · ENET'Com — Telecommunications Engineering

**Focus:** Applied Machine Learning · Radio KPI Analysis · Model Integration · Reproducible ML Research · FastAPI
