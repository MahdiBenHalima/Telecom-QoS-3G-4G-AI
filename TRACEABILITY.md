# Source traceability — Telecom QoS 3G/4G AI

This document separates **verified original implementation**, **source-derived reconstruction** and **portfolio-only validation code**. It is not a performance certification.

## Source audit

The source material reviewed for this public portfolio extraction consists of:
- `ai/sample.ipynb` in the original AI project archive;
- `ai/DT1_4G_clean_scored.csv`, `ai/DT2_3G_clean_labeled.csv`, and raw workbook/CSV names (data kept private);
- `ai/rf_3g_pipeline.joblib` and `ai/label_encoder_3g.joblib` (artifacts kept private);
- `backend/main.py`, `backend/core/model_loader.py`, `backend/services/predictor_3g.py`, `backend/services/predictor_4g.py`, `backend/services/chat_service.py` (original application);
- Flutter source files from the mobile application (excluded from this public AI-only repository).

The files named above are **original source locations**, not necessarily files published here. This is a privacy- and permissions-conscious extraction, not the full original application.

| Topic | Original source | Public representation | Fidelity / limitations |
| --- | --- | --- | --- |
| 4G RSRP/RSRQ/throughput classification | `ai/sample.ipynb` and `backend/services/predictor_4g.py` | `backend/services/predictor_4g.py` | Same KPI bins, score maps, 30/30/40 weights and final decision thresholds |
| 3G KPI classes | `ai/sample.ipynb` and `backend/services/predictor_3g.py` | `backend/services/predictor_3g.py`, `training/train_3g.py` | RSCP / Ec/N0 thresholds traced from original code |
| 3G training labels | `ai/sample.ipynb` | `training/train_3g.py:label_3g` | Derived labels, not human-annotated ground truth |
| 3G input feature contract | `ai/sample.ipynb` section “Features / Target” | `training/train_3g.py:FEATURES` and `backend/services/predictor_3g.py` | Seven named and ordered source features |
| 3G cleaning | Original notebook steps 2–6 | `training/train_3g.py` | Column renaming, first comma-separated reading, numeric conversion, time and KPI filtering |
| 3G model experiments | Notebook comparison cell | `training/train_3g.py` | Logistic Regression, Decision Tree, Random Forest. Original notebook optionally added XGBoost; not included in portable public baseline |
| Model export | Original notebook `best_pipe` and `joblib.dump` cells | `training/train_3g.py` | RF n_estimators=200, random_state=42, class_weight=balanced; fitted on all filtered samples |
| Local API | Original `backend/main.py` | `backend/main.py` | Public extraction preserving /predict and /chat use cases, not a byte-for-byte identical private application |
| Rule-based chat | Original `backend/services/chat_service.py` | `backend/services/chat_service.py` | Heuristic explanations, not generative AI |
| Mobile interface | Original `mobile/lib/` Flutter sources | Not published | Full app remains outside the AI-only public release |
| Synthetic tests | No equivalent historical measurement experiment claimed | `tests/` | Newly written engineering checks, not actual 3G/4G field-performance results |

## 3G prediction inputs

The original training notebook uses these seven features, in this order:

1. `band_active`
2. `channel_active`
3. `scrambling_active`
4. `rscp_active`
5. `ecno_active`
6. `rscp_detected`
7. `scrambling_detected`

Time is parsed to filter unusable records but **not** included in the trained model's input feature list.

## Real and synthetic evidence boundaries

- The original CSV/XLSX datasets are not public. Their provenance and redistribution rights have not been established for open distribution.
- The original trained `.joblib` artifacts are not redistributed. The public training module can rebuild compatible artifacts locally from authorized source data.
- No genuine private-field evaluation score is published or inferred.
- Tests use small, deliberately fabricated records for software validation.
- The 4G score is a deterministic rubric. Calling it a trained 4G ML model would be incorrect.
- The 3G target is generated from the same RSCP/EcN0 features used by the model. Predicting it does not establish independent QoS ground truth.

## Validation boundaries

The public test suite can verify endpoint contracts, class thresholds, sample 4G arithmetic, preprocessing, 3G training and artifact generation on artificial records. It cannot validate field conditions, class balance, leakage avoidance, deployment quality, mobile integration, or generalization to unseen cell sites.

For experimental-method caveats and reproduction instructions see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
