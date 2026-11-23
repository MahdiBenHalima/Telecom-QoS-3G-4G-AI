# Reproducing the original 3G ML experiment

This is a **source-derived reconstruction**, not a report of newly measured accuracy.

The original academic notebook `ai/sample.ipynb` implements these steps:

1. Read `DT2.csv` with semicolon separators; normalize the original 3G column names.
2. For multiple-value strings, keep the first comma-separated value; convert KPI fields to numeric.
3. Remove rows missing valid time, RSCP or Ec/N0.
4. Create a **derived QoS class** from RSCP and Ec/N0. Score each from 1 to 4, then use the mean.
5. Encode the derived class, split 80/20 with `random_state=42`, stratified by class.
6. Preprocess numerics with median imputation and standard scaling; encode the band via imputation and one-hot encoding.
7. Compare Logistic Regression, Decision Tree and Random Forest (accuracy and weighted F1).
8. Train the selected Random Forest (200 trees, balanced class weights) on all rows and export `.joblib` artifacts.

The source notebook also optionally includes XGBoost when installed; the portable reconstruction covers the three scikit-learn classifiers. It reproduces the **implemented final Random Forest export**, not an independently selected winner from test performance.

## Train on your authorized, local dataset

```bash
pip install -r requirements.txt
python -m training.train_3g --data "path/to/DT2.csv" --output backend/models
```

The resulting `evaluation_3g.json` is local and must be reviewed before publishing scores. The raw data and fitted model should only be redistributed with appropriate permission.

## Statistical caveats

- Targets are engineered from the **same KPI variables used as model inputs**. A high score means the model learns this labeling policy; it does not demonstrate QoS perception or objective operator-level SLA performance.
- The original split is randomized; repeated measurements from nearby times or the same cell may be correlated across train and test. Prefer a grouped or chronological evaluation for future experiments.
- No real-data accuracy/F1 figures are claimed here because they have not been independently verified for this public release.
- `joblib` deserialization can execute arbitrary code: load only artifacts from trusted sources.
