"""Reconstruct the original 3G supervised-learning experiment from DT2.csv.

Data is NOT bundled: use a local copy that you are authorized to process.
The target is engineered from the RSCP and Ec/N0 input KPIs, not an external QoS label.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RENAMES = {
    "Band (active)": "band_active",
    "Channel number (active)": "channel_active",
    "Scrambling code (active)": "scrambling_active",
    "RSCP (active)": "rscp_active",
    "Ec/N0 (active)": "ecno_active",
    "RSCP (detected)": "rscp_detected",
    "Scrambling code (detected)": "scrambling_detected",
}
FEATURES = ["band_active", "channel_active", "scrambling_active", "rscp_active",
            "ecno_active", "rscp_detected", "scrambling_detected"]
NUMERIC = [col for col in FEATURES if col != "band_active"]


def first_value(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return np.nan
    return value.split(",")[0].strip()


def load_3g_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAMES)
    missing = sorted(set(FEATURES + ["Time"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    for col in FEATURES:
        df[col] = df[col].map(first_value)
    df["Time"] = pd.to_datetime(df["Time"].str.strip(), format="%H:%M:%S.%f", errors="coerce")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Time", "rscp_active", "ecno_active"]).reset_index(drop=True)


def quality_score_rscp(value: float) -> int:
    return 4 if value >= -75 else 3 if value >= -85 else 2 if value >= -95 else 1


def quality_score_ecno(value: float) -> int:
    return 4 if value >= -6 else 3 if value >= -9 else 2 if value >= -12 else 1


def label_3g(df: pd.DataFrame) -> pd.Series:
    scores = 0.5 * df["rscp_active"].map(quality_score_rscp) + 0.5 * df["ecno_active"].map(quality_score_ecno)
    return pd.Series(np.select(
        [scores >= 3.2, scores >= 2.2],
        ["Connexion recommandée", "Connexion possible mais instable"],
        default="Connexion faible / non recommandée"), index=df.index)


def make_pipeline(model):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    processor = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, ["band_active"])])
    return Pipeline([("preprocessor", processor), ("model", model)])


def train(data: Path, output: Path, *, evaluate: bool = True):
    df = load_3g_data(data)
    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(label_3g(df))
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }
    results = []
    if evaluate:
        counts = pd.Series(y).value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError("Need at least two classes and two samples per class for stratified evaluation")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        for name, estimator in models.items():
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            results.append({"model": name, "accuracy": float(accuracy_score(y_test, pred)),
                            "f1_weighted": float(f1_score(y_test, pred, average="weighted"))})
    # Mirror the original notebook: export a Random Forest refitted on all available rows.
    final_model = make_pipeline(models["RandomForest"])
    final_model.fit(X, y)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "rf_3g_pipeline.joblib")
    joblib.dump(encoder, output / "label_encoder_3g.joblib")
    if evaluate:
        (output / "evaluation_3g.json").write_text(json.dumps({
            "n_rows": int(len(df)), "label_origin": "derived_from_input_kpis",
            "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
            "comparison": results, "exported_model": "RandomForest"
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Private DT2.csv source")
    parser.add_argument("--output", type=Path, default=Path("backend/models"))
    args = parser.parse_args()
    results = train(args.data, args.output)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
