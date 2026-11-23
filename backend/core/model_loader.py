from pathlib import Path
import joblib

def load_3g_model():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_path = model_dir / "rf_3g_pipeline.joblib"
    encoder_path = model_dir / "label_encoder_3g.joblib"
    if not model_path.exists() or not encoder_path.exists():
        return None, None, False
    try:
        return joblib.load(model_path), joblib.load(encoder_path), True
    except Exception:
        return None, None, False
