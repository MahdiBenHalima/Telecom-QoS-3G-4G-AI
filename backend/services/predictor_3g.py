import pandas as pd

def classify_rscp(x: float) -> str:
    if x >= -75: return "Très bonne"
    if -85 <= x < -75: return "Bonne"
    if -95 <= x < -85: return "Moyenne"
    return "Mauvaise"

def classify_ecno(x: float) -> str:
    if x >= -6: return "Très bonne"
    if -9 <= x < -6: return "Bonne"
    if -12 <= x < -9: return "Moyenne"
    return "Mauvaise"

def predict_3g(model_3g, label_encoder_3g, band_active, channel_active, scrambling_active,
               rscp_active, ecno_active, rscp_detected=None, scrambling_detected=None):
    input_df = pd.DataFrame([{"band_active": band_active, "channel_active": channel_active,
         "scrambling_active": scrambling_active, "rscp_active": rscp_active,
         "ecno_active": ecno_active, "rscp_detected": rscp_detected,
         "scrambling_detected": scrambling_detected}])
    y_pred = model_3g.predict(input_df)
    decision = label_encoder_3g.inverse_transform(y_pred)[0]
    return {"classes": {"rscp_class": classify_rscp(rscp_active), "ecno_class": classify_ecno(ecno_active)},
            "decision_finale": str(decision)}
