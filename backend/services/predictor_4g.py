def classify_rsrp(x: float) -> str:
    if x >= -80: return "Très bonne"
    if -90 < x < -80: return "Bonne"
    if -100 < x <= -90: return "Moyenne"
    return "Mauvaise"

def classify_rsrq(x: float) -> str:
    if x >= -5: return "Très bonne"
    if -10 < x < -5: return "Acceptable"
    if -14 < x <= -10: return "Assez bien"
    return "Mauvaise"

def classify_throughput(x: float) -> str:
    if x >= 30: return "Très bonne"
    if 25 < x < 30: return "Acceptable"
    if 20 < x <= 25: return "Assez bien"
    if 10 < x <= 20: return "Mauvaise"
    return "Très faible"

rsrp_score_map = {"Très bonne": 4, "Bonne": 3, "Moyenne": 2, "Mauvaise": 1}
rsrq_score_map = {"Très bonne": 4, "Acceptable": 3, "Assez bien": 2, "Mauvaise": 1}
throughput_score_map = {"Très bonne": 5, "Acceptable": 4, "Assez bien": 3, "Mauvaise": 2, "Très faible": 1}

def final_decision_4g(score: float) -> str:
    if score >= 3.5: return "Connexion recommandée"
    if score >= 2.3: return "Connexion possible mais instable"
    return "Connexion faible / non recommandée"

def predict_4g(rsrp: float, rsrq: float, throughput_dl: float):
    a, b, c = classify_rsrp(rsrp), classify_rsrq(rsrq), classify_throughput(throughput_dl)
    ra, rb, rc = rsrp_score_map[a], rsrq_score_map[b], throughput_score_map[c]
    score = 0.30 * ra + 0.30 * rb + 0.40 * rc
    return {"classes": {"rsrp_class": a, "rsrq_class": b, "throughput_class": c},
            "scores": {"rsrp_score": ra, "rsrq_score": rb, "throughput_score": rc, "global_score": round(score, 2)},
            "decision_finale": final_decision_4g(score)}
