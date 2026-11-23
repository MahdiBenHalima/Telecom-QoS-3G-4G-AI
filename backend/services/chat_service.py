def normalize_text(text: str) -> str:
    return text.lower().strip()

def get_worst_kpi_4g(classes: dict) -> str:
    ranking = {"Très bonne": 4, "Bonne": 3, "Acceptable": 3, "Assez bien": 2,
               "Moyenne": 2, "Mauvaise": 1, "Très faible": 0}
    kpis = {"RSRP": classes.get("rsrp_class"), "RSRQ": classes.get("rsrq_class"),
            "Throughput": classes.get("throughput_class")}
    return min(kpis, key=lambda k: ranking.get(kpis[k], 999))

def get_worst_kpi_3g(classes: dict) -> str:
    ranking = {"Très bonne": 4, "Bonne": 3, "Moyenne": 2, "Mauvaise": 1}
    kpis = {"RSCP": classes.get("rscp_class"), "Ec/N0": classes.get("ecno_class")}
    return min(kpis, key=lambda k: ranking.get(kpis[k], 999))

def generate_chat_answer(question: str, network_type: str, decision_finale: str, classes: dict,
                         scores: dict | None = None, zone: str | None = None) -> str:
    if network_type not in ("3G", "4G"):
        return "Type de réseau non reconnu."
    zone_text = f"dans la zone {zone}" if zone else "dans cette zone"
    worst = get_worst_kpi_4g(classes) if network_type == "4G" else get_worst_kpi_3g(classes)
    if "kpi" in normalize_text(question) or "critique" in normalize_text(question):
        return f"Le KPI le plus critique en {network_type} {zone_text} est {worst}."
    if "pourquoi" in normalize_text(question) or "cause" in normalize_text(question):
        return (f"En {network_type}, {zone_text}, la décision est '{decision_finale}'. "
                f"Le KPI le plus dégradé est {worst}.")
    return (f"Assistant technique {network_type}: décision {zone_text} : "
            f"'{decision_finale}'. KPI le plus sensible : {worst}.")
