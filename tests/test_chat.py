from backend.services.chat_service import generate_chat_answer

def test_chat_uses_rule_based_response():
    answer = generate_chat_answer("Quel KPI est critique ?", "4G",
        "Connexion faible / non recommandée",
        {"rsrp_class": "Mauvaise", "rsrq_class": "Acceptable", "throughput_class": "Très faible"})
    assert "Throughput" in answer
