from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_status():
    response = client.get("/")
    assert response.status_code == 200
    assert "model_3g_loaded" in response.json()


def test_4g_endpoint_contract():
    response = client.post("/predict", json={
        "network_type": "4G", "rsrp": -86, "rsrq": -9,
        "throughput_dl": 22, "zone": "Synthetic demo zone"
    })
    assert response.status_code == 200
    result = response.json()
    assert result["scores"]["global_score"] == 3.0
    assert result["decision_finale"] == "Connexion possible mais instable"
    assert result["network_type"] == "4G"
    assert "explication" in result


def test_4g_requires_all_metrics():
    response = client.post("/predict", json={
        "network_type": "4G", "rsrp": -86, "rsrq": -9
    })
    assert response.status_code == 400


def test_chat_is_rule_based():
    response = client.post("/chat", json={
        "network_type": "4G",
        "question": "Quel KPI est critique ?",
        "decision_finale": "Connexion faible / non recommandée",
        "classes": {
            "rsrp_class": "Mauvaise",
            "rsrq_class": "Acceptable",
            "throughput_class": "Très faible"
        }
    })
    assert response.status_code == 200
    assert "Throughput" in response.json()["answer"]
