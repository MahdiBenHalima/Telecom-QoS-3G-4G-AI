from backend.services.predictor_4g import predict_4g

def test_good_connection():
    result = predict_4g(-70, -4, 35)
    assert result["scores"]["global_score"] == 4.4
    assert result["decision_finale"] == "Connexion recommandée"

def test_weak_connection():
    result = predict_4g(-110, -16, 5)
    assert result["decision_finale"] == "Connexion faible / non recommandée"

def test_weights():
    result = predict_4g(-86, -9, 22)
    assert result["scores"]["global_score"] == 3.0
