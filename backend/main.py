from fastapi import FastAPI, HTTPException
from backend.schemas.network_request import NetworkRequest
from backend.core.model_loader import load_3g_model
from backend.services.predictor_4g import predict_4g
from backend.services.predictor_3g import predict_3g
from backend.services.explanation_service import generate_explanation_3g, generate_explanation_4g
from backend.schemas.chat_request import ChatRequest
from backend.services.chat_service import generate_chat_answer

app = FastAPI(title="Smart Telecom QoS API")
model_3g, label_encoder_3g, model_3g_loaded = load_3g_model()

@app.get("/")
def root():
    return {"message": "Smart Telecom QoS API is running", "model_3g_loaded": model_3g_loaded}

@app.post("/predict")
def predict_network(data: NetworkRequest):
    if data.network_type == "4G":
        if any(v is None for v in [data.rsrp, data.rsrq, data.throughput_dl]):
            raise HTTPException(status_code=400, detail="Pour la 4G, il faut fournir rsrp, rsrq et throughput_dl.")
        result = predict_4g(data.rsrp, data.rsrq, data.throughput_dl)
        return {"network_type": "4G", "zone": data.zone, "band": data.band,
                "input": {"rsrp": data.rsrp, "rsrq": data.rsrq, "throughput_dl": data.throughput_dl},
                **result,
                "explication": generate_explanation_4g(result["classes"]["rsrp_class"],
                result["classes"]["rsrq_class"], result["classes"]["throughput_class"],
                result["decision_finale"], data.zone)}
    if not model_3g_loaded:
        raise HTTPException(status_code=503, detail="Modèle 3G non disponible. Voir README.")
    required = [data.band_active, data.channel_active, data.scrambling_active,
                data.rscp_active, data.ecno_active]
    if any(v is None for v in required):
        raise HTTPException(status_code=400, detail="Champs 3G obligatoires manquants.")
    result = predict_3g(model_3g, label_encoder_3g,
        data.band_active, data.channel_active, data.scrambling_active, data.rscp_active,
        data.ecno_active, data.rscp_detected, data.scrambling_detected)
    return {"network_type": "3G", "zone": data.zone, "input": data.model_dump(exclude_none=True),
            **result,
            "explication": generate_explanation_3g(result["classes"]["rscp_class"],
            result["classes"]["ecno_class"], result["decision_finale"], data.zone)}

@app.post("/chat")
def chat_assistant(data: ChatRequest):
    return {"network_type": data.network_type, "question": data.question,
            "answer": generate_chat_answer(data.question, data.network_type,
                data.decision_finale, data.classes, data.scores, data.zone)}
