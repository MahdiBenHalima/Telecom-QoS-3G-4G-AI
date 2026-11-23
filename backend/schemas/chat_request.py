from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    network_type: str
    question: str
    decision_finale: str
    classes: dict
    scores: Optional[dict] = None
    zone: Optional[str] = None
