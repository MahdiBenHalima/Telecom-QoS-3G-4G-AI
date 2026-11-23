from typing import Literal, Optional
from pydantic import BaseModel

class NetworkRequest(BaseModel):
    network_type: Literal["3G", "4G"]
    band_active: Optional[str] = None
    channel_active: Optional[float] = None
    scrambling_active: Optional[float] = None
    rscp_active: Optional[float] = None
    ecno_active: Optional[float] = None
    rscp_detected: Optional[float] = None
    scrambling_detected: Optional[float] = None
    band: Optional[str] = None
    rsrp: Optional[float] = None
    rsrq: Optional[float] = None
    throughput_dl: Optional[float] = None
    zone: Optional[str] = None
