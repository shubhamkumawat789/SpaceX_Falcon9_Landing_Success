
from pydantic import BaseModel
from typing import Optional

class LandingRequest(BaseModel):
    payload_mass_kg: float
    flight_number: int
    launch_site: str
    orbit: str
    booster_version: Optional[str] = None
    reused_count: int = 0
    cores: int = 1
    gridfins: int = 0
    legs: int = 0
    block: Optional[int] = None
    launch_year: int
    launch_month: int

class LandingResponse(BaseModel):
    probability: float
    predicted_label: int
