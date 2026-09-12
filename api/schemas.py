"""
Day 15 — Pydantic request/response schemas for the FastAPI service.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            "Mapping of ACO-selected feature name -> value. Must contain "
            "exactly the 35 features listed in GET / model-info. Extra keys "
            "are ignored; missing keys raise a 422 error."
        ),
        examples=[
            {
                "Destination Port": 443.0,
                "Flow Duration": 120000.0,
                "Flow Bytes/s": 5000.0,
            }
        ],
    )


class PredictResponse(BaseModel):
    predicted_label: str = Field(..., description="Model's top predicted traffic class")
    prob_attack: float = Field(..., description="P(attack) = 1 - P(BENIGN), in [0, 1]")
    confidence: float = Field(..., description="Margin between top-1 and top-2 class probabilities")
    traffic_density: float = Field(..., description="Normalized Flow Bytes/s, in [0, 1]")
    risk_score: float = Field(..., description="Fuzzy-inferred risk score, 1-100")
    risk_level: str = Field(..., description="Low / Medium / High / Critical")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    n_features: int
    selected_features: list[str]
    label_mapping: dict[int, str]
    density_feature: str
