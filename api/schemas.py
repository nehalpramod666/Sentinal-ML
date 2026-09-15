"""
Day 15 — Pydantic request/response schemas for the FastAPI service.
"""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Raw flow feature values for the 35 ACO-selected features. Field names
    match the columns in reports/selected_features.csv exactly."""

    features: dict[str, float] = Field(
        ...,
        description=(
            "Mapping of ACO-selected feature name -> value. Must contain "
            "exactly the 35 features listed in GET /model-info. Extra keys "
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

    @field_validator("features")
    @classmethod
    def validate_no_extreme_values(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject values so extreme they indicate malformed input rather than
        real traffic data — e.g. absolute values beyond 1e15, which is far
        outside anything a real flow statistic could plausibly be (recall
        from Day 5's EDA that even the most extreme legitimate features
        topped out around 1e8-1e9). Note: negative values are intentionally
        NOT blanket-rejected here — legitimate features like Down/Up Ratio
        deltas don't apply, and Day 3's preprocessing already handles the
        specific known-corrupted columns (Flow Duration, Flow Bytes/s,
        Fwd Header Length) at the data layer, not here."""
        for feature_name, value in v.items():
            if abs(value) > 1e15:
                raise ValueError(
                    f"Feature '{feature_name}' has an implausible value ({value}); "
                    "must have absolute value <= 1e15"
                )

        return v


class PredictResponse(BaseModel):
    predicted_label: str = Field(
        ...,
        description="Model's top predicted traffic class",
    )
    prob_attack: float = Field(
        ...,
        description="P(attack) = 1 - P(BENIGN), in [0, 1]",
    )
    confidence: float = Field(
        ...,
        description="Margin between top-1 and top-2 class probabilities",
    )
    traffic_density: float = Field(
        ...,
        description="Normalized Flow Bytes/s, in [0, 1]",
    )
    risk_score: float = Field(
        ...,
        description="Fuzzy-inferred risk score, 0-100",
    )
    risk_level: str = Field(
        ...,
        description="Low / Medium / High / Critical",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    n_features: int
    selected_features: list[str]
    label_mapping: dict[int, str]
    density_feature: str
