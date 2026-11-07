"""FastAPI service exposing the credit-risk model.

    uvicorn api.main:app --reload
    # interactive docs at http://127.0.0.1:8000/docs

The service returns a probability of default, the corresponding risk band and
the binary decision at the tuned threshold. Model internals (feature values,
SHAP vectors, artefact paths) are deliberately not exposed.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.models.predict import load_artifacts, score_records

LOGGER = logging.getLogger("credit-risk-api")

HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
LoanIntent = Literal[
    "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
]
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]


class CustomerRequest(BaseModel):
    """Loan application attributes available at decision time."""

    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Gross annual income")
    person_home_ownership: HomeOwnership
    person_emp_length: float | None = Field(
        None, ge=0, le=60, description="Years in current employment"
    )
    loan_intent: LoanIntent
    loan_grade: LoanGrade
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float | None = Field(
        None, ge=0, le=100, description="Offered annual interest rate (%)"
    )
    loan_percent_income: float | None = Field(
        None, gt=0, le=1, description="Loan amount / income; derived when omitted"
    )
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)

    @field_validator("person_home_ownership", "loan_intent", "loan_grade",
                     "cb_person_default_on_file", mode="before")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper() if isinstance(value, str) else value

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 29,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "C",
                "loan_amnt": 12000,
                "loan_int_rate": 13.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Scoring result for one application."""

    default_probability: float = Field(..., ge=0, le=1)
    non_default_probability: float = Field(..., ge=0, le=1)
    risk_category: str
    prediction: int = Field(..., ge=0, le=1, description="1 = predicted default")
    decision_threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the artefacts at start-up so the first request is not slow."""
    try:
        _, feature_config = load_artifacts()
        app.state.model_name = feature_config["selected_model"]
        app.state.model_loaded = True
        LOGGER.info("Loaded model: %s", app.state.model_name)
    except FileNotFoundError as exc:  # pragma: no cover - start-up path
        app.state.model_name = None
        app.state.model_loaded = False
        LOGGER.error("Model artefacts unavailable: %s", exc)
    yield


app = FastAPI(
    title="Credit Risk & Loan Default API",
    description="Probability of default and risk segmentation for loan applicants.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Service banner."""
    return {
        "service": "Credit Risk & Loan Default API",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness/readiness probe: is the model available?"""
    loaded = bool(getattr(app.state, "model_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=getattr(app.state, "model_name", None),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(customer: CustomerRequest) -> PredictionResponse:
    """Score one loan application."""
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artefacts are not available. Run the training pipeline first.",
        )
    try:
        result = score_records(customer.model_dump(exclude_none=True))[0]
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Scoring failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not score this application: {exc}",
        ) from exc
    return PredictionResponse(**result)
