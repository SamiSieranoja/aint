from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class SystemProfile(BaseModel):
    biometric_realtime: float = Field(default=0.0, ge=0.0, le=1.0)
    public_space: float = Field(default=0.0, ge=0.0, le=1.0)
    social_scoring: float = Field(default=0.0, ge=0.0, le=1.0)
    public_authority: float = Field(default=0.0, ge=0.0, le=1.0)
    targets_vulnerable: float = Field(default=0.0, ge=0.0, le=1.0)
    subliminal_manipulation: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion_recognition: float = Field(default=0.0, ge=0.0, le=1.0)
    workplace_use: float = Field(default=0.0, ge=0.0, le=1.0)
    education_use: float = Field(default=0.0, ge=0.0, le=1.0)
    predictive_policing: float = Field(default=0.0, ge=0.0, le=1.0)
    employment_use: float = Field(default=0.0, ge=0.0, le=1.0)
    automates_decisions: float = Field(default=0.0, ge=0.0, le=1.0)
    affects_individuals: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_infrastructure: float = Field(default=0.0, ge=0.0, le=1.0)
    law_enforcement: float = Field(default=0.0, ge=0.0, le=1.0)
    migration_border: float = Field(default=0.0, ge=0.0, le=1.0)
    credit_insurance: float = Field(default=0.0, ge=0.0, le=1.0)
    justice_democratic: float = Field(default=0.0, ge=0.0, le=1.0)
    is_chatbot: float = Field(default=0.0, ge=0.0, le=1.0)
    generates_synthetic: float = Field(default=0.0, ge=0.0, le=1.0)
    interacts_with_people: float = Field(default=0.0, ge=0.0, le=1.0)
    biometric_categorisation: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_clips_assertion(self) -> str:
        slots = []
        for field_name, value in self.model_dump().items():
            clips_name = field_name.replace("_", "-")
            slots.append(f"({clips_name} {value:.4f})")
        return "(ai-system " + " ".join(slots) + ")"


class MatchedRule(BaseModel):
    rule_id: str
    risk_tier: str
    article: str
    reason: str
    confidence: float


class ClassificationResult(BaseModel):
    primary_tier: str
    primary_confidence: float
    all_matches: list[MatchedRule]
    secondary_obligations: list[str] = Field(default_factory=list)
    uncertainty_note: Optional[str] = None


class QuestionChoice(BaseModel):
    value: str
    label: str


class DependsOn(BaseModel):
    question: str
    values: list[str]


class Question(BaseModel):
    id: str
    text: str
    type: str  # "fuzzy_yes_no" or "choice"
    clips_slot: Optional[str] = None
    choices: list[QuestionChoice] = Field(default_factory=list)
    depends_on: Optional[DependsOn] = None
    help: Optional[str] = None
