from __future__ import annotations
import pytest
from aiact.engine import ClassificationEngine
from aiact.models import SystemProfile

@pytest.fixture(scope="module")
def engine():
    return ClassificationEngine()


def test_social_scoring_unacceptable(engine):
    profile = SystemProfile(social_scoring=1.0, public_authority=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "UNACCEPTABLE" in tiers
    rule_ids = {m.rule_id for m in matches}
    assert "R-ART5-C" in rule_ids


def test_rtbi_public_space_unacceptable(engine):
    profile = SystemProfile(biometric_realtime=1.0, public_space=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "UNACCEPTABLE" in tiers
    rule_ids = {m.rule_id for m in matches}
    assert "R-ART5-H" in rule_ids


def test_rtbi_unknown_public_space_confidence(engine):
    profile = SystemProfile(biometric_realtime=1.0, public_space=0.5)
    matches = engine.classify(profile)
    rtbi_match = next((m for m in matches if m.rule_id == "R-ART5-H"), None)
    assert rtbi_match is not None
    assert abs(rtbi_match.confidence - 0.5) < 0.01


def test_employment_high_risk(engine):
    profile = SystemProfile(employment_use=1.0, automates_decisions=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "HIGH" in tiers


def test_chatbot_limited_risk(engine):
    profile = SystemProfile(is_chatbot=1.0, interacts_with_people=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "LIMITED" in tiers
    assert "UNACCEPTABLE" not in tiers


def test_no_indicators_minimal(engine):
    profile = SystemProfile()  # all 0.0
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert tiers == {"MINIMAL"}
    assert any(m.rule_id == "R-DEFAULT" for m in matches)


def test_emotion_recognition_workplace_unacceptable(engine):
    profile = SystemProfile(emotion_recognition=1.0, workplace_use=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "UNACCEPTABLE" in tiers


def test_deepfake_limited_risk(engine):
    profile = SystemProfile(generates_synthetic=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "LIMITED" in tiers


def test_employment_chatbot_dual_tier(engine):
    profile = SystemProfile(employment_use=0.9, automates_decisions=0.8,
                            is_chatbot=1.0, interacts_with_people=1.0)
    matches = engine.classify(profile)
    tiers = {m.risk_tier for m in matches}
    assert "HIGH" in tiers
    assert "LIMITED" in tiers
