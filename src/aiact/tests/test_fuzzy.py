from __future__ import annotations
import pytest
from aiact.fuzzy import map_answer, aggregate_results, ANSWER_CERTAINTY
from aiact.models import MatchedRule


def test_map_answer_all_values():
    assert map_answer("yes") == 1.0
    assert map_answer("probably_yes") == 0.8
    assert map_answer("unknown") == 0.5
    assert map_answer("probably_no") == 0.2
    assert map_answer("no") == 0.0


def test_map_answer_unknown_key():
    assert map_answer("garbage") == 0.0


def test_aggregate_selects_highest_tier():
    matches = [
        MatchedRule(rule_id="R1", risk_tier="LIMITED", article="A", reason="r", confidence=0.9),
        MatchedRule(rule_id="R2", risk_tier="HIGH",    article="B", reason="r", confidence=0.7),
    ]
    result = aggregate_results(matches)
    assert result.primary_tier == "HIGH"
    assert result.primary_confidence == 0.7


def test_aggregate_unacceptable_wins():
    matches = [
        MatchedRule(rule_id="R1", risk_tier="HIGH",         article="A", reason="r", confidence=0.9),
        MatchedRule(rule_id="R2", risk_tier="UNACCEPTABLE", article="B", reason="r", confidence=0.5),
    ]
    result = aggregate_results(matches)
    assert result.primary_tier == "UNACCEPTABLE"


def test_uncertainty_note_when_low_confidence():
    matches = [
        MatchedRule(rule_id="R1", risk_tier="HIGH", article="A", reason="r", confidence=0.5),
    ]
    result = aggregate_results(matches)
    assert result.uncertainty_note is not None
    assert "confidence" in result.uncertainty_note.lower()


def test_no_uncertainty_note_when_high_confidence():
    matches = [
        MatchedRule(rule_id="R1", risk_tier="HIGH", article="A", reason="r", confidence=0.9),
    ]
    result = aggregate_results(matches)
    assert result.uncertainty_note is None


def test_secondary_obligations_detected():
    matches = [
        MatchedRule(rule_id="R1", risk_tier="HIGH",    article="Annex III", reason="r", confidence=0.9),
        MatchedRule(rule_id="R2", risk_tier="LIMITED", article="Art. 50",   reason="r", confidence=0.8),
    ]
    result = aggregate_results(matches)
    assert result.primary_tier == "HIGH"
    assert len(result.secondary_obligations) == 1
    assert "LIMITED" in result.secondary_obligations[0]


def test_empty_matches_returns_minimal():
    result = aggregate_results([])
    assert result.primary_tier == "MINIMAL"
