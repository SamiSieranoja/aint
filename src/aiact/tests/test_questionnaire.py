from __future__ import annotations
import pytest
from pathlib import Path
import yaml
from aiact.questionnaire import QuestionnaireRunner, load_questions, _should_ask
from aiact.models import Question, QuestionChoice, DependsOn

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_questions_count():
    questions = load_questions()
    assert len(questions) >= 15


def test_fuzzy_yes_no_gets_choices():
    questions = load_questions()
    fuzzy_questions = [q for q in questions if q.type == "fuzzy_yes_no"]
    for q in fuzzy_questions:
        values = {c.value for c in q.choices}
        assert {"yes", "probably_yes", "unknown", "probably_no", "no"} == values


def test_depends_on_skips_when_parent_no():
    questions = load_questions()
    # public_space depends on biometric_realtime being yes/probably_yes/unknown
    ps_q = next(q for q in questions if q.id == "public_space")
    assert not _should_ask(ps_q, {"biometric_realtime": "no"})
    assert not _should_ask(ps_q, {"biometric_realtime": "probably_no"})
    assert _should_ask(ps_q, {"biometric_realtime": "yes"})
    assert _should_ask(ps_q, {"biometric_realtime": "unknown"})


def test_build_profile_maps_answers():
    runner = QuestionnaireRunner()
    answers = {
        "employment_use": "yes",
        "automates_decisions": "probably_yes",
        "affects_individuals": "unknown",
    }
    profile = runner.build_profile(answers)
    assert profile.employment_use == 1.0
    assert profile.automates_decisions == 0.8
    assert profile.affects_individuals == 0.5


def test_build_profile_no_answer_stays_zero():
    runner = QuestionnaireRunner()
    profile = runner.build_profile({})
    assert profile.employment_use == 0.0
    assert profile.social_scoring == 0.0


@pytest.mark.parametrize("fixture_file", list(FIXTURES_DIR.glob("*.yaml")))
def test_fixture_classification(fixture_file, engine_fixture):
    data = yaml.safe_load(fixture_file.read_text())
    answers = data["answers"]
    expected = data["expected"]

    runner = QuestionnaireRunner()
    profile = runner.build_profile(answers)

    from aiact import fuzzy
    matches = engine_fixture.classify(profile)
    result = fuzzy.aggregate_results(matches)

    assert result.primary_tier == expected["primary_tier"], (
        f"Fixture {fixture_file.name}: expected {expected['primary_tier']}, got {result.primary_tier}"
    )
    assert result.primary_confidence >= expected["min_confidence"], (
        f"Fixture {fixture_file.name}: confidence {result.primary_confidence} < {expected['min_confidence']}"
    )
