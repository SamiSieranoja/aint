from __future__ import annotations
from pathlib import Path
import clips
from aiact.models import SystemProfile, MatchedRule

RULES_DIR = Path(__file__).parent / "rules"
LOAD_ORDER = ["templates.clp", "unacceptable.clp", "high_risk.clp", "limited_risk.clp", "default.clp"]


class ClassificationEngine:
    def __init__(self, rules_dir: Path = RULES_DIR) -> None:
        self._env = clips.Environment()
        self._env.define_function(lambda a, b: min(a, b), "min-float")
        for filename in LOAD_ORDER:
            path = rules_dir / filename
            self._env.load(str(path))

    def classify(self, profile: SystemProfile) -> list[MatchedRule]:
        self._env.reset()
        self._env.assert_string(profile.to_clips_assertion())
        self._env.run()
        results: list[MatchedRule] = []
        for fact in self._env.facts():
            if fact.template.name == "matched-rule":
                slots = dict(fact)
                results.append(MatchedRule(
                    rule_id=slots["rule-id"],
                    risk_tier=slots["risk-tier"],
                    article=slots["article"],
                    reason=slots["reason"],
                    confidence=float(slots["confidence"]),
                ))
        return results
