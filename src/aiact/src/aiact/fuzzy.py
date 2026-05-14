from __future__ import annotations
from aiact.models import MatchedRule, ClassificationResult

ANSWER_CERTAINTY: dict[str, float] = {
    "yes": 1.0,
    "probably_yes": 0.8,
    "unknown": 0.5,
    "probably_no": 0.2,
    "no": 0.0,
}

TIER_PRIORITY: dict[str, int] = {
    "UNACCEPTABLE": 4,
    "HIGH": 3,
    "LIMITED": 2,
    "MINIMAL": 1,
}

UNCERTAINTY_THRESHOLD = 0.65


def map_answer(answer_value: str) -> float:
    return ANSWER_CERTAINTY.get(answer_value, 0.0)


def aggregate_results(matches: list[MatchedRule]) -> ClassificationResult:
    if not matches:
        fallback = MatchedRule(
            rule_id="R-DEFAULT",
            risk_tier="MINIMAL",
            article="N/A",
            reason="No indicators found",
            confidence=0.9,
        )
        return ClassificationResult(
            primary_tier="MINIMAL",
            primary_confidence=0.9,
            all_matches=[fallback],
        )

    by_tier: dict[str, list[MatchedRule]] = {}
    for m in matches:
        by_tier.setdefault(m.risk_tier, []).append(m)

    primary_tier = max(by_tier.keys(), key=lambda t: TIER_PRIORITY.get(t, 0))
    primary_confidence = max(m.confidence for m in by_tier[primary_tier])

    secondary_obligations: list[str] = []
    for tier, tier_matches in by_tier.items():
        if tier == primary_tier:
            continue
        articles = ", ".join(sorted({m.article for m in tier_matches}))
        secondary_obligations.append(f"{tier} ({articles})")

    uncertainty_note: str | None = None
    if primary_confidence < UNCERTAINTY_THRESHOLD:
        uncertainty_note = (
            "Classification confidence is low due to uncertain answers. "
            "Consider reviewing with a legal expert specialising in EU AI Act compliance."
        )

    return ClassificationResult(
        primary_tier=primary_tier,
        primary_confidence=primary_confidence,
        all_matches=matches,
        secondary_obligations=secondary_obligations,
        uncertainty_note=uncertainty_note,
    )
