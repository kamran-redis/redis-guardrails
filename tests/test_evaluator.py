from redis_guardrails.evaluator import decide
from redis_guardrails.models import Match


def _match(**overrides) -> Match:
    defaults = dict(
        rule_id="rule-1",
        category="cat-1",
        action="BLOCK",
        distance=0.1,
        threshold=0.5,
        chunk_id="input-0",
        evaluated_text="some text",
    )
    defaults.update(overrides)
    return Match(**defaults)


def test_no_matches_returns_allow():
    decision = decide([])
    assert decision.action == "ALLOW"
    assert decision.primary_match is None
    assert decision.matches == []


def test_match_outside_threshold_is_ignored():
    match = _match(distance=0.9, threshold=0.5)
    decision = decide([match])
    assert decision.action == "ALLOW"
    assert decision.matches == []


def test_keeps_strongest_match_per_rule_across_chunks():
    weaker = _match(rule_id="rule-1", distance=0.4, chunk_id="input-0")
    stronger = _match(rule_id="rule-1", distance=0.1, chunk_id="input-1")
    decision = decide([weaker, stronger])
    assert len(decision.matches) == 1
    assert decision.matches[0].distance == 0.1
    assert decision.matches[0].chunk_id == "input-1"


def test_keeps_strongest_match_per_category():
    weaker = _match(rule_id="rule-1", category="cat-1", distance=0.4)
    stronger = _match(rule_id="rule-2", category="cat-1", distance=0.1)
    decision = decide([weaker, stronger])
    assert len(decision.matches) == 1
    assert decision.matches[0].rule_id == "rule-2"


def test_block_beats_flag_beats_allow():
    block = _match(rule_id="r-block", category="c-block", action="BLOCK", distance=0.45)
    flag = _match(rule_id="r-flag", category="c-flag", action="FLAG", distance=0.1)
    decision = decide([block, flag])
    assert decision.action == "BLOCK"
    assert decision.primary_match.rule_id == "r-block"
    assert len(decision.matches) == 2  # both matched categories are reported


def test_primary_match_tie_break_by_margin_then_id():
    # Both BLOCK, same distance-from-threshold margin (0.4), so the tie
    # breaks on the smaller rule_id.
    a = _match(rule_id="rule-b", category="cat-a", action="BLOCK", distance=0.1, threshold=0.5)
    b = _match(rule_id="rule-a", category="cat-b", action="BLOCK", distance=0.1, threshold=0.5)
    decision = decide([a, b])
    assert decision.primary_match.rule_id == "rule-a"


def test_category_collapse_never_discards_a_higher_priority_action():
    # Same category, different actions: a BLOCK match must survive category
    # collapse even if a same-category FLAG match has a smaller distance —
    # picking the category's representative by raw distance alone would
    # silently drop the BLOCK signal before action-priority is ever applied.
    block_match = _match(rule_id="rule-block", category="shared-cat", action="BLOCK", distance=0.4, threshold=0.5)
    flag_match = _match(rule_id="rule-flag", category="shared-cat", action="FLAG", distance=0.1, threshold=0.5)
    decision = decide([block_match, flag_match])
    assert decision.action == "BLOCK"


def test_category_collapse_is_order_independent_on_exact_distance_ties():
    # Same category, same distance, different actions — result must not
    # depend on which order the matches are passed in.
    block_match = _match(rule_id="rule-block", category="shared-cat", action="BLOCK", distance=0.3, threshold=0.5)
    flag_match = _match(rule_id="rule-flag", category="shared-cat", action="FLAG", distance=0.3, threshold=0.5)
    assert decide([block_match, flag_match]).action == "BLOCK"
    assert decide([flag_match, block_match]).action == "BLOCK"
