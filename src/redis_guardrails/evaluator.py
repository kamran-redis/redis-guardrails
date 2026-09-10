from redis_guardrails.models import Action, Decision, Match

_ACTION_PRIORITY: dict[Action, int] = {"BLOCK": 3, "FLAG": 2, "ALLOW": 1}


def decide(matches: list[Match]) -> Decision:
    within_threshold = [m for m in matches if m.distance <= m.threshold]
    if not within_threshold:
        return Decision(action="ALLOW", primary_match=None, matches=[])

    strongest_per_rule = _strongest_per_key(within_threshold, key=lambda m: m.rule_id)
    strongest_per_category = _strongest_per_key(strongest_per_rule, key=lambda m: m.category)

    top_action = max(
        (m.action for m in strongest_per_category), key=lambda a: _ACTION_PRIORITY[a]
    )
    candidates = [m for m in strongest_per_category if m.action == top_action]
    primary = _pick_primary(candidates)

    return Decision(action=top_action, primary_match=primary, matches=strongest_per_category)


def _strongest_per_key(matches: list[Match], key) -> list[Match]:
    best: dict[str, Match] = {}
    for m in matches:
        k = key(m)
        if k not in best or m.distance < best[k].distance:
            best[k] = m
    return list(best.values())


def _pick_primary(candidates: list[Match]) -> Match:
    def sort_key(m: Match) -> tuple[float, str]:
        margin = m.threshold - m.distance
        return (-margin, m.rule_id)

    return sorted(candidates, key=sort_key)[0]
