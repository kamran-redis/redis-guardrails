from redis_guardrails.models import Action, Decision, Match

_ACTION_PRIORITY: dict[Action, int] = {"BLOCK": 3, "FLAG": 2, "ALLOW": 1}


def decide(matches: list[Match]) -> Decision:
    within_threshold = [m for m in matches if m.distance <= m.threshold]
    if not within_threshold:
        return Decision(action="ALLOW", primary_match=None, matches=[])

    strongest_per_rule = _strongest_per_rule(within_threshold)
    strongest_per_category = _strongest_per_category(strongest_per_rule)

    top_action = max(
        (m.action for m in strongest_per_category), key=lambda a: _ACTION_PRIORITY[a]
    )
    candidates = [m for m in strongest_per_category if m.action == top_action]
    primary = _pick_primary(candidates)

    return Decision(action=top_action, primary_match=primary, matches=strongest_per_category)


def _strongest_per_rule(matches: list[Match]) -> list[Match]:
    # Action and category are invariant per rule_id (one guardrail = one
    # category/action), so a rule-level tie only affects which chunk_id
    # is reported, not the decision — but it must still be deterministic,
    # so ties break on chunk_id.
    best: dict[str, Match] = {}
    for m in matches:
        key = (m.distance, m.chunk_id)
        current = best.get(m.rule_id)
        if current is None or key < (current.distance, current.chunk_id):
            best[m.rule_id] = m
    return list(best.values())


def _category_sort_key(m: Match) -> tuple[int, float, str]:
    return (-_ACTION_PRIORITY[m.action], m.distance, m.rule_id)


def _strongest_per_category(matches: list[Match]) -> list[Match]:
    # Rank by action priority FIRST, distance second, rule_id last. Picking
    # by raw min-distance alone (ignoring action) lets a same-category
    # FLAG/ALLOW match with a smaller distance silently eliminate a BLOCK
    # match before action-priority is ever applied — a genuine BLOCK signal
    # would vanish just because a same-category FLAG happened to be closer.
    # This also fixes order-dependence on exact distance ties, which the
    # plain "m.distance < best[k].distance" comparison left undefined.
    best: dict[str, Match] = {}
    for m in matches:
        current = best.get(m.category)
        if current is None or _category_sort_key(m) < _category_sort_key(current):
            best[m.category] = m
    return list(best.values())


def _pick_primary(candidates: list[Match]) -> Match:
    def sort_key(m: Match) -> tuple[float, str]:
        margin = m.threshold - m.distance
        return (-margin, m.rule_id)

    return sorted(candidates, key=sort_key)[0]
