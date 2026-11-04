"""Authoritative-finalist queue policy for Optimizer 2.3.

The cheap scout and optional physical stages may already provide a meaningful
ranking.  Role-protected rows (source baseline, naturally-clear champion, DRC
boundary insurance) are *insurance* and must not displace the best available
higher-fidelity candidate from the first authoritative validation slot.
"""

from __future__ import annotations

from collections.abc import Iterable


def optimizer_authoritative_queue_ids(
    ranked_ids: Iterable[str],
    protected_ids: Iterable[str],
    *,
    preserve_ranked_leader: bool,
) -> list[str]:
    """Return a stable authoritative queue over candidate IDs.

    When ``preserve_ranked_leader`` is true, the first item in ``ranked_ids`` is
    treated as the best candidate from a higher-fidelity ranking and is kept at
    the head of the queue.  Protected/insurance candidates are inserted
    immediately after it, in their declared priority order, then the remaining
    ranked candidates follow in their original order.

    Without a higher-fidelity ranking, protected candidates retain the historic
    behavior of being front-loaded before the ordinary queue.
    """

    ranked: list[str] = []
    seen: set[str] = set()
    for raw in ranked_ids:
        candidate_id = str(raw)
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        ranked.append(candidate_id)

    if not ranked:
        return []

    ranked_set = set(ranked)
    protected: list[str] = []
    protected_seen: set[str] = set()
    for raw in protected_ids:
        candidate_id = str(raw)
        if (
            not candidate_id
            or candidate_id in protected_seen
            or candidate_id not in ranked_set
        ):
            continue
        protected_seen.add(candidate_id)
        protected.append(candidate_id)

    if preserve_ranked_leader:
        leader = ranked[0]
        protected_tail = [candidate_id for candidate_id in protected if candidate_id != leader]
        reserved = {leader, *protected_tail}
        ordinary = [candidate_id for candidate_id in ranked[1:] if candidate_id not in reserved]
        return [leader, *protected_tail, *ordinary]

    protected_set = set(protected)
    ordinary = [candidate_id for candidate_id in ranked if candidate_id not in protected_set]
    return [*protected, *ordinary]


def optimizer_should_expand_authoritative_validation(
    *,
    validation_count: int,
    validation_cap: int,
    has_survivor: bool,
    intermediate_enabled: bool,
    intermediate_decisive: bool,
    ranked_leader_required: bool,
    ranked_leader_survived: bool,
) -> bool:
    """Decide whether progressive authoritative validation must open more slots.

    A decisive intermediate ranking may stop early only after its best-ranked
    candidate has itself survived authoritative validation.  This prevents an
    insurance candidate from satisfying the generic "some survivor exists"
    condition while the higher-fidelity leader failed or was never tested.
    """

    if int(validation_count) >= int(validation_cap):
        return False
    if ranked_leader_required and not ranked_leader_survived:
        return True
    return bool(
        not has_survivor
        or not intermediate_enabled
        or not intermediate_decisive
    )
