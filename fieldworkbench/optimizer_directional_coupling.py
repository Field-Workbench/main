"""Coupling topology for Optimizer 2.4 directional closure.

The bounded 1-D engine can only prove closure on the conditional slice on which
it was evaluated.  If another coupled freedom subsequently moves, that proof is
stale and the axis must be revisited.  This module derives coupling components
from the same freedom-capability metadata used by the fidelity planner so the
search does not accumulate another geometry-kind whitelist.

Phase 4 uses these components to cycle conditional directional searches until a
whole coupled family is stable.  Later phases reuse the same topology under
capable physical physics.
"""
from __future__ import annotations

from typing import Any, Iterable

from .optimizer_capabilities import optimizer_freedom_descriptor


DIRECTIONAL_COUPLING_SCHEMA_VERSION = 1


_PRIMARY_FAMILY_BY_KIND = {
    "current": "current",
    "turns": "turns",
    "axial_winding_length": "winding_distribution",
    "circular_diameter": "path_geometry",
    "racetrack_end_diameter": "path_geometry",
    "racetrack_straight_length": "path_geometry",
    "square_length": "path_geometry",
    "square_width": "path_geometry",
    "pair_spacing": "pair_spacing",
    "position_x": "position",
    "position_y": "position",
    "position_z": "position",
    "orientation_x": "orientation",
    "orientation_y": "orientation",
    "orientation_z": "orientation",
    "assembly_orientation_x": "orientation",
    "assembly_orientation_y": "orientation",
    "assembly_orientation_z": "orientation",
    "coil_orientation_x": "orientation",
    "coil_orientation_y": "orientation",
    "coil_orientation_z": "orientation",
}


def optimizer_directional_primary_family(variable: dict[str, Any]) -> str:
    """Return the structural family used for conditional-closure coupling."""
    kind = str(variable.get("kind", ""))
    return _PRIMARY_FAMILY_BY_KIND.get(kind, "unknown")


def optimizer_directional_coupling_descriptor(variable: dict[str, Any]) -> dict[str, Any]:
    """Return one variable's primary family and declared coupled families."""
    descriptor = optimizer_freedom_descriptor(variable)
    return {
        "id": str(variable.get("id", "")),
        "kind": str(variable.get("kind", "")),
        "label": str(variable.get("label", variable.get("id", "freedom"))),
        "primary_family": optimizer_directional_primary_family(variable),
        "coupling_families": [
            str(value) for value in list(descriptor.get("coupling_families", []) or [])
        ],
    }


def optimizer_directional_coupling_components(
    variables: Iterable[dict[str, Any]],
    *,
    eligible_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return connected components for conditionally coupled structural freedoms.

    Two freedoms are connected when they belong to the same structural family
    (for example racetrack end diameter and straight length are both
    ``path_geometry``) or either freedom declares the other's family as coupled.
    The graph is intentionally symmetric even when the capability descriptor is
    one-way; a conditional optimum can become stale in either direction.
    """
    ordered = [dict(item) for item in variables]
    allowed = None if eligible_ids is None else {str(value) for value in eligible_ids}
    descriptors = [
        optimizer_directional_coupling_descriptor(item)
        for item in ordered
        if str(item.get("id", "")) and (allowed is None or str(item.get("id", "")) in allowed)
    ]
    by_id = {str(item["id"]): item for item in descriptors}
    order = [str(item["id"]) for item in descriptors]
    adjacency: dict[str, set[str]] = {variable_id: set() for variable_id in order}

    for index, left_id in enumerate(order):
        left = by_id[left_id]
        left_family = str(left.get("primary_family", "unknown"))
        left_couplings = set(map(str, left.get("coupling_families", []) or []))
        for right_id in order[index + 1:]:
            right = by_id[right_id]
            right_family = str(right.get("primary_family", "unknown"))
            right_couplings = set(map(str, right.get("coupling_families", []) or []))
            same_family = bool(
                left_family == right_family
                and left_family not in {"", "unknown"}
            )
            declared = bool(
                right_family in left_couplings
                or left_family in right_couplings
                or "unknown" in left_couplings
                or "unknown" in right_couplings
            )
            if same_family or declared:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    components: list[dict[str, Any]] = []
    visited: set[str] = set()
    order_index = {value: index for index, value in enumerate(order)}
    for root in order:
        if root in visited:
            continue
        stack = [root]
        component_ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_ids.append(current)
            neighbours = sorted(
                adjacency.get(current, set()),
                key=lambda value: order_index.get(value, 10**9),
                reverse=True,
            )
            stack.extend(neighbours)
        component_ids.sort(key=lambda value: order_index.get(value, 10**9))
        components.append({
            "component_id": f"coupled_{len(components) + 1}",
            "variable_ids": component_ids,
            "families": sorted({
                str(by_id[value].get("primary_family", "unknown"))
                for value in component_ids
            }),
            "coupled": len(component_ids) > 1,
        })

    return {
        "schema_version": DIRECTIONAL_COUPLING_SCHEMA_VERSION,
        "variable_order": order,
        "descriptors": descriptors,
        "adjacency": {
            variable_id: sorted(
                values, key=lambda value: order_index.get(value, 10**9)
            )
            for variable_id, values in adjacency.items()
        },
        "components": components,
        "coupled_component_count": sum(
            1 for item in components if bool(item.get("coupled", False))
        ),
    }
