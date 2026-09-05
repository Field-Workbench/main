"""Persistent flat optimizer solution registry.

The optimizer sees the same physical setting at several stages and fidelities:
cheap magnetic reconnaissance, geometry-only DRC, optional physical promotion,
and authoritative full-model validation.  This module keeps one durable record
per *canonical structural solution* and attaches observations/provenance to that
record instead of building a parent/child candidate tree or passing disposable
stage-local lists between stages.

The registry is intentionally compact and JSON-friendly.  Large field arrays and
scene copies remain stage-local; the registry stores only canonical settings,
scalar observations, provenance references, DRC state, lifecycle metadata and
view tags for later ranking/decimation/UI use.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np


_PERIODIC_KINDS = {
    "orientation_x", "orientation_y", "orientation_z",
    "assembly_orientation_x", "assembly_orientation_y", "assembly_orientation_z",
    "coil_orientation_x", "coil_orientation_y", "coil_orientation_z",
}

# Higher number = more trustworthy for ranking.  Unknown future fidelity names
# remain below the explicit physical/authoritative tiers rather than silently
# outranking them.
_FIDELITY_PRIORITY = {
    "scout": 10,
    "fine_scout": 20,
    "physical_promotion": 30,
    "promotion": 30,
    "intermediate": 30,
    "authoritative": 40,
}


def _plain(value: Any) -> Any:
    """Return a compact JSON-friendly copy, omitting solver-sized arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 24:
            return value.tolist()
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            clean = _plain(item)
            if clean is not None:
                result[str(key)] = clean
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return None
        result = []
        for item in value:
            clean = _plain(item)
            if clean is not None:
                result.append(clean)
        return result
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _fidelity_rank(name: Any) -> int:
    text = str(name or "").strip().lower()
    if text in _FIDELITY_PRIORITY:
        return _FIDELITY_PRIORITY[text]
    if "authoritative" in text or "full" in text:
        return 40
    if "physical" in text or "promotion" in text or "bundled" in text:
        return 30
    if "fine" in text:
        return 20
    if "scout" in text or "centreline" in text:
        return 10
    return 0


class OptimizerCandidateRegistry:
    """Accumulate one flat durable record per canonical structural solution."""

    SCHEMA_VERSION = 3

    def __init__(self, structural_variables: Iterable[str | dict[str, Any]]):
        variables: list[dict[str, Any]] = []
        for item in structural_variables:
            if isinstance(item, dict):
                variable_id = str(item.get("id", ""))
                if not variable_id:
                    continue
                variables.append({
                    "id": variable_id,
                    "kind": str(item.get("kind", "")),
                    "label": str(item.get("label", variable_id)),
                    "unit": str(item.get("unit", "")),
                    "step": _plain(item.get("step")),
                    "base": _plain(item.get("base")),
                    "min": _plain(item.get("min")),
                    "max": _plain(item.get("max")),
                    "integer": bool(item.get("integer", False)),
                })
            else:
                variables.append({
                    "id": str(item), "kind": "", "label": str(item), "unit": "",
                    "step": None, "base": None, "min": None, "max": None,
                    "integer": False,
                })
        self.structural_variables = tuple(variables)
        self.structural_variable_ids = tuple(item["id"] for item in variables)
        self._variable_by_id = {item["id"]: item for item in variables}
        self._records: list[dict[str, Any]] = []
        self._by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        self._next_id = 1
        self.views: dict[str, list[str]] = {
            "finalists": [],
            "best_feasible": [],
            "best_overall": [],
            "drc_boundary": [],
            "source_baseline": [],
            "strong_basins": [],
        }

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        wrapped = (float(value) + 180.0) % 360.0 - 180.0
        # Prefer +0 over -0 in signatures/UI.
        return 0.0 if abs(wrapped) < 5.0e-12 else wrapped

    def _canonical_numeric(self, variable: dict[str, Any], raw: Any) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None

        step = variable.get("step")
        if step is None and bool(variable.get("integer", False)):
            step = 1.0
        try:
            step_value = float(step) if step is not None else None
        except (TypeError, ValueError):
            step_value = None
        if step_value is not None and (not math.isfinite(step_value) or step_value <= 0.0):
            step_value = None

        base = variable.get("base")
        try:
            base_value = float(base) if base is not None else 0.0
        except (TypeError, ValueError):
            base_value = 0.0
        if step_value is not None:
            value = base_value + round((value - base_value) / step_value) * step_value

        if str(variable.get("kind", "")) in _PERIODIC_KINDS:
            value = self._wrap_degrees(value)

        # Nine decimal places remains far below any user-visible geometry lattice,
        # while removing arithmetic noise from bisection/interpolation paths.
        return round(float(value), 9)

    def canonical_structural_values(self, values: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for variable_id in self.structural_variable_ids:
            variable = self._variable_by_id[variable_id]
            numeric = self._canonical_numeric(variable, values.get(variable_id))
            result[variable_id] = numeric
        return result

    def _signature(self, values: dict[str, Any]) -> tuple[Any, ...]:
        if not self.structural_variable_ids:
            return ("source",)
        canonical = self.canonical_structural_values(values)
        parts: list[Any] = []
        for variable_id in self.structural_variable_ids:
            parts.extend((variable_id, canonical.get(variable_id)))
        return tuple(parts)

    def _structural_setting_summary(self, values: dict[str, Any]) -> str:
        """Return a stable display summary containing structural freedoms only.

        Inner-solved excitation values (notably current groups) deliberately do not
        belong to a physical solution's identity.  They live on each fidelity
        observation in ``currents_a`` / ``solved_values`` instead.
        """
        canonical = self.canonical_structural_values(values)
        pieces: list[str] = []
        for variable in self.structural_variables:
            variable_id = str(variable["id"])
            value = canonical.get(variable_id)
            if value is None:
                continue
            if bool(variable.get("integer", False)):
                value_text = str(int(round(float(value))))
            else:
                value_text = f"{float(value):.7g}"
            label = str(variable.get("label") or variable_id)
            unit = str(variable.get("unit") or "").strip()
            pieces.append(f"{label}={value_text}{(' ' + unit) if unit else ''}")
        return "; ".join(pieces)

    def _structural_only_values(self, values: dict[str, Any]) -> dict[str, Any]:
        canonical = self.canonical_structural_values(values)
        return {
            variable_id: canonical.get(variable_id)
            for variable_id in self.structural_variable_ids
        }

    def _solved_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return non-structural values reported by an evaluator.

        These are metadata for that fidelity observation, never part of canonical
        solution identity.  In mixed searches this primarily captures inner-solved
        current-group values.
        """
        clean = _plain(values) or {}
        if not isinstance(clean, dict):
            return {}
        structural_ids = set(self.structural_variable_ids)
        return {
            str(key): value
            for key, value in clean.items()
            if str(key) not in structural_ids
        }

    def _new_record(self, values: dict[str, Any], setting_summary: str = "") -> dict[str, Any]:
        candidate_id = f"candidate_{self._next_id:06d}"
        self._next_id += 1
        structural_values = self.canonical_structural_values(values)
        record = {
            "id": candidate_id,
            "structural_values": structural_values,
            # Keep the master solution state purely structural. Inner-solved current
            # values belong to individual evaluations and must never overwrite the
            # canonical solution record as fidelity changes.
            "variable_values": self._structural_only_values(values),
            "setting_summary": self._structural_setting_summary(values),
            "provenance": [],
            "evaluations": [],
            "drc": {"status": "unknown", "minimum_clearance_mm": None, "reasons": []},
            "rejection_reasons": [],
            "tags": [],
            "retired_reason": None,
        }
        signature = self._signature(values)
        self._records.append(record)
        self._by_signature[signature] = record
        self._by_id[candidate_id] = record
        return record

    def get(self, candidate_id: str | None) -> dict[str, Any] | None:
        if not candidate_id:
            return None
        return self._by_id.get(str(candidate_id))

    def find_by_values(self, values: dict[str, Any]) -> dict[str, Any] | None:
        return self._by_signature.get(self._signature(values))

    def add_provenance(
        self,
        candidate_id: str | None,
        *,
        origin: str,
        stage: str = "",
        relation: str | None = None,
        related_solution_ids: Iterable[str] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        record = self.get(candidate_id)
        if record is None:
            return
        related = []
        for item in related_solution_ids:
            value = str(item)
            if value and value != record["id"] and value not in related:
                related.append(value)
        event = {
            "origin": str(origin or "unknown"),
            "stage": str(stage or ""),
            "relation": (str(relation) if relation else None),
            "related_solution_ids": related,
            "details": _plain(details or {}) or {},
        }
        if event not in record["provenance"]:
            record["provenance"].append(event)

    def ensure(
        self,
        values: dict[str, Any],
        *,
        setting_summary: str = "",
        origin: str | None = None,
        stage: str | None = None,
        related_solution_ids: Iterable[str] = (),
        relation: str | None = None,
        provenance_details: dict[str, Any] | None = None,
        # v1/internal compatibility: callers using parent_ids are flattened into
        # provenance references; no parent/child ownership is created.
        parent_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        signature = self._signature(values)
        record = self._by_signature.get(signature)
        if record is None:
            record = self._new_record(values, setting_summary)
        else:
            record["variable_values"].update(self._structural_only_values(values))
            # Regenerate from canonical structural values instead of accepting a
            # candidate summary that may contain a fidelity-specific solved current.
            record["setting_summary"] = self._structural_setting_summary(
                record.get("structural_values", {}) or values
            )
        related = list(map(str, related_solution_ids)) + list(map(str, parent_ids))
        event_details = dict(provenance_details or {})
        raw_structural = {
            variable_id: _plain(values.get(variable_id))
            for variable_id in self.structural_variable_ids
            if variable_id in values
        }
        canonical = record.get("structural_values", {}) or {}
        if raw_structural and any(
            raw_structural.get(variable_id) != canonical.get(variable_id)
            for variable_id in raw_structural
        ):
            event_details.setdefault("search_coordinate_alias", raw_structural)
        if origin or stage or related or relation or event_details:
            self.add_provenance(
                str(record["id"]),
                origin=str(origin or "unknown"),
                stage=str(stage or ""),
                relation=relation or ("derived_from" if related else None),
                related_solution_ids=related,
                details=event_details,
            )
        return record

    def observe_candidate(
        self,
        candidate: dict[str, Any],
        *,
        stage: str,
        fidelity: str,
        origin: str = "evaluation",
        model: str | None = None,
        sample_count: int | None = None,
        related_solution_ids: Iterable[str] = (),
        relation: str | None = None,
        provenance_details: dict[str, Any] | None = None,
        parent_ids: Iterable[str] = (),
    ) -> str:
        values = dict(candidate.get("variable_values", {}) or {})
        record = self.ensure(
            values,
            setting_summary=str(candidate.get("setting_summary", "")),
            origin=origin,
            stage=stage,
            related_solution_ids=related_solution_ids,
            relation=relation,
            provenance_details=provenance_details,
            parent_ids=parent_ids,
        )
        evaluation = {
            "stage": str(stage),
            "fidelity": str(fidelity),
            "fidelity_rank": int(_fidelity_rank(fidelity)),
            "model": str(model) if model else None,
            "sample_count": int(sample_count) if sample_count is not None else None,
            "feasible": bool(candidate.get("feasible", True)),
            "metrics": _plain(candidate.get("metrics", {}) or {}) or {},
            "currents_a": _plain(candidate.get("currents_a", {}) or {}) or {},
            "solved_values": self._solved_values(values),
            "search_index": _plain(candidate.get("search_index")),
        }
        fingerprint = (
            evaluation["stage"], evaluation["fidelity"], evaluation["model"],
            evaluation["sample_count"], evaluation["search_index"],
        )
        exists = any(
            (
                item.get("stage"), item.get("fidelity"), item.get("model"),
                item.get("sample_count"), item.get("search_index")
            ) == fingerprint
            for item in record["evaluations"]
        )
        if not exists:
            record["evaluations"].append(evaluation)
        return str(record["id"])

    def observe_candidate_for_id(
        self,
        candidate_id: str | None,
        candidate: dict[str, Any],
        *,
        stage: str,
        fidelity: str,
        origin: str = "evaluation",
        model: str | None = None,
        sample_count: int | None = None,
        related_solution_ids: Iterable[str] = (),
        relation: str | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> str | None:
        """Attach an evaluation to an already-canonical solution record.

        This is intentionally different from :meth:`observe_candidate`: callers that
        already know the physical solution identity (for example a DRC boundary row)
        must not let a nested evaluator's partial/aliased variable dictionary silently
        resolve the observation onto another record. Structural aliases and
        inner-solved values remain evaluation/provenance metadata; they do not mutate
        the master structural state, and identity stays with ``candidate_id``.
        """
        record = self.get(candidate_id)
        if record is None:
            return None

        raw_values = dict(candidate.get("variable_values", {}) or {})
        # Identity is already known. Do not let fidelity-specific solved values (or
        # periodic aliases from a nested evaluator) mutate the master solution state.
        #
        # When the evaluator reports a *complete* structural coordinate, however,
        # verify that it really canonicalizes to the record we were asked to attach
        # to.  Targeted observations are used for authoritative promotion where a
        # stale row->registry mapping would otherwise be particularly dangerous: the
        # field metrics could silently land on the wrong physical solution.  Reduced
        # nested evaluators are still allowed to report only a subset of coordinates;
        # their identity is supplied by the already-known parent record.
        structural_ids = set(self.structural_variable_ids)
        if structural_ids and structural_ids.issubset(raw_values):
            observed_structural = self.canonical_structural_values(raw_values)
            expected_structural = dict(record.get("structural_values", {}) or {})
            mismatches = [
                variable_id
                for variable_id in self.structural_variable_ids
                if observed_structural.get(variable_id) != expected_structural.get(variable_id)
            ]
            if mismatches:
                details = ", ".join(
                    f"{variable_id}: expected {expected_structural.get(variable_id)!r}, "
                    f"observed {observed_structural.get(variable_id)!r}"
                    for variable_id in mismatches
                )
                raise ValueError(
                    "Targeted optimizer observation does not match its canonical "
                    f"solution {record['id']}: {details}."
                )

        self.add_provenance(
            str(record["id"]),
            origin=origin,
            stage=stage,
            relation=relation,
            related_solution_ids=related_solution_ids,
            details=provenance_details,
        )
        evaluation = {
            "stage": str(stage),
            "fidelity": str(fidelity),
            "fidelity_rank": int(_fidelity_rank(fidelity)),
            "model": str(model) if model else None,
            "sample_count": int(sample_count) if sample_count is not None else None,
            "feasible": bool(candidate.get("feasible", True)),
            "metrics": _plain(candidate.get("metrics", {}) or {}) or {},
            "currents_a": _plain(candidate.get("currents_a", {}) or {}) or {},
            "solved_values": self._solved_values(raw_values),
            "search_index": _plain(candidate.get("search_index")),
        }
        fingerprint = (
            evaluation["stage"], evaluation["fidelity"], evaluation["model"],
            evaluation["sample_count"], evaluation["search_index"],
        )
        exists = any(
            (
                item.get("stage"), item.get("fidelity"), item.get("model"),
                item.get("sample_count"), item.get("search_index")
            ) == fingerprint
            for item in record["evaluations"]
        )
        if not exists:
            record["evaluations"].append(evaluation)
        return str(record["id"])

    def observe_rejection(
        self,
        values: dict[str, Any],
        *,
        stage: str,
        reason: str,
        origin: str = "evaluation",
        setting_summary: str = "",
    ) -> str:
        record = self.ensure(
            values,
            setting_summary=setting_summary,
            origin=origin,
            stage=stage,
        )
        if str(reason) not in record["rejection_reasons"]:
            record["rejection_reasons"].append(str(reason))
        return str(record["id"])

    def mark_drc(
        self,
        values: dict[str, Any],
        probe: dict[str, Any],
        *,
        stage: str = "geometry-only DRC",
        origin: str = "drc_probe",
        related_solution_ids: Iterable[str] = (),
        relation: str | None = None,
        provenance_details: dict[str, Any] | None = None,
        parent_ids: Iterable[str] = (),
    ) -> str:
        record = self.ensure(
            values,
            origin=origin,
            stage=stage,
            related_solution_ids=related_solution_ids,
            relation=relation,
            provenance_details=provenance_details,
            parent_ids=parent_ids,
        )
        feasible = bool(probe.get("feasible"))
        record["drc"] = {
            "status": "clear" if feasible else "blocked",
            "minimum_clearance_mm": _plain(probe.get("minimum_clearance_mm")),
            "reasons": _plain(list(probe.get("reasons", []) or [])) or [],
        }
        return str(record["id"])

    def mark_drc_for_id(
        self,
        candidate_id: str | None,
        probe: dict[str, Any],
        *,
        stage: str = "geometry-only DRC",
        origin: str = "drc_probe",
        related_solution_ids: Iterable[str] = (),
        relation: str | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> str | None:
        """Attach DRC state to an already-known canonical structural solution."""
        record = self.get(candidate_id)
        if record is None:
            return None
        self.add_provenance(
            str(record["id"]),
            origin=origin,
            stage=stage,
            relation=relation,
            related_solution_ids=related_solution_ids,
            details=provenance_details,
        )
        feasible = bool(probe.get("feasible"))
        record["drc"] = {
            "status": "clear" if feasible else "blocked",
            "minimum_clearance_mm": _plain(probe.get("minimum_clearance_mm")),
            "reasons": _plain(list(probe.get("reasons", []) or [])) or [],
        }
        return str(record["id"])

    def set_retired(self, candidate_id: str | None, reason: str | None) -> None:
        record = self.get(candidate_id)
        if record is not None:
            record["retired_reason"] = str(reason) if reason else None

    def preferred_evaluations(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Return evaluations from the highest available fidelity tier.

        This prevents a rosy scout estimate from continuing to rank a solution after
        a more trustworthy physical/authoritative evaluation exists.
        """
        evaluations = [
            item for item in list(record.get("evaluations", []) or [])
            if isinstance(item.get("metrics"), dict)
        ]
        if not evaluations:
            return []
        best_rank = max(
            int(item.get("fidelity_rank", _fidelity_rank(item.get("fidelity"))))
            for item in evaluations
        )
        return [
            item for item in evaluations
            if int(item.get("fidelity_rank", _fidelity_rank(item.get("fidelity")))) == best_rank
        ]

    def lifecycle_state(self, record: dict[str, Any]) -> dict[str, Any]:
        evaluations = list(record.get("evaluations", []) or [])
        preferred = self.preferred_evaluations(record)
        has_authoritative = any(
            _fidelity_rank(item.get("fidelity")) >= _FIDELITY_PRIORITY["authoritative"]
            for item in evaluations
        )
        magnetic_state = "evaluated" if evaluations else (
            "rejected" if record.get("rejection_reasons") else "unevaluated"
        )
        if preferred and not any(bool(item.get("feasible", True)) for item in preferred):
            magnetic_state = "rejected"
        return {
            "magnetic": magnetic_state,
            "drc": str((record.get("drc", {}) or {}).get("status", "unknown")),
            "authoritative": "evaluated" if has_authoritative else "pending",
            "active_frontier": record.get("retired_reason") is None,
            "retired_reason": record.get("retired_reason"),
        }

    def lifecycle_counts(self) -> dict[str, int]:
        """Return coherent whole-registry counts for live optimizer progress.

        ``active``, ``retired`` and ``blocked`` are mutually exclusive current
        lifecycle buckets, so together they always equal ``discovered``. A
        solution is *blocked* when DRC or an evaluated hard gate has rejected it;
        otherwise an explicit retirement reason moves it out of the active
        frontier. ``authoritative`` is an independent fidelity tally and may
        overlap any lifecycle bucket.

        These counts intentionally describe the durable master solution
        population, never a stage-local evaluation batch.
        """
        discovered = int(len(self._records))
        active = 0
        retired = 0
        blocked = 0
        authoritative = 0
        for record in self._records:
            state = self.lifecycle_state(record)
            is_blocked = (
                str(state.get("drc", "unknown")) == "blocked"
                or str(state.get("magnetic", "unevaluated")) == "rejected"
                or bool(record.get("rejection_reasons"))
            )
            if is_blocked:
                blocked += 1
            elif record.get("retired_reason") is not None:
                retired += 1
            else:
                active += 1
            if str(state.get("authoritative", "pending")) == "evaluated":
                authoritative += 1
        return {
            "solutions_discovered": discovered,
            "solutions_active": int(active),
            "solutions_retired": int(retired),
            "solutions_blocked": int(blocked),
            "solutions_authoritative": int(authoritative),
        }

    def tag(self, candidate_id: str | None, *tags: str) -> None:
        record = self.get(candidate_id)
        if record is None:
            return
        for tag in map(str, tags):
            if tag and tag not in record["tags"]:
                record["tags"].append(tag)

    def add_to_view(self, view: str, candidate_id: str | None) -> None:
        if not candidate_id:
            return
        bucket = self.views.setdefault(str(view), [])
        candidate_id = str(candidate_id)
        if candidate_id not in bucket:
            bucket.append(candidate_id)

    def import_payload(self, payload: dict[str, Any], *, stage_prefix: str = "scout") -> dict[str, str]:
        """Merge another flat registry and return old-id -> canonical local-id mapping.

        Schema-v1 tree lineage is accepted and converted into provenance references;
        schema-v2 provenance is translated through the merged canonical IDs.
        """
        id_map: dict[str, str] = {}
        records = list((payload or {}).get("candidates", []) or [])
        for incoming in records:
            values = dict(incoming.get("structural_values", {}) or incoming.get("variable_values", {}) or {})
            record = self.ensure(
                values,
                setting_summary=str(incoming.get("setting_summary", "")),
                origin="nested_scout",
                stage=stage_prefix,
            )
            id_map[str(incoming.get("id", ""))] = str(record["id"])
            for evaluation in list(incoming.get("evaluations", []) or []):
                imported = copy.deepcopy(evaluation)
                imported["stage"] = f"{stage_prefix}: {str(imported.get('stage', '')).strip()}".strip()
                imported["fidelity_rank"] = int(
                    imported.get("fidelity_rank", _fidelity_rank(imported.get("fidelity")))
                )
                if imported not in record["evaluations"]:
                    record["evaluations"].append(imported)
            incoming_drc = incoming.get("drc", {}) or {}
            if incoming_drc.get("status") in {"clear", "blocked"}:
                record["drc"] = _plain(incoming_drc) or record["drc"]
            for reason in list(incoming.get("rejection_reasons", []) or []):
                if str(reason) not in record["rejection_reasons"]:
                    record["rejection_reasons"].append(str(reason))
            for tag in list(incoming.get("tags", []) or []):
                self.tag(record["id"], str(tag))
            if incoming.get("retired_reason") and not record.get("retired_reason"):
                record["retired_reason"] = str(incoming.get("retired_reason"))

        # Translate provenance after every physical solution has a local canonical ID.
        for incoming in records:
            new_id = id_map.get(str(incoming.get("id", "")))
            if not new_id:
                continue
            provenance = list(incoming.get("provenance", []) or [])
            # v1 compatibility: flatten parent ownership into a derived-from event.
            if not provenance and incoming.get("origins"):
                for item in list(incoming.get("origins", []) or []):
                    provenance.append({
                        "origin": item.get("origin", "nested_scout"),
                        "stage": item.get("stage", ""),
                    })
            if incoming.get("parent_ids"):
                provenance.append({
                    "origin": "legacy_lineage",
                    "stage": stage_prefix,
                    "relation": "derived_from",
                    "related_solution_ids": list(incoming.get("parent_ids", []) or []),
                })
            for event in provenance:
                related = [
                    id_map.get(str(old_id), str(old_id))
                    for old_id in list(event.get("related_solution_ids", []) or [])
                ]
                self.add_provenance(
                    new_id,
                    origin=str(event.get("origin", "nested_scout")),
                    stage=f"{stage_prefix}: {str(event.get('stage', '')).strip()}".strip(),
                    relation=(str(event.get("relation")) if event.get("relation") else None),
                    related_solution_ids=related,
                    details=dict(event.get("details", {}) or {}),
                )
        return id_map

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def payload(self) -> dict[str, Any]:
        records = copy.deepcopy(self._records)
        for record in records:
            original = self.get(record.get("id")) or record
            record["lifecycle"] = self.lifecycle_state(original)
            preferred = self.preferred_evaluations(original)
            record["preferred_fidelity"] = (
                str(preferred[0].get("fidelity", "")) if preferred else None
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "candidate_count": int(len(records)),
            "solution_count": int(len(records)),
            "lifecycle_counts": self.lifecycle_counts(),
            "structural_variable_ids": list(self.structural_variable_ids),
            "structural_variables": copy.deepcopy(list(self.structural_variables)),
            "canonicalization": "structural-only + step-snapped + periodic-angle-normalized; inner-solved values live per evaluation",
            "views": copy.deepcopy(self.views),
            "candidates": records,
        }
