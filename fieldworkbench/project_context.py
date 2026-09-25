"""Project-scoped suite context and resource registry.

Project state is independent of any particular editor window. The existing
``StudioAdapter`` remains the native data owner and ``.magpy.json`` remains the
on-disk container; this module supplies stable project/resource identities and
the common context shared by suite applications.
"""

from __future__ import annotations

import copy
import hashlib
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


RESOURCE_MODEL_OBJECT = "model_object"
RESOURCE_SNAPSHOT = "snapshot"
RESOURCE_SAVED_MAP = "saved_map"
RESOURCE_EXPOSURE_ANALYSIS = "exposure_analysis"
RESOURCE_EXPOSURE = "exposure"
RESOURCE_DATASET = "dataset"
RESOURCE_CALIBRATION = "calibration"
RESOURCE_WAVEFORM = "waveform"


@dataclass(frozen=True)
class ProjectResource:
    """One addressable resource exposed by a :class:`ProjectContext`.

    ``key`` is suite-scoped and namespaced by resource kind, while ``native_id``
    is the identifier already stored in the current native scene document.
    The registry can expose native project resources added by later suite phases.
    """

    key: str
    kind: str
    native_id: str
    name: str
    persistent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ProjectResourceRegistry:
    """Build stable views over the resources currently stored by one adapter."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._resources: tuple[ProjectResource, ...] = ()
        self._signature: tuple[Any, ...] = ()
        self.refresh()

    @staticmethod
    def resource_key(kind: str, native_id: str) -> str:
        return f"{str(kind).strip()}:{str(native_id).strip()}"

    @staticmethod
    def _fallback_native_id(kind: str, index: int, payload: Any) -> str:
        """Return a deterministic read-only identity for malformed resource data.

        Current Workbench objects/exposures/datasets/snapshots already carry
        native IDs.  This fallback lets the suite tree remain addressable if an
        older or partially damaged optional resource lacks one, without writing
        a synthetic ID back into the scene merely by opening it.
        """

        digest = hashlib.sha1(
            repr((kind, index, payload)).encode("utf-8", errors="replace")
        ).hexdigest()[:12]
        return f"fallback-{digest}"

    def _native_id(self, kind: str, index: int, payload: dict[str, Any]) -> str:
        value = str(payload.get("id", "")).strip()
        return value or self._fallback_native_id(kind, index, payload)

    def _object_resources(self) -> Iterable[ProjectResource]:
        try:
            rows = self.adapter.list_objects()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_MODEL_OBJECT, index, row)
            label = str(row.get("label") or native_id or "Object")
            try:
                type_label = self.adapter.object_type_label(native_id)
            except Exception:
                type_label = str(row.get("type") or "Object")
            yield ProjectResource(
                key=self.resource_key(RESOURCE_MODEL_OBJECT, native_id),
                kind=RESOURCE_MODEL_OBJECT,
                native_id=native_id,
                name=label,
                metadata={"type": str(row.get("type", "")), "type_label": str(type_label)},
            )

    def _snapshot_resources(
        self, *, live_status: bool = False
    ) -> Iterable[ProjectResource]:
        # list_snapshots() computes freshness fingerprints and can be expensive.
        # Use stored snapshot records for normal registry refreshes and request
        # live status only for explicit/event-driven dashboard rebuilds.
        if live_status:
            try:
                rows = self.adapter.list_snapshots()
            except Exception:
                rows = []
        else:
            stored = getattr(self.adapter, "snapshots", [])
            rows = [copy.deepcopy(item) for item in stored if isinstance(item, dict)]
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_SNAPSHOT, index, row)
            yield ProjectResource(
                key=self.resource_key(RESOURCE_SNAPSHOT, native_id),
                kind=RESOURCE_SNAPSHOT,
                native_id=native_id,
                name=str(row.get("name") or native_id or "Snapshot"),
                metadata={
                    "status": str(row.get("status") or ""),
                    "source_object_id": str(row.get("source_object_id") or ""),
                },
            )

    def _saved_map_resources(self) -> Iterable[ProjectResource]:
        try:
            rows = self.adapter.list_saved_maps()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_SAVED_MAP, index, row)
            yield ProjectResource(
                key=self.resource_key(RESOURCE_SAVED_MAP, native_id),
                kind=RESOURCE_SAVED_MAP,
                native_id=native_id,
                name=str(row.get("name") or native_id or "Saved map"),
                metadata=copy.deepcopy(row),
            )

    def _exposure_analysis_resources(self) -> Iterable[ProjectResource]:
        # Avoid live freshness hashing during ordinary project-registry refreshes.
        rows = getattr(self.adapter, "exposure_analyses", [])
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_EXPOSURE_ANALYSIS, index, row)
            metadata = {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key != "payload_b64"
            }
            yield ProjectResource(
                key=self.resource_key(RESOURCE_EXPOSURE_ANALYSIS, native_id),
                kind=RESOURCE_EXPOSURE_ANALYSIS,
                native_id=native_id,
                name=str(row.get("name") or native_id or "Exposure analysis"),
                metadata=metadata,
            )

    def _exposure_resources(self) -> Iterable[ProjectResource]:
        try:
            rows = self.adapter.list_exposures()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_EXPOSURE, index, row)
            yield ProjectResource(
                key=self.resource_key(RESOURCE_EXPOSURE, native_id),
                kind=RESOURCE_EXPOSURE,
                native_id=native_id,
                name=str(row.get("name") or native_id or "Exposure"),
                metadata=copy.deepcopy(row),
            )

    def _dataset_resources(self) -> Iterable[ProjectResource]:
        try:
            rows = self.adapter.list_measurement_datasets()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_DATASET, index, row)
            name = str(
                row.get("name")
                or row.get("label")
                or row.get("sensor_label")
                or native_id
                or "Dataset"
            )
            yield ProjectResource(
                key=self.resource_key(RESOURCE_DATASET, native_id),
                kind=RESOURCE_DATASET,
                native_id=native_id,
                name=name,
                metadata=copy.deepcopy(row),
            )

    def _calibration_resources(self) -> Iterable[ProjectResource]:
        try:
            rows = self.adapter.list_measurement_calibrations()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_CALIBRATION, index, row)
            yield ProjectResource(
                key=self.resource_key(RESOURCE_CALIBRATION, native_id),
                kind=RESOURCE_CALIBRATION,
                native_id=native_id,
                name=str(row.get("label") or native_id or "Calibration"),
                metadata=copy.deepcopy(row),
            )

    def _waveform_resources(self) -> Iterable[ProjectResource]:
        """Expose first-class waveform resources stored with the project."""
        try:
            rows = self.adapter.list_waveforms()
        except Exception:
            rows = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            native_id = self._native_id(RESOURCE_WAVEFORM, index, row)
            yield ProjectResource(
                key=self.resource_key(RESOURCE_WAVEFORM, native_id),
                kind=RESOURCE_WAVEFORM,
                native_id=native_id,
                name=str(row.get("name") or native_id or "Waveform"),
                metadata=copy.deepcopy(row),
            )


    def refresh(
        self, *, live_snapshot_status: bool = False
    ) -> tuple[ProjectResource, ...]:
        resources = tuple(
            list(self._object_resources())
            + list(self._snapshot_resources(live_status=live_snapshot_status))
            + list(self._saved_map_resources())
            + list(self._exposure_analysis_resources())
            + list(self._exposure_resources())
            + list(self._dataset_resources())
            + list(self._calibration_resources())
            + list(self._waveform_resources())
        )
        self._resources = resources
        self._signature = tuple(
            (
                item.key,
                item.name,
                item.persistent,
                tuple(sorted((str(key), repr(value)) for key, value in item.metadata.items())),
            )
            for item in resources
        )
        return resources

    def all(
        self, *, refresh: bool = False, live_snapshot_status: bool = False
    ) -> tuple[ProjectResource, ...]:
        if refresh:
            self.refresh(live_snapshot_status=live_snapshot_status)
        return self._resources

    def by_kind(self, kind: str, *, refresh: bool = False) -> tuple[ProjectResource, ...]:
        resources = self.all(refresh=refresh)
        requested = str(kind)
        return tuple(item for item in resources if item.kind == requested)

    def get(self, key: str, *, refresh: bool = False) -> ProjectResource | None:
        for item in self.all(refresh=refresh):
            if item.key == str(key):
                return item
        return None

    def counts(self, *, refresh: bool = False) -> dict[str, int]:
        resources = self.all(refresh=refresh)
        result: dict[str, int] = {}
        for item in resources:
            result[item.kind] = result.get(item.kind, 0) + 1
        return result

    def signature(self, *, refresh: bool = True) -> tuple[Any, ...]:
        if refresh:
            self.refresh()
        return self._signature


class ProjectContext:
    """Project-scoped state shared by suite applications.

    The context is intentionally UI-toolkit agnostic. It owns the shared
    adapter/path/save baseline and tracks top-level windows weakly so every suite
    application receives project state directly.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        path: str | Path | None = None,
        untitled_number: int = 1,
        saved_signature: str | None = None,
        project_id: str | None = None,
        display_label: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.path = None if path is None else Path(path).expanduser().resolve()
        self.untitled_number = max(1, int(untitled_number))
        self.project_id = str(project_id or uuid4().hex)
        self.display_label = str(display_label or "").strip()
        self.saved_signature = (
            str(saved_signature)
            if saved_signature is not None
            else str(adapter.signature())
        )
        self.resources = ProjectResourceRegistry(adapter)
        self._windows: dict[str, list[weakref.ReferenceType[Any]]] = {}

    def display_name(self) -> str:
        if self.path is not None:
            return self.path.name
        if self.display_label:
            return self.display_label
        return "Untitled" if self.untitled_number <= 1 else f"Untitled {self.untitled_number}"

    def is_dirty(self) -> bool:
        try:
            return str(self.adapter.signature()) != self.saved_signature
        except Exception:
            return False

    def update_document_state(
        self,
        *,
        path: str | Path | None,
        saved_signature: str,
    ) -> None:
        self.path = None if path is None else Path(path).expanduser().resolve()
        self.saved_signature = str(saved_signature)

    def mark_saved(self, path: str | Path | None = None) -> None:
        if path is not None:
            self.path = Path(path).expanduser().resolve()
        self.saved_signature = str(self.adapter.signature())

    def register_window(self, role: str, window: Any) -> None:
        """Associate a live suite window with this project without owning it."""

        role_name = str(role or "window")
        self._prune_windows(role_name)
        try:
            reference = weakref.ref(window)
        except TypeError:
            return
        existing = self._windows.setdefault(role_name, [])
        if any(item() is window for item in existing):
            return
        existing.append(reference)
        # Lightweight attributes are useful to future suite windows and tests,
        # and do not impose any Qt dependency on this module.
        try:
            setattr(window, "project_context", self)
            setattr(window, "project_id", self.project_id)
            setattr(window, "project_role", role_name)
        except Exception:
            pass

    def _prune_windows(self, role: str | None = None) -> None:
        roles = [str(role)] if role is not None else list(self._windows)
        for role_name in roles:
            live = [reference for reference in self._windows.get(role_name, []) if reference() is not None]
            if live:
                self._windows[role_name] = live
            else:
                self._windows.pop(role_name, None)

    def windows(self, role: str | None = None) -> tuple[Any, ...]:
        self._prune_windows(role)
        if role is not None:
            return tuple(
                window
                for reference in self._windows.get(str(role), [])
                if (window := reference()) is not None
            )
        result: list[Any] = []
        seen: set[int] = set()
        for references in self._windows.values():
            for reference in references:
                window = reference()
                if window is None or id(window) in seen:
                    continue
                seen.add(id(window))
                result.append(window)
        return tuple(result)
