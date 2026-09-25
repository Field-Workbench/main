"""Single-signal waveform workspace used by the Field Workbench suite.

The Waveform Editor presents waveforms in physical electrical terms.  The
sampled trace remains an internal playback representation; normal authoring is
performed through embedded waveform tools rather than a raw sample table.
Playback remains a floating-point physical waveform. Imported DAC chunks may
also retain their original unsigned codes and translation settings in the
authoring recipe so those chunks can be reopened and edited losslessly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...waveform_io import WaveformDocument, WaveformIOError, WaveformSeries
from ...waveform_recipe import (
    AUTHORING_RECIPE_KEY,
    generated_segment_from_settings,
    make_authoring_recipe,
    normalize_authoring_recipe,
    recipe_is_exact,
    sampled_segment,
    update_recipe_signal,
)
from ..core.native import (
    ComposedTrace,
    GeneratedTrace,
    NativeWaveformError,
    compose_generated_sequence,
    compose_generated_trace,
    trace_from_recipe_segment,
)
from .preview import WaveformPreviewCanvas


@dataclass
class _SignalState:
    """Mutable working copy of the one electrical signal being edited."""

    name: str
    time_s: list[float]
    values: list[float]
    quantity: str = "current"
    unit: str = "mA"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_series(cls, series: WaveformSeries) -> "_SignalState":
        return cls(
            name=str(series.name),
            time_s=[float(value) for value in series.time_s],
            values=[float(value) for value in series.values],
            quantity=str(series.quantity),
            unit=str(series.unit),
            metadata=dict(series.metadata),
        )

    def to_series(self) -> WaveformSeries:
        return WaveformSeries(
            self.name,
            self.time_s,
            self.values,
            quantity=self.quantity,
            unit=self.unit,
            metadata=dict(self.metadata or {}),
        )


class NativeWaveformEditor(QWidget):
    """Hold one native waveform working copy and its persistent graph.

    This widget intentionally contains no raw time/value editor.  The user
    authors signals through embedded Square, Sine, DC / pause, Action Potential,
    or Free Form tools (or imports an external waveform); this class only owns the
    resulting physical trace, metadata, and preview.
    """

    documentChanged = Signal()
    segmentEditRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._signal: _SignalState | None = None
        self._metadata: dict[str, Any] = {}
        self._source_format = ""
        self._source_path = ""
        self._loading = False
        self._dirty = False
        self._build_ui()
        self.new_document()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.plot = WaveformPreviewCanvas(self, empty_text="No waveform loaded")
        # Keep help local to each modebar button.  A tooltip on the canvas itself
        # covers most of the editor whenever the pointer pauses anywhere over the
        # plot, which is distracting and unlike the Plotly viewers we mirror.
        self.plot.setToolTip("")
        self.plot.segmentReordered.connect(self.reorder_recipe_segment)
        self.plot.segmentDoubleClicked.connect(self.segmentEditRequested)
        root.addWidget(self.plot, 1)

    def show_empty_preview(self) -> None:
        """Clear the graph when the project catalogue contains no waveform."""
        self._loading = True
        try:
            self._signal = None
            self._metadata = {}
            self._source_format = ""
            self._source_path = ""
            self._refresh_signal()
            self._dirty = False
        finally:
            self._loading = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _mark_changed(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self.documentChanged.emit()

    def _signal_state(self) -> _SignalState | None:
        return self._signal

    def _refresh_signal(self) -> None:
        self.plot.set_signal(self._signal)
        self._refresh_segments()

    def _refresh_segments(self) -> None:
        try:
            recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
            segments = recipe.get("segments", []) if recipe.get("integrity") == "exact" else []
        except Exception:
            segments = []
        self.plot.set_segments(segments)

    def set_recipe_editable(self, editable: bool) -> None:
        self.plot.set_segments_editable(editable)

    def _trace_changed(self) -> None:
        """Refresh after embedded-tool-driven trace changes."""
        self._refresh_segments()
        self.plot.update()
        self._mark_changed()

    def set_current_signal_definition(self, quantity: str, unit: str) -> None:
        """Update the working waveform's electrical type and display unit."""
        signal = self._signal_state()
        if signal is None:
            return
        quantity = str(quantity or "current").strip().lower()
        if quantity not in {"current", "voltage"}:
            raise WaveformIOError("Waveform signals must be Current or Voltage.")
        unit = str(unit or "").strip()
        changed = (signal.quantity, signal.unit) != (quantity, unit)
        signal.quantity = quantity
        signal.unit = unit
        if changed:
            recipe = update_recipe_signal(
                self._metadata.get(AUTHORING_RECIPE_KEY),
                quantity=quantity,
                unit=unit,
            )
            if recipe is not None:
                self._metadata[AUTHORING_RECIPE_KEY] = recipe
            self.plot.update()
            self._mark_changed()

    def new_document(self) -> None:
        """Create the internal starter trace for a new unsaved waveform.

        The two-point zero trace is not an authoring mode or a user-editable
        data table; it merely gives the information panel and builders a valid
        physical signal context until the first builder/import replaces it.
        """
        self._loading = True
        try:
            self._signal = _SignalState("Waveform", [0.0, 0.001], [0.0, 0.0], "current", "mA")
            self._metadata = {"created_by": "Field Workbench Waveform Editor"}
            self._source_format = ""
            self._source_path = ""
            self._refresh_signal()
            self._dirty = False
        finally:
            self._loading = False

    def load_document(self, document: WaveformDocument) -> None:
        self._loading = True
        try:
            self._signal = _SignalState.from_series(document.signal)
            self._metadata = dict(document.metadata)
            self._source_format = str(document.source_format)
            self._source_path = str(document.source_path)
            self._refresh_signal()
            self._dirty = False
        finally:
            self._loading = False

    def document(self, *, title: str = "") -> WaveformDocument:
        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        return WaveformDocument(
            [signal.to_series()],
            title=str(title).strip(),
            metadata=dict(self._metadata),
            source_format=self._source_format,
            source_path=self._source_path,
        )

    def current_signal(self) -> WaveformSeries:
        """Return the working waveform as one validated native signal."""
        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        return signal.to_series()

    def apply_generated_trace(self, trace: GeneratedTrace, *, append: bool = False) -> None:
        """Apply the exact physical trace shown by the unified builder.

        Replace and Append use the same composition helper as the live preview.
        Each chunk keeps its own internal representation; no user-visible or
        inherited timing resolution is required at a join.  Composition and
        its authoring recipe are validated before the working copy is changed.
        """
        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")

        first_chunk = append and self._is_placeholder()
        append = append and not first_chunk
        existing_start_s = float(signal.time_s[0])
        try:
            composed = compose_generated_trace(
                trace,
                append=append,
                existing_time_s=signal.time_s if append else None,
                existing_values=signal.values if append else None,
            )
        except NativeWaveformError as error:
            raise WaveformIOError(str(error)) from error

        boundary = composed.append_boundary_s
        recipe_boundary = 0.0 if boundary is None else boundary - existing_start_s
        segment = generated_segment_from_settings(
            dict(trace.settings),
            start_s=recipe_boundary if append else 0.0,
        )

        if append:
            existing_recipe = self._metadata.get(AUTHORING_RECIPE_KEY)
            try:
                recipe = normalize_authoring_recipe(existing_recipe) if recipe_is_exact(existing_recipe) else None
            except Exception:
                recipe = None
            if recipe is not None and recipe_is_exact(recipe):
                segments = [dict(item) for item in recipe.get("segments", [])]
            else:
                if recipe_boundary <= 0.0:
                    raise WaveformIOError("Existing waveform must have positive duration before a segment can be appended.")
                segments = [sampled_segment(
                    signal.time_s,
                    signal.values,
                    start_s=0.0,
                    description="Existing waveform",
                )]
            segments.append(segment)
            recipe = make_authoring_recipe(
                segments,
                quantity=signal.quantity,
                unit=signal.unit,
            )
        else:
            recipe = make_authoring_recipe(
                [segment],
                quantity=signal.quantity,
                unit=signal.unit,
            )

        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        if not append:
            self._source_format = ""
            self._source_path = ""
        self._metadata[AUTHORING_RECIPE_KEY] = recipe
        self._metadata["created_by"] = "Field Workbench Waveform Editor"
        self._trace_changed()

    def _is_placeholder(self) -> bool:
        signal = self._signal_state()
        return bool(
            signal is not None
            and AUTHORING_RECIPE_KEY not in self._metadata
            and not self._source_format
            and signal.time_s == [0.0, 0.001]
            and signal.values == [0.0, 0.0]
        )

    def is_placeholder(self) -> bool:
        """Return whether the working signal is still the untouched new-waveform stub."""
        return self._is_placeholder()

    def preview_appended_trace(self, trace: GeneratedTrace) -> ComposedTrace:
        signal = self.current_signal()
        if self._is_placeholder():
            return compose_generated_trace(trace, append=False)
        return compose_generated_trace(
            trace,
            append=True,
            existing_time_s=signal.time_s,
            existing_values=signal.values,
        )

    def show_generated_preview(
        self,
        trace: GeneratedTrace,
        *,
        segment_index: int | None = None,
    ) -> tuple[ComposedTrace, float | None]:
        """Show a non-destructive generator preview in the persistent main plot.

        The working signal and authoring recipe are not modified.  Generator
        controls can therefore update this preview continuously and Apply/Cancel
        can either commit or discard the proposal without a second viewer.
        """
        signal = self.current_signal()
        if segment_index is None:
            composed = self.preview_appended_trace(trace)
            boundary = composed.append_boundary_s
        else:
            composed = self.preview_generated_recipe_segment(int(segment_index), trace)
            boundary = None
        self.plot.set_trace(
            composed.time_s,
            composed.values,
            unit=signal.unit,
            quantity=signal.quantity,
            append_boundary_s=boundary,
        )
        # The committed chunk geometry can be stale while a segment duration is
        # being edited, so suppress drag/reorder overlays until Apply or Cancel.
        self.plot.set_segments([])
        self.plot.set_segments_editable(False)
        return composed, boundary

    def focus_generated_preview(
        self,
        trace: GeneratedTrace,
        *,
        segment_index: int | None = None,
    ) -> None:
        """Focus the persistent viewer around one generated preview component.

        Millisecond-scale components can be effectively invisible when appended
        to a waveform lasting several seconds.  This helper temporarily frames
        the proposed component with a little neighboring context while keeping
        Home available to restore the whole waveform.
        """

        signal = self.current_signal()
        duration = float(trace.settings.get("duration_s", 0.0) or 0.0)
        if duration <= 0.0:
            return

        if segment_index is None:
            start_s = (
                float(signal.time_s[0])
                if self._is_placeholder()
                else float(signal.time_s[-1])
            )
        else:
            start_s = float(signal.time_s[0])
            try:
                recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
                segments = list(recipe.get("segments", []))
                if 0 <= int(segment_index) < len(segments):
                    start_s += float(segments[int(segment_index)].get("start_s", 0.0) or 0.0)
            except Exception:
                return

        self.plot.focus_time_range(
            start_s,
            start_s + duration,
            context_before_s=duration * 2.0,
            context_after_s=duration * 0.5,
        )

    def show_document_preview(self, document: WaveformDocument) -> tuple[ComposedTrace, float | None]:
        """Show one imported signal as a non-destructive prospective append.

        The imported signal keeps every physical source sample, but its signal
        type and unit must already match the waveform-level encoding selected
        before authoring began. Even the first imported chunk cannot redefine
        the waveform encoding.
        """
        signal = self._signal_state()
        incoming = document.signal
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        if (incoming.quantity, incoming.unit) != (signal.quantity, signal.unit):
            raise WaveformIOError("Imported components must use the waveform's confirmed signal type and units.")
        trace = GeneratedTrace(
            tuple(float(value) for value in incoming.time_s),
            tuple(float(value) for value in incoming.values),
            {"shape": "sampled", "duration_s": float(incoming.time_s[-1] - incoming.time_s[0])},
        )
        composed = self.preview_appended_trace(trace)
        boundary = composed.append_boundary_s
        self.plot.set_trace(
            composed.time_s,
            composed.values,
            unit=str(signal.unit),
            quantity=str(signal.quantity),
            append_boundary_s=boundary,
        )
        self.plot.set_segments([])
        self.plot.set_segments_editable(False)
        return composed, boundary

    def restore_working_preview(self, *, editable: bool = False) -> None:
        """Restore the persistent viewer to the committed working waveform."""
        self._refresh_signal()
        self.plot.set_segments_editable(bool(editable))

    def append_document(self, document: WaveformDocument, *, description: str = "Imported waveform") -> None:
        """Append an imported signal as one lossless, reorderable recipe chunk."""
        signal = self._signal_state()
        incoming = document.signal
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        if (incoming.quantity, incoming.unit) != (signal.quantity, signal.unit):
            raise WaveformIOError("Imported components must use the waveform's confirmed signal type and units.")
        trace = GeneratedTrace(
            tuple(float(value) for value in incoming.time_s),
            tuple(float(value) for value in incoming.values),
            {"shape": "sampled", "duration_s": float(incoming.time_s[-1] - incoming.time_s[0])},
        )
        first_chunk = self._is_placeholder()
        composed = self.preview_appended_trace(trace)
        if first_chunk:
            segments: list[dict[str, Any]] = []
            start_s = 0.0
        else:
            try:
                recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
                segments = [dict(item) for item in recipe["segments"]]
            except Exception:
                duration = float(signal.time_s[-1] - signal.time_s[0])
                segments = [sampled_segment(signal.time_s, signal.values, start_s=0.0, description="Existing waveform")]
            start_s = float(signal.time_s[-1] - signal.time_s[0])
        segments.append(sampled_segment(incoming.time_s, incoming.values, start_s=start_s, description=description))
        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        self._metadata[AUTHORING_RECIPE_KEY] = make_authoring_recipe(
            segments, quantity=signal.quantity, unit=signal.unit
        )
        self._source_format = ""
        self._source_path = ""
        self._trace_changed()

    def show_recipe_segment_preview(
        self,
        segment: dict[str, Any],
        *,
        segment_index: int | None = None,
    ) -> tuple[ComposedTrace, float | None]:
        """Preview one exact authored chunk without changing the working copy."""

        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        try:
            canonical = make_authoring_recipe(
                [{**dict(segment), "start_s": 0.0}],
                quantity=signal.quantity,
                unit=signal.unit,
            )["segments"][0]
            trace = trace_from_recipe_segment(canonical)
        except Exception as error:
            raise WaveformIOError(f"Unable to preview authored waveform segment: {error}") from error

        if segment_index is None:
            composed = self.preview_appended_trace(trace)
            boundary = composed.append_boundary_s
        else:
            composed, _recipe = self._compose_recipe_segment_edit(int(segment_index), canonical)
            boundary = None
        self.plot.set_trace(
            composed.time_s,
            composed.values,
            unit=signal.unit,
            quantity=signal.quantity,
            append_boundary_s=boundary,
        )
        self.plot.set_segments([])
        self.plot.set_segments_editable(False)
        return composed, boundary

    def append_recipe_segment(self, segment: dict[str, Any]) -> None:
        """Append a canonical editable recipe chunk such as an imported DAC."""

        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        try:
            canonical = make_authoring_recipe(
                [{**dict(segment), "start_s": 0.0}],
                quantity=signal.quantity,
                unit=signal.unit,
            )["segments"][0]
            trace = trace_from_recipe_segment(canonical)
        except Exception as error:
            raise WaveformIOError(f"Unable to append authored waveform segment: {error}") from error

        first_chunk = self._is_placeholder()
        composed = self.preview_appended_trace(trace)
        if first_chunk:
            segments: list[dict[str, Any]] = []
            start_s = 0.0
        else:
            try:
                recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
                segments = [dict(item) for item in recipe["segments"]]
            except Exception:
                duration = float(signal.time_s[-1] - signal.time_s[0])
                segments = [
                    sampled_segment(
                        signal.time_s,
                        signal.values,
                        start_s=0.0,
                        description="Existing waveform",
                    )
                ]
            start_s = float(signal.time_s[-1] - signal.time_s[0])
        canonical["start_s"] = start_s
        segments.append(canonical)
        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        self._metadata[AUTHORING_RECIPE_KEY] = make_authoring_recipe(
            segments,
            quantity=signal.quantity,
            unit=signal.unit,
        )
        self._source_format = ""
        self._source_path = ""
        self._trace_changed()

    def reorder_recipe_segment(self, source_index: int, target_index: int) -> None:
        """Move one authored chunk and rebuild the physical waveform."""
        signal = self._signal_state()
        if signal is None:
            return
        try:
            recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
            segments = [dict(item) for item in recipe["segments"]]
            source = int(source_index)
            target = int(target_index)
            if source == target:
                return
            moved = segments.pop(source)
            segments.insert(target, moved)
        except Exception:
            return
        try:
            self._replace_recipe_segments(segments)
        except WaveformIOError:
            return

    def _replace_recipe_segments(self, segments: list[dict[str, Any]]) -> None:
        """Replace the exact authored chunk sequence and rebuild its physical trace.

        Chunk order is the authoring source of truth.  Start times are therefore
        recalculated after every reorder/clone/delete instead of copied from the
        old recipe.  Deleting the final chunk returns to the ordinary untouched
        authoring stub while retaining the waveform-level signal definition.
        """

        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")

        if not segments:
            signal.time_s[:] = [0.0, 0.001]
            signal.values[:] = [0.0, 0.0]
            self._metadata.pop(AUTHORING_RECIPE_KEY, None)
            self._source_format = ""
            self._source_path = ""
            self._metadata["created_by"] = "Field Workbench Waveform Editor"
            self._trace_changed()
            return

        rebuilt: list[dict[str, Any]] = []
        traces: list[GeneratedTrace] = []
        start_s = 0.0
        try:
            for index, original in enumerate(segments):
                item = copy.deepcopy(dict(original))
                item["start_s"] = start_s
                canonical = make_authoring_recipe(
                    [item],
                    quantity=signal.quantity,
                    unit=signal.unit,
                )["segments"][0]
                trace = trace_from_recipe_segment(canonical)
                rebuilt.append(canonical)
                traces.append(trace)
                start_s += float(canonical["duration_s"])
            composed = compose_generated_sequence(traces)
            updated = make_authoring_recipe(
                rebuilt,
                quantity=signal.quantity,
                unit=signal.unit,
            )
        except Exception as error:
            raise WaveformIOError(f"Unable to rebuild waveform components: {error}") from error

        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        self._metadata[AUTHORING_RECIPE_KEY] = updated
        self._source_format = ""
        self._source_path = ""
        self._metadata["created_by"] = "Field Workbench Waveform Editor"
        self._trace_changed()

    def delete_recipe_segment(self, segment_index: int) -> None:
        """Delete one exact authored chunk and rebuild the remaining waveform."""

        recipe = self.exact_authoring_recipe()
        if recipe is None:
            raise WaveformIOError("This waveform does not have an exact authored component recipe to edit.")
        segments = [copy.deepcopy(dict(item)) for item in recipe.get("segments", [])]
        index = int(segment_index)
        if index < 0 or index >= len(segments):
            raise WaveformIOError("The authored waveform component no longer exists.")
        segments.pop(index)
        self._replace_recipe_segments(segments)

    def clone_recipe_segment(self, segment_index: int) -> int:
        """Duplicate one exact authored chunk immediately after the source chunk."""

        recipe = self.exact_authoring_recipe()
        if recipe is None:
            raise WaveformIOError("This waveform does not have an exact authored component recipe to edit.")
        segments = [copy.deepcopy(dict(item)) for item in recipe.get("segments", [])]
        index = int(segment_index)
        if index < 0 or index >= len(segments):
            raise WaveformIOError("The authored waveform component no longer exists.")
        segments.insert(index + 1, copy.deepcopy(segments[index]))
        self._replace_recipe_segments(segments)
        return index + 1

    def exact_generated_recipe(self) -> dict[str, Any] | None:
        """Return the current exact, fully generated authoring recipe if editable."""
        value = self._metadata.get(AUTHORING_RECIPE_KEY)
        try:
            recipe = normalize_authoring_recipe(value)
        except Exception:
            return None
        if recipe.get("integrity") != "exact":
            return None
        if not recipe.get("segments") or any(segment.get("kind") != "generator" for segment in recipe["segments"]):
            return None
        return recipe

    def exact_authoring_recipe(self) -> dict[str, Any] | None:
        """Return any exact chunk recipe, including lossless imported chunks."""
        try:
            recipe = normalize_authoring_recipe(self._metadata.get(AUTHORING_RECIPE_KEY))
        except Exception:
            return None
        return recipe if recipe.get("integrity") == "exact" else None

    def _compose_generated_recipe_edit(
        self,
        segment_index: int,
        replacement: GeneratedTrace,
    ) -> tuple[ComposedTrace, dict[str, Any]]:
        """Build a prospective full waveform after replacing one authored chunk."""
        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        recipe = self.exact_authoring_recipe()
        if recipe is None:
            raise WaveformIOError(
                "This waveform no longer has an exact authored component recipe that can be reopened in the generator editor."
            )
        segments = list(recipe.get("segments", []))
        index = int(segment_index)
        if index < 0 or index >= len(segments):
            raise WaveformIOError("The generated waveform segment no longer exists.")

        traces: list[GeneratedTrace] = []
        rebuilt_segments: list[dict[str, Any]] = []
        start_s = 0.0
        for current_index, segment in enumerate(segments):
            try:
                trace = replacement if current_index == index else trace_from_recipe_segment(segment)
                if current_index == index or segment.get("kind") == "generator":
                    canonical = generated_segment_from_settings(trace.settings, start_s=start_s)
                else:
                    canonical = dict(segment)
                    canonical["start_s"] = start_s
            except (NativeWaveformError, ValueError, KeyError, TypeError) as error:
                raise WaveformIOError(f"Unable to rebuild generated waveform segment {current_index + 1}: {error}") from error
            traces.append(trace)
            rebuilt_segments.append(canonical)
            start_s += float(canonical["duration_s"])

        try:
            composed = compose_generated_sequence(traces)
            updated_recipe = make_authoring_recipe(
                rebuilt_segments,
                quantity=signal.quantity,
                unit=signal.unit,
            )
        except (NativeWaveformError, ValueError) as error:
            raise WaveformIOError(str(error)) from error
        return composed, updated_recipe

    def _compose_recipe_segment_edit(
        self,
        segment_index: int,
        replacement_segment: dict[str, Any],
    ) -> tuple[ComposedTrace, dict[str, Any]]:
        """Build a prospective waveform after replacing any exact authored chunk."""

        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        recipe = self.exact_authoring_recipe()
        if recipe is None:
            raise WaveformIOError("This waveform no longer has an exact authored component recipe to edit.")
        segments = [dict(item) for item in recipe.get("segments", [])]
        index = int(segment_index)
        if index < 0 or index >= len(segments):
            raise WaveformIOError("The authored waveform segment no longer exists.")

        rebuilt: list[dict[str, Any]] = []
        traces: list[GeneratedTrace] = []
        start_s = 0.0
        for current_index, original in enumerate(segments):
            item = dict(replacement_segment if current_index == index else original)
            item["start_s"] = start_s
            try:
                canonical = make_authoring_recipe(
                    [item],
                    quantity=signal.quantity,
                    unit=signal.unit,
                )["segments"][0]
                trace = trace_from_recipe_segment(canonical)
            except Exception as error:
                raise WaveformIOError(
                    f"Unable to rebuild waveform segment {current_index + 1}: {error}"
                ) from error
            rebuilt.append(canonical)
            traces.append(trace)
            start_s += float(canonical["duration_s"])

        try:
            composed = compose_generated_sequence(traces)
            updated = make_authoring_recipe(
                rebuilt,
                quantity=signal.quantity,
                unit=signal.unit,
            )
        except Exception as error:
            raise WaveformIOError(str(error)) from error
        return composed, updated

    def replace_recipe_segment(self, segment_index: int, replacement_segment: dict[str, Any]) -> None:
        """Commit an edit to any exact authored chunk and rebuild the waveform."""

        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        composed, recipe = self._compose_recipe_segment_edit(segment_index, replacement_segment)
        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        self._source_format = ""
        self._source_path = ""
        self._metadata[AUTHORING_RECIPE_KEY] = recipe
        self._metadata["created_by"] = "Field Workbench Waveform Editor"
        self._trace_changed()

    def preview_generated_recipe_segment(
        self,
        segment_index: int,
        replacement: GeneratedTrace,
    ) -> ComposedTrace:
        """Return the exact full-waveform preview for an authored-segment edit."""
        composed, _recipe = self._compose_generated_recipe_edit(segment_index, replacement)
        return composed

    def replace_generated_recipe_segment(
        self,
        segment_index: int,
        replacement: GeneratedTrace,
    ) -> None:
        """Commit one builder edit while regenerating all recipe-backed chunks."""
        signal = self._signal_state()
        if signal is None:
            raise WaveformIOError("No waveform is loaded.")
        composed, recipe = self._compose_generated_recipe_edit(segment_index, replacement)

        signal.time_s[:] = composed.time_s
        signal.values[:] = composed.values
        self._source_format = ""
        self._source_path = ""
        self._metadata[AUTHORING_RECIPE_KEY] = recipe
        self._metadata["created_by"] = "Field Workbench Waveform Editor"
        self._trace_changed()

    def current_value_defaults(self) -> tuple[float, float]:
        """Return useful low/high defaults for the embedded waveform tools."""
        signal = self._signal_state()
        if signal is None or not signal.values:
            return -1.0, 1.0
        low = min(signal.values)
        high = max(signal.values)
        if math.isclose(low, high):
            span = max(abs(low), 1.0)
            return low - span, high + span
        return float(low), float(high)

    def apply_workbench_theme(self, _theme: str) -> None:
        self.plot.update()
