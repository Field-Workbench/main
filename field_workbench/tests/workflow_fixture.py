"""Stable small scene used by calculation/UI workflow tests.

The application default is intentionally a richer user-facing HM demonstration
scene.  Most focused tests use this compact scene so their setup remains about
the behaviour under test rather than the contents of the demonstration file.
"""

from __future__ import annotations

from typing import Any

from fieldworkbench.studio_adapter import default_coil_spec


def compact_workflow_scene() -> dict[str, Any]:
    sensor_positions = [
        [0.0, 0.0, round(-0.15 + index * 0.005, 6)] for index in range(61)
    ]
    return {
        "objects": [
            {
                "id": "coil_a",
                "type": "current.Circle",
                "params": {
                    "diameter": 0.2,
                    "current": 0.07,
                    "position": [0.0, 0.0, -0.05],
                },
                "style": {"label": "Lower coil"},
            },
            {
                "id": "coil_b",
                "type": "current.Circle",
                "params": {
                    "diameter": 0.2,
                    "current": 0.07,
                    "position": [0.0, 0.0, 0.05],
                },
                "style": {"label": "Upper coil"},
            },
            {
                "id": "axis_sensor",
                "type": "Sensor",
                "params": {"position": sensor_positions},
                "style": {"label": "Axis sensor 1", "size": 0.55},
            },
        ],
        "field_workbench": {
            "version": 12,
            "coil_specs": {
                "coil_a": default_coil_spec(0.07),
                "coil_b": default_coil_spec(0.07),
            },
        },
    }
