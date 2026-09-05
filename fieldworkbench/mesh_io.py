"""Small, dependency-free mesh import helpers for Field Workbench.

STL files do not declare a unit.  The desktop UI therefore asks the user which
unit the source coordinates use and passes the corresponding SI scale here.
OBJ imports support vertices and polygon faces; polygons are triangulated with
a simple fan, which is appropriate for the ordinary mechanical meshes this
prototype is intended to display.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable


class MeshImportError(ValueError):
    """Raised when a supported mesh file cannot be decoded."""


def load_mesh(path: str | Path, unit_scale: float = 0.001) -> tuple[list[list[float]], list[list[int]]]:
    """Load an STL or OBJ mesh and return SI-metre vertices and triangle faces."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".stl":
        vertices, faces = _load_stl(file_path.read_bytes())
    elif suffix == ".obj":
        vertices, faces = _load_obj(file_path.read_text(encoding="utf-8", errors="replace"))
    else:
        raise MeshImportError("This version imports STL and OBJ meshes.")
    if not vertices or not faces:
        raise MeshImportError("The file does not contain a triangulated surface.")
    scale = float(unit_scale)
    return [[coordinate * scale for coordinate in vertex] for vertex in vertices], faces


def _deduplicate(triangles: Iterable[Iterable[Iterable[float]]]):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    indices: dict[tuple[float, float, float], int] = {}
    for triangle in triangles:
        face: list[int] = []
        for raw_vertex in triangle:
            vertex = tuple(float(value) for value in raw_vertex)
            if len(vertex) != 3:
                raise MeshImportError("A mesh vertex does not have three coordinates.")
            index = indices.get(vertex)
            if index is None:
                index = len(vertices)
                indices[vertex] = index
                vertices.append(list(vertex))
            face.append(index)
        if len(face) == 3 and len(set(face)) == 3:
            faces.append(face)
    return vertices, faces


def _load_stl(data: bytes):
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + count * 50
        if count > 0 and expected <= len(data):
            triangles = []
            for index in range(count):
                offset = 84 + index * 50 + 12
                values = struct.unpack_from("<9f", data, offset)
                triangles.append((values[0:3], values[3:6], values[6:9]))
            return _deduplicate(triangles)

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MeshImportError("The STL file is neither valid binary nor ASCII STL.") from error
    raw_vertices: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                raw_vertices.append([float(value) for value in parts[1:]])
            except ValueError as error:
                raise MeshImportError("The ASCII STL contains an invalid vertex.") from error
    if len(raw_vertices) % 3:
        raise MeshImportError("The ASCII STL has an incomplete triangle.")
    triangles = [raw_vertices[index : index + 3] for index in range(0, len(raw_vertices), 3)]
    return _deduplicate(triangles)


def _load_obj(text: str):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as error:
                raise MeshImportError(f"Invalid OBJ vertex on line {line_number}.") from error
        elif parts[0] == "f" and len(parts) >= 4:
            polygon: list[int] = []
            for token in parts[1:]:
                try:
                    raw_index = int(token.split("/", 1)[0])
                except ValueError as error:
                    raise MeshImportError(f"Invalid OBJ face on line {line_number}.") from error
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if index < 0 or index >= len(vertices):
                    raise MeshImportError(f"OBJ face on line {line_number} references a missing vertex.")
                polygon.append(index)
            for index in range(1, len(polygon) - 1):
                triangle = [polygon[0], polygon[index], polygon[index + 1]]
                if len(set(triangle)) == 3:
                    faces.append(triangle)
    return vertices, faces
