#!/usr/bin/env python3
"""Generate the checked-in LPBA40 Brain View mesh cache.

This is a developer-only asset build. The normal Field Workbench application
loads the resulting NPZ and does not import or distribute VTK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fieldworkbench.anatomy_frames import sri24_ras_mm_to_head_mm  # noqa: E402
from fieldworkbench.brain_mesh import (  # noqa: E402
    LPBA40_MESH_CACHE_FRAME,
    LPBA40_MESH_CACHE_RESOURCE,
    LPBA40_MESH_CACHE_STRIDE,
    LPBA40_MESH_CACHE_VERSION,
)
from fieldworkbench.sri24_atlas import (  # noqa: E402
    SRI24_LABEL_ASSET_DIR,
    SRI24_LABEL_MANIFEST,
    load_sri24_parcellation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vtk_region_meshes(visual_stride: int):
    try:
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        from vtkmodules.vtkCommonCore import vtkVersion
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D
    except ImportError as error:
        raise RuntimeError(
            "VTK is required only to regenerate this developer asset. "
            "Install requirements-mesh-build.txt first."
        ) from error

    atlas = load_sri24_parcellation("lpba40")
    sampled = np.ascontiguousarray(
        np.asarray(atlas.volume)[
            ::visual_stride, ::visual_stride, ::visual_stride
        ],
        dtype=np.int16,
    )
    label_ids = np.asarray(
        [int(value) for value in atlas.present_label_ids if int(value) != 0],
        dtype=np.int32,
    )
    missing = sorted(set(int(value) for value in label_ids) - set(int(value) for value in np.unique(sampled)))
    if missing:
        raise RuntimeError(
            "The requested visual stride removed LPBA40 labels: "
            + ", ".join(map(str, missing))
        )

    image = vtkImageData()
    image.SetDimensions(*sampled.shape)
    image.SetSpacing(
        float(visual_stride), float(visual_stride), float(visual_stride)
    )
    scalars = numpy_to_vtk(sampled.ravel(order="F"), deep=True)
    scalars.SetName("LPBA40")
    image.GetPointData().SetScalars(scalars)

    surface = vtkSurfaceNets3D()
    surface.SetInputData(image)
    surface.SetOutputMeshTypeToTriangles()
    surface.SetNumberOfLabels(len(label_ids))
    for index, label_id in enumerate(label_ids):
        surface.SetLabel(index, int(label_id))
    surface.SmoothingOn()
    surface.Update()

    polydata = surface.GetOutput()
    boundary_array = polydata.GetCellData().GetArray("BoundaryLabels")
    points_obj = polydata.GetPoints()
    polys_obj = polydata.GetPolys()
    if (
        boundary_array is None
        or boundary_array.GetNumberOfComponents() != 2
        or points_obj is None
        or polys_obj is None
    ):
        raise RuntimeError("VTK SurfaceNets returned an incomplete LPBA40 surface.")

    surface_points = np.asarray(vtk_to_numpy(points_obj.GetData()), dtype=float)
    boundary_labels = np.asarray(vtk_to_numpy(boundary_array), dtype=np.int32)
    connectivity = np.asarray(
        vtk_to_numpy(polys_obj.GetConnectivityArray()), dtype=np.int64
    )
    offsets = np.asarray(
        vtk_to_numpy(polys_obj.GetOffsetsArray()), dtype=np.int64
    )
    if len(offsets) < 2 or not np.all(np.diff(offsets) == 3):
        raise RuntimeError("VTK SurfaceNets did not return triangular faces.")
    triangles = connectivity.reshape(-1, 3)
    if len(triangles) != len(boundary_labels):
        raise RuntimeError("Surface connectivity and boundary labels disagree.")

    atlas_ras_mm = (
        surface_points @ np.asarray(atlas.affine_ras_mm[:3, :3], dtype=float).T
        + np.asarray(atlas.affine_ras_mm[:3, 3], dtype=float)
    )
    canonical_head_mm = sri24_ras_mm_to_head_mm(atlas_ras_mm)

    vertices_by_label: list[np.ndarray] = []
    triangles_by_label: list[np.ndarray] = []
    for label_id in label_ids:
        first = boundary_labels[:, 0] == int(label_id)
        second = boundary_labels[:, 1] == int(label_id)
        face_indices = np.flatnonzero(first | second)
        if not len(face_indices):
            raise RuntimeError(f"LPBA40 label {int(label_id)} produced no shell.")
        region_faces = np.array(triangles[face_indices], copy=True)
        reverse = second[face_indices]
        if np.any(reverse):
            region_faces[reverse] = region_faces[reverse][:, [0, 2, 1]]
        used, inverse = np.unique(region_faces.reshape(-1), return_inverse=True)
        vertices_by_label.append(
            np.asarray(canonical_head_mm[used], dtype=np.float32)
        )
        triangles_by_label.append(
            inverse.reshape(-1, 3).astype(np.int32, copy=False)
        )

    return (
        atlas,
        sampled.shape,
        label_ids,
        vertices_by_label,
        triangles_by_label,
        vtkVersion.GetVTKVersion(),
    )


def _offsets(arrays: list[np.ndarray]) -> np.ndarray:
    values = np.empty(len(arrays) + 1, dtype=np.int64)
    values[0] = 0
    values[1:] = np.cumsum([len(array) for array in arrays], dtype=np.int64)
    return values


def _write_cache(path: Path, visual_stride: int) -> dict[str, object]:
    (
        _atlas,
        source_shape,
        label_ids,
        vertices_by_label,
        triangles_by_label,
        vtk_version,
    ) = _vtk_region_meshes(visual_stride)
    vertices = np.ascontiguousarray(np.concatenate(vertices_by_label), dtype=np.float32)
    triangles = np.ascontiguousarray(
        np.concatenate(triangles_by_label), dtype=np.int32
    )
    vertex_offsets = _offsets(vertices_by_label)
    triangle_offsets = _offsets(triangles_by_label)
    volume_path = SRI24_LABEL_ASSET_DIR / "lpba40.nii.gz"
    labels_path = SRI24_LABEL_ASSET_DIR / "LPBA40-labels.txt"

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            mesh_version=np.asarray(LPBA40_MESH_CACHE_VERSION, dtype=np.int32),
            atlas_key=np.asarray("lpba40"),
            coordinate_frame=np.asarray(LPBA40_MESH_CACHE_FRAME),
            source_stride=np.asarray(visual_stride, dtype=np.int32),
            source_shape=np.asarray(source_shape, dtype=np.int32),
            source_volume_sha256=np.asarray(_sha256(volume_path)),
            source_label_table_sha256=np.asarray(_sha256(labels_path)),
            label_ids=label_ids,
            vertices=vertices,
            vertex_offsets=vertex_offsets,
            triangles=triangles,
            triangle_offsets=triangle_offsets,
        )
    temporary.replace(path)

    with np.load(path, allow_pickle=False) as archive:
        if set(int(value) for value in archive["label_ids"]) != set(
            int(value) for value in label_ids
        ):
            raise RuntimeError("Written cache did not preserve the LPBA40 labels.")
        if int(archive["triangle_offsets"][-1]) != len(triangles):
            raise RuntimeError("Written cache failed triangle-offset validation.")

    return {
        "sha256": _sha256(path),
        "kind": "derived_visual_mesh_cache",
        "mesh_version": LPBA40_MESH_CACHE_VERSION,
        "source_stride": int(visual_stride),
        "source_shape": list(map(int, source_shape)),
        "coordinate_frame": LPBA40_MESH_CACHE_FRAME,
        "label_count": int(len(label_ids)),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "source_volume": volume_path.name,
        "source_volume_sha256": _sha256(volume_path),
        "source_label_table": labels_path.name,
        "source_label_table_sha256": _sha256(labels_path),
        "generator": f"VTK SurfaceNets3D {vtk_version}",
        "smoothing": True,
    }


def _update_manifest(metadata: dict[str, object]) -> None:
    manifest = json.loads(SRI24_LABEL_MANIFEST.read_text(encoding="utf-8"))
    assets = manifest.setdefault("assets", {})
    if not isinstance(assets, dict):
        raise RuntimeError("The SRI24 asset manifest has an invalid asset table.")
    assets[LPBA40_MESH_CACHE_RESOURCE.name] = metadata
    SRI24_LABEL_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LPBA40_MESH_CACHE_RESOURCE,
        help="Output NPZ path (defaults to the bundled application asset).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=LPBA40_MESH_CACHE_STRIDE,
        help="Atlas-voxel visual stride.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least one")

    output = args.output.resolve()
    source = SRI24_LABEL_ASSET_DIR / "lpba40.nii.gz"
    print(f"LPBA40 source: {source.name}")
    print(f"Visual stride: {args.stride}")
    metadata = _write_cache(output, args.stride)
    if output == LPBA40_MESH_CACHE_RESOURCE.resolve():
        _update_manifest(metadata)
    print(f"Labels: {metadata['label_count']}")
    print(f"Vertices: {metadata['vertex_count']:,}")
    print(f"Triangles: {metadata['triangle_count']:,}")
    print(f"Writing: {output}")
    print(f"SHA-256: {metadata['sha256']}")
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
