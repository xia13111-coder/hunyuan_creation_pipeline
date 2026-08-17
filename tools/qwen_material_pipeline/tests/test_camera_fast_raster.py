from __future__ import annotations

import numpy as np
import pytest

from qwen_material_pipeline.evidence.camera_fast_raster import (
    FastCameraRasterError,
    _analysis_basis,
    _registry_bounds,
    _triangulate_faces,
)


def test_analysis_basis_matches_default_renderer_axes() -> None:
    basis = _analysis_basis("z", "-y")

    assert np.allclose(basis, np.eye(3), atol=1e-12)


def test_analysis_basis_rejects_parallel_front_and_up() -> None:
    with pytest.raises(FastCameraRasterError, match="non-zero"):
        _analysis_basis("z", "z")


def test_triangulate_faces_preserves_mesh_vertex_offsets() -> None:
    triangles = _triangulate_faces(
        [4, 3],
        [0, 1, 2, 3, 1, 4, 2],
        point_count=5,
        vertex_offset=10,
    )

    assert triangles.tolist() == [
        [10, 11, 12],
        [10, 12, 13],
        [11, 14, 12],
    ]


def test_triangulate_faces_rejects_invalid_topology() -> None:
    with pytest.raises(FastCameraRasterError, match="invalid mesh point"):
        _triangulate_faces(
            [3],
            [0, 1, 4],
            point_count=4,
            vertex_offset=0,
        )


def test_registry_bounds_cover_the_complete_rigid_assembly() -> None:
    center, diagonal = _registry_bounds(
        [
            {"part_id": "P0001", "world_bbox": [[-1, -2, 0], [0, 1, 2]]},
            {"part_id": "P0002", "world_bbox": [[2, -1, -2], [3, 4, 1]]},
        ]
    )

    assert center.tolist() == [1.0, 1.0, 0.0]
    assert diagonal == pytest.approx(np.linalg.norm([4.0, 6.0, 4.0]))
