from __future__ import annotations

import numpy as np

from qwen_material_pipeline.segmentation.cad_mesh_templates import (
    _project_points,
    _rasterize_faces,
)


def test_perspective_projection_uses_sealed_camera_without_mesh_motion() -> None:
    view = {
        "camera_position": [0.0, 0.0, 10.0],
        "camera_look_at_target": [0.0, 0.0, 0.0],
        "camera_up_axis": [0.0, 1.0, 0.0],
        "camera_projection_mode": "perspective",
        "focal_length_mm": 20.955,
    }
    points = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    projected, valid = _project_points(points, view=view, width=100, height=100)

    assert valid.tolist() == [True, True, True, True]
    np.testing.assert_allclose(
        projected,
        np.asarray([[40.0, 60.0], [60.0, 60.0], [60.0, 40.0], [40.0, 40.0]]),
    )


def test_rasterizer_unions_complete_mesh_faces_without_other_part_occlusion() -> None:
    points = np.asarray(
        [[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]]
    )

    mask = _rasterize_faces(
        points,
        np.ones(4, dtype=bool),
        [4],
        [0, 1, 2, 3],
        width=16,
        height=16,
    )

    assert np.all(mask[2:11, 2:11] == 255)
    assert np.count_nonzero(mask) == 81
