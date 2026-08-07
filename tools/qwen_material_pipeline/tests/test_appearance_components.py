"""Tests for photo-supported Part-ID appearance component construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from qwen_material_pipeline.evidence.appearance_components import (
    build_appearance_components,
    _part_color,
)


class AppearanceComponentsTest(unittest.TestCase):
    def _fixture(self, root: Path, *, iou: float = 0.95, p95: float = 4.0):
        ids = np.zeros((64, 64, 3), dtype=np.uint8)
        for part_id, bounds in {
            "P0001": (8, 10, 26, 54),
            "P0002": (26, 10, 44, 54),
            "P0003": (44, 10, 56, 54),
        }.items():
            red, green, blue = _part_color(part_id)
            left, top, right, bottom = bounds
            ids[top:bottom, left:right] = (blue, green, red)
        ids_path = root / "part_ids.png"
        cv2.imwrite(str(ids_path), ids)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :] = (30, 180, 40)  # BGR green paint
        image[10:54, 44:56] = (20, 110, 230)  # BGR warm/orange part
        image_path = root / "reference.png"
        mask_path = root / "foreground.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), np.full((64, 64), 255, dtype=np.uint8))

        registry = {
            "schema_version": "qwen-material-parts/v1",
            "part_count": 3,
            "parts": [{"part_id": value} for value in ("P0001", "P0002", "P0003")],
            "render_set": {
                "views": [
                    {
                        "view_id": "front",
                        "part_ids": str(ids_path),
                        "part_ids_raw": str(ids_path),
                        "camera_calibration": {"reference_view_id": "front"},
                    }
                ]
            },
        }
        registry_path = root / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        manifest = {
            "source_views": [
                {
                    "id": "front",
                    "image": str(image_path),
                    "confirmed_mask": {"path": str(mask_path)},
                }
            ]
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = {
            "final_rendered_registry": str(registry_path),
            "views": [
                {
                    "reference_view_id": "front",
                    "complete_alignment_passed": False,
                    "final": {
                        "projection_iou": iou,
                        "boundary_p95_px": p95,
                        "whole_asset_similarity": {
                            "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                        },
                    },
                }
            ],
        }
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return registry_path, manifest_path, report_path

    def test_adjacent_same_colour_parts_make_one_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, manifest, report = self._fixture(Path(directory))
            document = build_appearance_components(
                rendered_registry=registry,
                reference_manifest=manifest,
                camera_report=report,
            )
        self.assertEqual(document["summary"]["component_count"], 1)
        component = document["components"][0]
        self.assertEqual(component["member_part_ids"], ["P0001", "P0002"])
        memberships = {row["part_id"]: row for row in document["part_memberships"]}
        self.assertEqual(memberships["P0001"]["component_id"], component["component_id"])
        self.assertEqual(memberships["P0002"]["component_id"], component["component_id"])
        self.assertIsNone(memberships["P0003"]["component_id"])
        self.assertTrue(document["contract"]["per_part_geometric_warp_applied"] is False)
        self.assertTrue(document["contract"]["material_identity_mutated"] is False)

    def test_downweighted_view_cannot_create_component_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, manifest, report = self._fixture(Path(directory), iou=0.90, p95=12.0)
            document = build_appearance_components(
                rendered_registry=registry,
                reference_manifest=manifest,
                camera_report=report,
            )
        self.assertEqual(document["camera_alignment"]["front"]["tier"], "downweighted_box_correspondence")
        self.assertEqual(document["summary"]["component_count"], 0)
        self.assertEqual(document["summary"]["accepted_link_count"], 0)
