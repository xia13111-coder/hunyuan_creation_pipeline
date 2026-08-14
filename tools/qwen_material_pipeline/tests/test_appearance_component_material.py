from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from qwen_material_pipeline.evidence.appearance_component_material import (
    AppearanceComponentMaterialError,
    apply_fixed_component_mdl_choices,
    build_component_material_inputs,
    filter_components_for_material_evidence,
)
from qwen_material_pipeline.workflows.appearance_component_qwen import (
    run_component_qwen_rerank,
)


class _Generation:
    def __init__(self, text: str) -> None:
        self.text = text

    def metadata(self) -> dict[str, object]:
        return {"backend": "fake"}


class _ComponentRunner:
    model_identity = {"backend": "fake"}

    def generate_with_metadata(self, _payload: object) -> _Generation:
        return _Generation(
            json.dumps(
                {
                    "schema_version": "qwen-part-id-material-rerank-batch/v2",
                    "selections": [
                        {
                            "part_id": "AC_green",
                            "candidate_index": 1,
                            "confidence": 0.91,
                        }
                    ],
                }
            )
        )


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class AppearanceComponentMaterialTests(unittest.TestCase):
    def _documents(self, root: Path, *, same_registry: bool = True):
        image = root / "photo.png"
        mask = root / "mask.png"
        crop = root / "crop.png"
        Image.new("RGB", (32, 24), (42, 118, 55)).save(image)
        Image.new("L", (32, 24), 255).save(mask)
        Image.new("RGB", (64, 48), (42, 118, 55)).save(crop)
        registry_sha = "a" * 64
        components_unsigned = {
            "schema_version": "qwen-part-id-appearance-components/v1",
            # Match the actual producer contract in appearance_components.py.
            "status": "COMPLETED",
            "assignment_unit": "part_id",
            "inputs": {"rendered_registry_sha256": registry_sha},
            "components": [
                {
                    "component_id": "AC_green",
                    "member_part_ids": ["P0001", "P0002"],
                    "anchor_part_id": "P0001",
                    "appearance_family": "green",
                    "canonical_reference_rgb": [0.16, 0.40, 0.20],
                    "membership_authority": "rigid_projection",
                    "supporting_view_ids": ["front", "side"],
                    "total_trusted_pixels": 2048,
                }
            ],
        }
        components = {
            **components_unsigned,
            "integrity": {"document_sha256": _sha(components_unsigned)},
        }
        evidence_unsigned = {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "sam3_role": "whole_workpiece_foreground",
            "inputs": [
                {
                    "label": "rendered_registry",
                    "document_sha256": registry_sha if same_registry else "b" * 64,
                }
            ],
            "parts": [
                {
                    "part_id": part_id,
                    "status": "observed",
                    "descriptor": {
                        "roughness_hint": 0.42,
                        "metallicity_hint": 0.10,
                    },
                    "observations": [
                        {
                            "view_id": "front",
                            "image": str(image),
                            "mask": str(mask),
                            "box_mask": str(mask),
                            "crop": str(crop),
                            "mask_sha256": "c" * 64,
                            "trusted_foreground_pixels": 512,
                            "selected_for_material_inference": True,
                        }
                    ],
                }
                for part_id in ("P0001", "P0002")
            ],
        }
        evidence = {
            **evidence_unsigned,
            "integrity": {"document_sha256": _sha(evidence_unsigned)},
        }
        return components, evidence

    def test_builds_aggregate_inputs_and_applies_one_immutable_mdl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components, evidence = self._documents(root)
            catalog = root / "catalog.json"
            catalog.write_text("{}", encoding="utf-8")
            material_root = root / "materials"
            material_root.mkdir()
            component_evidence, request = build_component_material_inputs(
                appearance_components=components,
                part_id_evidence=evidence,
                catalog=catalog,
                material_root=material_root,
                output_dir=root / "generated",
            )
            self.assertEqual(component_evidence["summary"]["component_count"], 1)
            self.assertEqual(request["groups"][0]["group_id"], "AC_green")
            self.assertEqual(len(request["groups"][0]["observations"]), 2)
            self.assertTrue(
                Path(component_evidence["parts"][0]["observations"][0]["crop"]).is_file()
            )
            material_id = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
            base_plan = {
                "assignments": [
                    {
                        "part_id": part_id,
                        "material_id": "mdl:Metals/Chrome.mdl#Chrome",
                        "status": "auto",
                        "provenance": {},
                    }
                    for part_id in ("P0001", "P0002")
                ],
                "provenance": {},
            }
            base_audit = {
                "parts": [
                    {"part_id": part_id, "status": "independently_selected"}
                    for part_id in ("P0001", "P0002")
                ],
                "summary": {},
                "coating_consistency_gate": {"summary": {}},
            }
            retrieval = {
                "schema_version": "qwen-visual-material-retrieval-result/v1",
                "groups": [
                    {
                        "group_id": "AC_green",
                        "fused_ranking": [{"material_id": material_id, "rank": 1}],
                    }
                ]
            }
            retrieval["integrity"] = {
                "result_sha256": _sha(retrieval)
            }
            qwen = {
                "schema_version": "qwen-appearance-component-rerank/v1",
                "component_evidence_sha256": component_evidence["integrity"][
                    "document_sha256"
                ],
                "choices": {"AC_green": material_id},
                "selections": [
                    {"part_id": "AC_green", "confidence": 0.91}
                ],
            }
            qwen["integrity"] = {"document_sha256": _sha(qwen)}
            plan, audit = apply_fixed_component_mdl_choices(
                base_plan=base_plan,
                base_audit=base_audit,
                appearance_components=components,
                part_id_evidence=evidence,
                component_evidence=component_evidence,
                component_retrieval=retrieval,
                component_qwen_choices=qwen,
            )
            self.assertEqual(
                {row["material_id"] for row in plan["assignments"]}, {material_id}
            )
            self.assertTrue(plan["photo_appearance_components_used"])
            self.assertFalse(plan["coating_consistency_used"])
            self.assertEqual(audit["coating_consistency_gate"]["status"], "PASS")
            self.assertTrue(
                audit["coating_consistency_gate"][
                    "replaced_legacy_source_appearance_coating_gate"
                ]
            )
            for assignment in plan["assignments"]:
                provenance = assignment["provenance"]
                self.assertEqual(provenance["photo_appearance_component_id"], "AC_green")
                self.assertTrue(provenance["immutable_mdl_after_component_selection"])
                self.assertEqual(
                    provenance["mdl_parameter_candidates"]["candidates"],
                    [
                        {
                            "candidate_id": "H0",
                            "kind": "native_mdl",
                            "material_id": material_id,
                            "parameters": {},
                        }
                    ],
                )
            self.assertEqual(audit["summary"]["appearance_component_count"], 1)

            independent_plan, independent_audit = apply_fixed_component_mdl_choices(
                base_plan=base_plan,
                base_audit=base_audit,
                appearance_components=components,
                part_id_evidence=evidence,
                component_evidence=component_evidence,
                component_retrieval=retrieval,
                component_qwen_choices=qwen,
                authorized_component_ids=[],
            )
            self.assertEqual(
                {row["material_id"] for row in independent_plan["assignments"]},
                {"mdl:Metals/Chrome.mdl#Chrome"},
            )
            self.assertFalse(independent_plan["photo_appearance_components_used"])
            selection_audit = independent_audit[
                "appearance_component_mdl_selection"
            ]
            self.assertEqual(selection_audit["selections"], [])
            self.assertEqual(selection_audit["authorized_component_ids"], [])
            self.assertEqual(selection_audit["excluded_component_ids"], ["AC_green"])

    def test_component_qwen_reuses_bounded_candidate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components, evidence = self._documents(root)
            catalog_path = root / "catalog.json"
            catalog_path.write_text("{}", encoding="utf-8")
            material_root = root / "materials"
            material_root.mkdir()
            component_evidence, _request = build_component_material_inputs(
                appearance_components=components,
                part_id_evidence=evidence,
                catalog=catalog_path,
                material_root=material_root,
                output_dir=root / "generated",
            )
            first = "mdl:Miscellaneous/Paint_Gloss.mdl#Paint_Gloss"
            second = "mdl:Plastics/Vinyl.mdl#Vinyl"
            retrieval = {
                "groups": [
                    {
                        "group_id": "AC_green",
                        "fused_ranking": [
                            {"material_id": first, "rank": 1},
                            {"material_id": second, "rank": 2},
                        ],
                    }
                ],
                "catalog": {},
            }
            catalog = {
                "materials": [
                    {"material_id": first, "family": "paint"},
                    {"material_id": second, "family": "plastic"},
                ]
            }
            result = run_component_qwen_rerank(
                component_evidence=component_evidence,
                retrieval=retrieval,
                catalog=catalog,
                runner=_ComponentRunner(),
                model="fake",
                output_dir=root / "qwen",
                candidate_count=2,
            )
            self.assertEqual(
                result["schema_version"], "qwen-appearance-component-rerank/v1"
            )
            self.assertEqual(result["choices"], {"AC_green": first})
            self.assertFalse(result["mdl_parameter_mutation_allowed"])

    def test_refuses_cross_run_component_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components, evidence = self._documents(root, same_registry=False)
            catalog = root / "catalog.json"
            catalog.write_text("{}", encoding="utf-8")
            material_root = root / "materials"
            material_root.mkdir()
            with self.assertRaisesRegex(
                AppearanceComponentMaterialError, "same camera-calibrated"
            ):
                build_component_material_inputs(
                    appearance_components=components,
                    part_id_evidence=evidence,
                    catalog=catalog,
                    material_root=material_root,
                    output_dir=root / "generated",
                )

    def test_accepts_file_and_canonical_registry_hash_conventions(self) -> None:
        """The two producer stages seal the same registry differently."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components, evidence = self._documents(root)
            registry_path = root / "camera_calibrated_registry.json"
            registry_document = {"parts": [{"part_id": "P0001"}], "render_set": {}}
            registry_path.write_text(json.dumps(registry_document), encoding="utf-8")
            components["inputs"]["rendered_registry"] = str(registry_path)
            components["inputs"]["rendered_registry_sha256"] = hashlib.sha256(
                registry_path.read_bytes()
            ).hexdigest()
            components["integrity"] = {
                "document_sha256": _sha(
                    {key: value for key, value in components.items() if key != "integrity"}
                )
            }
            evidence["inputs"][0]["path"] = str(registry_path)
            evidence["inputs"][0]["document_sha256"] = _sha(registry_document)
            evidence["integrity"] = {
                "document_sha256": _sha(
                    {key: value for key, value in evidence.items() if key != "integrity"}
                )
            }
            catalog = root / "catalog.json"
            catalog.write_text("{}", encoding="utf-8")
            material_root = root / "materials"
            material_root.mkdir()
            component_evidence, _ = build_component_material_inputs(
                appearance_components=components,
                part_id_evidence=evidence,
                catalog=catalog,
                material_root=material_root,
                output_dir=root / "generated",
            )
            self.assertEqual(component_evidence["summary"]["component_count"], 1)

    def test_filters_members_without_selected_material_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components, evidence = self._documents(root)
            evidence["parts"][1]["status"] = "unobserved"
            filtered = filter_components_for_material_evidence(
                appearance_components=components,
                part_id_evidence=evidence,
            )
            self.assertEqual(filtered["components"], [])
            row = filtered["material_evidence_filter"]["components"][0]
            self.assertFalse(row["retained"])
            self.assertEqual(row["excluded_member_count"], 1)


if __name__ == "__main__":
    unittest.main()
