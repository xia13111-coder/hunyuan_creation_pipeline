from __future__ import annotations

import unittest

from qwen_material_pipeline.qwen.client import validate_material_plan
from qwen_material_pipeline.core.staged_analysis import (
    BATCH_SCHEMA_VERSION,
    GROUP_MATERIAL_SCHEMA_VERSION,
    PALETTE_SCHEMA_VERSION,
    StagedAnalysisError,
    detect_material_collapse,
    merge_staged_results,
    normalize_part_palette_batch,
    validate_group_materials,
    validate_palette,
    validate_part_palette_batch,
)


def palette_document(group_count: int = 3) -> dict:
    templates = [
        ("G01", "metal", "white", "painted", "white painted metal"),
        ("G02", "metal", "yellow", "painted", "yellow painted metal"),
        ("G03", "rubber", "cyan", "smooth", "cyan smooth hose"),
    ]
    return {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": "ref_single",
        "groups": [
            {
                "group_id": group_id,
                "family_hint": family,
                "base_color": color,
                "finish_hint": finish,
                "visual_description": description,
                "boxes": [[10 + index * 100, 20, 90 + index * 100, 200]],
                "confidence": 0.95,
            }
            for index, (group_id, family, color, finish, description) in enumerate(
                templates[:group_count]
            )
        ],
    }


def mapping(
    part_id: str,
    group_id: str | None = "G01",
    *,
    confidence: float = 0.91,
    status: str = "matched",
    reason_code: str = "shape_and_location",
) -> dict:
    unknown = status == "unknown"
    return {
        "part_id": part_id,
        "group_id": None if unknown else group_id,
        "mapping_confidence": confidence,
        "evidence_view_id": None if unknown else "ref_single",
        "evidence_box_index": None if unknown else 0,
        "status": status,
        "reason_code": reason_code,
    }


def batch_document(batch_id: str, mappings: list[dict]) -> dict:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "mappings": mappings,
    }


def material_document(*selections: tuple[str, str, float, bool]) -> dict:
    return {
        "schema_version": GROUP_MATERIAL_SCHEMA_VERSION,
        "selections": [
            {
                "group_id": group_id,
                "material_id": material_id,
                "confidence": confidence,
                "confirmed": confirmed,
            }
            for group_id, material_id, confidence, confirmed in selections
        ],
    }


class PaletteValidationTests(unittest.TestCase):
    def test_valid_palette_is_canonicalized(self) -> None:
        palette = palette_document()
        palette["groups"][0]["visual_description"] = "  white painted metal  "

        result = validate_palette(
            palette, allowed_reference_view_ids={"ref_single", "ref_other"}
        )

        self.assertEqual(
            result["groups"][0]["visual_description"], "white painted metal"
        )
        self.assertEqual(result["groups"][0]["confidence"], 0.95)

    def test_palette_rejects_extra_fields_and_duplicate_groups(self) -> None:
        palette = palette_document()
        palette["unexpected"] = True
        with self.assertRaisesRegex(StagedAnalysisError, "unexpected=.*unexpected"):
            validate_palette(palette)

        palette = palette_document()
        palette["groups"][1]["group_id"] = "G01"
        with self.assertRaisesRegex(StagedAnalysisError, "Duplicate palette group_id"):
            validate_palette(palette)

    def test_palette_requires_real_reference_and_valid_boxes(self) -> None:
        palette = palette_document()
        palette["source_view_id"] = "cad_iso"
        with self.assertRaisesRegex(StagedAnalysisError, "user reference"):
            validate_palette(palette)

        palette = palette_document()
        palette["groups"][0]["boxes"] = [[50, 50, 50, 100]]
        with self.assertRaisesRegex(StagedAnalysisError, "positive width"):
            validate_palette(palette)

        with self.assertRaisesRegex(StagedAnalysisError, "unknown source_view_id"):
            validate_palette(
                palette_document(), allowed_reference_view_ids={"ref_other"}
            )

    def test_palette_rejects_whole_image_placeholder_box(self) -> None:
        palette = palette_document()
        palette["groups"][0]["boxes"] = [[0, 0, 1000, 1000]]
        with self.assertRaisesRegex(StagedAnalysisError, "whole-image"):
            validate_palette(palette)

    def test_palette_enforces_closed_enums(self) -> None:
        palette = palette_document()
        palette["groups"][0]["base_color"] = "chartreuse"
        with self.assertRaisesRegex(StagedAnalysisError, "base_color must be one of"):
            validate_palette(palette)


class BatchValidationTests(unittest.TestCase):
    def test_batch_requires_exact_target_coverage(self) -> None:
        batch = batch_document("B01", [mapping("P0001")])
        with self.assertRaisesRegex(
            StagedAnalysisError, r"missing=.*P0002.*unexpected=\[\]"
        ):
            validate_part_palette_batch(
                batch,
                target_part_ids={"P0001", "P0002"},
                palette=palette_document(),
            )

        batch["mappings"].append(mapping("P9999"))
        with self.assertRaisesRegex(StagedAnalysisError, "unexpected=.*P9999"):
            validate_part_palette_batch(
                batch,
                target_part_ids={"P0001", "P0002"},
                palette=palette_document(),
            )

    def test_batch_rejects_duplicate_parts(self) -> None:
        batch = batch_document("B01", [mapping("P0001"), mapping("P0001")])
        with self.assertRaisesRegex(StagedAnalysisError, "Duplicate mapping"):
            validate_part_palette_batch(
                batch,
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

    def test_unknown_mapping_must_clear_group_and_evidence(self) -> None:
        unknown = mapping(
            "P0001",
            confidence=0.20,
            status="unknown",
            reason_code="occluded",
        )
        result = validate_part_palette_batch(
            batch_document("B01", [unknown]),
            target_part_ids={"P0001"},
            palette=palette_document(),
        )
        self.assertIsNone(result["mappings"][0]["group_id"])

        unknown["group_id"] = "G01"
        with self.assertRaisesRegex(StagedAnalysisError, "must use null group"):
            validate_part_palette_batch(
                batch_document("B01", [unknown]),
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

    def test_non_unknown_mapping_requires_palette_box_and_status_threshold(
        self,
    ) -> None:
        invalid_group = mapping("P0001", "G99")
        with self.assertRaisesRegex(StagedAnalysisError, "unknown group_id"):
            validate_part_palette_batch(
                batch_document("B01", [invalid_group]),
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

        invalid_confidence = mapping("P0001", confidence=0.70)
        with self.assertRaisesRegex(StagedAnalysisError, "matched requires"):
            validate_part_palette_batch(
                batch_document("B01", [invalid_confidence]),
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

        invalid_box = mapping("P0001")
        invalid_box["evidence_box_index"] = 4
        with self.assertRaisesRegex(StagedAnalysisError, "box_index is invalid"):
            validate_part_palette_batch(
                batch_document("B01", [invalid_box]),
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

    def test_compact_model_mapping_derives_only_mechanical_fields(self) -> None:
        events: list[dict] = []
        result = normalize_part_palette_batch(
            {
                "mappings": [
                    {
                        "part_id": "P0001",
                        "group_id": "G01",
                        "mapping_confidence": "90%",
                        "evidence_box_index": "0",
                        "reason_code": "shape_and_location",
                    }
                ]
            },
            expected_batch_id="B01",
            target_part_ids={"P0001"},
            palette=palette_document(),
            audit_events=events,
        )

        self.assertEqual(result["schema_version"], BATCH_SCHEMA_VERSION)
        self.assertEqual(result["batch_id"], "B01")
        self.assertEqual(result["mappings"][0]["evidence_view_id"], "ref_single")
        self.assertEqual(result["mappings"][0]["status"], "matched")
        self.assertEqual(result["mappings"][0]["mapping_confidence"], 0.9)
        self.assertIn("normalized_confidence_format", events[0]["changes"])

    def test_legacy_derived_field_mismatches_do_not_change_semantic_citation(
        self,
    ) -> None:
        legacy = mapping("P0001", confidence=0.70, status="matched")
        legacy["evidence_view_id"] = "cad_iso"
        events: list[dict] = []

        result = normalize_part_palette_batch(
            batch_document("WRONG", [legacy]),
            expected_batch_id="B01",
            target_part_ids={"P0001"},
            palette=palette_document(),
            audit_events=events,
        )

        normalized = result["mappings"][0]
        self.assertEqual(normalized["group_id"], "G01")
        self.assertEqual(normalized["evidence_box_index"], 0)
        self.assertEqual(normalized["status"], "review")
        self.assertEqual(normalized["evidence_view_id"], "ref_single")
        self.assertIn("derived_status", events[0]["changes"])
        self.assertIn("derived_evidence_view_id", events[0]["changes"])

    def test_legacy_review_status_is_never_promoted_to_matched(self) -> None:
        legacy = mapping("P0001", confidence=0.95, status="review")
        result = normalize_part_palette_batch(
            batch_document("B01", [legacy]),
            expected_batch_id="B01",
            target_part_ids={"P0001"},
            palette=palette_document(),
        )

        normalized = result["mappings"][0]
        self.assertEqual(normalized["status"], "review")
        self.assertEqual(normalized["group_id"], "G01")
        self.assertLess(normalized["mapping_confidence"], 0.85)

    def test_sub_256_match_is_downgraded_to_cited_review(self) -> None:
        result = normalize_part_palette_batch(
            batch_document("B01", [mapping("P0001")]),
            expected_batch_id="B01",
            target_part_ids={"P0001"},
            palette=palette_document(),
            visible_pixels_by_part={"P0001": 100},
        )

        normalized = result["mappings"][0]
        self.assertEqual(normalized["status"], "review")
        self.assertEqual(normalized["reason_code"], "too_small")
        self.assertEqual(normalized["group_id"], "G01")
        self.assertEqual(normalized["evidence_box_index"], 0)
        self.assertLess(normalized["mapping_confidence"], 0.85)

    def test_invalid_palette_citation_is_never_guessed(self) -> None:
        invalid = mapping("P0001")
        invalid["evidence_box_index"] = 99
        with self.assertRaisesRegex(StagedAnalysisError, "invalid for group G01"):
            normalize_part_palette_batch(
                batch_document("B01", [invalid]),
                expected_batch_id="B01",
                target_part_ids={"P0001"},
                palette=palette_document(),
            )

        events: list[dict] = []
        result = normalize_part_palette_batch(
            batch_document("B01", [invalid]),
            expected_batch_id="B01",
            target_part_ids={"P0001"},
            palette=palette_document(),
            quarantine_invalid_rows=True,
            audit_events=events,
        )
        self.assertEqual(result["mappings"][0]["status"], "unknown")
        self.assertIsNone(result["mappings"][0]["group_id"])
        self.assertEqual(events[0]["action"], "quarantined")


class MaterialSelectionTests(unittest.TestCase):
    def test_selection_ids_are_strictly_whitelisted(self) -> None:
        selections = material_document(("G01", "MAT_WHITE", 0.92, True))
        result = validate_group_materials(
            selections,
            palette=palette_document(),
            allowed_material_ids={"MAT_WHITE"},
        )
        self.assertTrue(result["selections"][0]["confirmed"])

        selections["selections"][0]["material_id"] = "INVENTED"
        with self.assertRaisesRegex(StagedAnalysisError, "unknown material_id"):
            validate_group_materials(
                selections,
                palette=palette_document(),
                allowed_material_ids={"MAT_WHITE"},
            )


class MergeTests(unittest.TestCase):
    def test_merge_excludes_model_and_forced_unknowns_from_material_plan(self) -> None:
        batch = batch_document(
            "B01",
            [
                mapping("P0001", "G01"),
                mapping(
                    "P0002",
                    confidence=0.20,
                    status="unknown",
                    reason_code="occluded",
                ),
            ],
        )
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=material_document(("G01", "MAT_WHITE", 0.92, True)),
            allowed_material_ids={"MAT_WHITE"},
            all_part_ids={"P0001", "P0002", "P0020"},
            forced_unknown_parts={"P0020": "no_cad_render"},
            orientation_confidence=0.96,
        )

        self.assertEqual(
            [item["part_id"] for item in result["material_plan"]["assignments"]],
            ["P0001"],
        )
        self.assertEqual(result["material_plan"]["assignments"][0]["status"], "auto")
        self.assertEqual(
            result["unknown_parts"],
            [
                {"part_id": "P0002", "reason_code": "occluded"},
                {"part_id": "P0020", "reason_code": "no_cad_render"},
            ],
        )
        self.assertFalse(
            any(
                assignment["status"] == "unknown"
                for assignment in result["material_plan"]["assignments"]
            )
        )
        validate_material_plan(result["material_plan"], {"P0001"}, {"MAT_WHITE"})

    def test_unconfirmed_high_confidence_selection_is_capped_to_review(self) -> None:
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", [mapping("P0001")])],
            batch_targets={"B01": {"P0001"}},
            material_selections=material_document(("G01", "MAT_WHITE", 0.99, False)),
            all_part_ids={"P0001"},
        )
        assignment = result["material_plan"]["assignments"][0]
        self.assertEqual(assignment["status"], "review")
        self.assertLess(assignment["confidence"], 0.85)
        validate_material_plan(result["material_plan"], {"P0001"}, {"MAT_WHITE"})

    def test_missing_or_low_confidence_material_becomes_unknown(self) -> None:
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", [mapping("P0001")])],
            batch_targets={"B01": {"P0001"}},
            material_selections=material_document(),
            all_part_ids={"P0001"},
        )
        self.assertEqual(result["material_plan"]["assignments"], [])
        self.assertEqual(
            result["unknown_parts"],
            [{"part_id": "P0001", "reason_code": "missing_material_selection"}],
        )

        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", [mapping("P0001")])],
            batch_targets={"B01": {"P0001"}},
            material_selections=material_document(("G01", "MAT_WHITE", 0.40, True)),
            all_part_ids={"P0001"},
        )
        self.assertEqual(
            result["unknown_parts"][0]["reason_code"],
            "low_combined_confidence",
        )

    def test_merge_requires_disjoint_complete_batch_declarations(self) -> None:
        batches = [
            batch_document("B01", [mapping("P0001")]),
            batch_document("B02", [mapping("P0002")]),
        ]
        with self.assertRaisesRegex(StagedAnalysisError, "multiple batch targets"):
            merge_staged_results(
                palette=palette_document(),
                batches=batches,
                batch_targets={"B01": {"P0001"}, "B02": {"P0001"}},
                material_selections=material_document(),
                all_part_ids={"P0001"},
            )

        with self.assertRaisesRegex(StagedAnalysisError, "exactly cover all parts"):
            merge_staged_results(
                palette=palette_document(),
                batches=[batches[0]],
                batch_targets={"B01": {"P0001"}},
                material_selections=material_document(),
                all_part_ids={"P0001", "P0002"},
            )

    def test_multi_batch_merge_is_stably_sorted(self) -> None:
        result = merge_staged_results(
            palette=palette_document(),
            batches=[
                batch_document("B02", [mapping("P0002", "G02")]),
                batch_document("B01", [mapping("P0001", "G01")]),
            ],
            batch_targets={"B01": {"P0001"}, "B02": {"P0002"}},
            material_selections=material_document(
                ("G01", "MAT_WHITE", 0.92, True),
                ("G02", "MAT_YELLOW", 0.92, True),
            ),
            all_part_ids={"P0001", "P0002"},
        )
        self.assertEqual(
            [item["part_id"] for item in result["material_plan"]["assignments"]],
            ["P0001", "P0002"],
        )


class CollapseDetectionTests(unittest.TestCase):
    def test_old_all_yellow_pattern_is_detected_and_safely_recovered(self) -> None:
        mappings = [mapping(f"P{index:04d}", "G02") for index in range(1, 9)]
        materials = material_document(("G02", "MAT_YELLOW", 0.90, True))
        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
        )
        self.assertTrue(diagnostic["detected"])
        self.assertTrue(diagnostic["recovery_required"])
        self.assertEqual(diagnostic["dominant_group_share"], 1.0)
        self.assertEqual(diagnostic["dominant_material_share"], 1.0)

        batch = batch_document("B01", mappings)
        part_ids = {item["part_id"] for item in mappings}
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch],
            batch_targets={"B01": part_ids},
            material_selections=materials,
            all_part_ids=part_ids,
        )
        self.assertEqual(result["material_plan"]["assignments"], [])
        self.assertEqual(
            {record["part_id"] for record in result["unknown_parts"]},
            part_ids,
        )
        self.assertEqual(
            {record["reason_code"] for record in result["unknown_parts"]},
            {"material_collapse_recovery"},
        )
        collapse = result["audit"]["collapse_check"]
        self.assertTrue(collapse["detected"])
        self.assertTrue(collapse["recovery_applied"])
        self.assertEqual(collapse["unsafe_group_ids"], ["G02"])
        self.assertEqual(collapse["recovered_to_unknown_part_ids"], sorted(part_ids))

    def test_non_authorable_review_concentration_is_audited_not_rejected(
        self,
    ) -> None:
        """Conditional concentration is not collapse when nothing can apply."""

        mappings = [
            *[
                mapping(
                    f"P{index:04d}",
                    "G01",
                    confidence=0.74,
                    status="review",
                    reason_code="ambiguous",
                )
                for index in range(1, 28)
            ],
            *[
                mapping(
                    f"P{index:04d}",
                    "G02",
                    confidence=0.72,
                    status="review",
                    reason_code="ambiguous",
                )
                for index in range(28, 34)
            ],
        ]
        materials = material_document(
            ("G01", "MAT_WHITE", 0.0, True),
            ("G02", "MAT_YELLOW", 0.0, True),
        )
        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
            total_part_count=137,
        )

        self.assertFalse(diagnostic["detected"])
        self.assertEqual(diagnostic["mapped_assignment_count"], 0)
        self.assertEqual(diagnostic["raw_mapped_assignment_count"], 33)
        self.assertGreater(diagnostic["raw_dominant_group_share"], 0.80)
        self.assertEqual(diagnostic["eligible_coverage_share"], 0.0)
        self.assertEqual(diagnostic["authorable_material_group_count"], 0)
        self.assertEqual(diagnostic["raw_to_eligible_retention_share"], 0.0)
        self.assertFalse(diagnostic["evidence_starvation"])

    def test_partial_authorable_evidence_starvation_requires_recovery(self) -> None:
        mappings = [
            mapping("P0001", "G01"),
            *[
                mapping(f"P{index:04d}", "G02")
                for index in range(2, 9)
            ],
        ]
        materials = material_document(
            ("G01", "MAT_WHITE", 0.90, True),
            ("G02", "MAT_YELLOW", 0.0, False),
        )

        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
        )

        self.assertFalse(diagnostic["detected"])
        self.assertTrue(diagnostic["evidence_starvation"])
        self.assertTrue(diagnostic["recovery_required"])
        self.assertEqual(diagnostic["mapped_assignment_count"], 1)
        self.assertEqual(diagnostic["raw_mapped_assignment_count"], 8)
        self.assertEqual(diagnostic["authorable_material_group_count"], 1)
        self.assertEqual(diagnostic["raw_to_eligible_retention_share"], 0.125)
        self.assertIn(
            "authoring evidence starved after confidence filtering",
            diagnostic["recovery_reasons"],
        )

        part_ids = {item["part_id"] for item in mappings}
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": part_ids},
            material_selections=materials,
            all_part_ids=part_ids,
        )
        collapse = result["audit"]["collapse_check"]
        self.assertFalse(collapse["detected"])
        self.assertTrue(collapse["recovery_required"])
        self.assertTrue(collapse["evidence_starvation"])

    def test_evidence_starvation_retention_threshold_is_strict(self) -> None:
        mappings = [
            mapping("P0001", "G01"),
            mapping("P0002", "G01"),
            *[
                mapping(f"P{index:04d}", "G02")
                for index in range(3, 11)
            ],
        ]
        materials = material_document(
            ("G01", "MAT_WHITE", 0.90, True),
            ("G02", "MAT_YELLOW", 0.0, False),
        )

        boundary = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
        )
        self.assertEqual(boundary["raw_to_eligible_retention_share"], 0.20)
        self.assertFalse(boundary["evidence_starvation"])
        self.assertFalse(boundary["detected"])

        stricter = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
            minimum_eligible_retention_share=0.21,
        )
        self.assertTrue(stricter["evidence_starvation"])
        self.assertTrue(stricter["recovery_required"])
        self.assertFalse(stricter["detected"])

    def test_evidence_starvation_requires_minimum_raw_signal(self) -> None:
        mappings = [
            mapping("P0001", "G01"),
            *[
                mapping(f"P{index:04d}", "G02")
                for index in range(2, 8)
            ],
        ]
        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=material_document(
                ("G01", "MAT_WHITE", 0.90, True),
                ("G02", "MAT_YELLOW", 0.0, False),
            ),
        )

        self.assertEqual(diagnostic["raw_mapped_assignment_count"], 7)
        self.assertFalse(diagnostic["evidence_starvation"])
        self.assertFalse(diagnostic["detected"])

    def test_evidence_starvation_threshold_validation(self) -> None:
        for invalid in (True, 0.0, 1.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    StagedAnalysisError,
                    "minimum_eligible_retention_share must be in",
                ):
                    detect_material_collapse(
                        palette=palette_document(),
                        mappings=[],
                        material_selections=material_document(),
                        minimum_eligible_retention_share=invalid,
                    )

    def test_different_visible_colors_share_identity_only_through_review(self) -> None:
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        materials = material_document(
            ("G01", "MAT_FIRST", 0.90, True),
            ("G02", "MAT_FIRST", 0.90, True),
        )
        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=materials,
        )
        self.assertFalse(diagnostic["detected"])
        self.assertTrue(diagnostic["recovery_required"])
        self.assertIn(
            "different palette colors",
            diagnostic["recovery_reasons"][0],
        )
        self.assertEqual(diagnostic["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(
            diagnostic["color_material_collisions"],
            [
                {
                    "material_id": "MAT_FIRST",
                    "group_ids": ["G01", "G02"],
                    "base_colors": ["white", "yellow"],
                    "affected_part_ids": ["P0001", "P0002"],
                }
            ],
        )

        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=materials,
            all_part_ids={"P0001", "P0002"},
        )
        self.assertEqual(result["unknown_parts"], [])
        self.assertEqual(
            [assignment["status"] for assignment in result["material_plan"]["assignments"]],
            ["review", "review"],
        )
        collapse = result["audit"]["collapse_check"]
        self.assertTrue(collapse["recovery_applied"])
        self.assertEqual(
            collapse["review_required_part_ids"],
            ["P0001", "P0002"],
        )

    def test_reviewed_tunable_identity_can_serve_different_colors(self) -> None:
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        materials = material_document(
            ("G01", "MAT_TUNABLE", 0.90, True),
            ("G02", "MAT_TUNABLE", 0.90, True),
        )

        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=materials,
            all_part_ids={"P0001", "P0002"},
            parameter_authoring_evidence={
                "G01": {"base_color": "white", "roughness_class": "satin"},
                "G02": {"base_color": "yellow", "roughness_class": "satin"},
            },
        )

        self.assertEqual(result["unknown_parts"], [])
        self.assertEqual(
            [
                assignment["status"]
                for assignment in result["material_plan"]["assignments"]
            ],
            ["auto", "auto"],
        )
        collapse = result["audit"]["collapse_check"]
        self.assertFalse(collapse["detected"])
        self.assertFalse(collapse["recovery_required"])
        self.assertEqual(collapse["review_required_group_ids"], [])
        self.assertEqual(collapse["color_material_collisions"], [])
        self.assertEqual(
            collapse["color_tunable_identity_shares"],
            [
                {
                    "material_id": "MAT_TUNABLE",
                    "group_ids": ["G01", "G02"],
                    "base_colors": ["white", "yellow"],
                    "affected_part_ids": ["P0001", "P0002"],
                }
            ],
        )

    def test_partial_parameter_evidence_does_not_exempt_color_collision(self) -> None:
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        result = merge_staged_results(
            palette=palette_document(),
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=material_document(
                ("G01", "MAT_TUNABLE", 0.90, True),
                ("G02", "MAT_TUNABLE", 0.90, True),
            ),
            all_part_ids={"P0001", "P0002"},
            parameter_authoring_evidence={
                "G01": {"base_color": "white", "roughness_class": "satin"}
            },
        )

        collapse = result["audit"]["collapse_check"]
        self.assertTrue(collapse["recovery_required"])
        self.assertEqual(collapse["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(collapse["color_tunable_identity_shares"], [])
        self.assertEqual(
            [
                assignment["status"]
                for assignment in result["material_plan"]["assignments"]
            ],
            ["review", "review"],
        )

    def test_wrong_authored_color_does_not_exempt_color_collision(self) -> None:
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        diagnostic = detect_material_collapse(
            palette=palette_document(),
            mappings=mappings,
            material_selections=material_document(
                ("G01", "MAT_TUNABLE", 0.90, True),
                ("G02", "MAT_TUNABLE", 0.90, True),
            ),
            parameter_authoring_evidence={
                "G01": {"base_color": "white", "roughness_class": "satin"},
                "G02": {"base_color": "white", "roughness_class": "satin"},
            },
        )

        self.assertTrue(diagnostic["recovery_required"])
        self.assertEqual(diagnostic["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(diagnostic["color_tunable_identity_shares"], [])

    def test_same_color_incompatible_material_families_require_recovery(self) -> None:
        palette = palette_document()
        palette["groups"][1]["base_color"] = "white"
        palette["groups"][1]["family_hint"] = "rubber"
        palette["groups"][1]["finish_hint"] = "smooth"
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        result = merge_staged_results(
            palette=palette,
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=material_document(
                ("G01", "MAT_WHITE_METAL", 0.90, True),
                ("G02", "MAT_WHITE_METAL", 0.90, True),
            ),
            all_part_ids={"P0001", "P0002"},
            parameter_authoring_evidence={
                "G01": {"base_color": "white", "roughness_class": "satin"},
                "G02": {"base_color": "white", "roughness_class": "glossy"},
            },
        )

        collapse = result["audit"]["collapse_check"]
        self.assertEqual(collapse["color_material_collisions"], [])
        self.assertEqual(collapse["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(
            collapse["material_identity_collisions"],
            [
                {
                    "material_id": "MAT_WHITE_METAL",
                    "group_ids": ["G01", "G02"],
                    "family_hints": ["metal", "rubber"],
                    "finish_hints": ["painted", "smooth"],
                    "reason_codes": ["incompatible_material_families"],
                    "affected_part_ids": ["P0001", "P0002"],
                }
            ],
        )

    def test_same_family_painted_and_bare_treatments_require_recovery(self) -> None:
        palette = palette_document()
        palette["groups"][1]["base_color"] = "white"
        palette["groups"][1]["finish_hint"] = "bare"
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=material_document(
                ("G01", "MAT_WHITE", 0.90, True),
                ("G02", "MAT_WHITE", 0.90, True),
            ),
            parameter_authoring_evidence={
                "G01": {"base_color": "white", "roughness_class": "satin"},
                "G02": {"base_color": "white", "roughness_class": "satin"},
            },
        )

        self.assertEqual(diagnostic["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(
            diagnostic["material_identity_collisions"][0]["reason_codes"],
            ["incompatible_surface_treatments"],
        )

    def test_same_color_matte_and_glossy_require_parameter_evidence(self) -> None:
        palette = palette_document()
        palette["groups"][0]["base_color"] = "gray"
        palette["groups"][0]["finish_hint"] = "matte"
        palette["groups"][1]["base_color"] = "gray"
        palette["groups"][1]["finish_hint"] = "glossy"
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        materials = material_document(
            ("G01", "MAT_GRAY", 0.90, True),
            ("G02", "MAT_GRAY", 0.90, True),
        )

        diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=materials,
        )
        self.assertEqual(diagnostic["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(
            diagnostic["material_identity_collisions"][0]["reason_codes"],
            ["incompatible_roughness_finishes"],
        )

        parameterized = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=materials,
            parameter_authoring_evidence={
                "G01": {"base_color": "gray", "roughness_class": "matte"},
                "G02": {"base_color": "gray", "roughness_class": "glossy"},
            },
        )
        self.assertEqual(parameterized["review_required_group_ids"], [])
        self.assertEqual(parameterized["material_identity_collisions"], [])
        self.assertEqual(
            parameterized["parameterized_finish_shares"][0]["roughness_classes"],
            {"G01": "matte", "G02": "glossy"},
        )

    def test_collision_covers_group_with_low_palette_confidence(self) -> None:
        palette = palette_document()
        palette["groups"][1]["confidence"] = 0.50
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        materials = material_document(
            ("G01", "MAT_FIRST", 0.90, True),
            ("G02", "MAT_FIRST", 0.90, True),
        )

        result = merge_staged_results(
            palette=palette,
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001", "P0002"}},
            material_selections=materials,
            all_part_ids={"P0001", "P0002"},
        )

        collapse = result["audit"]["collapse_check"]
        self.assertEqual(collapse["review_required_group_ids"], ["G01", "G02"])
        self.assertEqual(
            collapse["color_material_collisions"][0]["affected_part_ids"],
            ["P0001", "P0002"],
        )
        self.assertEqual(collapse["review_required_part_ids"], ["P0001"])
        self.assertEqual(
            result["unknown_parts"],
            [{"part_id": "P0002", "reason_code": "low_combined_confidence"}],
        )

    def test_color_collision_audit_is_stably_sorted_across_input_orders(
        self,
    ) -> None:
        palette = palette_document()
        palette["groups"].append(
            {
                "group_id": "G04",
                "family_hint": "metal",
                "base_color": "red",
                "finish_hint": "painted",
                "visual_description": "red painted metal",
                "boxes": [[310, 20, 390, 200]],
                "confidence": 0.95,
            }
        )
        mappings = [
            mapping("P0004", "G04"),
            mapping("P0002", "G02"),
            mapping("P0003", "G03"),
            mapping("P0001", "G01"),
        ]
        selections = (
            ("G04", "MAT_A", 0.90, True),
            ("G02", "MAT_Z", 0.90, True),
            ("G03", "MAT_A", 0.90, True),
            ("G01", "MAT_Z", 0.90, True),
        )
        expected_collisions = [
            {
                "material_id": "MAT_A",
                "group_ids": ["G03", "G04"],
                "base_colors": ["cyan", "red"],
                "affected_part_ids": ["P0003", "P0004"],
            },
            {
                "material_id": "MAT_Z",
                "group_ids": ["G01", "G02"],
                "base_colors": ["white", "yellow"],
                "affected_part_ids": ["P0001", "P0002"],
            },
        ]

        diagnostics = []
        results = []
        for reverse in (False, True):
            ordered_palette = {
                **palette,
                "groups": list(
                    reversed(palette["groups"])
                    if reverse
                    else palette["groups"]
                ),
            }
            ordered_mappings = list(reversed(mappings)) if reverse else mappings
            ordered_selections = tuple(reversed(selections)) if reverse else selections
            diagnostics.append(
                detect_material_collapse(
                    palette=ordered_palette,
                    mappings=ordered_mappings,
                    material_selections=material_document(*ordered_selections),
                )
            )
            results.append(
                merge_staged_results(
                    palette=ordered_palette,
                    batches=[batch_document("B01", ordered_mappings)],
                    batch_targets={"B01": {f"P{index:04d}" for index in range(1, 5)}},
                    material_selections=material_document(*ordered_selections),
                    all_part_ids={f"P{index:04d}" for index in range(1, 5)},
                )
            )

        for diagnostic in diagnostics:
            self.assertEqual(
                diagnostic["review_required_group_ids"],
                ["G01", "G02", "G03", "G04"],
            )
            self.assertEqual(
                diagnostic["color_material_collisions"], expected_collisions
            )
        for result in results:
            collapse = result["audit"]["collapse_check"]
            self.assertEqual(
                collapse["review_required_group_ids"],
                ["G01", "G02", "G03", "G04"],
            )
            self.assertEqual(
                collapse["review_required_part_ids"],
                ["P0001", "P0002", "P0003", "P0004"],
            )
            self.assertEqual(
                collapse["color_material_collisions"], expected_collisions
            )

    def test_same_visible_color_can_share_one_material_without_recovery(self) -> None:
        palette = palette_document()
        palette["groups"][1]["base_color"] = "white"
        mappings = [mapping("P0001", "G01"), mapping("P0002", "G02")]
        diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=material_document(
                ("G01", "MAT_FIRST", 0.90, True),
                ("G02", "MAT_FIRST", 0.90, True),
            ),
        )

        self.assertFalse(diagnostic["detected"])
        self.assertFalse(diagnostic["recovery_required"])
        self.assertEqual(diagnostic["color_material_collisions"], [])

    def test_one_group_reference_may_legitimately_use_one_material(self) -> None:
        palette = palette_document(group_count=1)
        mappings = [mapping(f"P{index:04d}") for index in range(1, 9)]
        diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=material_document(("G01", "MAT_WHITE", 0.90, True)),
        )
        self.assertFalse(diagnostic["detected"])

    def test_filtering_multiple_palette_groups_to_one_is_explicitly_audited(
        self,
    ) -> None:
        palette = palette_document(group_count=1)
        mappings = [mapping("P0001")]
        materials = material_document(("G01", "MAT_WHITE", 0.90, True))

        legacy_diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=materials,
        )
        self.assertFalse(legacy_diagnostic["detected"])
        self.assertFalse(legacy_diagnostic["palette_filter_context_supplied"])

        diagnostic = detect_material_collapse(
            palette=palette,
            mappings=mappings,
            material_selections=materials,
            pre_filter_palette_group_count=3,
        )
        self.assertTrue(diagnostic["detected"])
        self.assertTrue(diagnostic["palette_filter_collapse"])
        self.assertEqual(diagnostic["pre_filter_palette_group_count"], 3)
        self.assertEqual(diagnostic["filtered_palette_group_count"], 1)
        self.assertIn("filtering reduced", diagnostic["reasons"][0])

        result = merge_staged_results(
            palette=palette,
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001"}},
            material_selections=materials,
            all_part_ids={"P0001"},
            pre_filter_palette_group_count=3,
        )
        self.assertEqual(result["material_plan"]["assignments"], [])
        self.assertEqual(
            result["unknown_parts"],
            [
                {
                    "part_id": "P0001",
                    "reason_code": "material_collapse_recovery",
                }
            ],
        )
        self.assertTrue(result["audit"]["collapse_check"]["recovery_applied"])

    def test_filter_collapse_is_recovered_without_eligible_mapping(self) -> None:
        palette = palette_document(group_count=1)
        palette["groups"][0]["confidence"] = 0.50
        mappings = [mapping("P0001")]
        materials = material_document(("G01", "MAT_WHITE", 0.90, True))

        result = merge_staged_results(
            palette=palette,
            batches=[batch_document("B01", mappings)],
            batch_targets={"B01": {"P0001"}},
            material_selections=materials,
            all_part_ids={"P0001"},
            pre_filter_palette_group_count=3,
        )

        collapse = result["audit"]["collapse_check"]
        self.assertTrue(collapse["detected"])
        self.assertTrue(collapse["recovery_required"])
        self.assertEqual(collapse["unsafe_group_ids"], ["G01"])
        self.assertEqual(
            result["unknown_parts"],
            [
                {
                    "part_id": "P0001",
                    "reason_code": "material_collapse_recovery",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
