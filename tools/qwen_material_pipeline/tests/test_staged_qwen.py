from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from qwen_material_pipeline.core.staged_analysis import (
    PALETTE_SCHEMA_VERSION,
    StagedAnalysisError,
)
from qwen_material_pipeline.qwen.client import QwenContentParseError
from qwen_material_pipeline.qwen.staged import (
    MATERIAL_CHOICE_SCHEMA_VERSION,
    LocalStagedQwenClient,
    build_group_material_payload,
    build_palette_payload,
    build_part_palette_payload,
    validate_group_material_choice,
)
from qwen_material_pipeline.qwen.local_vl import LocalGenerationResult


PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
).decode("ascii")


def palette() -> dict:
    return {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": "ref_single",
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "metal",
                "base_color": "white",
                "finish_hint": "painted",
                "visual_description": "white painted metal",
                "boxes": [[10, 20, 400, 900]],
                "confidence": 0.94,
            }
        ],
    }


def test_palette_payload_contains_one_image_and_closed_enums() -> None:
    payload = build_palette_payload(
        "qwen-test", {"id": "ref_single", "image": PNG_DATA_URL}
    )
    content = payload["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 1
    prompt = content[-1]["text"]
    assert PALETTE_SCHEMA_VERSION in prompt
    assert "family_hint" in prompt
    assert "0..1000" in prompt
    assert "one original camera view, not a contact sheet" in prompt
    assert "one single, visibly homogeneous object surface" in prompt
    assert "neighboring colors/materials" in prompt
    assert "black or dark painted structural parts" in prompt
    assert "white or light-gray modules" in prompt
    assert "85 percent" in prompt


def test_part_payload_is_four_images_and_exact_batch_scope() -> None:
    payload = build_part_palette_payload(
        "qwen-test",
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[
            {
                "part_id": "P0001",
                "evidence_visible_pixels": 300,
                "prim_path": "/Secret/Prim",
                "renders": [{"visible_pixels": 500, "image": "/secret.png"}],
            }
        ],
        batch_id="B01",
    )
    content = payload["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 4
    prompt = content[-1]["text"]
    assert "P0001" in prompt
    assert '"evidence_visible_pixels": 300' in prompt
    assert "/Secret/Prim" not in prompt
    assert "/secret.png" not in prompt
    assert "same asset" in prompt
    assert "NOT the target-part number" in prompt
    assert '"G01": [0]' in prompt
    assert (
        "Do not return schema_version, batch_id, status, or evidence_view_id" in prompt
    )


def test_part_payload_rejects_more_than_four_targets() -> None:
    with pytest.raises(StagedAnalysisError, match="1..4"):
        build_part_palette_payload(
            "qwen-test",
            reference_view={"id": "ref_single", "image": PNG_DATA_URL},
            cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
            part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
            batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
            palette=palette(),
            target_parts=[{"part_id": f"P{i:04d}"} for i in range(5)],
            batch_id="B01",
        )


def test_part_payload_accepts_one_bounded_multiview_support_sheet() -> None:
    payload = build_part_palette_payload(
        "qwen-test",
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        support_reference_view={
            "id": "ref_support_multiview",
            "image": PNG_DATA_URL,
        },
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001"}],
        batch_id="B01",
    )
    content = payload["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 5
    assert "IDENTITY/LOCALIZATION ONLY" in content[2]["text"]


def test_material_payload_labels_previews_by_exact_id() -> None:
    candidates = [
        {
            "material_id": "MAT_WHITE",
            "display_name": "White paint",
            "thumbnail_image": PNG_DATA_URL,
            "mdl_path": "/secret/material.mdl",
            "surface_interpretation": "applied_paint",
        }
    ]
    payload = build_group_material_payload(
        "qwen-test",
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=palette()["groups"][0],
        candidate_materials=candidates,
    )
    texts = [
        item["text"]
        for item in payload["messages"][1]["content"]
        if item["type"] == "text"
    ]
    assert any("material_id: MAT_WHITE" in text for text in texts)
    assert "/secret/material.mdl" not in texts[-1]
    assert '"surface_interpretation": "applied_paint"' in texts[-1]


def test_material_payload_forbids_assumed_tuning_for_immutable_mdl() -> None:
    group = {
        **palette()["groups"][0],
        "mvinverse_pbr_context": {
            "selected_mdl_parameters_mutable": False,
            "library_defaults_must_match_reference": True,
        },
    }
    payload = build_group_material_payload(
        "qwen-test",
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=group,
        candidate_materials=[
            {
                "material_id": "MAT_FIXED",
                "appearance_profile": {
                    "base_color_srgb": [0.2, 0.5, 0.2],
                    "roughness": 0.4,
                },
            }
        ],
    )
    prompt = payload["messages"][1]["content"][-1]["text"]

    assert "parameters will be immutable" in prompt
    assert "do not assume any later recoloring" in prompt


def test_material_payload_visual_objective_ignores_physical_family() -> None:
    group = {
        **palette()["groups"][0],
        "material_selection_objective": "visual_similarity",
        "mvinverse_pbr_context": {
            "selected_mdl_parameters_mutable": False,
            "library_defaults_must_match_reference": True,
        },
    }
    payload = build_group_material_payload(
        "gpt-5.6",
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=group,
        candidate_materials=[
            {"material_id": "MAT_VISUAL", "family": "unrelated_family"}
        ],
    )
    prompt = payload["messages"][1]["content"][-1]["text"]

    assert "Optimize visible similarity only" in prompt
    assert "engineering plausibility are not constraints" in prompt
    assert "Never prefer semantic plausibility over the visible pixels" in prompt
    assert "Match physical family first" not in prompt


def test_material_payload_requests_calibrated_non_placeholder_confidence() -> None:
    payload = build_group_material_payload(
        "qwen-test",
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=palette()["groups"][0],
        candidate_materials=[{"material_id": "MAT_WHITE"}],
    )
    prompt = payload["messages"][1]["content"][-1]["text"]

    assert "calibrated confidence" in prompt
    assert "Do not copy the illustrative confidence value" in prompt
    assert '"confidence": 0.75' in prompt
    assert '"confidence": 0.0' not in prompt


def test_material_choice_is_strictly_whitelisted() -> None:
    choice = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": "G01",
        "material_id": "MAT_WHITE",
        "confidence": 0.9,
    }
    assert (
        validate_group_material_choice(
            choice, group_id="G01", allowed_material_ids=["MAT_WHITE"]
        )["confidence"]
        == 0.9
    )
    choice["material_id"] = "INVENTED"
    with pytest.raises(StagedAnalysisError, match="unknown material_id"):
        validate_group_material_choice(
            choice, group_id="G01", allowed_material_ids=["MAT_WHITE"]
        )


def test_material_choice_normalizes_unique_omitted_mdl_export_syntax() -> None:
    candidate = (
        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Orange"
    )
    result = validate_group_material_choice(
        {
            "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
            "group_id": "G01",
            "material_id": "mdl:vMaterials_2/Metal/Steel_Painted_Orange",
            "confidence": 0.9,
        },
        group_id="G01",
        allowed_material_ids=[candidate],
    )

    assert result["material_id"] == candidate


def test_material_choice_rejects_ambiguous_module_only_alias() -> None:
    with pytest.raises(StagedAnalysisError, match="unknown material_id"):
        validate_group_material_choice(
            {
                "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
                "group_id": "G01",
                "material_id": "mdl:vMaterials_2/Metal/Steel_Painted",
                "confidence": 0.9,
            },
            group_id="G01",
            allowed_material_ids=[
                (
                    "mdl:vMaterials_2/Metal/Steel_Painted.mdl"
                    "#Steel_Painted_Orange"
                ),
                (
                    "mdl:vMaterials_2/Metal/Steel_Painted.mdl"
                    "#Steel_Painted_Army_Green"
                ),
            ],
        )


def test_local_client_saves_raw_and_validates_palette(tmp_path: Path) -> None:
    raw = json.dumps(palette())
    client = LocalStagedQwenClient(
        model="qwen-test", runner=lambda _payload: raw, raw_output_dir=tmp_path
    )
    result = client.extract_palette({"id": "ref_single", "image": PNG_DATA_URL})
    assert result["groups"][0]["group_id"] == "G01"
    assert (tmp_path / "01_palette.raw.txt").read_text() == raw
    audit = json.loads((tmp_path / "01_palette.parse.json").read_text())
    assert audit["normalization"] == "none"
    assert audit["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert audit["normalized_sha256"] == audit["raw_sha256"]
    assert audit["strict_json_status"] == "valid_object"


def test_local_client_unwraps_exact_fenced_palette_and_preserves_raw(
    tmp_path: Path,
) -> None:
    normalized = json.dumps(palette())
    raw = f"\n```json\n{normalized}\n```\t"
    client = LocalStagedQwenClient(
        model="qwen-test", runner=lambda _payload: raw, raw_output_dir=tmp_path
    )

    result = client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    )

    assert result == palette()
    assert (tmp_path / "01_palette.raw.txt").read_text() == raw
    audit = json.loads((tmp_path / "01_palette.parse.json").read_text())
    assert audit["normalization"] == "exact_markdown_json_fence_removed"
    assert audit["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert audit["normalized_sha256"] == hashlib.sha256(
        normalized.encode()
    ).hexdigest()
    assert audit["strict_json_status"] == "valid_object"


def test_local_client_retries_nonexact_fence_without_accepting_it(
    tmp_path: Path,
) -> None:
    raw = "Result:\n```json\n{}\n```"
    responses = iter((raw, json.dumps(palette())))
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: next(responses),
        raw_output_dir=tmp_path,
    )

    result = client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    )

    assert result == palette()
    assert (tmp_path / "01_palette.raw.txt").read_text() == raw
    audit = json.loads((tmp_path / "01_palette.parse.json").read_text())
    assert audit["strict_json_status"] == "not_parsed_transport_rejected"
    assert audit["strict_json_valid"] is False
    assert audit["error_reason"] == "nonexact_markdown_fence"
    retry_audit = json.loads(
        (tmp_path / "01_palette_retry1.parse.json").read_text()
    )
    assert retry_audit["strict_json_valid"] is True
    assert (tmp_path / "01_palette_retry1.raw.txt").is_file()


def test_material_choice_accepts_exact_fence_without_relaxing_allowlist(
    tmp_path: Path,
) -> None:
    choice = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": "G01",
        "material_id": "MAT_WHITE",
        "confidence": 0.9,
    }
    normalized = json.dumps(choice)
    raw = f"```json\n{normalized}\n```"
    client = LocalStagedQwenClient(
        model="qwen-test", runner=lambda _payload: raw, raw_output_dir=tmp_path
    )

    result = client.choose_group_material(
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=palette()["groups"][0],
        candidate_materials=[{"material_id": "MAT_WHITE"}],
        run_label="reverse",
    )

    assert result == choice
    stage = "03_material_G01_reverse"
    assert (tmp_path / f"{stage}.raw.txt").read_text() == raw
    audit = json.loads((tmp_path / f"{stage}.parse.json").read_text())
    assert audit["normalization"] == "exact_markdown_json_fence_removed"
    assert audit["strict_json_status"] == "valid_object"

    bad_choice = {**choice, "material_id": "INVENTED"}
    bad_raw = f"```json\n{json.dumps(bad_choice)}\n```"
    bad_client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: bad_raw,
        raw_output_dir=tmp_path / "bad",
    )
    with pytest.raises(StagedAnalysisError, match="unknown material_id"):
        bad_client.choose_group_material(
            reference_crop_view={
                "id": "ref_group_G01",
                "image": PNG_DATA_URL,
            },
            group=palette()["groups"][0],
            candidate_materials=[{"material_id": "MAT_WHITE"}],
            run_label="reverse",
        )
    bad_audit = json.loads(
        (
            tmp_path
            / "bad"
            / "03_material_G01_reverse.parse.json"
        ).read_text()
    )
    assert bad_audit["strict_json_status"] == "valid_object"
    assert bad_audit["schema_validation_status"] == "invalid"
    assert bad_audit["schema_valid"] is False


def test_material_choice_uses_one_causal_validator_repair(
    tmp_path: Path,
) -> None:
    invalid = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": "G01",
        "material_id": "INVENTED",
        "confidence": 0.9,
    }
    valid = {**invalid, "material_id": "MAT_WHITE"}
    responses = iter((json.dumps(invalid), json.dumps(valid)))
    payloads = []

    def runner(payload):
        payloads.append(payload)
        return next(responses)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    result = client.choose_group_material(
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=palette()["groups"][0],
        candidate_materials=[{"material_id": "MAT_WHITE"}],
        run_label="reverse",
    )

    assert result == valid
    assert len(payloads) == 2
    repair_text = payloads[1]["messages"][1]["content"][-1]["text"]
    assert "VALIDATOR REPAIR" in repair_text
    assert "INVENTED" in repair_text
    assert "MAT_WHITE" in repair_text
    assert client.repair_events[-1]["action"] == (
        "accepted_validated_material_choice_retry"
    )
    first_audit = json.loads(
        (tmp_path / "03_material_G01_reverse.parse.json").read_text()
    )
    retry_audit = json.loads(
        (tmp_path / "03_material_G01_reverse_retry1.parse.json").read_text()
    )
    assert first_audit["schema_validation_status"] == "invalid"
    assert first_audit["schema_valid"] is False
    assert retry_audit["schema_validation_status"] == "valid"
    assert retry_audit["schema_valid"] is True


def test_material_choice_repairs_nonexact_transport_once(
    tmp_path: Path,
) -> None:
    valid = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": "G01",
        "material_id": "MAT_WHITE",
        "confidence": 0.9,
    }
    responses = iter(
        (
            "Here is the result:\n```json\n"
            + json.dumps(valid)
            + "\n```",
            json.dumps(valid),
        )
    )
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: next(responses),
        raw_output_dir=tmp_path,
    )

    result = client.choose_group_material(
        reference_crop_view={"id": "ref_group_G01", "image": PNG_DATA_URL},
        group=palette()["groups"][0],
        candidate_materials=[{"material_id": "MAT_WHITE"}],
        run_label="forward",
    )

    assert result == valid
    first_audit = json.loads(
        (tmp_path / "03_material_G01_forward.parse.json").read_text()
    )
    assert first_audit["strict_json_valid"] is False
    assert first_audit["error_reason"] == "nonexact_markdown_fence"
    retry_audit = json.loads(
        (tmp_path / "03_material_G01_forward_retry1.parse.json").read_text()
    )
    assert retry_audit["schema_valid"] is True


def test_local_client_palette_run_label_preserves_each_raw_view(tmp_path: Path) -> None:
    documents = []

    def runner(_payload):
        document = palette()
        document["source_view_id"] = f"ref_{len(documents) + 1}"
        documents.append(document)
        return json.dumps(document)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    client.extract_palette({"id": "ref_1", "image": PNG_DATA_URL}, run_label="01_ref_1")
    client.extract_palette({"id": "ref_2", "image": PNG_DATA_URL}, run_label="02_ref_2")
    assert (tmp_path / "01_palette_01_ref_1.raw.txt").is_file()
    assert (tmp_path / "01_palette_02_ref_2.raw.txt").is_file()


def test_validated_palette_checkpoint_is_atomic_hash_bound_and_reused(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    checkpoint_dir = tmp_path / "checkpoints"
    identity_sha256 = "a" * 64
    calls = 0

    def runner(_payload):
        nonlocal calls
        calls += 1
        return json.dumps(palette())

    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=runner,
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
    )
    expected = client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    )

    assert calls == 1
    checkpoint_path = checkpoint_dir / "01_palette.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["schema_version"] == (
        "qwen-validated-stage-checkpoint/v1"
    )
    assert checkpoint["stage_name"] == "01_palette"
    assert checkpoint["stage_kind"] == "palette"
    assert checkpoint["checkpoint_identity_sha256"] == identity_sha256
    assert checkpoint["result"] == expected
    assert checkpoint["provenance"]["source_stage_name"] == "01_palette"
    assert not Path(
        checkpoint["provenance"]["raw_output"]["relative_path"]
    ).is_absolute()
    assert not Path(
        checkpoint["provenance"]["parse_audit"]["relative_path"]
    ).is_absolute()
    audit = json.loads((raw_dir / "01_palette.parse.json").read_text())
    assert audit["schema_validation_status"] == "valid"
    assert audit["schema_valid"] is True

    reuse_client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: pytest.fail("runner must not be called"),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
        reuse_checkpoints=True,
    )
    assert reuse_client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    ) == expected


def test_validated_checkpoint_fails_closed_on_input_or_artifact_change(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    checkpoint_dir = tmp_path / "checkpoints"
    identity_sha256 = "b" * 64
    creator = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: json.dumps(palette()),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
    )
    creator.extract_palette({"id": "ref_single", "image": PNG_DATA_URL})

    reuse_client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: pytest.fail("runner must not be called"),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
        reuse_checkpoints=True,
    )
    changed_image = (
        "data:image/png;base64,"
        + base64.b64encode(b"different-image-bytes").decode("ascii")
    )
    with pytest.raises(
        StagedAnalysisError, match="input payload SHA-256 mismatch"
    ) as changed:
        reuse_client.extract_palette(
            {"id": "ref_single", "image": changed_image}
        )
    assert changed.value.stage_name == "01_palette"
    assert changed.value.raw_output_path == raw_dir / "01_palette.raw.txt"
    assert changed.value.checkpoint_path == (
        checkpoint_dir / "01_palette.checkpoint.json"
    )

    (raw_dir / "01_palette.raw.txt").write_text("tampered")
    with pytest.raises(
        StagedAnalysisError, match="raw output artifact SHA-256 mismatch"
    ) as tampered:
        reuse_client.extract_palette(
            {"id": "ref_single", "image": PNG_DATA_URL}
        )
    assert tampered.value.checkpoint_context["stage_name"] == "01_palette"


def test_mapping_checkpoint_caches_only_final_normalized_repair(
    tmp_path: Path,
) -> None:
    initial = {
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": 0.9,
                "evidence_box_index": 3,
                "reason_code": "shape_and_location",
            }
        ]
    }
    repaired_compact = {
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": "85%",
                "evidence_box_index": "0",
                "reason_code": "shape_and_location",
            }
        ]
    }
    responses = iter((json.dumps(initial), json.dumps(repaired_compact)))
    raw_dir = tmp_path / "raw"
    checkpoint_dir = tmp_path / "checkpoints"
    identity_sha256 = "c" * 64
    mapping_kwargs = {
        "reference_view": {"id": "ref_single", "image": PNG_DATA_URL},
        "cad_view": {"id": "cad_iso", "image": PNG_DATA_URL},
        "part_id_view": {"id": "part_ids_iso", "image": PNG_DATA_URL},
        "batch_sheet_view": {
            "id": "batch_parts_B01",
            "image": PNG_DATA_URL,
        },
        "palette": palette(),
        "target_parts": [
            {"part_id": "P0001", "renders": [{"visible_pixels": 500}]}
        ],
        "batch_id": "B01",
    }
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: next(responses),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
    )
    expected = client.map_part_batch(**mapping_kwargs)

    checkpoint_path = checkpoint_dir / "02_map_B01.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert not (checkpoint_dir / "02_map_B01_retry1.checkpoint.json").exists()
    assert checkpoint["provenance"]["source_stage_name"] == (
        "02_map_B01_retry1"
    )
    assert checkpoint["result"] == expected
    assert expected["mappings"][0]["mapping_confidence"] == 0.85
    assert expected["mappings"][0]["status"] == "matched"
    retry_audit = json.loads(
        (raw_dir / "02_map_B01_retry1.parse.json").read_text()
    )
    assert retry_audit["schema_valid"] is True

    reuse_client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: pytest.fail("runner must not be called"),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
        reuse_checkpoints=True,
    )
    assert reuse_client.map_part_batch(**mapping_kwargs) == expected


def test_material_repair_checkpoint_uses_original_stage_key_and_reuses(
    tmp_path: Path,
) -> None:
    invalid = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": "G01",
        "material_id": "INVENTED",
        "confidence": 0.9,
    }
    valid = {**invalid, "material_id": "MAT_WHITE"}
    responses = iter((json.dumps(invalid), json.dumps(valid)))
    raw_dir = tmp_path / "raw"
    checkpoint_dir = tmp_path / "checkpoints"
    identity_sha256 = "d" * 64
    material_kwargs = {
        "reference_crop_view": {
            "id": "ref_group_G01",
            "image": PNG_DATA_URL,
        },
        "group": palette()["groups"][0],
        "candidate_materials": [{"material_id": "MAT_WHITE"}],
        "run_label": "reverse",
    }
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: next(responses),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
    )
    assert client.choose_group_material(**material_kwargs) == valid

    checkpoint_path = (
        checkpoint_dir / "03_material_G01_reverse.checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    assert not (
        checkpoint_dir
        / "03_material_G01_reverse_retry1.checkpoint.json"
    ).exists()
    assert checkpoint["provenance"]["source_stage_name"] == (
        "03_material_G01_reverse_retry1"
    )

    reuse_client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: pytest.fail("runner must not be called"),
        raw_output_dir=raw_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity_sha256=identity_sha256,
        reuse_checkpoints=True,
    )
    assert reuse_client.choose_group_material(**material_kwargs) == valid


def test_palette_retries_only_truncation_with_bounded_token_growth(
    tmp_path: Path,
) -> None:
    valid = palette()
    responses = iter(
        (
            ('{"schema_version":', 1024, False),
            (json.dumps(valid), 731, True),
        )
    )
    budgets = []
    callbacks = []

    class MetadataRunner:
        max_new_tokens = 1024

        def __call__(self, _payload):
            raise AssertionError("metadata generation path must be used")

        def generate_with_metadata(self, _payload, *, max_new_tokens=None):
            budgets.append(max_new_tokens)
            text, generated_tokens, eos_detected = next(responses)
            return LocalGenerationResult(
                text=text,
                generated_tokens=generated_tokens,
                max_new_tokens=max_new_tokens,
                hit_token_limit=generated_tokens >= max_new_tokens,
                eos_detected=eos_detected,
            )

    runner = MetadataRunner()
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=runner,
        raw_output_dir=tmp_path,
        max_new_tokens_ceiling=4096,
        generation_event_callback=lambda event: callbacks.append(dict(event)),
    )
    result = client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL},
        run_label="front",
    )

    assert result == valid
    assert budgets == [1024, 2048]
    assert [event["status"] for event in client.generation_events] == [
        "truncated_retry",
        "valid",
    ]
    assert callbacks == client.generation_events
    assert (tmp_path / "01_palette_front.raw.txt").read_text() == (
        '{"schema_version":'
    )
    assert (tmp_path / "01_palette_front_retry1.raw.txt").read_text() == (
        json.dumps(valid)
    )
    retry_metadata = json.loads(
        (tmp_path / "01_palette_front_retry1.generation.json").read_text()
    )
    assert retry_metadata["generated_tokens"] == 731
    assert retry_metadata["eos_detected"] is True
    assert retry_metadata["status"] == "valid"


def test_palette_truncation_retries_land_exactly_on_non_power_of_two_ceiling(
    tmp_path: Path,
) -> None:
    budgets = []

    class MetadataRunner:
        max_new_tokens = 1024

        def __call__(self, _payload):
            raise AssertionError("metadata generation path must be used")

        def generate_with_metadata(self, _payload, *, max_new_tokens=None):
            budgets.append(max_new_tokens)
            return LocalGenerationResult(
                text='{"groups": [',
                generated_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                hit_token_limit=True,
                eos_detected=False,
            )

    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=MetadataRunner(),
        raw_output_dir=tmp_path,
        max_new_tokens_ceiling=2500,
    )
    with pytest.raises(StagedAnalysisError, match="remained truncated.*ceiling=2500"):
        client.extract_palette({"id": "ref_single", "image": PNG_DATA_URL})

    assert budgets == [1024, 2048, 2500]
    assert [event["status"] for event in client.generation_events] == [
        "truncated_retry",
        "truncated_retry",
        "truncated_exhausted",
    ]


def test_palette_retries_complete_schema_invalid_json_once_with_causal_prompt(
    tmp_path: Path,
) -> None:
    invalid = {"error": "unable to inspect image", "usable": False}
    valid = palette()
    responses = iter((json.dumps(invalid), json.dumps(valid)))
    payloads: list[dict] = []

    class MetadataRunner:
        max_new_tokens = 1024

        def __call__(self, _payload):
            raise AssertionError("metadata generation path must be used")

        def generate_with_metadata(self, payload, *, max_new_tokens=None):
            payloads.append(deepcopy(payload))
            return LocalGenerationResult(
                text=next(responses),
                generated_tokens=18,
                max_new_tokens=max_new_tokens,
                hit_token_limit=False,
                eos_detected=True,
            )

    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=MetadataRunner(),
        raw_output_dir=tmp_path,
        max_new_tokens_ceiling=4096,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_identity_sha256="e" * 64,
    )
    assert client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    ) == valid

    assert len(payloads) == 2
    assert len(payloads[1]["messages"]) == len(payloads[0]["messages"]) + 2
    assert payloads[1]["messages"][-2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(invalid)}],
    }
    correction = payloads[1]["messages"][-1]["content"][0]["text"]
    assert "Source view_id (must be preserved exactly): ref_single" in correction
    assert "palette fields are invalid" in correction
    assert "strict JSON object only" in correction
    assert "no Markdown fences" in correction
    assert "Do not return error, usable, status" in correction
    assert [event["status"] for event in client.generation_events] == [
        "schema_retry",
        "valid",
    ]
    assert client.generation_events[0]["error_reason"] == (
        "palette_schema_invalid"
    )
    initial_audit = json.loads(
        (tmp_path / "01_palette.parse.json").read_text()
    )
    retry_audit = json.loads(
        (tmp_path / "01_palette_retry1.parse.json").read_text()
    )
    assert initial_audit["strict_json_valid"] is True
    assert initial_audit["schema_validation_status"] == "invalid"
    assert initial_audit["schema_valid"] is False
    assert "palette fields are invalid" in initial_audit["schema_error"]
    assert retry_audit["strict_json_valid"] is True
    assert retry_audit["schema_validation_status"] == "valid"
    assert retry_audit["schema_valid"] is True
    checkpoint = json.loads(
        (
            tmp_path / "checkpoints" / "01_palette.checkpoint.json"
        ).read_text()
    )
    assert checkpoint["provenance"]["source_stage_name"] == (
        "01_palette_retry1"
    )


def test_palette_schema_retry_failure_is_unusable_and_not_retried_again(
    tmp_path: Path,
) -> None:
    responses = iter(
        (
            json.dumps({"error": "first", "usable": False}),
            json.dumps({"error": "second", "usable": False}),
        )
    )
    calls = 0

    class MetadataRunner:
        max_new_tokens = 1024

        def __call__(self, _payload):
            raise AssertionError("metadata generation path must be used")

        def generate_with_metadata(self, _payload, *, max_new_tokens=None):
            nonlocal calls
            calls += 1
            return LocalGenerationResult(
                text=next(responses),
                generated_tokens=18,
                max_new_tokens=max_new_tokens,
                hit_token_limit=False,
                eos_detected=True,
            )

    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=MetadataRunner(),
        raw_output_dir=tmp_path,
        max_new_tokens_ceiling=4096,
    )
    with pytest.raises(StagedAnalysisError, match="palette fields are invalid"):
        client.extract_palette({"id": "ref_single", "image": PNG_DATA_URL})

    assert calls == 2
    assert [event["status"] for event in client.generation_events] == [
        "schema_retry",
        "invalid",
    ]
    assert all(
        event["error_reason"] == "palette_schema_invalid"
        for event in client.generation_events
    )
    retry_audit = json.loads(
        (tmp_path / "01_palette_retry1.parse.json").read_text()
    )
    assert retry_audit["strict_json_valid"] is True
    assert retry_audit["schema_validation_status"] == "invalid"
    assert retry_audit["schema_valid"] is False
    assert not (tmp_path / "01_palette_retry2.raw.txt").exists()
    assert not (tmp_path / "palette.json").exists()


def test_palette_valid_response_has_no_schema_retry(tmp_path: Path) -> None:
    calls = 0

    class MetadataRunner:
        max_new_tokens = 1024

        def __call__(self, _payload):
            raise AssertionError("metadata generation path must be used")

        def generate_with_metadata(self, _payload, *, max_new_tokens=None):
            nonlocal calls
            calls += 1
            return LocalGenerationResult(
                text=json.dumps(palette()),
                generated_tokens=120,
                max_new_tokens=max_new_tokens,
                hit_token_limit=False,
                eos_detected=True,
            )

    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=MetadataRunner(),
        raw_output_dir=tmp_path,
        max_new_tokens_ceiling=4096,
    )
    assert client.extract_palette(
        {"id": "ref_single", "image": PNG_DATA_URL}
    ) == palette()

    assert calls == 1
    assert [event["status"] for event in client.generation_events] == ["valid"]
    assert not (tmp_path / "01_palette_retry1.raw.txt").exists()


def test_part_mapping_retries_once_without_relaxing_validator(tmp_path: Path) -> None:
    invalid = {
        "schema_version": "qwen-part-palette-map/v1",
        "batch_id": "B01",
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": 0.9,
                "evidence_view_id": "ref_single",
                "evidence_box_index": 3,
                "status": "matched",
                "reason_code": "shape_and_location",
            }
        ],
    }
    repaired = deepcopy(invalid)
    repaired["mappings"][0]["evidence_box_index"] = 0
    responses = iter((json.dumps(invalid), json.dumps(repaired)))
    payloads = []

    def runner(payload):
        payloads.append(payload)
        return next(responses)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    result = client.map_part_batch(
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001", "renders": [{"visible_pixels": 500}]}],
        batch_id="B01",
    )
    assert result["mappings"][0]["evidence_box_index"] == 0
    assert len(payloads) == 2
    repair_text = payloads[1]["messages"][1]["content"][-1]["text"]
    assert "VALIDATOR REPAIR" in repair_text
    assert "invalid for group G01" in repair_text
    assert (tmp_path / "02_map_B01.raw.txt").exists()
    assert (tmp_path / "02_map_B01_retry1.raw.txt").exists()


def test_part_mapping_retries_extra_json_data_with_same_strict_parser(
    tmp_path: Path,
) -> None:
    valid = {
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": 0.9,
                "evidence_box_index": 0,
                "reason_code": "shape_and_location",
            }
        ]
    }
    malformed = json.dumps(valid) + "\ntrailing explanation"
    responses = iter((malformed, json.dumps(valid)))
    payloads = []

    def runner(payload):
        payloads.append(deepcopy(payload))
        return next(responses)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    result = client.map_part_batch(
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001", "renders": [{"visible_pixels": 500}]}],
        batch_id="B01",
    )

    assert result["mappings"][0]["group_id"] == "G01"
    assert len(payloads) == 2
    retry_text = payloads[1]["messages"][-1]["content"][0]["text"]
    assert "FORMAT REPAIR" in retry_text
    assert "second object, trailing text" in retry_text
    first_audit = json.loads(
        (tmp_path / "02_map_B01.parse.json").read_text()
    )
    retry_audit = json.loads(
        (tmp_path / "02_map_B01_retry1.parse.json").read_text()
    )
    assert first_audit["strict_json_valid"] is False
    assert retry_audit["strict_json_valid"] is True


def test_part_mapping_normalizes_compact_threshold_fields_without_retry(
    tmp_path: Path,
) -> None:
    compact = {
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": "85%",
                "evidence_box_index": "0",
                "reason_code": "shape_and_location",
            }
        ]
    }
    payloads = []

    def runner(payload):
        payloads.append(payload)
        return json.dumps(compact)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    result = client.map_part_batch(
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001", "renders": [{"visible_pixels": 500}]}],
        batch_id="B01",
    )

    assert len(payloads) == 1
    assert result["mappings"][0]["status"] == "matched"
    assert result["mappings"][0]["evidence_view_id"] == "ref_single"
    assert client.repair_events[0]["action"] == "normalized_model_rows"


def test_part_mapping_downgrades_sub_256_match_without_retry(tmp_path: Path) -> None:
    legacy = {
        "schema_version": "qwen-part-palette-map/v1",
        "batch_id": "B01",
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": 0.9,
                "evidence_view_id": "ref_single",
                "evidence_box_index": 0,
                "status": "matched",
                "reason_code": "shape_and_location",
            }
        ],
    }
    calls = 0

    def runner(_payload):
        nonlocal calls
        calls += 1
        return json.dumps(legacy)

    client = LocalStagedQwenClient(
        model="qwen-test", runner=runner, raw_output_dir=tmp_path
    )
    result = client.map_part_batch(
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001", "renders": [{"visible_pixels": 100}]}],
        batch_id="B01",
    )

    mapping = result["mappings"][0]
    assert calls == 1
    assert mapping["status"] == "review"
    assert mapping["reason_code"] == "too_small"
    assert mapping["group_id"] == "G01"
    assert mapping["evidence_box_index"] == 0
    assert mapping["mapping_confidence"] < 0.85


def test_part_mapping_quarantines_invalid_retry_rows(tmp_path: Path) -> None:
    initial = {
        "schema_version": "qwen-part-palette-map/v1",
        "batch_id": "B01",
        "mappings": [
            {
                "part_id": "P0001",
                "group_id": "G01",
                "mapping_confidence": 0.9,
                "evidence_view_id": "ref_single",
                "evidence_box_index": 3,
                "status": "matched",
                "reason_code": "shape_and_location",
            }
        ],
    }
    retry = deepcopy(initial)
    retry["mappings"][0].update({"group_id": "G99", "evidence_box_index": 0})
    responses = iter((json.dumps(initial), json.dumps(retry)))
    client = LocalStagedQwenClient(
        model="qwen-test",
        runner=lambda _payload: next(responses),
        raw_output_dir=tmp_path,
    )
    result = client.map_part_batch(
        reference_view={"id": "ref_single", "image": PNG_DATA_URL},
        cad_view={"id": "cad_iso", "image": PNG_DATA_URL},
        part_id_view={"id": "part_ids_iso", "image": PNG_DATA_URL},
        batch_sheet_view={"id": "batch_parts_B01", "image": PNG_DATA_URL},
        palette=palette(),
        target_parts=[{"part_id": "P0001", "renders": [{"visible_pixels": 100}]}],
        batch_id="B01",
    )
    mapping = result["mappings"][0]
    assert mapping == {
        "part_id": "P0001",
        "group_id": None,
        "mapping_confidence": 0.0,
        "evidence_view_id": None,
        "evidence_box_index": None,
        "status": "unknown",
        "reason_code": "too_small",
    }
    assert client.repair_events[0]["action"] == "quarantined_invalid_rows"
    assert client.repair_events[0]["quarantined_parts"] == ["P0001"]
