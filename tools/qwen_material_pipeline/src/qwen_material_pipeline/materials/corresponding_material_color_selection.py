"""Select per-scope colour intensity from registered actual-CAD renders.

Every candidate keeps the selected MDL identities fixed and changes only an
audited absolute-linear intensity gain per sealed scope.  Selection is local:
a component shares one winning gain, while an independent part may choose its
own.  The authority is registered render evidence, never the candidate's name
or the nominal target RGB alone.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .component_mdl_tournament import score_component_render
from .corresponding_material_color import (
    AUDIT_SCHEMA_VERSION as COLOR_AUDIT_SCHEMA_VERSION,
    CorrespondingMaterialColorError,
    _verify_document_integrity,
)
from .part_id_parameter_tournament import score_part_id_render
from ..usd.material_common import canonical_sha256


SCHEMA_VERSION = "qwen-corresponding-material-color-render-selection/v1"
AUDIT_SCHEMA_VERSION = "qwen-corresponding-material-color-render-selection-audit/v2"
MINIMUM_SCORABLE_SCOPE_PIXELS = 32
MINIMUM_SCOPE_APPEARANCE_SCORE = 0.50
MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS = 16
MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE = 0.35


class CorrespondingMaterialColorSelectionError(ValueError):
    """Raised when colour-render candidates violate the sealed contract."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    gains_by_scope: Mapping[str, float]
    plan: Mapping[str, Any]
    audit: Mapping[str, Any]
    rendered_registry: Mapping[str, Any]
    paths: Mapping[str, str]
    hashes: Mapping[str, str]


def _local_quality_gate(
    *, scope_id: str, score: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit local appearance without turning a quality miss into data loss.

    Malformed or unbound evidence remains a hard error.  A well-formed actual-
    CAD result below the appearance floor is instead returned as ``FAIL`` so
    the selector can retain the best measured candidate, mark it for review,
    and still produce the complete asset needed by whole-view QA.
    """
    raw_pixels = score.get("comparison_pixel_count", 0)
    raw_appearance = score.get("appearance_score")
    if (
        isinstance(raw_pixels, bool)
        or not isinstance(raw_pixels, int)
        or raw_pixels < 0
        or isinstance(raw_appearance, bool)
        or not isinstance(raw_appearance, (int, float))
        or not math.isfinite(float(raw_appearance))
    ):
        raise CorrespondingMaterialColorSelectionError(
            f"winning scope {scope_id} has malformed local quality evidence"
        )
    failures: list[str] = []
    if (
        raw_pixels >= MINIMUM_SCORABLE_SCOPE_PIXELS
        and float(raw_appearance) < MINIMUM_SCOPE_APPEARANCE_SCORE
    ):
        failures.append("scope_appearance_below_floor")
    member_records: list[dict[str, Any]] = []
    raw_members = score.get("member_scores", [])
    if not isinstance(raw_members, list):
        raise CorrespondingMaterialColorSelectionError(
            f"winning scope {scope_id} has malformed member scores"
        )
    for index, raw_member in enumerate(raw_members):
        member = _mapping(raw_member, f"{scope_id}.member_scores[{index}]")
        part_id = _text(member.get("part_id"), f"{scope_id}.member part_id")
        member_pixels = member.get("comparison_pixel_count", 0)
        member_appearance = member.get("appearance_score")
        if (
            isinstance(member_pixels, bool)
            or not isinstance(member_pixels, int)
            or member_pixels < 0
            or isinstance(member_appearance, bool)
            or not isinstance(member_appearance, (int, float))
            or not math.isfinite(float(member_appearance))
        ):
            raise CorrespondingMaterialColorSelectionError(
                f"winning scope {scope_id} has malformed quality for {part_id}"
            )
        passed = (
            member_pixels < MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
            or float(member_appearance)
            >= MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
        )
        if not passed:
            failures.append(f"component_member_below_floor:{part_id}")
        member_records.append(
            {
                "part_id": part_id,
                "comparison_pixel_count": member_pixels,
                "appearance_score": float(member_appearance),
                "evaluated": (
                    member_pixels >= MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
                ),
                "passed": passed,
            }
        )
    gate = {
        "status": "PASS" if not failures else "FAIL",
        "comparison_pixel_count": raw_pixels,
        "appearance_score": float(raw_appearance),
        "scope_evaluated": raw_pixels >= MINIMUM_SCORABLE_SCOPE_PIXELS,
        "minimum_scorable_scope_pixels": MINIMUM_SCORABLE_SCOPE_PIXELS,
        "minimum_scope_appearance_score": MINIMUM_SCOPE_APPEARANCE_SCORE,
        "minimum_scorable_component_member_pixels": (
            MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
        ),
        "minimum_component_member_appearance_score": (
            MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
        ),
        "member_scores": member_records,
        "failure_reasons": failures,
    }
    return gate


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrespondingMaterialColorSelectionError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CorrespondingMaterialColorSelectionError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorrespondingMaterialColorSelectionError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CorrespondingMaterialColorSelectionError(
            f"JSON root must be an object: {resolved}"
        )
    return value


def _assignments(plan: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(plan.get("assignments"), f"{label}.assignments")
    ):
        row = _mapping(raw, f"{label}.assignments[{index}]")
        part_id = _text(row.get("part_id"), f"{label}.assignments[{index}].part_id")
        if part_id in output:
            raise CorrespondingMaterialColorSelectionError(
                f"{label} contains duplicate Part ID {part_id}"
            )
        output[part_id] = row
    return output


def _scopes(audit: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_array(audit.get("scopes"), f"{label}.scopes")):
        row = _mapping(raw, f"{label}.scopes[{index}]")
        scope_id = _text(row.get("scope_id"), f"{label}.scopes[{index}].scope_id")
        if scope_id in output:
            raise CorrespondingMaterialColorSelectionError(
                f"{label} contains duplicate scope {scope_id}"
            )
        output[scope_id] = row
    return output


def load_rendered_color_candidate(
    directory: Path, source_plan_sha256: str
) -> Candidate:
    root = directory.expanduser().resolve(strict=True)
    expected = {
        "plan": root / "part_id_material_plan.color.json",
        "audit": root / "corresponding_material_color_audit.json",
        "apply_report": root / "apply_report.json",
        "rendered_registry": root / "renders" / "part_registry.rendered.json",
    }
    for label, path in expected.items():
        if not path.is_file() or path.is_symlink():
            raise CorrespondingMaterialColorSelectionError(
                f"candidate {root.name} lacks regular {label}: {path}"
            )
    plan = _read_object(expected["plan"])
    audit = _read_object(expected["audit"])
    apply_report = _read_object(expected["apply_report"])
    registry = _read_object(expected["rendered_registry"])
    asset = (
        Path(_text(apply_report.get("output_usd"), "apply_report.output_usd"))
        .expanduser()
        .resolve(strict=True)
    )
    if asset.parent != root or not asset.is_file() or asset.is_symlink():
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} apply report points outside its candidate directory"
        )
    expected["asset"] = asset
    try:
        _verify_document_integrity(audit, f"candidate {root.name} audit")
    except CorrespondingMaterialColorError as exc:
        raise CorrespondingMaterialColorSelectionError(str(exc)) from exc
    if audit.get("schema_version") != COLOR_AUDIT_SCHEMA_VERSION:
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} has unsupported colour audit"
        )
    if audit.get("source_plan_sha256") != source_plan_sha256:
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} source plan mismatch"
        )
    if audit.get("output_plan_sha256") != canonical_sha256(plan):
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} plan/audit mismatch"
        )
    scopes = _scopes(audit, f"candidate {root.name}")
    policy = _mapping(audit.get("policy"), f"candidate {root.name}.policy")
    declared_scope_gains = policy.get("linear_intensity_gains_by_scope")
    if declared_scope_gains is None:
        global_gain = policy.get("linear_intensity_gain")
        declared_scope_gains = {scope_id: global_gain for scope_id in scopes}
    declared_scope_gains = _mapping(
        declared_scope_gains,
        f"candidate {root.name}.policy.linear_intensity_gains_by_scope",
    )
    if set(declared_scope_gains) != set(scopes):
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} gains do not cover its colour scopes"
        )
    gains_by_scope: dict[str, float] = {}
    for scope_id, raw_gain in declared_scope_gains.items():
        if (
            isinstance(raw_gain, bool)
            or not isinstance(raw_gain, (int, float))
            or not math.isfinite(float(raw_gain))
            or not 0.1 <= float(raw_gain) <= 8.0
        ):
            raise CorrespondingMaterialColorSelectionError(
                f"candidate {root.name} has invalid gain for {scope_id}"
            )
        gain = float(raw_gain)
        scope_audit = _mapping(
            scopes[str(scope_id)].get("color_parameter_audit"),
            f"candidate {root.name}.{scope_id}.color_parameter_audit",
        )
        raw_scope_gain = scope_audit.get("linear_intensity_gain")
        if (
            isinstance(raw_scope_gain, bool)
            or not isinstance(raw_scope_gain, (int, float))
            or not math.isfinite(float(raw_scope_gain))
            or float(raw_scope_gain) != gain
        ):
            raise CorrespondingMaterialColorSelectionError(
                f"candidate {root.name} scope gain mismatch for {scope_id}"
            )
        gains_by_scope[str(scope_id)] = gain
    file_hashes = {label: _sha256_file(path) for label, path in expected.items()}
    if apply_report.get("plan_sha256") != canonical_sha256(plan):
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} apply report does not bind its plan"
        )
    if apply_report.get("output_sha256") != file_hashes["asset"]:
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} applied asset hash mismatch"
        )
    if (
        Path(_text(registry.get("asset_usd"), "registry.asset_usd")).resolve()
        != expected["asset"]
    ):
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} render registry points at another asset"
        )
    if registry.get("asset_sha256") != file_hashes["asset"]:
        raise CorrespondingMaterialColorSelectionError(
            f"candidate {root.name} render registry asset hash mismatch"
        )
    return Candidate(
        candidate_id=root.name,
        gains_by_scope=gains_by_scope,
        plan=plan,
        audit=audit,
        rendered_registry=registry,
        paths={label: str(path) for label, path in expected.items()},
        hashes=file_hashes,
    )


def _load_candidate(directory: Path, source_plan_sha256: str) -> Candidate:
    """Backward-compatible private alias for existing callers and tests."""

    return load_rendered_color_candidate(directory, source_plan_sha256)


def select_render_calibrated_color_plan(
    *,
    source_plan: Mapping[str, Any],
    candidates: Sequence[Candidate],
    part_id_evidence: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    minimum_candidate_count: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge the best registered gain independently for every colour scope."""

    if minimum_candidate_count < 1 or len(candidates) < minimum_candidate_count:
        raise CorrespondingMaterialColorSelectionError(
            f"render selection needs at least {minimum_candidate_count} candidates"
        )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CorrespondingMaterialColorSelectionError("candidate IDs must be unique")
    source_assignments = _assignments(source_plan, "source_plan")
    reference_scopes = _scopes(candidates[0].audit, candidates[0].candidate_id)
    reference_scope_contract = {
        scope_id: {
            "member_part_ids": list(scope["member_part_ids"]),
            "material_id": scope["material_id"],
            "target_srgb": list(scope["target_srgb"]),
        }
        for scope_id, scope in reference_scopes.items()
    }
    candidate_assignments: dict[str, dict[str, Mapping[str, Any]]] = {}
    candidate_scopes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for candidate in candidates:
        scopes = _scopes(candidate.audit, candidate.candidate_id)
        contract = {
            scope_id: {
                "member_part_ids": list(scope["member_part_ids"]),
                "material_id": scope["material_id"],
                "target_srgb": list(scope["target_srgb"]),
            }
            for scope_id, scope in scopes.items()
        }
        if contract != reference_scope_contract:
            raise CorrespondingMaterialColorSelectionError(
                f"candidate {candidate.candidate_id} changes the colour scopes"
            )
        assignments = _assignments(candidate.plan, candidate.candidate_id)
        if set(assignments) != set(source_assignments):
            raise CorrespondingMaterialColorSelectionError(
                f"candidate {candidate.candidate_id} changes Part-ID exact cover"
            )
        for part_id, source in source_assignments.items():
            if assignments[part_id].get("material_id") != source.get("material_id"):
                raise CorrespondingMaterialColorSelectionError(
                    f"candidate {candidate.candidate_id} changes material {part_id}"
                )
        candidate_assignments[candidate.candidate_id] = assignments
        candidate_scopes[candidate.candidate_id] = scopes

    output = copy.deepcopy(dict(source_plan))
    output_assignments = _assignments(output, "output")
    selection_records: list[dict[str, Any]] = []
    selected_gain_counts: dict[str, int] = {}
    parameterized_ids: set[str] = set()
    review_scope_ids: list[str] = []
    review_part_ids: set[str] = set()
    for scope_id in sorted(reference_scopes):
        scope = reference_scopes[scope_id]
        members = sorted(
            _text(value, f"{scope_id}.member") for value in scope["member_part_ids"]
        )
        scores: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(members) == 1:
                score = score_part_id_render(
                    part_id=members[0],
                    evidence=part_id_evidence,
                    spatial_mapping_report=spatial_mapping_report,
                    rendered_registry=candidate.rendered_registry,
                )
            else:
                score = score_component_render(
                    component_id=scope_id,
                    member_part_ids=members,
                    evidence=part_id_evidence,
                    spatial_mapping_report=spatial_mapping_report,
                    rendered_registry=candidate.rendered_registry,
                )
            scores.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "linear_intensity_gain": candidate.gains_by_scope[scope_id],
                    "score": score,
                    "rendered_registry_sha256": candidate.hashes["rendered_registry"],
                    "applied_asset_sha256": candidate.hashes["asset"],
                }
            )
        winner = min(
            scores,
            key=lambda row: (
                -float(row["score"]["appearance_score"]),
                float(row["linear_intensity_gain"]),
                str(row["candidate_id"]),
            ),
        )
        winner_id = str(winner["candidate_id"])
        local_quality_gate = _local_quality_gate(
            scope_id=scope_id,
            score=_mapping(winner["score"], f"{scope_id}.winning score"),
        )
        if local_quality_gate["status"] != "PASS":
            review_scope_ids.append(scope_id)
            review_part_ids.update(members)
        selected_gain_counts[str(winner["linear_intensity_gain"])] = (
            selected_gain_counts.get(str(winner["linear_intensity_gain"]), 0) + 1
        )
        for part_id in members:
            selected = candidate_assignments[winner_id][part_id]
            parameters = selected.get("parameters")
            if not isinstance(parameters, Mapping) or not parameters:
                raise CorrespondingMaterialColorSelectionError(
                    f"winning scope {scope_id} has no parameters for {part_id}"
                )
            output_assignments[part_id]["parameters"] = copy.deepcopy(dict(parameters))
            provenance = dict(
                _mapping(
                    output_assignments[part_id].get("provenance"),
                    f"{part_id}.provenance",
                )
            )
            provenance["corresponding_material_color_render_selection"] = {
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_id,
                "selected_candidate_id": winner_id,
                "linear_intensity_gain": winner["linear_intensity_gain"],
                "material_id_unchanged": True,
                "local_quality_gate_status": local_quality_gate["status"],
                "best_available_result_retained": (
                    local_quality_gate["status"] != "PASS"
                ),
            }
            output_assignments[part_id]["provenance"] = provenance
            parameterized_ids.add(part_id)
        selection_records.append(
            {
                "scope_id": scope_id,
                "member_part_ids": members,
                "material_id": scope["material_id"],
                "target_srgb": scope["target_srgb"],
                "selected_candidate_id": winner_id,
                "selected_linear_intensity_gain": winner["linear_intensity_gain"],
                "selected_appearance_score": winner["score"]["appearance_score"],
                "local_quality_gate": local_quality_gate,
                "candidates": scores,
            }
        )
    expected_parameterized = {
        part_id
        for scope in reference_scopes.values()
        for part_id in scope["member_part_ids"]
    }
    if parameterized_ids != expected_parameterized:
        raise CorrespondingMaterialColorSelectionError(
            "selected scopes do not exactly cover corresponding materials"
        )
    for part_id, assignment in output_assignments.items():
        if part_id not in parameterized_ids and assignment.get("parameters") not in (
            None,
            {},
        ):
            raise CorrespondingMaterialColorSelectionError(
                f"render selection parameterized non-corresponding Part ID {part_id}"
            )
    provenance = dict(_mapping(output.get("provenance"), "output.provenance"))
    provenance["corresponding_material_color_render_selection"] = {
        "schema_version": SCHEMA_VERSION,
        "source_plan_sha256": canonical_sha256(source_plan),
        "candidate_ids": sorted(candidate_ids),
        "colour_scope_count": len(reference_scopes),
        "parameterized_part_count": len(parameterized_ids),
        "material_identity_changes": 0,
        "selection_authority": "registered_actual_cad_part_scope_appearance_score",
        "quality_rejection_behavior": (
            "retain_best_rendered_candidate_and_continue_with_review"
        ),
        "review_required_scope_ids": sorted(review_scope_ids),
    }
    output["provenance"] = provenance
    local_quality_status = "REVIEW" if review_scope_ids else "PASS"
    audit_unsigned = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": local_quality_status,
        "source_plan_sha256": canonical_sha256(source_plan),
        "output_plan_sha256": canonical_sha256(output),
        "part_id_evidence_sha256": canonical_sha256(part_id_evidence),
        "spatial_mapping_report_sha256": canonical_sha256(spatial_mapping_report),
        "candidate_count": len(candidates),
        "candidate_ids": sorted(candidate_ids),
        "summary": {
            "colour_scope_count": len(reference_scopes),
            "parameterized_part_count": len(parameterized_ids),
            "material_identity_change_count": 0,
            "local_quality_gate_status": local_quality_status,
            "local_quality_pass_scope_count": (
                len(reference_scopes) - len(review_scope_ids)
            ),
            "local_quality_review_scope_count": len(review_scope_ids),
            "local_quality_review_part_count": len(review_part_ids),
            "local_quality_review_scope_ids": sorted(review_scope_ids),
            "local_quality_review_part_ids": sorted(review_part_ids),
            "quality_rejection_behavior": (
                "retain_best_rendered_candidate_and_continue_with_review"
            ),
            "minimum_scorable_scope_pixels": MINIMUM_SCORABLE_SCOPE_PIXELS,
            "minimum_scope_appearance_score": MINIMUM_SCOPE_APPEARANCE_SCORE,
            "minimum_scorable_component_member_pixels": (
                MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
            ),
            "minimum_component_member_appearance_score": (
                MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
            ),
            "selected_gain_scope_counts": selected_gain_counts,
        },
        "selections": selection_records,
    }
    audit = {
        **audit_unsigned,
        "integrity": {"document_sha256": canonical_sha256(audit_unsigned)},
    }
    return output, audit


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--part-id-evidence", type=Path, required=True)
    parser.add_argument("--spatial-mapping-report", type=Path, required=True)
    parser.add_argument("--candidate-dir", action="append", type=Path, required=True)
    parser.add_argument("--minimum-candidate-count", type=int, default=2)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_plan = _read_object(args.source_plan)
    source_plan_sha256 = canonical_sha256(source_plan)
    candidates = [
        load_rendered_color_candidate(path, source_plan_sha256)
        for path in args.candidate_dir
    ]
    output, audit = select_render_calibrated_color_plan(
        source_plan=source_plan,
        candidates=candidates,
        part_id_evidence=_read_object(args.part_id_evidence),
        spatial_mapping_report=_read_object(args.spatial_mapping_report),
        minimum_candidate_count=args.minimum_candidate_count,
    )
    _write_object(args.output_plan, output)
    _write_object(args.audit, audit)
    print(
        "Render-calibrated corresponding-material colour selection complete: "
        f"{audit['summary']['parameterized_part_count']} parts / "
        f"{audit['summary']['colour_scope_count']} scopes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
