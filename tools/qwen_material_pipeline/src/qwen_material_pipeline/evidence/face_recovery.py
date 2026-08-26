"""Fail-closed face-subset recovery for unattended material authoring.

The normal confidence gate deliberately rejects a whole-Mesh material binding
when topology suggests that one prim may contain multiple appearances.  This
module handles that narrow case without weakening the whole-part gate:

* load deterministic surface-region projections from ``face_region_evidence``;
* reuse only trusted reference-to-CAD registrations from the spatial gate;
* require raw-photo and MVInverse-albedo agreement in multiple views;
* reject label changes under registration and pixel perturbations; and
* emit subset-only assignments that preserve the parent Mesh binding.

MVInverse remains image-space evidence.  It supplies albedo/PBR observations;
the deterministic CAD region AOV supplies the face localization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from qwen_material_pipeline.mvinverse.autonomy import parameterize_auto_material_plan
from qwen_material_pipeline.mvinverse.evidence import (
    MVInverseEvidenceError,
    validate_mvinverse_evidence,
)
from qwen_material_pipeline.materials.tuning import tuning_profile_for_material
from qwen_material_pipeline.evidence.spatial import (
    SpatialMappingError,
    _base_color_labels,
    _pixel_label,
    _verify_report,
)


SCHEMA_VERSION = "qwen-face-material-recovery/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"
FACE_REGION_SCHEMA_VERSION = "qwen-face-region-evidence/v1"
CONFIDENCE_GATE_SCHEMA_VERSION = "qwen-material-confidence-gate/v1"
GROUP_MATERIAL_SCHEMA_VERSION = "qwen-palette-material/v1"
MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION = (
    "qwen-derived-material-selection-confidence/v1"
)
SURFACE_PATCH_METHOD = "smooth_edge_plus_seed_normal_coherence/v2"
MINIMUM_TRUSTED_REGISTERED_VIEWS = 2
INSUFFICIENT_TRUSTED_VIEWS_REASON = "INSUFFICIENT_TRUSTED_REGISTERED_VIEWS"

DEFAULT_POLICY: dict[str, float | int | bool] = {
    "minimum_projected_pixels": 256,
    "minimum_support_views": 2,
    "minimum_raw_color_share": 0.70,
    "minimum_albedo_color_share": 0.70,
    "minimum_color_margin": 0.30,
    "enable_shadow_compensation": True,
    "shadow_minimum_raw_color_share": 0.55,
    "shadow_minimum_albedo_color_share": 0.65,
    "projection_shift_pixels": 2,
    "projection_scale_delta": 0.01,
    "maximum_patch_normal_deviation_degrees": 55.0,
    "minimum_spatial_override_support_views": 2,
    "minimum_spatial_override_registration_views": 2,
    "minimum_unknown_spatial_override_support_views": 3,
    "minimum_unknown_spatial_override_registration_views": 3,
    "minimum_spatial_override_projected_pixels": 512,
    "minimum_spatial_override_color_share": 0.65,
    "minimum_spatial_override_color_margin": 0.30,
    "minimum_spatial_override_model_confidence": 0.60,
    "minimum_spatial_override_material_views": 2,
    "minimum_spatial_override_material_confidence": 0.90,
    "minimum_spatial_override_material_margin": 0.10,
    "minimum_override_patch_projected_pixels": 128,
    "require_override_strict_support": True,
}

_ALLOWED_CANDIDATE_REASON_CODES = {
    "MAPPING_BELOW_AUTO",
    "MULTI_MATERIAL_RISK",
    "GEOMETRY_MULTI_MATERIAL_RISK",
}

_ALLOWED_SPATIAL_OVERRIDE_REASON_CODES = {
    "MODEL_STATUS_REQUIRES_REVIEW",
    "MODEL_CONFIDENCE_BELOW_AUTO",
    "MAPPING_BELOW_AUTO",
    "CANDIDATE_MARGIN_BELOW_AUTO",
    "CROSS_VIEW_MATERIAL_CONFLICT",
    "INDEPENDENT_VIEW_PREDICTIONS_UNAVAILABLE",
    "INSUFFICIENT_INDEPENDENT_REFERENCES",
    "NO_INDEPENDENT_REFERENCE_EVIDENCE",
    "RETRIEVAL_TOP_DISAGREEMENT",
    "MATERIAL_ORDER_DISAGREEMENT",
    "MATERIAL_CHOICE_BELOW_AUTO",
    "CANDIDATE_MARGIN_UNAVAILABLE",
    "MULTI_MATERIAL_RISK",
    "GEOMETRY_MULTI_MATERIAL_RISK",
}


class FaceMaterialRecoveryError(ValueError):
    """Raised when face evidence cannot be consumed without guessing."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FaceMaterialRecoveryError(f"input is not canonical JSON: {exc}") from exc


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FaceMaterialRecoveryError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FaceMaterialRecoveryError(f"{label} must be an array")
    return value


def _read_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise FaceMaterialRecoveryError(
            f"unable to resolve face-region manifest: {path!r}"
        ) from exc
    if not resolved.is_file():
        raise FaceMaterialRecoveryError(
            f"face-region manifest is not a file: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FaceMaterialRecoveryError(
            f"unable to read face-region manifest: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise FaceMaterialRecoveryError("face-region manifest must be an object")
    return value, resolved


def _resolve_relative(owner: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FaceMaterialRecoveryError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FaceMaterialRecoveryError(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise FaceMaterialRecoveryError(f"{label} is not a file: {resolved}")
    return resolved


def _validated_policy(
    policy: Mapping[str, Any] | None,
) -> dict[str, float | int | bool]:
    output = dict(DEFAULT_POLICY)
    if policy is not None:
        unknown = set(policy) - set(output)
        if unknown:
            raise FaceMaterialRecoveryError(
                f"unknown face-material policy fields: {sorted(unknown)}"
            )
        output.update(policy)
    for name in (
        "minimum_projected_pixels",
        "minimum_support_views",
        "projection_shift_pixels",
        "minimum_spatial_override_support_views",
        "minimum_spatial_override_registration_views",
        "minimum_unknown_spatial_override_support_views",
        "minimum_unknown_spatial_override_registration_views",
        "minimum_spatial_override_projected_pixels",
        "minimum_spatial_override_material_views",
        "minimum_override_patch_projected_pixels",
    ):
        value = output[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FaceMaterialRecoveryError(
                f"policy.{name} must be a non-negative integer"
            )
    if int(output["minimum_projected_pixels"]) < 1:
        raise FaceMaterialRecoveryError(
            "policy.minimum_projected_pixels must be positive"
        )
    if int(output["minimum_support_views"]) < 2:
        raise FaceMaterialRecoveryError(
            "policy.minimum_support_views must be at least 2"
        )
    if int(output["minimum_spatial_override_support_views"]) < 2:
        raise FaceMaterialRecoveryError(
            "policy.minimum_spatial_override_support_views must be at least 2"
        )
    if int(output["minimum_spatial_override_registration_views"]) < 2:
        raise FaceMaterialRecoveryError(
            "policy.minimum_spatial_override_registration_views must be at least 2"
        )
    if int(output["minimum_unknown_spatial_override_support_views"]) < 3:
        raise FaceMaterialRecoveryError(
            "policy.minimum_unknown_spatial_override_support_views must be at least 3"
        )
    if int(output["minimum_unknown_spatial_override_registration_views"]) < 3:
        raise FaceMaterialRecoveryError(
            "policy.minimum_unknown_spatial_override_registration_views must be "
            "at least 3"
        )
    if int(output["minimum_spatial_override_material_views"]) < 2:
        raise FaceMaterialRecoveryError(
            "policy.minimum_spatial_override_material_views must be at least 2"
        )
    for name in (
        "minimum_spatial_override_projected_pixels",
        "minimum_override_patch_projected_pixels",
    ):
        if int(output[name]) < 1:
            raise FaceMaterialRecoveryError(f"policy.{name} must be positive")
    for name in ("enable_shadow_compensation", "require_override_strict_support"):
        if type(output[name]) is not bool:
            raise FaceMaterialRecoveryError(f"policy.{name} must be boolean")
    for name in (
        "minimum_raw_color_share",
        "minimum_albedo_color_share",
        "minimum_color_margin",
        "shadow_minimum_raw_color_share",
        "shadow_minimum_albedo_color_share",
        "projection_scale_delta",
        "minimum_spatial_override_color_share",
        "minimum_spatial_override_color_margin",
        "minimum_spatial_override_model_confidence",
        "minimum_spatial_override_material_confidence",
        "minimum_spatial_override_material_margin",
    ):
        value = output[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise FaceMaterialRecoveryError(f"policy.{name} must be in [0,1]")
        output[name] = float(value)
    maximum_deviation = output["maximum_patch_normal_deviation_degrees"]
    if (
        isinstance(maximum_deviation, bool)
        or not isinstance(maximum_deviation, (int, float))
        or not math.isfinite(float(maximum_deviation))
        or not 0.0 <= float(maximum_deviation) <= 90.0
    ):
        raise FaceMaterialRecoveryError(
            "policy.maximum_patch_normal_deviation_degrees must be in [0,90]"
        )
    output["maximum_patch_normal_deviation_degrees"] = float(maximum_deviation)
    if float(output["shadow_minimum_raw_color_share"]) > float(
        output["minimum_raw_color_share"]
    ):
        raise FaceMaterialRecoveryError(
            "shadow raw threshold cannot exceed strict threshold"
        )
    if float(output["shadow_minimum_albedo_color_share"]) > float(
        output["minimum_albedo_color_share"]
    ):
        raise FaceMaterialRecoveryError(
            "shadow albedo threshold cannot exceed strict threshold"
        )
    return output


def _open_bgr(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise FaceMaterialRecoveryError(f"unable to decode {label}: {path}")
    return image


def _palette_groups(palette: Mapping[str, Any]) -> list[dict[str, Any]]:
    if palette.get("schema_version") != "qwen-material-palette/v1":
        raise FaceMaterialRecoveryError("canonical palette schema is unsupported")
    groups = _array(palette.get("groups"), "canonical_palette.groups")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(groups):
        group = _object(raw, f"canonical_palette.groups[{index}]")
        group_id = group.get("group_id")
        base_color = group.get("base_color")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in seen
            or not isinstance(base_color, str)
            or not base_color
        ):
            raise FaceMaterialRecoveryError("canonical palette groups are invalid")
        seen.add(group_id)
        result.append(
            {
                "group_id": group_id,
                "base_color": base_color,
                "visual_description": str(group.get("visual_description", group_id)),
            }
        )
    if not result:
        raise FaceMaterialRecoveryError("canonical palette has no groups")
    return result


def _score_mask(
    image: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    *,
    minimum_share: float,
    minimum_margin: float,
) -> dict[str, Any]:
    y, x = np.where(mask > 0)
    labels = Counter(_pixel_label(pixel) for pixel in image[y, x])
    sample_count = int(len(x))
    rows: list[dict[str, Any]] = []
    for group in groups:
        accepted = _base_color_labels(str(group["base_color"]))
        matching = sum(labels[label] for label in accepted)
        rows.append(
            {
                "group_id": group["group_id"],
                "base_color": group["base_color"],
                "matching_pixels": int(matching),
                "color_share": round(
                    matching / sample_count if sample_count else 0.0, 8
                ),
            }
        )
    rows.sort(key=lambda item: (-float(item["color_share"]), str(item["group_id"])))
    best = rows[0]
    runner_share = float(rows[1]["color_share"]) if len(rows) > 1 else 0.0
    margin = float(best["color_share"]) - runner_share
    resolved = (
        str(best["group_id"])
        if float(best["color_share"]) >= minimum_share and margin >= minimum_margin
        else None
    )
    return {
        "sampled_pixels": sample_count,
        "color_counts": dict(sorted(labels.items())),
        "group_scores": rows,
        "best_group_id": best["group_id"],
        "best_color_share": best["color_share"],
        "color_margin": round(margin, 8),
        "resolved_group_id": resolved,
    }


def _score_pair(
    raw_image: np.ndarray,
    albedo_image: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    *,
    raw_minimum_share: float,
    albedo_minimum_share: float,
    minimum_margin: float,
) -> dict[str, Any]:
    return {
        "raw": _score_mask(
            raw_image,
            mask,
            groups,
            minimum_share=raw_minimum_share,
            minimum_margin=minimum_margin,
        ),
        "mvinverse_albedo": _score_mask(
            albedo_image,
            mask,
            groups,
            minimum_share=albedo_minimum_share,
            minimum_margin=minimum_margin,
        ),
    }


def _pair_group(pair: Mapping[str, Any]) -> str | None:
    raw = _object(pair.get("raw"), "pair.raw").get("resolved_group_id")
    albedo = _object(pair.get("mvinverse_albedo"), "pair.mvinverse_albedo").get(
        "resolved_group_id"
    )
    return str(raw) if isinstance(raw, str) and raw == albedo else None


def _shift_mask(mask: np.ndarray, x: float, y: float) -> np.ndarray:
    return cv2.warpAffine(
        mask,
        np.asarray([[1.0, 0.0, x], [0.0, 1.0, y]], dtype=np.float32),
        (int(mask.shape[1]), int(mask.shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    y, x = np.where(mask > 0)
    if not len(x):
        return mask.copy()
    center = (float(np.mean(x)), float(np.mean(y)))
    matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
    return cv2.warpAffine(
        mask,
        matrix,
        (int(mask.shape[1]), int(mask.shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _face_indices(patch: Mapping[str, Any], face_count: int) -> list[int]:
    ranges = _array(patch.get("face_ranges"), "surface_patch.face_ranges")
    result: list[int] = []
    for raw in ranges:
        pair = _array(raw, "surface_patch.face_ranges[]")
        if (
            len(pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in pair
            )
            or pair[0] < 0
            or pair[1] < pair[0]
            or pair[1] >= face_count
        ):
            raise FaceMaterialRecoveryError("surface patch has invalid face ranges")
        result.extend(range(pair[0], pair[1] + 1))
    if len(result) != len(set(result)) or len(result) != patch.get("face_count"):
        raise FaceMaterialRecoveryError(
            "surface patch face references are inconsistent"
        )
    explicit = patch.get("face_indices")
    if explicit is not None and list(explicit) != result:
        raise FaceMaterialRecoveryError("surface patch explicit/range faces disagree")
    return result


def _candidate_decisions(
    confidence_gate: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if confidence_gate.get("schema_version") != CONFIDENCE_GATE_SCHEMA_VERSION:
        raise FaceMaterialRecoveryError("confidence-gate schema is unsupported")
    gate_policy = _object(confidence_gate.get("policy"), "confidence_gate.policy")
    minimum_model = float(gate_policy.get("auto_model_confidence", 0.90))
    minimum_mapping = float(gate_policy.get("review_mapping_confidence", 0.60))
    minimum_choice = float(gate_policy.get("auto_material_choice_confidence", 0.85))
    minimum_margin = float(gate_policy.get("minimum_candidate_margin", 0.15))
    minimum_references = int(gate_policy.get("minimum_independent_references", 2))
    result: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(
        _array(confidence_gate.get("decisions"), "confidence_gate.decisions")
    ):
        decision = _object(raw, f"confidence_gate.decisions[{index}]")
        part_id = decision.get("part_id")
        if not isinstance(part_id, str) or not part_id:
            raise FaceMaterialRecoveryError("confidence decision has invalid part_id")
        if part_id in seen:
            raise FaceMaterialRecoveryError(f"duplicate confidence decision: {part_id}")
        seen.add(part_id)
        reasons = set(
            _array(decision.get("reason_codes"), f"decision {part_id}.reason_codes")
        )
        model = decision.get("model")
        mapping = decision.get("mapping")
        choice = decision.get("material_choice")
        geometry = decision.get("geometry_risk")
        if not all(
            isinstance(item, Mapping) for item in (model, mapping, choice, geometry)
        ):
            continue
        model = _object(model, f"decision {part_id}.model")
        mapping = _object(mapping, f"decision {part_id}.mapping")
        choice = _object(choice, f"decision {part_id}.material_choice")
        geometry = _object(geometry, f"decision {part_id}.geometry_risk")
        risk = geometry.get("risk")
        material_id = decision.get("material_id")
        group_id = mapping.get("group_id")
        candidate_margin = decision.get("candidate_margin")
        eligible = (
            decision.get("decision") == "preserve"
            and decision.get("multi_material_risk") is True
            and isinstance(risk, Mapping)
            and risk.get("multi_material_risk") is True
            and reasons
            and reasons <= _ALLOWED_CANDIDATE_REASON_CODES
            and model.get("status") == "auto"
            and isinstance(model.get("confidence"), (int, float))
            and not isinstance(model.get("confidence"), bool)
            and float(model["confidence"]) >= minimum_model
            and isinstance(model.get("independent_reference_count"), int)
            and model["independent_reference_count"] >= minimum_references
            and isinstance(group_id, str)
            and bool(group_id)
            and isinstance(mapping.get("confidence"), (int, float))
            and not isinstance(mapping.get("confidence"), bool)
            and float(mapping["confidence"]) >= minimum_mapping
            and choice.get("confirmed") is True
            and choice.get("forward_material_id") == material_id
            and choice.get("reverse_material_id") == material_id
            and isinstance(choice.get("forward_confidence"), (int, float))
            and isinstance(choice.get("reverse_confidence"), (int, float))
            and min(
                float(choice["forward_confidence"]), float(choice["reverse_confidence"])
            )
            >= minimum_choice
            and isinstance(candidate_margin, (int, float))
            and not isinstance(candidate_margin, bool)
            and float(candidate_margin) >= minimum_margin
            and isinstance(material_id, str)
            and bool(material_id)
        )
        if eligible:
            result[part_id] = decision
    return result


def _group_material_candidates(
    *,
    group_materials: Mapping[str, Any] | None,
    material_choice_audit: Mapping[str, Any] | None,
    allowed_material_ids: set[str],
    policy: Mapping[str, float | int | bool],
) -> dict[str, dict[str, Any]]:
    """Return only group choices with independent, order-stable Qwen evidence."""

    if group_materials is None and material_choice_audit is None:
        return {}
    if group_materials is None or material_choice_audit is None:
        raise FaceMaterialRecoveryError(
            "spatial override requires both group materials and material-choice audit"
        )
    if group_materials.get("schema_version") != GROUP_MATERIAL_SCHEMA_VERSION:
        raise FaceMaterialRecoveryError("group-material schema is unsupported")
    minimum_confidence = float(policy["minimum_spatial_override_material_confidence"])
    minimum_margin = float(policy["minimum_spatial_override_material_margin"])
    minimum_views = int(policy["minimum_spatial_override_material_views"])
    result: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(
        _array(group_materials.get("selections"), "group_materials.selections")
    ):
        selection = _object(raw, f"group_materials.selections[{index}]")
        group_id = selection.get("group_id")
        material_id = selection.get("material_id")
        confidence = selection.get("confidence")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in seen
            or not isinstance(material_id, str)
            or not material_id
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise FaceMaterialRecoveryError("group-material selections are invalid")
        seen.add(group_id)
        audit = material_choice_audit.get(group_id)
        if not isinstance(audit, Mapping):
            continue
        audit = _object(audit, f"material_choice_audit[{group_id}]")
        forward = audit.get("forward")
        reverse = audit.get("reverse")
        independent = audit.get("independent_view_choices")
        if not isinstance(forward, Mapping) or not isinstance(reverse, Mapping):
            continue
        if not isinstance(independent, Sequence) or isinstance(
            independent, (str, bytes)
        ):
            continue
        forward = _object(forward, f"material_choice_audit[{group_id}].forward")
        reverse = _object(reverse, f"material_choice_audit[{group_id}].reverse")
        direction_confidences = [forward.get("confidence"), reverse.get("confidence")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in direction_confidences
        ):
            continue
        raw_derived_confidence = audit.get("selection_confidence")
        if raw_derived_confidence is None:
            derived_confidence = min(
                float(value) for value in direction_confidences
            )
        else:
            derivation = audit.get("confidence_derivation")
            if (
                isinstance(raw_derived_confidence, bool)
                or not isinstance(raw_derived_confidence, (int, float))
                or not math.isfinite(float(raw_derived_confidence))
                or not 0.0 <= float(raw_derived_confidence) <= 1.0
                or not isinstance(derivation, Mapping)
                or derivation.get("schema_version")
                != MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION
                or derivation.get("reported_confidence_is_authoritative")
                is not False
                or derivation.get("derived_confidence")
                != float(raw_derived_confidence)
                or float(confidence) != float(raw_derived_confidence)
            ):
                raise FaceMaterialRecoveryError(
                    f"group material {group_id} derived-confidence contract "
                    "is inconsistent"
                )
            derived_confidence = float(raw_derived_confidence)
        direction_match = all(
            record.get("group_id") == group_id
            and record.get("material_id") == material_id
            for record in (forward, reverse)
        )
        if (
            not direction_match
            and audit.get("confirmed") is True
            and audit.get("confirmation_basis")
            == "mvinverse_tunable_module_agreement"
            and audit.get("confirmed_material_id") == material_id
        ):
            selected_profile = tuning_profile_for_material(material_id)
            selected_module = material_id.split("#", 1)[0]
            direction_match = selected_profile is not None and all(
                isinstance(record.get("material_id"), str)
                and str(record["material_id"]).split("#", 1)[0] == selected_module
                and tuning_profile_for_material(str(record["material_id"]))
                == selected_profile
                for record in (forward, reverse)
            )
        view_ids: set[str] = set()
        view_confidences: list[float] = []
        view_margins: list[float] = []
        views_match = True
        for view_index, raw_view in enumerate(independent):
            if not isinstance(raw_view, Mapping):
                views_match = False
                break
            view = _object(
                raw_view,
                f"material_choice_audit[{group_id}].independent[{view_index}]",
            )
            view_id = view.get("view_id")
            view_confidence = view.get("confidence")
            candidate_margin = view.get("candidate_margin")
            association = view.get("mvinverse_association")
            if (
                not isinstance(view_id, str)
                or not view_id
                or view_id in view_ids
                or view.get("canonical_group_id") != group_id
                or view.get("material_id") != material_id
                or isinstance(view_confidence, bool)
                or not isinstance(view_confidence, (int, float))
                or not math.isfinite(float(view_confidence))
                or isinstance(candidate_margin, bool)
                or not isinstance(candidate_margin, (int, float))
                or not math.isfinite(float(candidate_margin))
                or not isinstance(association, Mapping)
                or association.get("status") != "matched"
                or not isinstance(association.get("matched_group_id"), str)
                or not association.get("matched_group_id")
                or (
                    isinstance(association.get("candidate_group_ids"), Sequence)
                    and not isinstance(
                        association.get("candidate_group_ids"), (str, bytes)
                    )
                    and association.get("matched_group_id")
                    not in association.get("candidate_group_ids")
                )
            ):
                views_match = False
                break
            view_ids.add(view_id)
            view_confidences.append(float(view_confidence))
            view_margins.append(float(candidate_margin))
        # Direction/view confidence numbers are Qwen self-reports retained for
        # audit only.  ``derived_confidence`` is the independently gated
        # authoring signal.
        all_confidences = [derived_confidence]
        eligible = (
            selection.get("confirmed") is True
            and audit.get("confirmed") is True
            and audit.get("chosen_retrieval_rank") == 1
            and audit.get("model_choice_matches_retrieval_top") is True
            and material_id in allowed_material_ids
            and direction_match
            and views_match
            and len(view_ids) >= minimum_views
            and min(all_confidences) >= minimum_confidence
            and min(view_margins, default=-1.0) >= minimum_margin
        )
        if eligible:
            result[group_id] = {
                "group_id": group_id,
                "material_id": material_id,
                "confidence": min(all_confidences),
                "independent_view_ids": sorted(view_ids),
                "minimum_candidate_margin": min(view_margins),
            }
    return result


def _spatial_override_candidates(
    *,
    confidence_gate: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    group_material_candidates: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, float | int | bool],
) -> dict[str, dict[str, Any]]:
    """Select review parts whose photo-space evidence corrects a Qwen mapping."""

    if not group_material_candidates:
        return {}
    if confidence_gate.get("schema_version") != CONFIDENCE_GATE_SCHEMA_VERSION:
        raise FaceMaterialRecoveryError("confidence-gate schema is unsupported")
    decisions: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(confidence_gate.get("decisions"), "confidence_gate.decisions")
    ):
        decision = _object(raw, f"confidence_gate.decisions[{index}]")
        part_id = decision.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in decisions:
            raise FaceMaterialRecoveryError(
                "confidence decisions have invalid part IDs"
            )
        decisions[part_id] = decision

    minimum_supports = int(policy["minimum_spatial_override_support_views"])
    minimum_registrations = int(policy["minimum_spatial_override_registration_views"])
    minimum_unknown_supports = int(
        policy["minimum_unknown_spatial_override_support_views"]
    )
    minimum_unknown_registrations = int(
        policy["minimum_unknown_spatial_override_registration_views"]
    )
    minimum_pixels = int(policy["minimum_spatial_override_projected_pixels"])
    minimum_share = float(policy["minimum_spatial_override_color_share"])
    minimum_margin = float(policy["minimum_spatial_override_color_margin"])
    minimum_model = float(policy["minimum_spatial_override_model_confidence"])
    result: dict[str, dict[str, Any]] = {}
    seen_parts: set[str] = set()
    for index, raw_part in enumerate(
        _array(spatial_mapping_report.get("parts"), "spatial.parts")
    ):
        part = _object(raw_part, f"spatial.parts[{index}]")
        part_id = part.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in seen_parts:
            raise FaceMaterialRecoveryError("spatial report has invalid part IDs")
        seen_parts.add(part_id)
        decision = decisions.get(part_id)
        if decision is None or decision.get("decision") not in {"review", "preserve"}:
            continue
        reasons = set(
            _array(decision.get("reason_codes"), f"decision {part_id}.reason_codes")
        )
        model = decision.get("model")
        geometry = decision.get("geometry_risk")
        mapping = decision.get("mapping")
        if not all(isinstance(value, Mapping) for value in (model, geometry, mapping)):
            continue
        model = _object(model, f"decision {part_id}.model")
        geometry = _object(geometry, f"decision {part_id}.geometry_risk")
        risk = geometry.get("risk")
        model_confidence = model.get("confidence")
        review_model_eligible = (
            reasons
            and reasons <= _ALLOWED_SPATIAL_OVERRIDE_REASON_CODES
            and model.get("status") in {"review", "auto"}
            and model.get("unknown_reason_code") is None
            and isinstance(model_confidence, (int, float))
            and not isinstance(model_confidence, bool)
            and math.isfinite(float(model_confidence))
            and float(model_confidence) >= minimum_model
            and isinstance(risk, Mapping)
            and isinstance(risk.get("multi_material_risk"), bool)
        )
        unknown_model_eligible = (
            decision.get("decision") == "preserve"
            and reasons == {"NO_MODEL_ASSIGNMENT"}
            and model.get("status") == "unknown"
            and model.get("unknown_reason_code") == "not_in_reference"
            and isinstance(risk, Mapping)
            and isinstance(risk.get("multi_material_risk"), bool)
        )
        if not review_model_eligible and not unknown_model_eligible:
            continue
        conflict_ids = part.get("conflict_view_ids")
        counts = part.get("resolved_support_counts")
        if (
            not isinstance(conflict_ids, Sequence)
            or isinstance(conflict_ids, (str, bytes))
            or list(conflict_ids)
            or not isinstance(counts, Mapping)
        ):
            continue
        supported_groups = [
            str(group_id)
            for group_id, count in counts.items()
            if isinstance(count, int) and not isinstance(count, bool) and count > 0
        ]
        if len(supported_groups) != 1:
            continue
        target_group_id = supported_groups[0]
        required_supports = (
            minimum_unknown_supports if unknown_model_eligible else minimum_supports
        )
        required_registrations = (
            minimum_unknown_registrations
            if unknown_model_eligible
            else minimum_registrations
        )
        if (
            counts.get(target_group_id, 0) < required_supports
            or target_group_id not in group_material_candidates
        ):
            continue
        observations: list[Mapping[str, Any]] = []
        valid = True
        registration_count = 0
        for raw_observation in _array(
            part.get("observations"), f"spatial part {part_id}.observations"
        ):
            observation = _object(raw_observation, f"spatial observation {part_id}")
            if observation.get("classification") != "resolved":
                continue
            if observation.get("canonical_group_id") != target_group_id:
                valid = False
                break
            group_scores = observation.get("group_scores")
            if not isinstance(group_scores, Sequence) or isinstance(
                group_scores, (str, bytes)
            ):
                valid = False
                break
            target_scores = [
                score
                for score in group_scores
                if isinstance(score, Mapping)
                and score.get("canonical_group_id") == target_group_id
            ]
            share = max(
                (float(score.get("color_share", -1.0)) for score in target_scores),
                default=-1.0,
            )
            projected_pixels = observation.get("projected_part_pixels")
            color_margin = observation.get("color_margin")
            registration = observation.get("registration_label_stable")
            if (
                isinstance(projected_pixels, bool)
                or not isinstance(projected_pixels, int)
                or projected_pixels < minimum_pixels
                or isinstance(color_margin, bool)
                or not isinstance(color_margin, (int, float))
                or not math.isfinite(float(color_margin))
                or float(color_margin) < minimum_margin
                or share < minimum_share
                or observation.get("perturbation_label_stable") is not True
                or registration is False
            ):
                valid = False
                break
            if registration is True:
                registration_count += 1
            observations.append(observation)
        if (
            not valid
            or len(observations) < required_supports
            or len(observations) != counts.get(target_group_id)
            or registration_count < required_registrations
        ):
            continue
        material = group_material_candidates[target_group_id]
        minimum_observed_share = min(
            max(
                float(score["color_share"])
                for score in observation["group_scores"]
                if score.get("canonical_group_id") == target_group_id
            )
            for observation in observations
        )
        result[part_id] = {
            "decision": decision,
            "candidate_source": "spatial_consensus_override",
            "spatial_override_lane": (
                "three_view_unknown_recovery"
                if unknown_model_eligible
                else "review_mapping_correction"
            ),
            "group_id": target_group_id,
            "material_id": material["material_id"],
            "confidence": float(material["confidence"]),
            "minimum_spatial_color_share": minimum_observed_share,
            "spatial_supporting_view_ids": sorted(
                str(observation["reference_view_id"]) for observation in observations
            ),
            "spatial_registration_stable_view_count": registration_count,
            "source_mapping_group_id": _object(
                mapping, f"decision {part_id}.mapping"
            ).get("group_id"),
        }
    return result


def _load_view_inputs(
    *,
    spatial_report: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    files: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    spatial_files = _array(
        _object(spatial_report.get("inputs"), "spatial.inputs").get("files"),
        "spatial.inputs.files",
    )
    reference_paths: dict[str, Path] = {}
    for raw in spatial_files:
        record = _object(raw, "spatial input file")
        label = record.get("label")
        if isinstance(label, str) and label.startswith("reference_image:"):
            view_id = label.split(":", 1)[1]
            path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
            reference_paths[view_id] = path

    mvinverse_views = _array(mvinverse_evidence.get("views"), "mvinverse.views")
    albedo_paths: dict[str, Path] = {}
    for index, raw in enumerate(mvinverse_views):
        view = _object(raw, f"mvinverse.views[{index}]")
        view_id = view.get("view_id")
        sources = _object(view.get("sources"), f"mvinverse view {view_id}.sources")
        albedo = _object(sources.get("albedo"), f"mvinverse view {view_id}.albedo")
        if not isinstance(view_id, str):
            raise FaceMaterialRecoveryError("MVInverse view has invalid view_id")
        path = Path(str(albedo.get("path", ""))).expanduser().resolve(strict=True)
        expected = albedo.get("sha256")
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise FaceMaterialRecoveryError(f"MVInverse albedo hash mismatch: {path}")
        albedo_paths[view_id] = path

    result: dict[str, dict[str, Any]] = {}
    for alignment in _array(
        spatial_report.get("view_alignments"), "spatial.view_alignments"
    ):
        alignment = _object(alignment, "spatial alignment")
        if alignment.get("trusted") is not True:
            continue
        reference_id = alignment.get("reference_view_id")
        render_id = alignment.get("selected_render_view_id")
        if not isinstance(reference_id, str) or not isinstance(render_id, str):
            raise FaceMaterialRecoveryError("trusted alignment has invalid view IDs")
        if reference_id not in reference_paths or reference_id not in albedo_paths:
            raise FaceMaterialRecoveryError(
                f"trusted view lacks raw or albedo image: {reference_id}"
            )
        raw_path = reference_paths[reference_id]
        albedo_path = albedo_paths[reference_id]
        raw_image = _open_bgr(raw_path, f"reference {reference_id}")
        albedo_image = _open_bgr(albedo_path, f"MVInverse albedo {reference_id}")
        if albedo_image.shape[:2] != raw_image.shape[:2]:
            albedo_image = cv2.resize(
                albedo_image,
                (int(raw_image.shape[1]), int(raw_image.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        files.extend(
            [
                {
                    "label": f"reference_image:{reference_id}",
                    "path": str(raw_path),
                    "sha256": _sha256_file(raw_path),
                },
                {
                    "label": f"mvinverse_albedo:{reference_id}",
                    "path": str(albedo_path),
                    "sha256": _sha256_file(albedo_path),
                },
            ]
        )
        result[reference_id] = {
            "reference_view_id": reference_id,
            "render_view_id": render_id,
            "quarter_turns_ccw": int(alignment.get("quarter_turns_ccw", -1)),
            "bbox_affine": np.asarray(alignment.get("bbox_affine"), dtype=np.float32),
            "ecc_warp": np.asarray(alignment.get("ecc_warp"), dtype=np.float32),
            "raw_image": raw_image,
            "albedo_image": albedo_image,
        }
        if (
            result[reference_id]["quarter_turns_ccw"] not in range(4)
            or result[reference_id]["bbox_affine"].shape != (2, 3)
            or result[reference_id]["ecc_warp"].shape != (2, 3)
        ):
            raise FaceMaterialRecoveryError("trusted alignment matrices are malformed")
    return result


def _load_region_labels(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    views: Mapping[str, Mapping[str, Any]],
    files: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    contract = _object(manifest.get("projection_contract"), "projection_contract")
    source_resolution = list(
        _array(contract.get("source_resolution"), "source_resolution")
    )
    if len(source_resolution) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in source_resolution
    ):
        raise FaceMaterialRecoveryError("projection source resolution is invalid")
    records: dict[str, Mapping[str, Any]] = {}
    for raw in _array(contract.get("views"), "projection_contract.views"):
        record = _object(raw, "projection view")
        view_id = record.get("view_id")
        if not isinstance(view_id, str) or view_id in records:
            raise FaceMaterialRecoveryError("projection view IDs are invalid")
        if (
            record.get("semantic_alignment_status") != "approximate_coverage_match"
            or not isinstance(record.get("total_coverage_relative_error"), (int, float))
            or float(record["total_coverage_relative_error"]) > 0.01
        ):
            raise FaceMaterialRecoveryError(
                f"projection semantic coverage is not trustworthy: {view_id}"
            )
        records[view_id] = record
    output: dict[str, np.ndarray] = {}
    width, height = source_resolution
    for view in views.values():
        render_id = str(view["render_view_id"])
        record = records.get(render_id)
        if record is None:
            raise FaceMaterialRecoveryError(
                f"face-region evidence lacks selected render view: {render_id}"
            )
        path = _resolve_relative(
            manifest_path, record.get("numeric_labels"), f"region labels {render_id}"
        )
        try:
            labels = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FaceMaterialRecoveryError(
                f"unable to load numeric labels: {path}"
            ) from exc
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise FaceMaterialRecoveryError(f"numeric labels are malformed: {path}")
        if np.any(labels < 0):
            raise FaceMaterialRecoveryError(
                f"numeric labels contain negative IDs: {path}"
            )
        if labels.shape != (height, width):
            labels = cv2.resize(
                labels.astype(np.int32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        output[render_id] = labels.astype(np.int32, copy=False)
        files.append(
            {
                "label": f"region_labels:{render_id}",
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    return output


def _filter_projection_compatible_views(
    *,
    manifest: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Keep only trusted alignments with deterministic face projections.

    A dense pose bank may select a render that was intentionally omitted from
    the bounded face-projection set.  Such a registration cannot localize
    faces and is therefore excluded rather than guessed or treated as a fatal
    error.  The caller still requires at least two independent compatible
    views before attempting recovery.
    """

    contract = _object(manifest.get("projection_contract"), "projection_contract")
    available: set[str] = set()
    for raw in _array(contract.get("views"), "projection_contract.views"):
        record = _object(raw, "projection view")
        view_id = record.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in available:
            raise FaceMaterialRecoveryError("projection view IDs are invalid")
        available.add(view_id)
    compatible: dict[str, Mapping[str, Any]] = {}
    excluded: list[str] = []
    for reference_id, view in views.items():
        render_id = str(view.get("render_view_id", ""))
        if render_id in available:
            compatible[reference_id] = view
        else:
            excluded.append(render_id)
    return compatible, sorted(set(excluded))


def _project_region(
    labels: np.ndarray,
    numeric_region_id: int,
    view: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (labels == numeric_region_id).astype(np.uint8) * 255
    mask = np.rot90(mask, int(view["quarter_turns_ccw"])).copy()
    output_size = (
        int(view["raw_image"].shape[1]),
        int(view["raw_image"].shape[0]),
    )
    bbox = cv2.warpAffine(
        mask,
        view["bbox_affine"],
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    refined = cv2.warpAffine(
        bbox,
        view["ecc_warp"],
        output_size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return bbox, refined


def _observe_region(
    *,
    numeric_region_id: int,
    target_group_id: str,
    groups: Sequence[Mapping[str, Any]],
    view: Mapping[str, Any],
    labels: np.ndarray,
    policy: Mapping[str, float | int | bool],
) -> dict[str, Any]:
    bbox_mask, refined_mask = _project_region(labels, numeric_region_id, view)
    projected_pixels = int(np.count_nonzero(refined_mask))
    base = {
        "reference_view_id": view["reference_view_id"],
        "render_view_id": view["render_view_id"],
        "projected_pixels": projected_pixels,
    }
    minimum_pixels = int(policy["minimum_projected_pixels"])
    if projected_pixels < minimum_pixels:
        return {
            **base,
            "classification": "insufficient_visibility",
            "reason_code": "projected_region_pixels_below_floor",
        }

    profiles = [
        (
            "strict",
            float(policy["minimum_raw_color_share"]),
            float(policy["minimum_albedo_color_share"]),
        )
    ]
    if policy["enable_shadow_compensation"]:
        profiles.append(
            (
                "mvinverse_shadow_compensated",
                float(policy["shadow_minimum_raw_color_share"]),
                float(policy["shadow_minimum_albedo_color_share"]),
            )
        )
    profile_audits: list[dict[str, Any]] = []
    minimum_margin = float(policy["minimum_color_margin"])
    for profile_name, raw_share, albedo_share in profiles:
        bbox_pair = _score_pair(
            view["raw_image"],
            view["albedo_image"],
            bbox_mask,
            groups,
            raw_minimum_share=raw_share,
            albedo_minimum_share=albedo_share,
            minimum_margin=minimum_margin,
        )
        refined_pair = _score_pair(
            view["raw_image"],
            view["albedo_image"],
            refined_mask,
            groups,
            raw_minimum_share=raw_share,
            albedo_minimum_share=albedo_share,
            minimum_margin=minimum_margin,
        )
        bbox_group = _pair_group(bbox_pair)
        refined_group = _pair_group(refined_pair)
        registration_state = (
            "stable"
            if bbox_group is not None and bbox_group == refined_group
            else (
                "conflict"
                if bbox_group is not None
                and refined_group is not None
                and bbox_group != refined_group
                else "inconclusive"
            )
        )

        variants: list[tuple[str, np.ndarray]] = []
        shift = int(policy["projection_shift_pixels"])
        if shift:
            variants.extend(
                (
                    (f"shift_{x}_{y}", _shift_mask(refined_mask, x, y))
                    for x, y in ((-shift, 0), (shift, 0), (0, -shift), (0, shift))
                )
            )
        scale_delta = float(policy["projection_scale_delta"])
        if scale_delta:
            variants.extend(
                (
                    ("scale_down", _scale_mask(refined_mask, 1.0 - scale_delta)),
                    ("scale_up", _scale_mask(refined_mask, 1.0 + scale_delta)),
                )
            )
        perturbations: list[dict[str, Any]] = []
        for label, mask in variants:
            pair = _score_pair(
                view["raw_image"],
                view["albedo_image"],
                mask,
                groups,
                raw_minimum_share=raw_share,
                albedo_minimum_share=albedo_share,
                minimum_margin=minimum_margin,
            )
            perturbations.append(
                {
                    "variant": label,
                    "projected_pixels": int(np.count_nonzero(mask)),
                    "resolved_group_id": _pair_group(pair),
                    "raw_best_color_share": pair["raw"]["best_color_share"],
                    "raw_color_margin": pair["raw"]["color_margin"],
                    "albedo_best_color_share": pair["mvinverse_albedo"][
                        "best_color_share"
                    ],
                    "albedo_color_margin": pair["mvinverse_albedo"]["color_margin"],
                }
            )
        perturbation_stable = all(
            item["projected_pixels"] >= minimum_pixels
            and item["resolved_group_id"] == target_group_id
            for item in perturbations
        )
        profile = {
            "profile": profile_name,
            "bbox": bbox_pair,
            "refined": refined_pair,
            "bbox_resolved_group_id": bbox_group,
            "refined_resolved_group_id": refined_group,
            "registration_state": registration_state,
            "perturbations": perturbations,
            "perturbation_stable": perturbation_stable,
        }
        profile_audits.append(profile)
        if (
            refined_group == target_group_id
            and registration_state != "conflict"
            and perturbation_stable
        ):
            return {
                **base,
                "classification": "support",
                "reason_code": (
                    "raw_and_mvinverse_strict_support"
                    if profile_name == "strict"
                    else "mvinverse_albedo_shadow_compensated_support"
                ),
                "support_profile": profile_name,
                "canonical_group_id": target_group_id,
                "profiles": profile_audits,
            }

    relaxed = profile_audits[-1]["refined"]
    raw_group = relaxed["raw"]["resolved_group_id"]
    albedo_group = relaxed["mvinverse_albedo"]["resolved_group_id"]
    contradictory = sorted(
        {
            value
            for value in (raw_group, albedo_group)
            if isinstance(value, str) and value != target_group_id
        }
    )
    if contradictory:
        return {
            **base,
            "classification": "conflict",
            "reason_code": "stable_non_target_material_observed",
            "canonical_group_ids": contradictory,
            "profiles": profile_audits,
        }
    return {
        **base,
        "classification": "unresolved",
        "reason_code": "region_material_evidence_below_threshold",
        "profiles": profile_audits,
    }


def build_face_material_recovery(
    *,
    base_material_plan: Mapping[str, Any],
    confidence_gate: Mapping[str, Any],
    face_region_manifest: str | Path,
    spatial_mapping_report: Mapping[str, Any],
    canonical_palette: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    allowed_material_ids: Iterable[str],
    group_materials: Mapping[str, Any] | None = None,
    material_choice_audit: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    allow_parameter_writes: bool = True,
) -> dict[str, Any]:
    """Recover only proven surface regions and return an augmented plan."""

    if not isinstance(allow_parameter_writes, bool):
        raise FaceMaterialRecoveryError("allow_parameter_writes must be boolean")
    selected_policy = _validated_policy(policy)
    allowed_material_id_set: set[str] = set()
    for value in allowed_material_ids:
        if not isinstance(value, str) or not value:
            raise FaceMaterialRecoveryError("allowed material IDs are invalid")
        allowed_material_id_set.add(value)
    if not allowed_material_id_set:
        raise FaceMaterialRecoveryError("allowed material IDs are empty")
    if base_material_plan.get("schema_version") != MATERIAL_PLAN_SCHEMA_VERSION:
        raise FaceMaterialRecoveryError("base material plan schema is unsupported")
    base_assignments = _array(
        base_material_plan.get("assignments"), "base_material_plan.assignments"
    )
    base_part_ids: set[str] = set()
    for raw in base_assignments:
        assignment = _object(raw, "base material assignment")
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or part_id in base_part_ids:
            raise FaceMaterialRecoveryError("base material plan part IDs are invalid")
        base_part_ids.add(part_id)

    try:
        _verify_report(spatial_mapping_report)
    except SpatialMappingError as exc:
        raise FaceMaterialRecoveryError(
            f"spatial mapping report failed validation: {exc}"
        ) from exc
    try:
        verified_mvinverse = validate_mvinverse_evidence(mvinverse_evidence)
    except MVInverseEvidenceError as exc:
        raise FaceMaterialRecoveryError(
            f"MVInverse evidence failed validation: {exc}"
        ) from exc

    manifest, manifest_path = _read_manifest(face_region_manifest)
    if manifest.get("schema_version") != FACE_REGION_SCHEMA_VERSION:
        raise FaceMaterialRecoveryError("face-region schema is unsupported")
    if manifest.get("surface_patch_method") != SURFACE_PATCH_METHOD:
        raise FaceMaterialRecoveryError(
            "face-region patches lack the required seed-normal coherence contract"
        )
    if (
        manifest.get("source_usd_unchanged") is not True
        or manifest.get("source_usd_sha256_before")
        != manifest.get("source_usd_sha256_after")
        or manifest.get("source_usd_sha256_before") != manifest.get("asset_sha256")
    ):
        raise FaceMaterialRecoveryError("face-region source USD integrity failed")
    contract = _object(manifest.get("projection_contract"), "projection_contract")
    projection_views = _array(contract.get("views"), "projection_contract.views")
    if not projection_views or manifest.get("projection_view_count") != len(
        projection_views
    ):
        raise FaceMaterialRecoveryError("face-region projection contract is incomplete")
    if manifest.get("registry_sha256") != contract.get("rendered_registry_sha256"):
        raise FaceMaterialRecoveryError("face-region registry hashes disagree")
    spatial_registry_hashes = {
        record.get("sha256")
        for record in _array(
            _object(spatial_mapping_report.get("inputs"), "spatial.inputs").get(
                "files"
            ),
            "spatial.inputs.files",
        )
        if isinstance(record, Mapping) and record.get("label") == "rendered_registry"
    }
    if spatial_registry_hashes != {manifest.get("registry_sha256")}:
        raise FaceMaterialRecoveryError(
            "face-region and spatial reports use different rendered registries"
        )

    groups = _palette_groups(canonical_palette)
    groups_by_id = {group["group_id"]: group for group in groups}
    evidence_groups = {
        group["group_id"]: group
        for group in _array(verified_mvinverse.get("groups"), "mvinverse.groups")
        if isinstance(group, Mapping) and isinstance(group.get("group_id"), str)
    }
    standard_candidates: dict[str, dict[str, Any]] = {}
    for part_id, decision in _candidate_decisions(confidence_gate).items():
        group_id = decision["mapping"]["group_id"]
        choice = decision["material_choice"]
        if (
            part_id not in base_part_ids
            and group_id in groups_by_id
            and group_id in evidence_groups
            and decision["material_id"] in allowed_material_id_set
        ):
            standard_candidates[part_id] = {
                "decision": decision,
                "candidate_source": "multi_material_subset_recovery",
                "group_id": group_id,
                "material_id": decision["material_id"],
                "confidence": min(
                    float(decision["model"]["confidence"]),
                    float(choice["forward_confidence"]),
                    float(choice["reverse_confidence"]),
                ),
                "spatial_supporting_view_ids": [],
                "source_mapping_group_id": group_id,
            }
    group_choices = _group_material_candidates(
        group_materials=group_materials,
        material_choice_audit=material_choice_audit,
        allowed_material_ids=allowed_material_id_set,
        policy=selected_policy,
    )
    spatial_candidates = _spatial_override_candidates(
        confidence_gate=confidence_gate,
        spatial_mapping_report=spatial_mapping_report,
        group_material_candidates=group_choices,
        policy=selected_policy,
    )
    spatial_candidates = {
        part_id: candidate
        for part_id, candidate in spatial_candidates.items()
        if part_id not in base_part_ids
        and candidate["group_id"] in groups_by_id
        and candidate["group_id"] in evidence_groups
    }
    overlap = set(standard_candidates) & set(spatial_candidates)
    if overlap:
        raise FaceMaterialRecoveryError(
            f"parts entered multiple recovery paths: {sorted(overlap)}"
        )
    candidates = {**standard_candidates, **spatial_candidates}
    document_hashes = {
        "base_material_plan": _sha256_document(base_material_plan),
        "confidence_gate": _sha256_document(confidence_gate),
        "spatial_mapping_report": _sha256_document(spatial_mapping_report),
        "canonical_palette": _sha256_document(canonical_palette),
        "mvinverse_evidence": _sha256_document(verified_mvinverse),
        "batches": _sha256_document(list(batches)),
    }
    if group_materials is not None and material_choice_audit is not None:
        document_hashes["group_materials"] = _sha256_document(group_materials)
        document_hashes["material_choice_audit"] = _sha256_document(
            material_choice_audit
        )

    files: list[dict[str, str]] = [
        {
            "label": "face_region_manifest",
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
    ]
    views = _load_view_inputs(
        spatial_report=spatial_mapping_report,
        mvinverse_evidence=verified_mvinverse,
        files=files,
    )
    views, excluded_unprojected_view_ids = (
        _filter_projection_compatible_views(
            manifest=manifest,
            views=views,
        )
    )
    if len(views) < MINIMUM_TRUSTED_REGISTERED_VIEWS:
        output_plan = {
            "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
            "assignments": [
                copy.deepcopy(dict(assignment)) for assignment in base_assignments
            ],
        }
        reason_codes = [INSUFFICIENT_TRUSTED_VIEWS_REASON]
        report = {
            "schema_version": SCHEMA_VERSION,
            "fail_closed": True,
            "policy": selected_policy,
            "inputs": {
                "files": sorted(files, key=lambda item: (item["label"], item["path"])),
                "document_sha256": document_hashes,
            },
            "recovery_gate": {
                "status": "SKIPPED_INSUFFICIENT_EVIDENCE",
                "decision": "preserve_base_material_plan",
                "reason_codes": reason_codes,
                "trusted_registered_view_count": len(views),
                "minimum_trusted_registered_view_count": (
                    MINIMUM_TRUSTED_REGISTERED_VIEWS
                ),
                "candidate_part_ids": sorted(candidates),
                "face_region_labels_loaded": False,
                "face_parameterization_attempted": False,
                "face_subset_assignments_emitted": 0,
                **(
                    {
                        "excluded_unprojected_trusted_view_ids": (
                            excluded_unprojected_view_ids
                        )
                    }
                    if excluded_unprojected_view_ids
                    else {}
                ),
            },
            "summary": {
                "status": "SKIPPED_INSUFFICIENT_EVIDENCE",
                "skip_reason_codes": reason_codes,
                "candidate_part_count": len(candidates),
                "multi_material_candidate_count": len(standard_candidates),
                "spatial_override_candidate_count": len(spatial_candidates),
                "recovered_part_count": 0,
                "recovered_subset_count": 0,
                "recovered_face_count": 0,
                "base_uniform_assignment_count": len(base_assignments),
                "output_assignment_count": len(output_plan["assignments"]),
                "parent_material_bindings_preserved": True,
            },
            "parts": [
                {
                    "part_id": part_id,
                    "candidate_source": candidate["candidate_source"],
                    "decision": "preserve",
                    "reason_codes": reason_codes,
                }
                for part_id, candidate in sorted(candidates.items())
            ],
            "parameterization": [],
            "material_plan": output_plan,
        }
        unsigned = copy.deepcopy(report)
        report["integrity"] = {"report_sha256": _sha256_document(unsigned)}
        return report

    labels_by_render = _load_region_labels(
        manifest=manifest,
        manifest_path=manifest_path,
        views=views,
        files=files,
    )

    manifest_parts: dict[str, Mapping[str, Any]] = {}
    for raw in _array(manifest.get("parts"), "face_region.parts"):
        part = _object(raw, "face-region part")
        part_id = part.get("part_id")
        if not isinstance(part_id, str) or part_id in manifest_parts:
            raise FaceMaterialRecoveryError("face-region part IDs are invalid")
        manifest_parts[part_id] = part

    selected_by_part: dict[str, dict[str, Any]] = {}
    part_audits: list[dict[str, Any]] = []
    for part_id, candidate in sorted(candidates.items()):
        decision = candidate["decision"]
        summary = manifest_parts.get(part_id)
        if summary is None:
            raise FaceMaterialRecoveryError(
                f"face-region evidence lacks part {part_id}"
            )
        part_path = _resolve_relative(
            manifest_path, summary.get("evidence"), f"face-region part {part_id}"
        )
        try:
            part_document = json.loads(part_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FaceMaterialRecoveryError(
                f"unable to read face-region part: {part_path}"
            ) from exc
        if not isinstance(part_document, dict):
            raise FaceMaterialRecoveryError(f"face-region part is invalid: {part_id}")
        if part_document.get("surface_patch_method") != SURFACE_PATCH_METHOD:
            raise FaceMaterialRecoveryError(
                f"face-region part lacks the required patch method: {part_id}"
            )
        files.append(
            {
                "label": f"face_region_part:{part_id}",
                "path": str(part_path),
                "sha256": _sha256_file(part_path),
            }
        )
        face_count = part_document.get("face_count")
        patches = _array(
            part_document.get("surface_patches"), f"{part_id}.surface_patches"
        )
        if (
            isinstance(face_count, bool)
            or not isinstance(face_count, int)
            or face_count < 1
            or face_count != summary.get("face_count")
            or len(patches) != summary.get("surface_patch_count")
        ):
            raise FaceMaterialRecoveryError(
                f"face-region part summary mismatch: {part_id}"
            )
        target_group_id = str(candidate["group_id"])
        is_spatial_override = (
            candidate["candidate_source"] == "spatial_consensus_override"
        )
        patch_audits: list[dict[str, Any]] = []
        selected_patches: list[dict[str, Any]] = []
        covered_faces: set[int] = set()
        all_patch_faces: set[int] = set()
        for raw_patch in patches:
            patch = _object(raw_patch, f"{part_id}.surface_patch")
            region_id = patch.get("region_id")
            numeric_region_id = patch.get("numeric_region_id")
            if (
                not isinstance(region_id, str)
                or isinstance(numeric_region_id, bool)
                or not isinstance(numeric_region_id, int)
                or numeric_region_id < 1
            ):
                raise FaceMaterialRecoveryError(
                    f"surface patch IDs are invalid: {part_id}"
                )
            face_indices = _face_indices(patch, face_count)
            overlap = all_patch_faces & set(face_indices)
            if overlap:
                raise FaceMaterialRecoveryError(
                    f"surface patches overlap for {part_id}: {sorted(overlap)[:8]}"
                )
            all_patch_faces.update(face_indices)
            deviation = patch.get("max_normal_deviation_degrees")
            if (
                isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(float(deviation))
            ):
                raise FaceMaterialRecoveryError(
                    f"patch normal evidence is invalid: {part_id}/{region_id}"
                )
            observations = [
                _observe_region(
                    numeric_region_id=numeric_region_id,
                    target_group_id=target_group_id,
                    groups=groups,
                    view=view,
                    labels=labels_by_render[str(view["render_view_id"])],
                    policy=selected_policy,
                )
                for view in views.values()
            ]
            eligible_supports = [
                observation
                for observation in observations
                if observation["classification"] == "support"
            ]
            if is_spatial_override:
                override_pixel_floor = int(
                    selected_policy["minimum_override_patch_projected_pixels"]
                )
                eligible_supports = [
                    observation
                    for observation in eligible_supports
                    if int(observation["projected_pixels"]) >= override_pixel_floor
                    and any(
                        profile.get("perturbations")
                        for profile in observation.get("profiles", [])
                    )
                    and all(
                        int(perturbation["projected_pixels"]) >= override_pixel_floor
                        for profile in observation.get("profiles", [])
                        for perturbation in profile.get("perturbations", [])
                    )
                ]
            supports = sorted(
                observation["reference_view_id"] for observation in eligible_supports
            )
            strict_supports = sorted(
                observation["reference_view_id"]
                for observation in eligible_supports
                if observation.get("support_profile") == "strict"
            )
            conflicts = sorted(
                observation["reference_view_id"]
                for observation in observations
                if observation["classification"] == "conflict"
            )
            reasons: list[str] = []
            if float(deviation) > float(
                selected_policy["maximum_patch_normal_deviation_degrees"]
            ):
                reasons.append("patch_normal_deviation_above_limit")
            if conflicts:
                reasons.append("cross_view_material_conflict")
            if len(supports) < int(selected_policy["minimum_support_views"]):
                reasons.append("insufficient_independent_region_support")
            if (
                is_spatial_override
                and selected_policy["require_override_strict_support"]
                and not strict_supports
            ):
                reasons.append("spatial_override_requires_strict_region_support")
            selected = not reasons
            if selected:
                selected_patches.append(
                    {
                        "region_id": region_id,
                        "numeric_region_id": numeric_region_id,
                        "face_indices": face_indices,
                        "face_count": len(face_indices),
                        "area_world": float(patch.get("area_world", 0.0)),
                        "supporting_view_ids": supports,
                        "strict_supporting_view_ids": strict_supports,
                    }
                )
                covered_faces.update(face_indices)
            patch_audits.append(
                {
                    "region_id": region_id,
                    "numeric_region_id": numeric_region_id,
                    "face_count": len(face_indices),
                    "area_world": float(patch.get("area_world", 0.0)),
                    "max_normal_deviation_degrees": float(deviation),
                    "decision": "auto_subset" if selected else "preserve",
                    "supporting_view_ids": supports,
                    "strict_supporting_view_ids": strict_supports,
                    "conflicting_view_ids": conflicts,
                    "reason_codes": reasons
                    or ["multi_view_region_evidence_sufficient"],
                    "observations": observations,
                }
            )
        if all_patch_faces != set(range(face_count)):
            raise FaceMaterialRecoveryError(
                f"surface patches do not exactly cover faces for {part_id}"
            )
        part_record = {
            "part_id": part_id,
            "candidate_source": candidate["candidate_source"],
            "target_group_id": target_group_id,
            "source_mapping_group_id": candidate.get("source_mapping_group_id"),
            "minimum_spatial_color_share": candidate.get("minimum_spatial_color_share"),
            "spatial_supporting_view_ids": candidate.get(
                "spatial_supporting_view_ids", []
            ),
            "target_base_color": groups_by_id[target_group_id]["base_color"],
            "face_count": face_count,
            "selected_patch_count": len(selected_patches),
            "selected_face_count": len(covered_faces),
            "selected_face_fraction": round(len(covered_faces) / face_count, 8),
            "selected_area_world": round(
                sum(patch["area_world"] for patch in selected_patches), 12
            ),
            "total_area_world": float(part_document.get("total_area_world", 0.0)),
            "patches": patch_audits,
        }
        part_record["selected_area_fraction"] = round(
            part_record["selected_area_world"]
            / max(1e-18, part_record["total_area_world"]),
            8,
        )
        part_audits.append(part_record)
        if selected_patches:
            selected_by_part[part_id] = {
                "candidate": candidate,
                "group_id": target_group_id,
                "face_indices": sorted(covered_faces),
                "selected_patches": selected_patches,
            }

    recovered_assignments: list[dict[str, Any]] = []
    parameterization_audit: list[dict[str, Any]] = []
    if selected_by_part:
        synthetic_assignments = []
        for part_id, selected in sorted(selected_by_part.items()):
            candidate = selected["candidate"]
            supporting_views = sorted(
                {
                    view_id
                    for patch in selected["selected_patches"]
                    for view_id in patch["supporting_view_ids"]
                }
            )
            synthetic_assignments.append(
                {
                    "part_id": part_id,
                    "material_id": candidate["material_id"],
                    "semantic": groups_by_id[selected["group_id"]][
                        "visual_description"
                    ],
                    "confidence": float(candidate["confidence"]),
                    "evidence_views": supporting_views,
                    "status": "auto",
                }
            )
        parameterized = parameterize_auto_material_plan(
            auto_material_plan={
                "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
                "assignments": synthetic_assignments,
            },
            batches=batches,
            palette=canonical_palette,
            mvinverse_evidence=verified_mvinverse,
            allowed_material_ids=allowed_material_id_set,
            part_group_overrides={
                part_id: selected["group_id"]
                for part_id, selected in selected_by_part.items()
            },
            allow_parameter_writes=allow_parameter_writes,
        )
        parameterization_audit = list(parameterized["decisions"])
        for assignment in parameterized["material_plan"]["assignments"]:
            part_id = assignment["part_id"]
            recovered_assignment = {
                    **assignment,
                    "preserve_parent_material_binding": True,
                    "face_subsets": [
                        {
                            "subset_name": f"AutoMaterial_{selected_by_part[part_id]['group_id']}",
                            "material_id": assignment["material_id"],
                            "semantic": assignment["semantic"],
                            "face_indices": selected_by_part[part_id]["face_indices"],
                        }
                    ],
                }
            if "parameters" in assignment:
                recovered_assignment["face_subsets"][0]["parameters"] = (
                    copy.deepcopy(assignment["parameters"])
                )
            recovered_assignments.append(recovered_assignment)

    output_plan = {
        "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
        "assignments": [copy.deepcopy(dict(item)) for item in base_assignments]
        + recovered_assignments,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fail_closed": True,
        "policy": selected_policy,
        "inputs": {
            "files": sorted(files, key=lambda item: (item["label"], item["path"])),
            "document_sha256": document_hashes,
        },
        "summary": {
            "candidate_part_count": len(candidates),
            "multi_material_candidate_count": len(standard_candidates),
            "spatial_override_candidate_count": len(spatial_candidates),
            "recovered_part_count": len(recovered_assignments),
            "recovered_subset_count": len(recovered_assignments),
            "recovered_face_count": sum(
                len(assignment["face_subsets"][0]["face_indices"])
                for assignment in recovered_assignments
            ),
            "base_uniform_assignment_count": len(base_assignments),
            "output_assignment_count": len(output_plan["assignments"]),
            "parent_material_bindings_preserved": True,
            **(
                {
                    "excluded_unprojected_trusted_view_ids": (
                        excluded_unprojected_view_ids
                    )
                }
                if excluded_unprojected_view_ids
                else {}
            ),
        },
        "parts": part_audits,
        "parameterization": parameterization_audit,
        "material_plan": output_plan,
    }
    unsigned = copy.deepcopy(report)
    report["integrity"] = {"report_sha256": _sha256_document(unsigned)}
    return report


__all__ = [
    "DEFAULT_POLICY",
    "FaceMaterialRecoveryError",
    "INSUFFICIENT_TRUSTED_VIEWS_REASON",
    "MINIMUM_TRUSTED_REGISTERED_VIEWS",
    "SCHEMA_VERSION",
    "build_face_material_recovery",
]
