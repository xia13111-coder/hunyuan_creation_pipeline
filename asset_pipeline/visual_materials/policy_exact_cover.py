"""Validation and recovery contracts for policy exact-cover plans."""

from __future__ import annotations

import colorsys
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..paths import unique_path
from .config import canonical_sha256, write_object
from .policy_contract import (
    APPLICABLE_ASSIGNMENT_STATUSES,
    CORROBORATED_SOURCE_MDL_TIER,
    CORROBORATED_SOURCE_MIN_PROVISIONAL_CONFIDENCE,
    CORROBORATED_SOURCE_PROVISIONAL_MATERIAL_BASIS,
    POLICY_FALLBACK_STATUS,
    POLICY_PLAN_MODE,
    POLICY_REPORT_SCHEMA_VERSION,
    SOURCE_ACCENT_CHROMATIC_COLORS,
    SOURCE_ACCENT_MAX_REGISTRY_FRACTION,
    SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT,
    SOURCE_ACCENT_MIN_OPACITY,
    SOURCE_ACCENT_MIN_SATURATION,
    SOURCE_ACCENT_MIN_SIGNATURE_COUNT,
    SOURCE_CORROBORATION_REASON_CODES,
    SOURCE_MDL_CONFIRMED_REASON_CODE,
    SOURCE_MDL_PROVISIONAL_REASON_CODES,
    SOURCE_MDL_REPLACEMENT_REASON_CODES,
    SOURCE_VISUAL_PRESERVE_ACTION,
)
from qwen_material_pipeline.evidence.palette_fusion import (
    is_verified_unresolved_pixel_chromatic_group,
)
from qwen_material_pipeline.materials.policy_exact_cover import (
    PolicyExactCoverError,
    build_policy_exact_cover,
)


def _policy_checkpoint_matches_requested_overrides(
    *,
    audit: Mapping[str, Any],
    requested_policy: Mapping[str, Any],
) -> bool:
    """Return whether a valid policy audit contains every requested override.

    The exact-cover audit stores the fully expanded policy, while the input
    document intentionally contains only workflow-owned overrides.  Comparing
    those overrides prevents a completed inference checkpoint from silently
    retaining an older fallback policy after the visual workflow changes.
    """

    effective_policy = audit.get("policy")
    if not isinstance(effective_policy, Mapping):
        raise RuntimeError("Policy exact-cover audit is missing its policy")
    return all(
        effective_policy.get(key) == value for key, value in requested_policy.items()
    )


def _archive_stale_policy_exact_cover_checkpoint(
    *,
    destination: Path,
    paths: Sequence[Path],
    reason: str,
) -> Path:
    """Reversibly archive stale policy-derived checkpoints before rebuilding."""

    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if not existing:
        raise RuntimeError("No stale policy checkpoint exists to archive")
    archive_root = destination / "analysis" / "recovery_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = unique_path(archive_root / "policy_exact_cover_change")
    archive_dir.mkdir(parents=False, exist_ok=False)
    archived: list[dict[str, str]] = []
    for source_path in existing:
        relative_path = source_path.relative_to(destination)
        archived_path = archive_dir / relative_path
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(archived_path)
        archived.append(
            {
                "original": str(relative_path),
                "archived": str(archived_path.relative_to(archive_dir)),
            }
        )
    write_object(
        archive_dir / "archive_manifest.json",
        {
            "schema_version": (
                "asset-pipeline-visual-material-policy-recovery-archive/v1"
            ),
            "status": "COMPLETED",
            "reason": reason,
            "archived": archived,
        },
    )
    return archive_dir

def _require_exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}: {value!r}")
    return value

def _policy_unit_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RuntimeError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _policy_fusion_color(value: str) -> str:
    normalized = value.strip().casefold()
    return {"cyan": "blue", "brown": "orange"}.get(normalized, normalized)


def _policy_pixel_color(red: int, green: int, blue: int) -> str:
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum
    value = maximum / 255.0
    saturation = 0.0 if maximum == 0 else delta / maximum
    if maximum < 55:
        return "black"
    if saturation < 0.14:
        if minimum > 215:
            return "white"
        if value > 0.68:
            return "silver"
        return "gray"
    hue = colorsys.rgb_to_hsv(
        red / 255.0,
        green / 255.0,
        blue / 255.0,
    )[0] * 360.0
    if hue < 15.0 or hue >= 345.0:
        return "red"
    if hue < 45.0:
        return "brown" if value < 0.55 else "orange"
    if hue < 70.0:
        return "yellow"
    if hue < 170.0:
        return "green"
    if hue < 200.0:
        return "cyan"
    if hue < 260.0:
        return "blue"
    return "pink"


def _policy_rgb_color(rgb: Sequence[float]) -> str:
    encoded = tuple(
        max(0, min(255, int(round(float(channel) * 255.0)))) for channel in rgb
    )
    return _policy_fusion_color(_policy_pixel_color(*encoded))


def _policy_linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _policy_source_signature(
    part: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    properties = part.get("existing_visual_material_properties")
    if not isinstance(properties, dict) or properties.get("shader_id") != (
        "UsdPreviewSurface"
    ):
        raise RuntimeError(f"{label} lacks an auditable UsdPreviewSurface")
    raw_diffuse = properties.get("diffuseColor")
    if not isinstance(raw_diffuse, list) or len(raw_diffuse) != 3:
        raise RuntimeError(f"{label} diffuseColor must contain three channels")
    diffuse = [
        _policy_unit_number(value, f"{label}.diffuseColor") for value in raw_diffuse
    ]
    metallic = _policy_unit_number(properties.get("metallic"), f"{label}.metallic")
    roughness = _policy_unit_number(
        properties.get("roughness"), f"{label}.roughness"
    )
    opacity = _policy_unit_number(properties.get("opacity"), f"{label}.opacity")
    if opacity < SOURCE_ACCENT_MIN_OPACITY:
        raise RuntimeError(f"{label} is not opaque enough for source preservation")
    linear_rgb = [_policy_linear_to_srgb(value) for value in diffuse]
    raw_color = _policy_rgb_color(diffuse)
    linear_color = _policy_rgb_color(linear_rgb)
    raw_saturation = colorsys.rgb_to_hsv(*diffuse)[1]
    linear_saturation = colorsys.rgb_to_hsv(*linear_rgb)[1]
    if (
        raw_color != linear_color
        or raw_color not in SOURCE_ACCENT_CHROMATIC_COLORS
        or raw_saturation < SOURCE_ACCENT_MIN_SATURATION
        or linear_saturation < SOURCE_ACCENT_MIN_SATURATION
    ):
        raise RuntimeError(
            f"{label} source colour is not a stable saturated chromatic family"
        )
    payload = {
        "shader_id": "UsdPreviewSurface",
        "diffuse_color": [round(value, 6) for value in diffuse],
        "metallic": round(metallic, 6),
        "roughness": round(roughness, 6),
        "opacity": round(opacity, 6),
    }
    return payload, raw_color


def _policy_geometry_signature(
    part: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    point_count = _require_exact_int(part.get("point_count"), f"{label}.point_count", minimum=1)
    face_count = _require_exact_int(part.get("face_count"), f"{label}.face_count", minimum=1)
    raw_bbox = part.get("world_bbox")
    if (
        not isinstance(raw_bbox, list)
        or len(raw_bbox) != 2
        or any(not isinstance(bound, list) or len(bound) != 3 for bound in raw_bbox)
    ):
        raise RuntimeError(f"{label}.world_bbox is invalid")
    bounds: list[list[float]] = []
    for bound_index, raw_bound in enumerate(raw_bbox):
        bound: list[float] = []
        for value in raw_bound:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"{label}.world_bbox[{bound_index}] is invalid"
                )
            bound.append(float(value))
        bounds.append(bound)
    if any(bounds[1][axis] < bounds[0][axis] for axis in range(3)):
        raise RuntimeError(f"{label}.world_bbox has inverted bounds")
    return {
        "point_count": point_count,
        "face_count": face_count,
        "sorted_bbox_extents": sorted(
            round(bounds[1][axis] - bounds[0][axis], 3) for axis in range(3)
        ),
    }


def _validate_corroborated_source_visual_assignments(
    *,
    assignments: list[Any],
    raw_parts: list[Any],
    audit: dict[str, Any],
    palette_fusion: dict[str, Any] | None,
    group_materials: dict[str, Any] | None = None,
    allow_high_confidence_provisional: bool = False,
) -> tuple[int, int]:
    preserve_assignments = {
        str(item.get("part_id")): item
        for item in assignments
        if isinstance(item, dict)
        and item.get("provenance", {}).get("tier") == "source_visual_preserve"
    }
    mdl_assignments = {
        str(item.get("part_id")): item
        for item in assignments
        if isinstance(item, dict)
        and item.get("provenance", {}).get("tier") == CORROBORATED_SOURCE_MDL_TIER
    }
    corroborated_assignments = {**preserve_assignments, **mdl_assignments}
    if set(preserve_assignments) & set(mdl_assignments):
        raise RuntimeError("Policy source-corroboration tiers overlap")
    if not corroborated_assignments:
        corroboration = audit.get("corroborated_source_visual")
        if isinstance(corroboration, dict) and corroboration.get("applied_part_ids"):
            raise RuntimeError(
                "Policy source-corroboration audit declares unapplied assignments"
            )
        return 0, 0
    if palette_fusion is None:
        raise RuntimeError(
            "Neutralized source visuals may be preserved only with palette fusion"
        )
    raw_group_selections = (
        group_materials.get("selections")
        if isinstance(group_materials, dict)
        else [
            {
                "group_id": record.get("canonical_group_id"),
                "material_id": record.get("confirmed_material_id"),
                "confirmed": True,
                "confidence": 0.0,
            }
            for assignment in mdl_assignments.values()
            for record in [
                assignment.get("provenance", {}).get(
                    "source_visual_corroboration", {}
                )
            ]
            if isinstance(record, dict)
            and isinstance(record.get("confirmed_material_id"), str)
        ]
    )
    if not isinstance(raw_group_selections, list):
        raise RuntimeError(
            "Policy source-corroboration group material selections are missing"
        )
    group_selections: dict[str, dict[str, Any]] = {}
    for raw_selection in raw_group_selections:
        if not isinstance(raw_selection, dict):
            raise RuntimeError(
                "Policy source-corroboration group material selection is invalid"
            )
        group_id = raw_selection.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError(
                "Policy source-corroboration group material identity is invalid"
            )
        if group_id in group_selections:
            if group_selections[group_id] == raw_selection:
                continue
            raise RuntimeError(
                "Policy source-corroboration group material identity is invalid"
            )
        group_selections[group_id] = raw_selection
    if palette_fusion.get("schema_version") != (
        "qwen-multiview-palette-fusion/v1"
    ):
        raise RuntimeError("Policy palette fusion has an unsupported schema")
    canonical = palette_fusion.get("canonical_palette")
    raw_groups = canonical.get("groups") if isinstance(canonical, dict) else None
    if (
        not isinstance(canonical, dict)
        or canonical.get("schema_version")
        != "qwen-canonical-material-palette/v1"
        or not isinstance(raw_groups, list)
    ):
        raise RuntimeError("Policy canonical palette is invalid")
    valid_groups_by_id: dict[str, dict[str, Any]] = {}
    group_ids_by_color: dict[str, list[str]] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise RuntimeError("Policy canonical palette contains a non-object group")
        group_id = raw_group.get("group_id")
        color_value = raw_group.get("base_color")
        views = raw_group.get("source_view_ids")
        distinct = raw_group.get("distinct_view_count")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in valid_groups_by_id
            or not isinstance(color_value, str)
            or not isinstance(views, list)
            or any(not isinstance(view, str) or not view for view in views)
            or isinstance(distinct, bool)
            or not isinstance(distinct, int)
            or distinct != len(set(views))
        ):
            raise RuntimeError("Policy canonical palette group support is invalid")
        color = _policy_fusion_color(color_value)
        unresolved_pixel_corroborated = (
            is_verified_unresolved_pixel_chromatic_group(raw_group)
        )
        if (
            raw_group.get("singleton") is False
            and distinct >= 2
            and color in SOURCE_ACCENT_CHROMATIC_COLORS
            and (
                str(raw_group.get("family_hint", "")).casefold()
                not in {"", "other", "unknown"}
                or unresolved_pixel_corroborated
            )
        ):
            valid_groups_by_id[group_id] = raw_group
            group_ids_by_color.setdefault(color, []).append(group_id)

    corroboration = audit.get("corroborated_source_visual")
    if not isinstance(corroboration, dict):
        raise RuntimeError("Policy source-corroboration audit is missing")
    thresholds = corroboration.get("thresholds")
    expected_thresholds = {
        "maximum_registry_fraction": SOURCE_ACCENT_MAX_REGISTRY_FRACTION,
        "minimum_source_signature_count": SOURCE_ACCENT_MIN_SIGNATURE_COUNT,
        "minimum_repeated_geometry_count": (
            SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT
        ),
        "minimum_raw_and_linear_saturation": SOURCE_ACCENT_MIN_SATURATION,
        "minimum_opacity": SOURCE_ACCENT_MIN_OPACITY,
    }
    if thresholds != expected_thresholds:
        raise RuntimeError("Policy source-corroboration thresholds are invalid")
    applied_ids = corroboration.get("applied_part_ids")
    preserved_ids = corroboration.get("preserved_part_ids")
    mdl_replacement_ids = corroboration.get("nvidia_mdl_replacement_part_ids")
    eligible_ids = corroboration.get("eligible_part_ids")
    groups = corroboration.get("groups")
    if (
        not isinstance(applied_ids, list)
        or any(not isinstance(value, str) for value in applied_ids)
        or len(set(applied_ids)) != len(applied_ids)
        or set(applied_ids) != set(corroborated_assignments)
        or not isinstance(preserved_ids, list)
        or any(not isinstance(value, str) for value in preserved_ids)
        or len(set(preserved_ids)) != len(preserved_ids)
        or set(preserved_ids) != set(preserve_assignments)
        or not isinstance(mdl_replacement_ids, list)
        or any(not isinstance(value, str) for value in mdl_replacement_ids)
        or len(set(mdl_replacement_ids)) != len(mdl_replacement_ids)
        or set(mdl_replacement_ids) != set(mdl_assignments)
        or set(preserved_ids) & set(mdl_replacement_ids)
        or not isinstance(eligible_ids, list)
        or any(not isinstance(value, str) for value in eligible_ids)
        or len(set(eligible_ids)) != len(eligible_ids)
        or not set(applied_ids).issubset(eligible_ids)
        or not isinstance(groups, list)
    ):
        raise RuntimeError("Policy source-corroboration part coverage is invalid")

    parts_by_id = {
        str(part.get("part_id")): part
        for part in raw_parts
        if isinstance(part, dict) and isinstance(part.get("part_id"), str)
    }
    maximum_signature_count = math.floor(
        len(raw_parts) * SOURCE_ACCENT_MAX_REGISTRY_FRACTION
    )
    if corroboration.get("maximum_source_signature_count") != (
        maximum_signature_count
    ):
        raise RuntimeError("Policy source signature maximum is inconsistent")

    audited_part_records: dict[str, dict[str, Any]] = {}
    for raw_group_record in groups:
        if not isinstance(raw_group_record, dict):
            raise RuntimeError("Policy source-corroboration group is invalid")
        group_id = raw_group_record.get("group_id")
        color = raw_group_record.get("canonical_color_family")
        palette_group = valid_groups_by_id.get(str(group_id))
        if (
            palette_group is None
            or not isinstance(color, str)
            or group_ids_by_color.get(color) != [group_id]
            or _policy_fusion_color(str(palette_group["base_color"])) != color
            or raw_group_record.get("canonical_source_view_ids")
            != sorted(palette_group["source_view_ids"])
            or raw_group_record.get("canonical_group_association_basis")
            != palette_group.get("association_basis")
        ):
            raise RuntimeError(
                "Policy source-corroboration canonical group is ambiguous"
            )
        signature_digest = raw_group_record.get(
            "source_visual_signature_sha256"
        )
        group_eligible_ids = raw_group_record.get("eligible_part_ids")
        signature_member_ids = raw_group_record.get(
            "signature_member_part_ids", group_eligible_ids
        )
        geometry_anchor_ids = raw_group_record.get("geometry_anchor_part_ids")
        geometry_cohorts = raw_group_record.get("geometry_cohorts")
        signature_count = _require_exact_int(
            raw_group_record.get("source_signature_count"),
            "Policy source signature count",
            minimum=SOURCE_ACCENT_MIN_SIGNATURE_COUNT,
        )
        if (
            signature_count > maximum_signature_count
            or not isinstance(signature_digest, str)
            or not isinstance(group_eligible_ids, list)
            or any(not isinstance(value, str) for value in group_eligible_ids)
            or not isinstance(signature_member_ids, list)
            or any(not isinstance(value, str) for value in signature_member_ids)
            or set(signature_member_ids) != set(group_eligible_ids)
            or not isinstance(geometry_cohorts, list)
        ):
            raise RuntimeError("Policy source signature cohort is invalid")
        matching_signature_count = 0
        for part_id, part in parts_by_id.items():
            try:
                payload, source_color = _policy_source_signature(
                    part, label=f"registry part {part_id}"
                )
            except RuntimeError:
                continue
            if canonical_sha256(payload) == signature_digest:
                matching_signature_count += 1
                if source_color != color:
                    raise RuntimeError(
                        "Policy source signature conflicts with canonical colour"
                    )
        if matching_signature_count != signature_count:
            raise RuntimeError("Policy source signature count is not reproducible")
        cohort_ids: set[str] = set()
        for raw_cohort in geometry_cohorts:
            if not isinstance(raw_cohort, dict):
                raise RuntimeError("Policy repeated geometry cohort is invalid")
            part_ids = raw_cohort.get("part_ids")
            repeat_count = _require_exact_int(
                raw_cohort.get("repeat_count"),
                "Policy repeated geometry count",
                minimum=SOURCE_ACCENT_MIN_GEOMETRY_REPEAT_COUNT,
            )
            geometry_digest = raw_cohort.get("geometry_signature_sha256")
            if (
                not isinstance(part_ids, list)
                or any(not isinstance(value, str) for value in part_ids)
                or len(set(part_ids)) != repeat_count
                or not isinstance(geometry_digest, str)
            ):
                raise RuntimeError("Policy repeated geometry cohort is invalid")
            for part_id in part_ids:
                part = parts_by_id.get(part_id)
                if part is None:
                    raise RuntimeError(
                        "Policy repeated geometry part is absent from registry"
                    )
                source_payload, source_color = _policy_source_signature(
                    part, label=f"registry part {part_id}"
                )
                geometry_payload = _policy_geometry_signature(
                    part, label=f"registry part {part_id}"
                )
                if (
                    canonical_sha256(source_payload) != signature_digest
                    or source_color != color
                    or canonical_sha256(geometry_payload) != geometry_digest
                ):
                    raise RuntimeError(
                        "Policy repeated geometry evidence is not reproducible"
                    )
                audited_part_records[part_id] = {
                    "canonical_group_id": group_id,
                    "canonical_color_family": color,
                    "canonical_group_association_basis": palette_group.get(
                        "association_basis"
                    ),
                    "canonical_source_view_ids": sorted(
                        palette_group["source_view_ids"]
                    ),
                    "source_visual_signature_sha256": signature_digest,
                    "source_signature_count": signature_count,
                    "registry_fraction": round(signature_count / len(raw_parts), 8),
                    "geometry_signature_sha256": geometry_digest,
                    "geometry_repeat_count": repeat_count,
                }
                cohort_ids.add(part_id)
        if geometry_anchor_ids is None:
            geometry_anchor_ids = sorted(cohort_ids)
        if (
            not isinstance(geometry_anchor_ids, list)
            or any(not isinstance(value, str) for value in geometry_anchor_ids)
            or set(geometry_anchor_ids) != cohort_ids
            or not cohort_ids.issubset(set(group_eligible_ids))
        ):
            raise RuntimeError(
                "Policy source-corroboration eligible cohort is inconsistent"
            )
        expansion_ids = sorted(set(group_eligible_ids) - cohort_ids)
        for part_id in expansion_ids:
            part = parts_by_id.get(part_id)
            if part is None:
                raise RuntimeError(
                    "Policy source-signature expansion part is absent from registry"
                )
            source_payload, source_color = _policy_source_signature(
                part, label=f"registry part {part_id}"
            )
            if (
                canonical_sha256(source_payload) != signature_digest
                or source_color != color
            ):
                raise RuntimeError(
                    "Policy source-signature expansion is not reproducible"
                )
            audited_part_records[part_id] = {
                "canonical_group_id": group_id,
                "canonical_color_family": color,
                "canonical_group_association_basis": palette_group.get(
                    "association_basis"
                ),
                "canonical_source_view_ids": sorted(
                    palette_group["source_view_ids"]
                ),
                "source_visual_signature_sha256": signature_digest,
                "source_signature_count": signature_count,
                "registry_fraction": round(signature_count / len(raw_parts), 8),
                "signature_expansion_basis": (
                    "exact_source_signature_with_repeated_geometry_anchor"
                ),
                "signature_expansion_anchor_part_ids": sorted(cohort_ids),
            }
    if set(audited_part_records) != set(eligible_ids):
        raise RuntimeError(
            "Policy source-corroboration audit does not exactly cover eligible parts"
        )

    for part_id, assignment in preserve_assignments.items():
        part = parts_by_id[part_id]
        provenance = assignment.get("provenance")
        reason_codes = provenance.get("reason_codes") if isinstance(
            provenance, dict
        ) else None
        declared_record = provenance.get("source_visual_corroboration") if isinstance(
            provenance, dict
        ) else None
        source_path = part.get("existing_visual_material")
        expected_binding_hash = canonical_sha256(
            {
                "part_id": part_id,
                "prim_path": part.get("prim_path"),
                "source_visual_material_prim_path": source_path,
            }
        )
        if (
            assignment.get("apply_action") != SOURCE_VISUAL_PRESERVE_ACTION
            or not isinstance(reason_codes, list)
            or not SOURCE_CORROBORATION_REASON_CODES.issubset(reason_codes)
            or declared_record != audited_part_records.get(part_id)
            or not isinstance(source_path, str)
            or assignment.get("source_visual_material_prim_path") != source_path
            or assignment.get("source_visual_material_binding_sha256")
            != expected_binding_hash
        ):
            raise RuntimeError(
                "Policy source-preserve assignment is not independently corroborated"
            )

    for part_id, assignment in mdl_assignments.items():
        part = parts_by_id[part_id]
        provenance = assignment.get("provenance")
        reason_codes = (
            provenance.get("reason_codes") if isinstance(provenance, dict) else None
        )
        declared_record = (
            provenance.get("source_visual_corroboration")
            if isinstance(provenance, dict)
            else None
        )
        source_material = (
            provenance.get("source_visual_material")
            if isinstance(provenance, dict)
            else None
        )
        source_path = part.get("existing_visual_material")
        expected_binding_hash = canonical_sha256(
            {
                "part_id": part_id,
                "prim_path": part.get("prim_path"),
                "source_visual_material_prim_path": source_path,
            }
        )
        expected_record = audited_part_records.get(part_id)
        legacy_confirmed_material_id = (
            declared_record.get("confirmed_material_id")
            if isinstance(declared_record, dict)
            else None
        )
        selected_material_id = (
            declared_record.get("selected_material_id")
            if isinstance(declared_record, dict)
            else None
        )
        if selected_material_id is None:
            selected_material_id = legacy_confirmed_material_id
        material_selection_basis = (
            declared_record.get("material_selection_basis")
            if isinstance(declared_record, dict)
            else None
        )
        if material_selection_basis is None and isinstance(
            legacy_confirmed_material_id, str
        ):
            material_selection_basis = "exact_forward_reverse_agreement"
        selection_confidence = (
            declared_record.get("selection_confidence")
            if isinstance(declared_record, dict)
            else None
        )
        declared_corroboration = (
            {
                key: value
                for key, value in declared_record.items()
                if key
                not in {
                    "confirmed_material_id",
                    "selected_material_id",
                    "material_selection_basis",
                    "selection_confidence",
                }
            }
            if isinstance(declared_record, dict)
            else None
        )
        group_id = (
            declared_corroboration.get("canonical_group_id")
            if isinstance(declared_corroboration, dict)
            else None
        )
        selection = group_selections.get(str(group_id))
        if selection_confidence is None and isinstance(selection, dict):
            selection_confidence = selection.get("confidence")
        selection_is_confirmed = (
            isinstance(reason_codes, list)
            and material_selection_basis == "exact_forward_reverse_agreement"
            and isinstance(selection, dict)
            and selection.get("confirmed") is True
            and SOURCE_MDL_CONFIRMED_REASON_CODE in reason_codes
            and not SOURCE_MDL_PROVISIONAL_REASON_CODES.intersection(reason_codes)
        )
        selection_is_provisional = (
            isinstance(reason_codes, list)
            and allow_high_confidence_provisional
            and material_selection_basis
            == CORROBORATED_SOURCE_PROVISIONAL_MATERIAL_BASIS
            and isinstance(selection, dict)
            and selection.get("confirmed") is False
            and isinstance(selection_confidence, (int, float))
            and not isinstance(selection_confidence, bool)
            and math.isfinite(float(selection_confidence))
            and CORROBORATED_SOURCE_MIN_PROVISIONAL_CONFIDENCE
            <= float(selection_confidence)
            <= 1.0
            and selection.get("confidence") == selection_confidence
            and SOURCE_MDL_PROVISIONAL_REASON_CODES.issubset(reason_codes)
            and SOURCE_MDL_CONFIRMED_REASON_CODE not in reason_codes
        )
        if (
            assignment.get("apply_action") == SOURCE_VISUAL_PRESERVE_ACTION
            or not isinstance(reason_codes, list)
            or not SOURCE_MDL_REPLACEMENT_REASON_CODES.issubset(reason_codes)
            or declared_corroboration != expected_record
            or provenance.get("canonical_group_id") != group_id
            or provenance.get("supporting_view_ids")
            != expected_record.get("canonical_source_view_ids")
            or provenance.get("canonical_group_assignment_basis")
            != "photo_corroborated_rare_repeated_source_visual"
            or not (selection_is_confirmed or selection_is_provisional)
            or not isinstance(selected_material_id, str)
            or not selected_material_id
            or assignment.get("material_id") != selected_material_id
            or not isinstance(selection, dict)
            or selection.get("material_id") != selected_material_id
            or selection.get("confidence") != selection_confidence
            or not isinstance(source_path, str)
            or not isinstance(source_material, dict)
            or source_material.get("material_prim_path") != source_path
            or source_material.get("binding_sha256") != expected_binding_hash
        ):
            raise RuntimeError(
                "Policy NVIDIA MDL replacement is not independently corroborated"
            )
    return len(preserve_assignments), len(mdl_assignments)


def _validate_policy_exact_cover_bundle(
    *,
    plan: dict[str, Any],
    audit: dict[str, Any],
    registry: dict[str, Any],
    staged_result: dict[str, Any],
    confidence_gate: dict[str, Any],
    base_plan: dict[str, Any],
    group_materials: dict[str, Any],
    mvinverse_pbr_evidence: dict[str, Any],
    whitelist: dict[str, Any],
    palette_fusion: dict[str, Any] | None = None,
    part_id_evidence: dict[str, Any] | None = None,
    expected_source_visual_strategy: str | None = None,
    expected_policy_overrides: Mapping[str, Any] | None = None,
    expected_immutable_mdl_after_selection: bool = False,
) -> int:
    """Validate a hash-bound policy plan before it may reach USD Apply."""

    if audit.get("schema_version") != POLICY_REPORT_SCHEMA_VERSION:
        raise RuntimeError(
            "Policy exact-cover audit has an unsupported schema_version"
        )
    provenance = plan.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("mode") != POLICY_PLAN_MODE:
        raise RuntimeError("Policy exact-cover plan has invalid provenance mode")
    expected_hashes = {
        "registry_asset_sha256": registry.get("asset_sha256"),
        "registry_sha256": canonical_sha256(registry),
        "staged_result_sha256": canonical_sha256(staged_result),
        "confidence_gate_sha256": canonical_sha256(confidence_gate),
        "base_plan_sha256": canonical_sha256(base_plan),
        "group_materials_sha256": canonical_sha256(group_materials),
        "mvinverse_pbr_evidence_sha256": canonical_sha256(
            mvinverse_pbr_evidence
        ),
        "whitelist_sha256": canonical_sha256(whitelist),
    }
    if palette_fusion is not None:
        expected_hashes["palette_fusion_sha256"] = canonical_sha256(
            palette_fusion
        )
    if part_id_evidence is not None:
        if not isinstance(expected_policy_overrides, Mapping):
            raise RuntimeError(
                "Policy Part-ID evidence convergence requires its exact source "
                "policy document"
            )
        unsigned_part_id_evidence = dict(part_id_evidence)
        part_id_integrity = unsigned_part_id_evidence.pop("integrity", None)
        raw_evidence_inputs = part_id_evidence.get("inputs")
        registry_input_hashes = [
            raw_input.get("document_sha256")
            for raw_input in (
                raw_evidence_inputs
                if isinstance(raw_evidence_inputs, list)
                else []
            )
            if isinstance(raw_input, dict)
            and raw_input.get("label") == "rendered_registry"
        ]
        if (
            part_id_evidence.get("schema_version")
            != "qwen-part-id-reference-evidence/v1"
            or part_id_evidence.get("assignment_unit") != "part_id"
            or not isinstance(part_id_integrity, dict)
            or part_id_integrity.get("document_sha256")
            != canonical_sha256(unsigned_part_id_evidence)
            or registry_input_hashes != [canonical_sha256(registry)]
        ):
            raise RuntimeError(
                "Policy Part-ID evidence is invalid or belongs to another registry"
            )
        expected_hashes["part_id_evidence_sha256"] = canonical_sha256(
            part_id_evidence
        )
        expected_hashes["source_policy_sha256"] = canonical_sha256(
            dict(expected_policy_overrides)
        )
    elif "part_id_evidence_sha256" in provenance:
        raise RuntimeError(
            "Policy exact-cover plan requires its bound Part-ID evidence"
        )
    for field, expected in expected_hashes.items():
        if provenance.get(field) != expected:
            raise RuntimeError(
                f"Policy exact-cover plan provenance mismatch for {field}"
            )
    policy = audit.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("Policy exact-cover audit is missing its policy")
    if provenance.get("policy_sha256") != canonical_sha256(policy):
        raise RuntimeError("Policy exact-cover policy hash mismatch")
    source_visual_strategy = policy.get("source_visual_strategy", "preserve")
    if source_visual_strategy not in {"preserve", "neutralize_unverified"}:
        raise RuntimeError(
            "Policy exact-cover source visual strategy is invalid"
        )
    if (
        expected_source_visual_strategy is not None
        and source_visual_strategy != expected_source_visual_strategy
    ):
        raise RuntimeError(
            "Policy exact-cover source visual strategy does not match the "
            "authorized workflow"
        )
    if (
        expected_policy_overrides is not None
        and not _policy_checkpoint_matches_requested_overrides(
            audit=audit,
            requested_policy=expected_policy_overrides,
        )
    ):
        raise RuntimeError(
            "Policy exact-cover effective policy does not match the requested "
            "workflow overrides"
        )
    input_hashes = audit.get("input_hashes")
    if not isinstance(input_hashes, dict) or input_hashes != provenance:
        raise RuntimeError("Policy exact-cover audit input hashes do not match plan")
    if audit.get("output_plan_sha256") != canonical_sha256(plan):
        raise RuntimeError("Policy exact-cover audit output plan hash mismatch")
    if expected_immutable_mdl_after_selection:
        immutable_summary = audit.get("summary")
        if (
            provenance.get("immutable_mdl_after_selection") is not True
            or not isinstance(immutable_summary, dict)
            or immutable_summary.get(
                "selected_mdl_library_defaults_locked"
            )
            is not True
            or immutable_summary.get(
                "mvinverse_parameterized_part_count"
            )
            != 0
        ):
            raise RuntimeError(
                "Policy exact-cover did not preserve immutable MDL library defaults"
            )

    raw_parts = registry.get("parts")
    assignments = plan.get("assignments")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise RuntimeError("Policy exact-cover registry has no parts")
    if not isinstance(assignments, list) or not assignments:
        raise RuntimeError("Policy exact-cover plan has no assignments")
    registry_ids = [
        item.get("part_id") if isinstance(item, dict) else None for item in raw_parts
    ]
    assignment_ids = [
        item.get("part_id") if isinstance(item, dict) else None
        for item in assignments
    ]
    if (
        any(not isinstance(part_id, str) for part_id in registry_ids)
        or len(set(registry_ids)) != len(registry_ids)
        or any(not isinstance(part_id, str) for part_id in assignment_ids)
        or len(set(assignment_ids)) != len(assignment_ids)
        or set(assignment_ids) != set(registry_ids)
    ):
        raise RuntimeError(
            "Policy exact-cover plan does not cover the registry exactly once"
        )
    if part_id_evidence is not None:
        raw_evidence_parts = part_id_evidence.get("parts")
        evidence_status_by_part: dict[str, str] = {}
        if not isinstance(raw_evidence_parts, list):
            raise RuntimeError("Policy Part-ID evidence has no parts")
        for index, raw_evidence_part in enumerate(raw_evidence_parts):
            if not isinstance(raw_evidence_part, dict):
                raise RuntimeError(
                    f"Policy Part-ID evidence part {index} is invalid"
                )
            part_id = raw_evidence_part.get("part_id")
            status = raw_evidence_part.get("status")
            observations = raw_evidence_part.get("observations")
            if (
                not isinstance(part_id, str)
                or not part_id
                or part_id in evidence_status_by_part
                or status not in {"observed", "unobserved"}
                or not isinstance(observations, list)
                or (status == "observed" and not observations)
                or (status == "unobserved" and observations)
            ):
                raise RuntimeError(
                    f"Policy Part-ID evidence part {index} is malformed"
                )
            evidence_status_by_part[part_id] = str(status)
        if set(evidence_status_by_part) != set(registry_ids):
            raise RuntimeError(
                "Policy Part-ID evidence does not exactly cover the registry"
            )
        assignment_by_part = {
            str(item["part_id"]): item
            for item in assignments
            if isinstance(item, dict)
        }
        group_keys = {"canonical_group_id", "material_region_group_id", "group_id"}
        unobserved_part_ids = {
            part_id
            for part_id, status in evidence_status_by_part.items()
            if status == "unobserved"
        }
        invalid_hidden_parts = []
        for part_id in sorted(unobserved_part_ids):
            assignment = assignment_by_part[part_id]
            assignment_provenance = assignment.get("provenance")
            if (
                assignment.get("status") != POLICY_FALLBACK_STATUS
                or any(assignment.get(key) is not None for key in group_keys)
                or not isinstance(assignment_provenance, dict)
                or any(
                    assignment_provenance.get(key) is not None
                    for key in group_keys
                )
            ):
                invalid_hidden_parts.append(part_id)
        convergence = audit.get("part_id_evidence_convergence")
        evidence_summary = part_id_evidence.get("summary")
        if (
            invalid_hidden_parts
            or not isinstance(evidence_summary, dict)
            or evidence_summary.get("registry_part_count") != len(registry_ids)
            or evidence_summary.get("unobserved_part_count")
            != len(unobserved_part_ids)
            or not isinstance(convergence, dict)
            or convergence.get("state") != "final_visibility_applied"
            or convergence.get("part_id_evidence_sha256")
            != canonical_sha256(part_id_evidence)
            or convergence.get("unobserved_part_count")
            != len(unobserved_part_ids)
        ):
            raise RuntimeError(
                "Policy Part-ID evidence convergence is inconsistent; "
                f"invalid hidden parts={invalid_hidden_parts[:20]}"
            )

    # A parent Mesh binding does not override an existing materialBind
    # GeomSubset. New registries therefore carry a hash-bound inventory of
    # those subsets, and the policy plan must cover their exact source
    # topology before Isaac Apply starts. Source-visual no-op assignments are
    # the only exception because Apply intentionally preserves their existing
    # parent and subset bindings.
    subset_field = "existing_material_bind_face_subsets"
    subset_contract_enabled = any(
        isinstance(part, dict) and subset_field in part for part in raw_parts
    )
    source_subset_part_count = 0
    source_subset_count = 0
    if subset_contract_enabled:
        assignment_by_id = {
            str(item["part_id"]): item
            for item in assignments
            if isinstance(item, dict) and isinstance(item.get("part_id"), str)
        }
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                raise RuntimeError("Policy exact-cover registry part is invalid")
            part_id = raw_part.get("part_id")
            source_subsets = raw_part.get(subset_field)
            if not isinstance(part_id, str) or not isinstance(source_subsets, list):
                raise RuntimeError(
                    "Policy exact-cover registry has an incomplete source-subset "
                    f"contract: {part_id!r}"
                )
            if not source_subsets:
                continue
            source_subset_part_count += 1
            source_subset_count += len(source_subsets)
            source_by_name: dict[str, list[int]] = {}
            claimed_faces: set[int] = set()
            face_count = raw_part.get("face_count")
            if (
                isinstance(face_count, bool)
                or not isinstance(face_count, int)
                or face_count <= 0
            ):
                raise RuntimeError(
                    f"Source-subset registry face count is invalid: {part_id}"
                )
            for raw_subset in source_subsets:
                if not isinstance(raw_subset, dict):
                    raise RuntimeError(
                        f"Source materialBind subset is invalid: {part_id}"
                    )
                subset_name = raw_subset.get("subset_name")
                face_indices = raw_subset.get("face_indices")
                if (
                    not isinstance(subset_name, str)
                    or not subset_name
                    or subset_name in source_by_name
                    or raw_subset.get("family_name") != "materialBind"
                    or raw_subset.get("element_type") != "face"
                    or not isinstance(face_indices, list)
                    or not face_indices
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 0
                        or index >= face_count
                        for index in face_indices
                    )
                    or len(set(face_indices)) != len(face_indices)
                    or bool(claimed_faces.intersection(face_indices))
                ):
                    raise RuntimeError(
                        "Source materialBind subset topology is invalid or "
                        f"overlapping: {part_id}.{subset_name}"
                    )
                source_by_name[subset_name] = list(face_indices)
                claimed_faces.update(face_indices)

            assignment = assignment_by_id[part_id]
            planned_subsets = assignment.get("face_subsets")
            if assignment.get("apply_action") == SOURCE_VISUAL_PRESERVE_ACTION:
                if planned_subsets not in (None, []):
                    raise RuntimeError(
                        "Source-visual preserve assignment must not replace source "
                        f"materialBind subsets: {part_id}"
                    )
                continue
            if not isinstance(planned_subsets, list) or not planned_subsets:
                raise RuntimeError(
                    "Policy exact-cover plan omits source materialBind subsets: "
                    f"{part_id}"
                )
            planned_by_name: dict[str, list[int]] = {}
            for planned_subset in planned_subsets:
                if not isinstance(planned_subset, dict):
                    raise RuntimeError(
                        f"Policy face-subset assignment is invalid: {part_id}"
                    )
                subset_name = planned_subset.get("subset_name")
                face_indices = planned_subset.get("face_indices")
                if (
                    not isinstance(subset_name, str)
                    or subset_name in planned_by_name
                    or not isinstance(face_indices, list)
                ):
                    raise RuntimeError(
                        f"Policy face-subset identity is invalid: {part_id}"
                    )
                planned_by_name[subset_name] = list(face_indices)
            if planned_by_name != source_by_name:
                raise RuntimeError(
                    "Policy exact-cover source materialBind subset topology "
                    f"mismatch: {part_id}"
                )

    allowed_statuses = APPLICABLE_ASSIGNMENT_STATUSES | {POLICY_FALLBACK_STATUS}
    invalid_statuses = sorted(
        {
            str(item.get("status", "review"))
            for item in assignments
            if not isinstance(item, dict)
            or item.get("status", "review") not in allowed_statuses
        }
    )
    if invalid_statuses:
        raise RuntimeError(
            "Policy exact-cover plan contains invalid assignment statuses: "
            f"{invalid_statuses}"
        )
    fallback_count = sum(
        1
        for item in assignments
        if isinstance(item, dict) and item.get("status") == POLICY_FALLBACK_STATUS
    )
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Policy exact-cover audit is missing its summary")
    if subset_contract_enabled and (
        _require_exact_int(
            summary.get("source_material_bind_subset_part_count"),
            "Policy audit source materialBind subset part count",
        )
        != source_subset_part_count
        or _require_exact_int(
            summary.get("source_material_bind_subset_count"),
            "Policy audit source materialBind subset count",
        )
        != source_subset_count
    ):
        raise RuntimeError(
            "Policy exact-cover source materialBind subset audit is inconsistent"
        )
    expected_count = len(raw_parts)
    if (
        summary.get("exact_cover") is not True
        or summary.get("all_materials_in_industrial_whitelist") is not True
        or _require_exact_int(
            summary.get("registry_part_count"),
            "Policy audit registry_part_count",
        )
        != expected_count
        or _require_exact_int(
            summary.get("output_assignment_count"),
            "Policy audit output_assignment_count",
        )
        != expected_count
        or _require_exact_int(
            summary.get("policy_fallback_count"),
            "Policy audit policy_fallback_count",
        )
        != fallback_count
    ):
        raise RuntimeError("Policy exact-cover audit summary is inconsistent")
    if expected_source_visual_strategy is not None:
        tier_counts = Counter(
            str(item.get("provenance", {}).get("tier"))
            for item in assignments
            if isinstance(item, dict)
            and isinstance(item.get("provenance"), dict)
        )
        preserve_count = tier_counts["source_visual_preserve"]
        corroborated_mdl_count = tier_counts[CORROBORATED_SOURCE_MDL_TIER]
        source_neutral_count = tier_counts[
            "source_preserve_unavailable_neutral_fallback"
        ]
        neutral_default_count = tier_counts["neutral_default"]
        corroborated_preserve_count, validated_corroborated_mdl_count = (
            _validate_corroborated_source_visual_assignments(
                assignments=assignments,
                raw_parts=raw_parts,
                audit=audit,
                palette_fusion=palette_fusion,
                group_materials=group_materials,
                allow_high_confidence_provisional=(
                    expected_immutable_mdl_after_selection
                ),
            )
            if source_visual_strategy == "neutralize_unverified"
            else (0, 0)
        )
        if (
            audit.get("source_visual_strategy") != source_visual_strategy
            or _require_exact_int(
                summary.get("source_visual_preserve_count"),
                "Policy audit source_visual_preserve_count",
            )
            != preserve_count
            or _require_exact_int(
                summary.get(
                    "source_preserve_unavailable_neutral_fallback_count"
                ),
                "Policy audit source neutral fallback count",
            )
            != source_neutral_count
            or _require_exact_int(
                summary.get("neutral_default_count"),
                "Policy audit neutral_default_count",
            )
            != neutral_default_count
            or (
                source_visual_strategy == "neutralize_unverified"
                and (
                    preserve_count != corroborated_preserve_count
                    or _require_exact_int(
                        summary.get(
                            "corroborated_source_visual_preserve_count",
                            0,
                        ),
                        "Policy audit corroborated source preserve count",
                    )
                    != corroborated_preserve_count
                    or corroborated_mdl_count != validated_corroborated_mdl_count
                    or _require_exact_int(
                        summary.get(
                            "corroborated_source_visual_nvidia_mdl_count",
                            0,
                        ),
                        "Policy audit corroborated NVIDIA MDL count",
                    )
                    != validated_corroborated_mdl_count
                    or _require_exact_int(
                        summary.get(
                            (
                                "corroborated_source_visual_"
                                "provisional_nvidia_mdl_count"
                            ),
                            0,
                        ),
                        "Policy audit provisional corroborated NVIDIA MDL count",
                    )
                    != sum(
                        1
                        for item in assignments
                        if isinstance(item, dict)
                        and item.get("provenance", {})
                        .get("source_visual_corroboration", {})
                        .get("material_selection_basis")
                        == CORROBORATED_SOURCE_PROVISIONAL_MATERIAL_BASIS
                    )
                    or sum(
                        1
                        for item in assignments
                        if isinstance(item, dict)
                        and item.get("apply_action")
                        == SOURCE_VISUAL_PRESERVE_ACTION
                    )
                    != preserve_count
                    or any(
                        isinstance(item, dict)
                        and item.get("apply_action")
                        == SOURCE_VISUAL_PRESERVE_ACTION
                        and item.get("provenance", {}).get("tier")
                        != "source_visual_preserve"
                        for item in assignments
                    )
                )
            )
        ):
            raise RuntimeError(
                "Policy exact-cover source visual strategy audit is inconsistent"
            )
    if part_id_evidence is not None:
        try:
            replay_plan, replay_audit = build_policy_exact_cover(
                registry=registry,
                staged_result=staged_result,
                confidence_gate=confidence_gate,
                whitelist=whitelist,
                policy=expected_policy_overrides,
                base_plan=base_plan,
                group_materials=group_materials,
                mvinverse_pbr_evidence=mvinverse_pbr_evidence,
                palette_fusion=palette_fusion,
                part_id_evidence=part_id_evidence,
                acknowledge_policy_fallback=True,
                immutable_mdl_after_selection=(
                    expected_immutable_mdl_after_selection
                ),
            )
        except PolicyExactCoverError as exc:
            raise RuntimeError(
                f"Policy Part-ID evidence replay failed closed: {exc}"
            ) from exc
        if plan != replay_plan or audit != replay_audit:
            raise RuntimeError(
                "Policy Part-ID evidence convergence differs from exact trusted replay"
            )
    return fallback_count

__all__ = [
    "_archive_stale_policy_exact_cover_checkpoint",
    "_policy_checkpoint_matches_requested_overrides",
    "_require_exact_int",
    "_validate_corroborated_source_visual_assignments",
    "_validate_policy_exact_cover_bundle",
]
