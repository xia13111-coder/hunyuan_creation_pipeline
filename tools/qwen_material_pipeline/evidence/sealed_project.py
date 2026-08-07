"""Hash-bound identity checks for sealed material-project evidence.

The evidence method is project data, not a process-wide constant.  Accepting a
new historical result by merely changing a string constant would make the
preservation gate too easy to desynchronise from the project that actually
owns the template, catalog, and dependency lock.  This module therefore binds
the method to the exact ``project.json`` bytes and cross-checks every owned
artifact recorded by the evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = "qwen-material-project/v2"


class SealedProjectBindingError(ValueError):
    """Raised when sealed evidence is not owned by one exact project file."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SealedProjectBindingError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SealedProjectBindingError(f"{label} must be a non-empty string")
    return value


def _recorded_file(value: Any, label: str) -> Path:
    raw = _text(value, label)
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SealedProjectBindingError(f"{label} does not exist: {raw}") from exc
    if not path.is_file():
        raise SealedProjectBindingError(f"{label} is not a file: {path}")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedProjectBindingError(
            f"Unable to read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise SealedProjectBindingError(f"{label} must be a JSON object: {path}")
    return document


def _project_owned_file(project: Path, value: Any, label: str) -> Path:
    relative = _text(value, f"sealed project {label}")
    project_dir = project.parent.resolve(strict=True)
    try:
        path = (project_dir / relative).resolve(strict=True)
        path.relative_to(project_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SealedProjectBindingError(
            f"sealed project {label} escapes its project directory"
        ) from exc
    if not path.is_file():
        raise SealedProjectBindingError(
            f"sealed project {label} is not a file: {path}"
        )
    return path


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SealedProjectBindingError(f"{label} must be a positive integer")
    return value


def _owned_hashed_record(
    project: Path,
    value: Any,
    label: str,
) -> tuple[Path, str]:
    record = dict(value) if isinstance(value, Mapping) else None
    if record is None or set(record) != {"path", "sha256"}:
        raise SealedProjectBindingError(
            f"sealed project {label} must contain exactly path and sha256"
        )
    path = _project_owned_file(project, record.get("path"), label)
    expected_sha256 = _sha256_text(
        record.get("sha256"),
        f"sealed project {label} sha256",
    )
    if _sha256_file(path) != expected_sha256:
        raise SealedProjectBindingError(f"sealed project {label} hash changed")
    return path, expected_sha256


def _has_trusted_palette_sample(document: Mapping[str, Any]) -> bool:
    groups = document.get("groups")
    if not isinstance(groups, list):
        return False
    for group in groups:
        if not isinstance(group, Mapping) or group.get("accepted") is not True:
            continue
        boxes = group.get("boxes")
        if (
            not isinstance(group.get("base_color"), str)
            or not group.get("base_color")
            or not isinstance(boxes, list)
        ):
            continue
        for box in boxes:
            if not isinstance(box, Mapping) or box.get("accepted") is not True:
                continue
            representative = box.get("representative_srgb")
            matching_pixels = box.get(
                "matching_pixel_count", box.get("matching_pixels")
            )
            normalized_box = box.get("box")
            if (
                isinstance(representative, list)
                and len(representative) == 3
                and all(
                    isinstance(channel, int)
                    and not isinstance(channel, bool)
                    and 0 <= channel <= 255
                    for channel in representative
                )
                and isinstance(matching_pixels, int)
                and not isinstance(matching_pixels, bool)
                and matching_pixels >= 64
                and isinstance(normalized_box, list)
                and len(normalized_box) == 4
                and all(
                    isinstance(component, (int, float))
                    and not isinstance(component, bool)
                    and 0 <= float(component) <= 1000
                    for component in normalized_box
                )
                and float(normalized_box[0]) < float(normalized_box[2])
                and float(normalized_box[1]) < float(normalized_box[3])
            ):
                return True
    return False


def _sealed_acceptance_evidence(
    project: Path,
    document: Mapping[str, Any],
    reference_roles: tuple[str, ...],
) -> dict[str, Any] | None:
    descriptor = document.get("acceptance_evidence")
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "manifest",
        "sha256",
    }:
        raise SealedProjectBindingError(
            "sealed project acceptance_evidence must contain exactly "
            "manifest and sha256"
        )
    manifest = _project_owned_file(
        project,
        descriptor.get("manifest"),
        "acceptance evidence manifest",
    )
    manifest_sha256 = _sha256_text(
        descriptor.get("sha256"),
        "sealed project acceptance evidence manifest sha256",
    )
    if _sha256_file(manifest) != manifest_sha256:
        raise SealedProjectBindingError(
            "sealed project acceptance evidence manifest hash changed"
        )
    evidence_document = _read_object(manifest, "acceptance evidence manifest")
    if set(evidence_document) != {"schema_version", "source_views"} or (
        evidence_document.get("schema_version")
        != "qwen-bundled-acceptance-evidence/v1"
    ):
        raise SealedProjectBindingError(
            "sealed project acceptance evidence schema is invalid"
        )
    project_references = document.get("references")
    assert isinstance(project_references, list)
    reference_hashes = {
        reference.get("role"): reference.get("sha256")
        for reference in project_references
        if isinstance(reference, Mapping)
    }
    raw_views = evidence_document.get("source_views")
    if not isinstance(raw_views, list) or len(raw_views) != len(reference_roles):
        raise SealedProjectBindingError(
            "sealed acceptance evidence must cover every reference"
        )
    normalized_views: list[dict[str, Any]] = []
    evidence_roles: list[str] = []
    for index, raw_view in enumerate(raw_views):
        view = dict(raw_view) if isinstance(raw_view, Mapping) else None
        required_fields = {
            "id",
            "reference_sha256",
            "palette_status",
            "palette_mask",
            "palette_path",
            "palette_artifacts",
        }
        if view is None or set(view) != required_fields:
            raise SealedProjectBindingError(
                f"sealed acceptance evidence source_views[{index}] is invalid"
            )
        role = _text(view.get("id"), f"acceptance evidence view[{index}] id")
        evidence_roles.append(role)
        reference_sha256 = _sha256_text(
            view.get("reference_sha256"),
            f"acceptance evidence {role} reference_sha256",
        )
        if reference_sha256 != reference_hashes.get(role):
            raise SealedProjectBindingError(
                f"acceptance evidence {role} reference hash changed"
            )
        if view.get("palette_status") != "usable":
            raise SealedProjectBindingError(
                f"acceptance evidence {role} palette is not usable"
            )
        mask, mask_sha256 = _owned_hashed_record(
            project, view.get("palette_mask"), f"{role} palette_mask"
        )
        palette, palette_sha256 = _owned_hashed_record(
            project, view.get("palette_path"), f"{role} palette_path"
        )
        artifacts = view.get("palette_artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "normalized",
            "normalized_evidence_audit",
        }:
            raise SealedProjectBindingError(
                f"acceptance evidence {role} palette_artifacts are invalid"
            )
        normalized, normalized_sha256 = _owned_hashed_record(
            project,
            artifacts.get("normalized"),
            f"{role} normalized palette",
        )
        audit, audit_sha256 = _owned_hashed_record(
            project,
            artifacts.get("normalized_evidence_audit"),
            f"{role} normalized evidence audit",
        )
        if palette != normalized or palette_sha256 != normalized_sha256:
            raise SealedProjectBindingError(
                f"acceptance evidence {role} palette differs from normalized"
            )
        palette_document = _read_object(palette, f"{role} normalized palette")
        if (
            palette_document.get("schema_version") != "qwen-material-palette/v1"
            or palette_document.get("source_view_id") != role
            or not isinstance(palette_document.get("groups"), list)
        ):
            raise SealedProjectBindingError(
                f"acceptance evidence {role} palette schema is invalid"
            )
        if not _has_trusted_palette_sample(
            _read_object(audit, f"{role} normalized evidence audit")
        ):
            raise SealedProjectBindingError(
                f"acceptance evidence {role} has no trusted palette sample"
            )
        normalized_views.append(
            {
                "id": role,
                "reference_sha256": reference_sha256,
                "palette_status": "usable",
                "palette_mask": str(mask),
                "palette_mask_sha256": mask_sha256,
                "palette_path": str(palette),
                "palette_path_sha256": palette_sha256,
                "palette_artifacts": {
                    "normalized": str(normalized),
                    "normalized_sha256": normalized_sha256,
                    "normalized_evidence_audit": str(audit),
                    "normalized_evidence_audit_sha256": audit_sha256,
                },
            }
        )
    if tuple(evidence_roles) != reference_roles:
        raise SealedProjectBindingError(
            "sealed acceptance evidence roles/order differ from project references"
        )
    return {
        "schema_version": "qwen-bundled-acceptance-evidence/v1",
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "source_views": normalized_views,
    }


def _sealed_acceptance_contract(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Parse the complete acceptance contract owned by one project hash."""

    raw_acceptance = document.get("acceptance")
    acceptance = (
        dict(raw_acceptance) if isinstance(raw_acceptance, Mapping) else None
    )
    expected_acceptance_fields = {
        "render",
        "view_mapping",
        "minimum_comparable_views",
    }
    if acceptance is None or set(acceptance) != expected_acceptance_fields:
        raise SealedProjectBindingError(
            "sealed project acceptance must contain exactly "
            f"{sorted(expected_acceptance_fields)}"
        )

    raw_render = acceptance.get("render")
    render = dict(raw_render) if isinstance(raw_render, Mapping) else None
    expected_render_fields = {
        "resolution",
        "views",
        "rt_subframes",
        "lighting_profile",
        "analysis_up_axis",
        "analysis_front_axis",
    }
    if render is None or set(render) != expected_render_fields:
        raise SealedProjectBindingError(
            "sealed project acceptance.render must contain exactly "
            f"{sorted(expected_render_fields)}"
        )

    resolution = _positive_integer(
        render.get("resolution"),
        "sealed project acceptance resolution",
    )
    rt_subframes = _positive_integer(
        render.get("rt_subframes"),
        "sealed project acceptance rt_subframes",
    )
    raw_views = _text(
        render.get("views"),
        "sealed project acceptance views",
    )
    views = raw_views.split(",")
    if (
        any(not view or view.strip() != view for view in views)
        or len(set(views)) != len(views)
    ):
        raise SealedProjectBindingError(
            "sealed project acceptance views must be unique comma-separated tokens"
        )
    lighting_profile = _text(
        render.get("lighting_profile"),
        "sealed project acceptance lighting_profile",
    )
    if lighting_profile not in {"geometry", "material-neutral"}:
        raise SealedProjectBindingError(
            "sealed project acceptance lighting_profile is unsupported"
        )
    supported_axes = {"x", "-x", "y", "-y", "z", "-z"}
    up_axis = _text(
        render.get("analysis_up_axis"),
        "sealed project acceptance analysis_up_axis",
    )
    front_axis = _text(
        render.get("analysis_front_axis"),
        "sealed project acceptance analysis_front_axis",
    )
    if up_axis not in supported_axes or front_axis not in supported_axes:
        raise SealedProjectBindingError(
            "sealed project acceptance analysis axis is unsupported"
        )
    if up_axis.lstrip("-") == front_axis.lstrip("-"):
        raise SealedProjectBindingError(
            "sealed project acceptance axes must not be parallel"
        )

    raw_references = document.get("references")
    if not isinstance(raw_references, list) or not raw_references:
        raise SealedProjectBindingError("sealed project references are invalid")
    reference_roles: list[str] = []
    for index, raw_reference in enumerate(raw_references):
        reference = (
            raw_reference if isinstance(raw_reference, Mapping) else None
        )
        role = reference.get("role") if reference is not None else None
        if not isinstance(role, str) or not role or role in reference_roles:
            raise SealedProjectBindingError(
                "sealed project reference roles are invalid or duplicated "
                f"at index {index}"
            )
        reference_roles.append(role)

    raw_mapping = acceptance.get("view_mapping")
    mapping = dict(raw_mapping) if isinstance(raw_mapping, Mapping) else None
    if mapping is None or set(mapping) != set(reference_roles):
        raise SealedProjectBindingError(
            "sealed project acceptance mapping must cover every reference role"
        )
    normalized_mapping: dict[str, str] = {}
    for reference_role, render_view in mapping.items():
        if not isinstance(reference_role, str) or not isinstance(render_view, str):
            raise SealedProjectBindingError(
                "sealed project acceptance mapping entries must be strings"
            )
        if render_view not in views:
            raise SealedProjectBindingError(
                "sealed project acceptance mapping targets an undeclared view"
            )
        normalized_mapping[reference_role] = render_view
    if len(set(normalized_mapping.values())) != len(normalized_mapping):
        raise SealedProjectBindingError(
            "sealed project acceptance mapping must be one-to-one"
        )
    if set(normalized_mapping.values()) != set(views):
        raise SealedProjectBindingError(
            "sealed project acceptance mapping must exactly cover render views"
        )
    minimum_comparable_views = _positive_integer(
        acceptance.get("minimum_comparable_views"),
        "sealed project acceptance minimum_comparable_views",
    )
    if minimum_comparable_views != len(normalized_mapping):
        raise SealedProjectBindingError(
            "sealed project acceptance minimum_comparable_views must equal "
            "the exact mapping size"
        )

    return (
        {
            "render": {
                "resolution": resolution,
                "views": raw_views,
                "rt_subframes": rt_subframes,
                "lighting_profile": lighting_profile,
                "analysis_up_axis": up_axis,
                "analysis_front_axis": front_axis,
            },
            "view_mapping": dict(sorted(normalized_mapping.items())),
            "minimum_comparable_views": minimum_comparable_views,
        },
        tuple(reference_roles),
    )


def validate_sealed_project_binding(
    evidence: Mapping[str, Any],
    *,
    expected_project_sha256: str | None = None,
    expected_project_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and return the project identity declared by sealed evidence.

    ``expected_project_sha256`` is normally taken from the canonical material
    plan and ``expected_project_path`` from its independently generated audit.
    Supplying both lets the orchestrator prove that evidence, plan, and audit
    all refer to the same immutable project.  The standalone final visual gate
    still validates the explicit project/hash pair embedded in the evidence.
    """

    project = _recorded_file(evidence.get("project"), "sealed evidence project")
    project_sha256 = _sha256_text(
        evidence.get("project_sha256"),
        "sealed evidence project_sha256",
    )
    actual_project_sha256 = _sha256_file(project)
    if project_sha256 != actual_project_sha256:
        raise SealedProjectBindingError(
            "sealed evidence project hash does not match the file"
        )
    if expected_project_sha256 is not None:
        if (
            _sha256_text(expected_project_sha256, "expected project SHA-256")
            != project_sha256
        ):
            raise SealedProjectBindingError(
                "sealed evidence project hash does not match the material plan"
            )
    if expected_project_path is not None:
        expected_path = _recorded_file(
            str(expected_project_path),
            "bundled audit project",
        )
        if expected_path != project:
            raise SealedProjectBindingError(
                "sealed evidence project does not match the bundled audit"
            )

    document = _read_object(project, "sealed material project")
    if document.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise SealedProjectBindingError(
            "sealed evidence project has an unsupported schema_version"
        )
    asset_id = _text(evidence.get("asset_id"), "sealed evidence asset_id")
    if document.get("asset_id") != asset_id:
        raise SealedProjectBindingError(
            "sealed evidence asset_id does not match the project"
        )
    project_evidence = document.get("evidence")
    if not isinstance(project_evidence, Mapping):
        raise SealedProjectBindingError(
            "sealed material project lacks an evidence contract"
        )
    method = _text(evidence.get("method"), "sealed evidence method")
    if _text(project_evidence.get("method"), "sealed project evidence method") != method:
        raise SealedProjectBindingError(
            "sealed evidence method does not match the hash-bound project"
        )
    historical_result_sha256 = _sha256_text(
        evidence.get("historical_result_sha256"),
        "sealed evidence historical_result_sha256",
    )
    if (
        _sha256_text(
            project_evidence.get("historical_result_sha256"),
            "sealed project historical_result_sha256",
        )
        != historical_result_sha256
    ):
        raise SealedProjectBindingError(
            "sealed evidence historical result does not match the project"
        )

    acceptance, reference_roles = _sealed_acceptance_contract(document)
    acceptance_evidence = _sealed_acceptance_evidence(
        project,
        document,
        reference_roles,
    )
    evidence_references = evidence.get("reference_sha256")
    if (
        not isinstance(evidence_references, Mapping)
        or set(evidence_references) != set(reference_roles)
    ):
        raise SealedProjectBindingError(
            "sealed evidence references do not exactly cover project roles"
        )

    artifact_paths: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}
    for artifact in ("template", "catalog", "dependency_lock"):
        project_artifact = _project_owned_file(
            project,
            document.get(artifact),
            artifact,
        )
        evidence_artifact = _recorded_file(
            evidence.get(artifact),
            f"sealed evidence {artifact}",
        )
        if evidence_artifact != project_artifact:
            raise SealedProjectBindingError(
                f"sealed evidence {artifact} does not match the project"
            )
        digest = _sha256_file(project_artifact)
        manifest_digest = _sha256_text(
            document.get(f"{artifact}_sha256"),
            f"sealed project {artifact}_sha256",
        )
        evidence_digest = _sha256_text(
            evidence.get(f"{artifact}_sha256"),
            f"sealed evidence {artifact}_sha256",
        )
        if digest != manifest_digest or digest != evidence_digest:
            raise SealedProjectBindingError(
                f"sealed {artifact} hash is stale or inconsistent"
            )
        artifact_paths[artifact] = str(project_artifact)
        artifact_sha256[artifact] = digest

    return {
        "project": str(project),
        "project_sha256": project_sha256,
        "asset_id": asset_id,
        "method": method,
        "historical_result_sha256": historical_result_sha256,
        "acceptance": acceptance,
        "acceptance_sha256": _canonical_sha256(acceptance),
        "acceptance_evidence": acceptance_evidence,
        "acceptance_evidence_sha256": (
            _canonical_sha256(acceptance_evidence)
            if acceptance_evidence is not None
            else None
        ),
        "reference_roles": list(reference_roles),
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
    }


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "SealedProjectBindingError",
    "validate_sealed_project_binding",
]
