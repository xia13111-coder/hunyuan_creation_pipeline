"""Discovery and fail-closed matching for sealed material projects."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..project_layout import ProjectLayout
from ..runtime import isaac_python, root_dir
from .references import sha256_file
from .sealed_dependencies import verify_sealed_dependency_lock


PROJECT_SCHEMA_VERSION = "qwen-material-project/v2"
_SOURCE_TOPOLOGY_ROLES = frozenset(
    {"pre_expansion", "occurrence_equivalent"}
)
_CANONICAL_RENDER_VIEWS = frozenset(
    {"front", "rear", "left", "right", "top", "iso"}
)
_SUPPORTED_EXPLICIT_RENDER_VIEWS = frozenset(
    {
        *_CANONICAL_RENDER_VIEWS,
        *(f"pose_a{azimuth:03d}_e015" for azimuth in (45, 135, 225, 315)),
        *(
            f"pose_a{azimuth:03d}_{suffix}"
            for azimuth in range(0, 360, 45)
            for suffix in (
                "e035",
                "e060",
                "e075",
                "e075_r180",
                "e082",
                "e082_r180",
                "e082_toproll",
                "e082_toproll_r180",
            )
        ),
    }
)


@dataclass(frozen=True)
class BundledMaterialProject:
    asset_id: str
    project_file: Path
    planner_module: str
    template: Path
    catalog: Path
    dependency_lock: Path
    dependency_lock_verification: dict[str, Any]
    material_root: Path
    render: dict[str, Any]
    acceptance: dict[str, Any]
    acceptance_evidence: dict[str, Any] | None
    document: dict[str, Any]
    source_representation_id: str
    source_registry_topology_role: str


def _read_project(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != PROJECT_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported bundled material project: {path}")
    return document


def _nvidia_materials_root(configured_root: Path) -> Path:
    candidate = configured_root.expanduser().resolve(strict=True)
    if candidate.name == "Base":
        candidate = candidate.parent
    if not (candidate / "Base").is_dir():
        raise FileNotFoundError(
            "Bundled materials require the NVIDIA Materials root "
            f"(containing Base): {candidate}"
        )
    return candidate


def _project_owned_file(project_dir: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Bundled project {label} is invalid")
    try:
        path = (project_dir / relative).resolve(strict=True)
        path.relative_to(project_dir.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Bundled project {label} escapes its project directory"
        ) from exc
    if not path.is_file():
        raise ValueError(f"Bundled project {label} is not a file: {path}")
    return path


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Bundled project {label} must be a lowercase SHA-256")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read bundled project {label}: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Bundled project {label} must be a JSON object")
    return document


def _acceptance_evidence_file(
    *,
    project_dir: Path,
    record: Any,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(
            f"Bundled project {label} must contain exactly path and sha256"
        )
    path = _project_owned_file(project_dir, record.get("path"), label)
    expected_sha256 = _sha256_text(record.get("sha256"), f"{label} sha256")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Bundled project {label} hash changed")
    return path, expected_sha256


def _audit_has_trusted_palette_sample(document: Mapping[str, Any]) -> bool:
    groups = document.get("groups")
    if not isinstance(groups, list):
        return False
    for group in groups:
        if not isinstance(group, dict) or group.get("accepted") is not True:
            continue
        base_color = group.get("base_color")
        boxes = group.get("boxes")
        if not isinstance(base_color, str) or not base_color or not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict) or box.get("accepted") is not True:
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
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= 255
                    for value in representative
                )
                and isinstance(matching_pixels, int)
                and not isinstance(matching_pixels, bool)
                and matching_pixels >= 64
                and isinstance(normalized_box, list)
                and len(normalized_box) == 4
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 <= float(value) <= 1000
                    for value in normalized_box
                )
                and float(normalized_box[0]) < float(normalized_box[2])
                and float(normalized_box[1]) < float(normalized_box[3])
            ):
                return True
    return False


def validate_bundled_acceptance_evidence(
    project: Mapping[str, Any] | dict[str, Any],
    *,
    project_file: Path,
    reference_paths_by_role: Mapping[str, Path] | None = None,
) -> dict[str, Any] | None:
    """Validate trusted palette/mask evidence owned by one sealed project."""

    descriptor = project.get("acceptance_evidence")
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict) or set(descriptor) != {"manifest", "sha256"}:
        raise ValueError(
            "Bundled project acceptance_evidence must contain exactly "
            "manifest and sha256"
        )
    project_dir = project_file.parent.resolve(strict=True)
    manifest = _project_owned_file(
        project_dir,
        descriptor.get("manifest"),
        "acceptance evidence manifest",
    )
    manifest_sha256 = _sha256_text(
        descriptor.get("sha256"),
        "acceptance evidence manifest sha256",
    )
    if sha256_file(manifest) != manifest_sha256:
        raise ValueError("Bundled project acceptance evidence manifest hash changed")
    document = _read_json_object(manifest, "acceptance evidence manifest")
    if set(document) != {"schema_version", "source_views"} or document.get(
        "schema_version"
    ) != "qwen-bundled-acceptance-evidence/v1":
        raise ValueError("Bundled project acceptance evidence schema is invalid")

    references = project.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("Bundled project references are invalid")
    expected_roles = [
        reference.get("role") if isinstance(reference, dict) else None
        for reference in references
    ]
    expected_hashes = {
        reference.get("role"): reference.get("sha256")
        for reference in references
        if isinstance(reference, dict)
    }
    source_views = document.get("source_views")
    if not isinstance(source_views, list) or len(source_views) != len(expected_roles):
        raise ValueError(
            "Bundled project acceptance evidence must cover every reference"
        )

    normalized_views: list[dict[str, Any]] = []
    actual_roles: list[str] = []
    for index, raw_view in enumerate(source_views):
        expected_view_keys = {
            "id",
            "reference_sha256",
            "palette_status",
            "palette_mask",
            "palette_path",
            "palette_artifacts",
        }
        if not isinstance(raw_view, dict) or set(raw_view) != expected_view_keys:
            raise ValueError(
                f"Bundled acceptance evidence source_views[{index}] has invalid fields"
            )
        role = raw_view.get("id")
        if not isinstance(role, str) or not role:
            raise ValueError("Bundled acceptance evidence view id is invalid")
        actual_roles.append(role)
        reference_sha256 = _sha256_text(
            raw_view.get("reference_sha256"),
            f"acceptance evidence {role} reference_sha256",
        )
        if reference_sha256 != expected_hashes.get(role):
            raise ValueError(
                f"Bundled acceptance evidence {role} reference hash changed"
            )
        if raw_view.get("palette_status") != "usable":
            raise ValueError(
                f"Bundled acceptance evidence {role} palette is not usable"
            )
        mask, mask_sha256 = _acceptance_evidence_file(
            project_dir=project_dir,
            record=raw_view.get("palette_mask"),
            label=f"acceptance evidence {role} palette_mask",
        )
        palette, palette_sha256 = _acceptance_evidence_file(
            project_dir=project_dir,
            record=raw_view.get("palette_path"),
            label=f"acceptance evidence {role} palette_path",
        )
        raw_artifacts = raw_view.get("palette_artifacts")
        if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != {
            "normalized",
            "normalized_evidence_audit",
        }:
            raise ValueError(
                f"Bundled acceptance evidence {role} palette_artifacts are invalid"
            )
        normalized_palette, normalized_palette_sha256 = _acceptance_evidence_file(
            project_dir=project_dir,
            record=raw_artifacts.get("normalized"),
            label=f"acceptance evidence {role} normalized palette",
        )
        audit, audit_sha256 = _acceptance_evidence_file(
            project_dir=project_dir,
            record=raw_artifacts.get("normalized_evidence_audit"),
            label=f"acceptance evidence {role} normalized evidence audit",
        )
        if palette != normalized_palette or palette_sha256 != normalized_palette_sha256:
            raise ValueError(
                f"Bundled acceptance evidence {role} palette_path differs from normalized"
            )
        palette_document = _read_json_object(palette, f"{role} normalized palette")
        if (
            palette_document.get("schema_version") != "qwen-material-palette/v1"
            or palette_document.get("source_view_id") != role
            or not isinstance(palette_document.get("groups"), list)
        ):
            raise ValueError(
                f"Bundled acceptance evidence {role} palette schema is invalid"
            )
        audit_document = _read_json_object(audit, f"{role} evidence audit")
        if not _audit_has_trusted_palette_sample(audit_document):
            raise ValueError(
                f"Bundled acceptance evidence {role} has no trusted palette samples"
            )

        if reference_paths_by_role is not None:
            reference_path = reference_paths_by_role.get(role)
            if reference_path is None:
                raise ValueError(
                    f"Bundled acceptance evidence has no runtime reference for {role}"
                )
            try:
                from PIL import Image, ImageOps

                with Image.open(reference_path) as reference_image:
                    reference_size = ImageOps.exif_transpose(reference_image).size
                with Image.open(mask) as mask_image:
                    mask_image = ImageOps.exif_transpose(mask_image)
                    if mask_image.size != reference_size:
                        raise ValueError(
                            f"Bundled acceptance evidence {role} mask size changed"
                        )
                    plane = (
                        mask_image.getchannel("A")
                        if "A" in mask_image.getbands()
                        and mask_image.getchannel("A").getextrema() != (255, 255)
                        else mask_image.convert("L")
                    )
                    histogram = plane.histogram()
                    white_pixels = histogram[255]
                    coverage = white_pixels / max(1, plane.width * plane.height)
                    if plane.getbbox() is None or not 0.015 <= coverage <= 0.80:
                        raise ValueError(
                            f"Bundled acceptance evidence {role} mask is not scoreable"
                        )
            except OSError as exc:
                raise ValueError(
                    f"Bundled acceptance evidence {role} mask cannot be decoded"
                ) from exc

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
                    "normalized": str(normalized_palette),
                    "normalized_sha256": normalized_palette_sha256,
                    "normalized_evidence_audit": str(audit),
                    "normalized_evidence_audit_sha256": audit_sha256,
                },
            }
        )
    if actual_roles != expected_roles:
        raise ValueError(
            "Bundled acceptance evidence roles/order differ from project references"
        )
    return {
        "schema_version": "qwen-bundled-acceptance-evidence/v1",
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "source_views": normalized_views,
    }


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an integer >= 0")
    return value


def validate_bundled_acceptance_contract(
    project: Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    """Return one strict, project-owned final visual acceptance contract."""

    acceptance = project.get("acceptance")
    expected_acceptance_keys = {
        "render",
        "view_mapping",
        "minimum_comparable_views",
    }
    if not isinstance(acceptance, dict) or set(acceptance) != expected_acceptance_keys:
        raise ValueError(
            "Bundled project acceptance contract must contain exactly "
            f"{sorted(expected_acceptance_keys)}"
        )
    render = acceptance.get("render")
    expected_render_keys = {
        "resolution",
        "views",
        "rt_subframes",
        "lighting_profile",
        "analysis_up_axis",
        "analysis_front_axis",
    }
    if not isinstance(render, dict) or set(render) != expected_render_keys:
        raise ValueError(
            "Bundled project acceptance render contract must contain exactly "
            f"{sorted(expected_render_keys)}"
        )
    resolution = render.get("resolution")
    rt_subframes = render.get("rt_subframes")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution <= 0
    ):
        raise ValueError("Bundled project acceptance resolution must be positive")
    if (
        isinstance(rt_subframes, bool)
        or not isinstance(rt_subframes, int)
        or rt_subframes <= 0
    ):
        raise ValueError("Bundled project acceptance rt_subframes must be positive")
    raw_views = render.get("views")
    if not isinstance(raw_views, str) or not raw_views:
        raise ValueError("Bundled project acceptance views must be non-empty")
    views = raw_views.split(",")
    if (
        any(not view or view.strip() != view for view in views)
        or len(set(views)) != len(views)
    ):
        raise ValueError(
            "Bundled project acceptance views must be unique comma-separated tokens"
        )
    if any(view not in _SUPPORTED_EXPLICIT_RENDER_VIEWS for view in views):
        raise ValueError(
            "Bundled project acceptance views must be explicit supported poses, "
            "not presets"
        )
    if render.get("lighting_profile") not in {"geometry", "material-neutral"}:
        raise ValueError("Bundled project acceptance lighting_profile is unsupported")
    supported_axes = {"x", "-x", "y", "-y", "z", "-z"}
    up_axis = render.get("analysis_up_axis")
    front_axis = render.get("analysis_front_axis")
    if up_axis not in supported_axes or front_axis not in supported_axes:
        raise ValueError("Bundled project acceptance analysis axis is unsupported")
    if str(up_axis).lstrip("-") == str(front_axis).lstrip("-"):
        raise ValueError("Bundled project acceptance axes must not be parallel")

    references = project.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("Bundled project references are invalid")
    reference_roles: list[str] = []
    for index, reference in enumerate(references):
        role = reference.get("role") if isinstance(reference, dict) else None
        if not isinstance(role, str) or not role or role in reference_roles:
            raise ValueError(
                f"Bundled project reference role[{index}] is invalid or duplicated"
            )
        reference_roles.append(role)

    raw_mapping = acceptance.get("view_mapping")
    if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(reference_roles):
        raise ValueError(
            "Bundled project acceptance mapping must cover every reference role exactly"
        )
    mapping: dict[str, str] = {}
    for reference_role, render_view in raw_mapping.items():
        if not isinstance(render_view, str) or render_view not in views:
            raise ValueError(
                "Bundled project acceptance mapping targets an undeclared render view"
            )
        mapping[reference_role] = render_view
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "Bundled project acceptance mapping must be one-to-one"
        )
    if set(mapping.values()) != set(views):
        raise ValueError(
            "Bundled project acceptance mapping must exactly cover render views"
        )
    minimum = acceptance.get("minimum_comparable_views")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum != len(mapping)
    ):
        raise ValueError(
            "Bundled project acceptance minimum_comparable_views must equal "
            "the exact mapping size"
        )
    return {
        "render": {
            "resolution": resolution,
            "views": raw_views,
            "rt_subframes": rt_subframes,
            "lighting_profile": render["lighting_profile"],
            "analysis_up_axis": up_axis,
            "analysis_front_axis": front_axis,
        },
        "view_mapping": dict(sorted(mapping.items())),
        "minimum_comparable_views": minimum,
    }


def _registry_topology(
    registry: dict[str, Any],
    *,
    label: str,
) -> dict[str, int | str]:
    """Return the path-bound occurrence topology of one registry.

    Aggregate counts alone do not distinguish assemblies whose Mesh paths have
    changed.  The digest deliberately excludes mutable render/material fields
    and binds each occurrence path to its point/face counts.
    """

    parts = registry.get("parts")
    if not isinstance(parts, list):
        raise ValueError(f"{label} parts are missing")
    part_count = _exact_nonnegative_int(
        registry.get("part_count"),
        f"{label}.part_count",
    )
    if part_count != len(parts):
        raise ValueError(f"{label}.part_count is inconsistent")

    identities: list[dict[str, int | str]] = []
    paths: set[str] = set()
    for index, raw_part in enumerate(parts):
        if not isinstance(raw_part, dict):
            raise ValueError(f"{label}.parts[{index}] must be an object")
        prim_path = raw_part.get("prim_path")
        if (
            not isinstance(prim_path, str)
            or not prim_path.startswith("/")
            or prim_path in paths
        ):
            raise ValueError(
                f"{label}.parts[{index}].prim_path is invalid or duplicated"
            )
        paths.add(prim_path)
        identities.append(
            {
                "prim_path": prim_path,
                "point_count": _exact_nonnegative_int(
                    raw_part.get("point_count"),
                    f"{label}.parts[{index}].point_count",
                ),
                "face_count": _exact_nonnegative_int(
                    raw_part.get("face_count"),
                    f"{label}.parts[{index}].face_count",
                ),
            }
        )
    identities.sort(key=lambda item: str(item["prim_path"]))
    return {
        "mesh_occurrences": part_count,
        "point_occurrence_count": sum(
            int(item["point_count"]) for item in identities
        ),
        "face_occurrence_count": sum(
            int(item["face_count"]) for item in identities
        ),
        "occurrence_path_topology_sha256": _canonical_sha256(identities),
    }


def _match_source_representation(
    *,
    expected: dict[str, Any],
    source_registry: dict[str, Any],
    occurrence_topology: dict[str, int | str],
) -> tuple[str, str]:
    """Match one explicitly sealed source-registry representation.

    A manual/physics workflow may already have expanded CAD instances.  That is
    not a topology exception: an ``occurrence_equivalent`` contract requires
    the source registry to have the exact same path-bound topology as the
    sealed occurrence registry.
    """

    contracts = expected.get("source_registry_contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError(
            "Bundled project source_registry_contracts are invalid"
        )
    actual_instance_roots = _exact_nonnegative_int(
        source_registry.get("instance_root_count"),
        "source registry instance_root_count",
    )
    seen_ids: set[str] = set()
    seen_instance_counts: set[int] = set()
    matching_contract: tuple[str, str] | None = None
    matched_count_but_changed = False
    for index, raw_contract in enumerate(contracts):
        if not isinstance(raw_contract, dict):
            raise ValueError(
                f"source_registry_contracts[{index}] must be an object"
            )
        representation_id = raw_contract.get("representation_id")
        topology_role = raw_contract.get("topology_role")
        expected_instance_roots = _exact_nonnegative_int(
            raw_contract.get("instance_root_count"),
            (
                f"source_registry_contracts[{index}]"
                ".instance_root_count"
            ),
        )
        if (
            not isinstance(representation_id, str)
            or not representation_id
            or representation_id in seen_ids
        ):
            raise ValueError(
                "Bundled project source representation IDs are invalid "
                "or duplicated"
            )
        if expected_instance_roots in seen_instance_counts:
            raise ValueError(
                "Bundled project source instance-root contracts are ambiguous"
            )
        if topology_role not in _SOURCE_TOPOLOGY_ROLES:
            raise ValueError(
                "Bundled project source topology role is unsupported: "
                f"{topology_role!r}"
            )
        if topology_role == "pre_expansion" and expected_instance_roots == 0:
            raise ValueError(
                "A pre_expansion source contract must contain instance roots"
            )
        if (
            topology_role == "occurrence_equivalent"
            and expected_instance_roots != 0
        ):
            raise ValueError(
                "An occurrence_equivalent source contract must be deinstanced"
            )
        seen_ids.add(representation_id)
        seen_instance_counts.add(expected_instance_roots)

        if actual_instance_roots != expected_instance_roots:
            continue
        if topology_role == "occurrence_equivalent":
            source_topology = _registry_topology(
                source_registry,
                label="source registry",
            )
            if source_topology != occurrence_topology:
                matched_count_but_changed = True
                continue
        matching_contract = (representation_id, topology_role)

    if matching_contract is None:
        detail = (
            "source registry is not occurrence-topology-equivalent"
            if matched_count_but_changed
            else (
                "source instance_root_count is not an accepted sealed "
                f"representation: {actual_instance_roots}"
            )
        )
        raise ValueError(f"Bundled project source topology changed: {detail}")
    return matching_contract


def match_bundled_project(
    *,
    source_cad: Path | None,
    references: Sequence[tuple[str, Path]],
    source_registry: dict[str, Any],
    occurrence_registry: dict[str, Any],
    configured_material_root: Path,
    isaac_root: Path | None = None,
    projects_root: Path | None = None,
) -> BundledMaterialProject | None:
    """Return an exact project match, or ``None`` for the generic workflow.

    All identities must match before a project is selected.  A different CAD
    revision or photograph set falls through to the generic inference path.
    Once the immutable inputs match, a topology mismatch fails closed.
    """

    if source_cad is None:
        return None
    if projects_root is not None:
        root = projects_root.expanduser().resolve(strict=True)
    else:
        configured_root = Path(
            os.environ.get(
                "MATERIAL_PROJECTS_ROOT",
                ProjectLayout.from_root(root_dir()).material_projects,
            )
        ).expanduser()
        if not configured_root.is_dir():
            return None
        root = configured_root.resolve(strict=True)
    for project_file in sorted(root.glob("*/project.json")):
        project = _read_project(project_file)
        source_identity = project.get("source_cad")
        if not isinstance(source_identity, dict):
            raise ValueError(
                f"Bundled project source identity is invalid: {project_file}"
            )
        if sha256_file(source_cad) != source_identity.get("sha256"):
            continue

        expected_references = project.get("references")
        if not isinstance(expected_references, list) or not expected_references:
            raise ValueError("Bundled project reference identities are invalid")
        acceptance = validate_bundled_acceptance_contract(project)
        expected_hashes: set[str] = set()
        for index, item in enumerate(expected_references):
            digest = item.get("sha256") if isinstance(item, dict) else None
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or digest in expected_hashes
            ):
                raise ValueError(
                    "Bundled project reference SHA-256 values are malformed "
                    f"or duplicated at index {index}"
                )
            expected_hashes.add(digest)
        actual_hashes = {sha256_file(path) for _reference_id, path in references}
        if actual_hashes != expected_hashes:
            continue

        expected = project.get("expected_assembly")
        if not isinstance(expected, dict):
            raise ValueError("Bundled project or occurrence registry is incomplete")
        try:
            metrics = _registry_topology(
                occurrence_registry,
                label="occurrence registry",
            )
        except ValueError as exc:
            raise ValueError(
                f"Bundled project {project['asset_id']!r} topology changed: {exc}"
            ) from exc
        mismatches = {
            key: (metrics.get(key), expected.get(key))
            for key in metrics
            if metrics.get(key) != expected.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Bundled project {project['asset_id']!r} topology changed: "
                f"{mismatches}"
            )
        source_representation_id, source_registry_topology_role = (
            _match_source_representation(
                expected=expected,
                source_registry=source_registry,
                occurrence_topology=metrics,
            )
        )

        project_dir = project_file.parent
        planner_module = project.get("planner_module")
        if not isinstance(planner_module, str) or not planner_module:
            raise ValueError("Bundled project planner_module is invalid")
        template = _project_owned_file(
            project_dir,
            project.get("template"),
            "template",
        )
        catalog = _project_owned_file(
            project_dir,
            project.get("catalog"),
            "catalog",
        )
        dependency_lock = _project_owned_file(
            project_dir,
            project.get("dependency_lock"),
            "dependency_lock",
        )
        if sha256_file(template) != project.get("template_sha256"):
            raise ValueError(
                f"Bundled project {project['asset_id']!r} template hash changed"
            )
        if sha256_file(catalog) != project.get("catalog_sha256"):
            raise ValueError(
                f"Bundled project {project['asset_id']!r} catalog hash changed"
            )
        material_root = (
            _nvidia_materials_root(configured_material_root)
            if project.get("material_root_scope") == "nvidia_materials"
            else configured_material_root.expanduser().resolve(strict=True)
        )
        render = project.get("render")
        if not isinstance(render, dict):
            raise ValueError("Bundled project render contract is invalid")
        expected_reference_hashes_by_role = {
            item["role"]: item.get("sha256")
            for item in expected_references
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        }
        actual_reference_hashes_by_id = {
            reference_id: sha256_file(path)
            for reference_id, path in references
        }
        if (
            len(actual_reference_hashes_by_id) != len(references)
            or actual_reference_hashes_by_id != expected_reference_hashes_by_role
            or set(actual_reference_hashes_by_id)
            != set(acceptance["view_mapping"])
        ):
            raise ValueError(
                "Bundled project input references do not match the "
                "hash-bound acceptance roles and hashes"
            )
        acceptance_evidence = validate_bundled_acceptance_evidence(
            project,
            project_file=project_file,
            reference_paths_by_role=dict(references),
        )
        effective_isaac_root = (
            isaac_root.expanduser().resolve(strict=True)
            if isaac_root is not None
            else isaac_python().expanduser().resolve(strict=True).parent
        )
        dependency_lock_verification = verify_sealed_dependency_lock(
            lock_path=dependency_lock,
            expected_lock_sha256=project.get("dependency_lock_sha256"),
            catalog_path=catalog,
            material_root=material_root,
            isaac_root=effective_isaac_root,
            expected_asset_id=str(project["asset_id"]),
        )
        return BundledMaterialProject(
            asset_id=str(project["asset_id"]),
            project_file=project_file.resolve(strict=True),
            planner_module=planner_module,
            template=template,
            catalog=catalog,
            dependency_lock=dependency_lock,
            dependency_lock_verification=dependency_lock_verification,
            material_root=material_root,
            render=render,
            acceptance=acceptance,
            acceptance_evidence=acceptance_evidence,
            document=project,
            source_representation_id=source_representation_id,
            source_registry_topology_role=source_registry_topology_role,
        )
    return None


__all__ = [
    "BundledMaterialProject",
    "match_bundled_project",
    "validate_bundled_acceptance_contract",
    "validate_bundled_acceptance_evidence",
]
