"""Actual-CAD scoring helpers for immutable appearance-component MDL choices.

Retrieval-bank renders are useful for narrowing the NVIDIA Base catalog, but
they cannot predict how a translucent or reflective MDL will look on the
actual CAD geometry under the registered camera and lighting.  This module
keeps the final decision evidence-bounded: it creates a candidate plan that
changes one appearance component's *MDL identity only*, and scores its member
Part-ID cores after a real CAD render.  A separate helper may build one
same-identity color-only H1 after identity selection, but only through a
reviewed color interface on the already selected material.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .part_id_parameter_tournament import (
    PartIdParameterTournamentError,
    score_part_id_render,
)
from .semantics import (
    FINISH_CLASSES,
    MaterialSemanticsError,
    catalog_matches_part_semantics,
    normalize_catalog_surface_semantics,
    normalize_part_material_semantics,
)
from .tuning import (
    color_parameters_for_target_srgb,
    parameter_policy_for_material,
    tuning_profile_for_material,
)


SCHEMA_VERSION = "qwen-appearance-component-actual-mdl-tournament/v1"
COLOR_SCHEMA_VERSION = "qwen-appearance-component-color-tournament/v1"
COLOR_ARTIFACT_BINDING_SCHEMA_VERSION = (
    "qwen-appearance-component-color-render-artifact-binding/v1"
)
COLOR_MINIMUM_SCORE_IMPROVEMENT = 0.015
COLOR_MAXIMUM_MEMBER_REGRESSION = 0.03
# Kept equal to the Part-ID projection authoring floor.  Rebinding must verify
# the production contract, not trust a floor copied into an input plan.
MINIMUM_APPLYABLE_REVIEW_CONFIDENCE = 0.60


class ComponentMdlTournamentError(ValueError):
    """Raised when a component candidate would violate immutable-MDL rules."""


@dataclass(frozen=True)
class ComponentColorRenderArtifactPaths:
    """Paths that prove one color candidate was applied and rendered."""

    plan: str | Path
    apply_plan: str | Path
    apply_report: str | Path
    look_usd: str | Path
    rendered_registry: str | Path


@dataclass(frozen=True)
class ComponentColorScoreEvidence:
    """Caller-owned capability used to replay scores from immutable artifacts."""

    evidence: Mapping[str, Any]
    spatial_mapping_report: Mapping[str, Any]
    artifact_root: str | Path
    h0_artifact: ComponentColorRenderArtifactPaths
    h1_artifacts_by_component: Mapping[str, ComponentColorRenderArtifactPaths]


@dataclass(frozen=True)
class _VerifiedComponentColorScoreEvidence:
    evidence: Mapping[str, Any]
    spatial_mapping_report: Mapping[str, Any]
    h0_rendered_registry: Mapping[str, Any]
    h1_rendered_registries_by_component: Mapping[str, Mapping[str, Any]]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _artifact_root(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ComponentMdlTournamentError("component color artifact root is invalid")
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ComponentMdlTournamentError(
            "component color artifact root cannot be a symlink"
        )
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ComponentMdlTournamentError(
            "component color artifact root cannot be resolved"
        ) from exc
    if not resolved.is_dir():
        raise ComponentMdlTournamentError(
            "component color artifact root is not a directory"
        )
    return resolved


def _read_component_color_artifact_file(
    *,
    artifact_root: Path,
    value: str | Path,
    label: str,
    json_object: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read one root-contained regular file once through an O_NOFOLLOW fd."""

    if not isinstance(value, (str, Path)) or not str(value):
        raise ComponentMdlTournamentError(f"{label} path is invalid")
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ComponentMdlTournamentError(f"{label} cannot be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ComponentMdlTournamentError(f"{label} cannot be resolved") from exc
    if not resolved.is_relative_to(artifact_root):
        raise ComponentMdlTournamentError(
            f"{label} is outside the component color artifact root"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ComponentMdlTournamentError(
            f"{label} could not be opened as a trusted artifact"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ComponentMdlTournamentError(
                f"{label} must be a single-link regular file"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ComponentMdlTournamentError(
            f"{label} changed while it was being read"
        ) from exc
    expected_identity = _stable_file_identity(before)
    if (
        _stable_file_identity(after) != expected_identity
        or _stable_file_identity(path_after) != expected_identity
    ):
        raise ComponentMdlTournamentError(
            f"{label} changed while it was being read"
        )

    file_binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
    }
    if not json_object:
        return file_binding, None
    try:
        document = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentMdlTournamentError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(document, dict):
        raise ComponentMdlTournamentError(f"{label} must contain a JSON object")
    try:
        file_binding["canonical_sha256"] = _canonical_sha256(document)
    except (TypeError, ValueError) as exc:
        raise ComponentMdlTournamentError(
            f"{label} cannot be canonically sealed"
        ) from exc
    return file_binding, document


def _reported_artifact_path(
    value: Any,
    *,
    expected: Path,
    label: str,
) -> None:
    if not isinstance(value, str) or not value:
        raise ComponentMdlTournamentError(f"{label} path is invalid")
    try:
        reported = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ComponentMdlTournamentError(f"{label} path cannot be resolved") from exc
    if reported != expected:
        raise ComponentMdlTournamentError(
            f"{label} does not identify the exact component color Look"
        )


def _load_component_color_render_artifact_binding(
    *,
    artifact_root: str | Path,
    artifact: ComponentColorRenderArtifactPaths,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(artifact, ComponentColorRenderArtifactPaths):
        raise ComponentMdlTournamentError(
            "component color render artifact capability is invalid"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256):
        raise ComponentMdlTournamentError(
            "component color artifact expected plan hash is invalid"
        )
    root = _artifact_root(artifact_root)
    files: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name, value in (
        ("plan", artifact.plan),
        ("apply_plan", artifact.apply_plan),
        ("apply_report", artifact.apply_report),
        ("rendered_registry", artifact.rendered_registry),
    ):
        binding, document = _read_component_color_artifact_file(
            artifact_root=root,
            value=value,
            label=f"component color {name}",
            json_object=True,
        )
        assert document is not None
        files[name] = binding
        documents[name] = document
    look_binding, _ = _read_component_color_artifact_file(
        artifact_root=root,
        value=artifact.look_usd,
        label="component color look_usd",
        json_object=False,
    )
    files["look_usd"] = look_binding

    plan = documents["plan"]
    apply_plan = documents["apply_plan"]
    apply_report = documents["apply_report"]
    rendered_registry = documents["rendered_registry"]
    if _canonical_sha256(plan) != expected_plan_sha256:
        raise ComponentMdlTournamentError(
            "component color artifact plan hash does not match its candidate"
        )
    if plan.get("assignments") != apply_plan.get("assignments"):
        raise ComponentMdlTournamentError(
            "component color artifact apply-plan assignments differ from its plan"
        )
    apply_plan_sha256 = _canonical_sha256(apply_plan)
    if apply_report.get("plan_sha256") != apply_plan_sha256:
        raise ComponentMdlTournamentError(
            "component color apply report is not bound to its apply plan"
        )
    look_path = Path(look_binding["path"])
    _reported_artifact_path(
        apply_report.get("output_usd"),
        expected=look_path,
        label="component color apply report output",
    )
    _reported_artifact_path(
        rendered_registry.get("asset_usd"),
        expected=look_path,
        label="component color rendered registry asset",
    )
    if rendered_registry.get("asset_sha256") != look_binding["sha256"]:
        raise ComponentMdlTournamentError(
            "component color rendered registry is not hash-bound to its Look"
        )

    binding = {
        "schema_version": COLOR_ARTIFACT_BINDING_SCHEMA_VERSION,
        "artifact_root": str(root),
        "expected_plan_sha256": expected_plan_sha256,
        "files": files,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding, rendered_registry


def build_component_color_render_artifact_binding(
    *,
    artifact_root: str | Path,
    artifact: ComponentColorRenderArtifactPaths,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Build the audit binding for one safely read, internally bound render."""

    binding, _rendered_registry = _load_component_color_render_artifact_binding(
        artifact_root=artifact_root,
        artifact=artifact,
        expected_plan_sha256=expected_plan_sha256,
    )
    return binding


def _assignments(
    plan: Mapping[str, Any],
    *,
    allow_parameter_overrides: bool = False,
) -> dict[str, dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise ComponentMdlTournamentError("material plan has an invalid schema")
    raw = plan.get("assignments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ComponentMdlTournamentError("material plan assignments are invalid")
    output: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ComponentMdlTournamentError(
                f"material plan assignment {index} is invalid"
            )
        # ``output`` is already a deep copy. Keep its native dict object so a
        # validated candidate mutation is reflected in the serialized plan
        # rather than only in this temporary lookup table.
        row = value if isinstance(value, dict) else dict(value)
        part_id = row.get("part_id")
        material_id = row.get("material_id")
        parameters = row.get("parameters")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
            or (parameters is not None and not isinstance(parameters, Mapping))
            or (
                not allow_parameter_overrides
                and isinstance(parameters, Mapping)
                and bool(parameters)
            )
        ):
            raise ComponentMdlTournamentError(
                f"material plan assignment {index} is not an immutable MDL binding"
            )
        output[part_id] = row
    if not output:
        raise ComponentMdlTournamentError("material plan has no assignments")
    return output


def _member_ids(member_part_ids: Sequence[str]) -> list[str]:
    if isinstance(member_part_ids, (str, bytes)):
        raise ComponentMdlTournamentError("component member Part-IDs are invalid")
    members = sorted(member_part_ids)
    if (
        len(members) < 2
        or len(members) != len(set(members))
        or any(not isinstance(part_id, str) or not part_id for part_id in members)
    ):
        raise ComponentMdlTournamentError(
            "component needs at least two unique non-empty Part-IDs"
        )
    return members


_METAL_SURFACE_TREATMENTS = frozenset(
    {
        "anodized",
        "bare",
        "conversion_coating",
        "galvanized",
        "oxidized",
        "plated",
    }
)
_FALLBACK_FINISH_ORDER = {
    "matte": 0,
    "satin": 1,
    "glossy": 2,
    "polished": 3,
    "brushed": 4,
    "smooth": 5,
    "rough": 6,
    "textured": 7,
    "weathered": 8,
    "unknown": 9,
}


def _semantic_gate_requested(
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    requested = (
        member_material_semantics is not None
        or catalog_materials_by_id is not None
    )
    if requested and (
        member_material_semantics is None or catalog_materials_by_id is None
    ):
        raise ComponentMdlTournamentError(
            "strict component semantics require both member semantics and the catalog"
        )
    return requested


def _strict_member_semantics(
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    *,
    expected_member_part_ids: Sequence[str] | None = None,
    preferred_finish: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    """Validate one resolved physical-material contract for a component."""

    if not isinstance(member_material_semantics, Mapping):
        raise ComponentMdlTournamentError(
            "component member material semantics must be a mapping"
        )
    if len(member_material_semantics) < 2:
        raise ComponentMdlTournamentError(
            "strict component semantics need at least two members"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for part_id, value in member_material_semantics.items():
        if not isinstance(part_id, str) or not part_id or part_id in normalized:
            raise ComponentMdlTournamentError(
                "component member material semantics have an invalid Part-ID"
            )
        try:
            semantics = normalize_part_material_semantics(value)
        except (MaterialSemanticsError, TypeError) as exc:
            raise ComponentMdlTournamentError(
                f"component member semantics are malformed for {part_id}: {exc}"
            ) from exc
        if (
            semantics["evidence_status"] != "observed"
            or semantics["substrate"] == "unknown"
            or semantics["surface_treatment"] == "unknown"
            or semantics["optical_behavior"] == "unknown"
        ):
            raise ComponentMdlTournamentError(
                f"component member semantics are not observed and resolved for {part_id}"
            )
        normalized[part_id] = semantics
    if expected_member_part_ids is not None:
        expected = set(_member_ids(expected_member_part_ids))
        if set(normalized) != expected:
            raise ComponentMdlTournamentError(
                "component member semantics do not exactly cover the candidate members"
            )

    treatments = {value["surface_treatment"] for value in normalized.values()}
    optical_behaviors = {value["optical_behavior"] for value in normalized.values()}
    known_finishes = {
        value["finish"]
        for value in normalized.values()
        if value["finish"] != "unknown"
    }
    if len(treatments) != 1 or len(optical_behaviors) != 1 or len(known_finishes) > 1:
        raise ComponentMdlTournamentError(
            "component member material semantics conflict"
        )
    treatment = next(iter(treatments))
    optical = next(iter(optical_behaviors))
    substrates = {value["substrate"] for value in normalized.values()}
    if len(substrates) != 1:
        raise ComponentMdlTournamentError(
            "component member material substrates conflict"
        )
    substrate = next(iter(substrates))
    if treatment == "paint":
        target_family = "paint"
    elif substrate == "metal" and treatment in _METAL_SURFACE_TREATMENTS:
        target_family = "metal"
    else:
        # The hierarchical catalog contract below is authoritative.  This
        # label is retained only for audit and reviewed H1 surface-class checks.
        target_family = substrate

    if preferred_finish is not None and (
        not isinstance(preferred_finish, str)
        or preferred_finish not in FINISH_CLASSES
        or preferred_finish == "unknown"
    ):
        raise ComponentMdlTournamentError(
            "preferred_finish must be a known catalog finish"
        )
    inferred_finish = next(iter(known_finishes), None)
    if (
        preferred_finish is not None
        and inferred_finish is not None
        and preferred_finish != inferred_finish
    ):
        raise ComponentMdlTournamentError(
            "preferred_finish conflicts with component member semantics"
        )
    return normalized, target_family, preferred_finish or inferred_finish


def _catalog_candidate_semantics(
    *,
    material_id: str,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]],
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    target_family: str,
) -> dict[str, Any] | None:
    """Return normalized metadata only for a fully compatible strict candidate."""

    if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
        return None
    raw_record = catalog_materials_by_id.get(material_id)
    if not isinstance(raw_record, Mapping):
        return None
    recorded_id = raw_record.get("material_id")
    if recorded_id is not None and recorded_id != material_id:
        return None
    family = raw_record.get("family")
    if not isinstance(family, str) or not family:
        return None
    raw_surface = raw_record.get("surface_semantics")
    if not isinstance(raw_surface, Mapping):
        return None
    try:
        surface = normalize_catalog_surface_semantics(raw_surface)
        if surface["confidence"] == "low":
            return None
        identifier_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", material_id.casefold())
            if token
        }
        # Defense in depth against a malformed catalog row claiming that a
        # visually convenient proxy belongs to an unrelated predicted class.
        # The checks are conditional, so a genuine liquid Water or glass
        # Mirror remains usable for those predicted material categories.
        if (
            "water" in identifier_tokens
            and (
                "liquid" not in surface["compatible_substrates"]
                or surface["optical_behavior"] != "transparent"
            )
        ) or (
            "mirror" in identifier_tokens
            and (
                "glass" not in surface["compatible_substrates"]
                or surface["optical_behavior"] != "opaque"
            )
        ) or (
            "grass" in identifier_tokens
            and surface["surface_treatment"] in {"paint", "powder_coat"}
        ):
            return None
        if not all(
            catalog_matches_part_semantics(surface, semantics)
            for semantics in member_material_semantics.values()
        ):
            return None
    except (MaterialSemanticsError, TypeError):
        return None
    return {
        "family": family.casefold(),
        "surface_semantics": surface,
    }


def _paint_alias_key(material_id: str, family: str) -> str:
    if family != "paint":
        return material_id.casefold()
    sub_identifier = material_id.rsplit("#", 1)[-1].casefold()
    if sub_identifier.endswith("_finish"):
        sub_identifier = sub_identifier[: -len("_finish")]
    return f"paint:{sub_identifier}"


def _strict_component_candidate_material_ids(
    *,
    baseline_material_id: str,
    retrieval_group: Mapping[str, Any],
    visual_compatibility: Mapping[str, Any] | None,
    maximum_candidates: int,
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]],
    preferred_finish: str | None,
) -> list[str]:
    normalized_members, target_family, target_finish = _strict_member_semantics(
        member_material_semantics,
        preferred_finish=preferred_finish,
    )
    if not isinstance(catalog_materials_by_id, Mapping):
        raise ComponentMdlTournamentError(
            "strict component material catalog must be a mapping"
        )
    limit = min(3, maximum_candidates)
    selected: list[str] = []
    selected_aliases: set[str] = set()

    def compatible_metadata(material_id: Any) -> dict[str, Any] | None:
        if not isinstance(material_id, str):
            return None
        return _catalog_candidate_semantics(
            material_id=material_id,
            catalog_materials_by_id=catalog_materials_by_id,
            member_material_semantics=normalized_members,
            target_family=target_family,
        )

    def add(material_id: Any) -> None:
        if len(selected) >= limit or not isinstance(material_id, str):
            return
        metadata = compatible_metadata(material_id)
        if metadata is None:
            return
        alias = _paint_alias_key(material_id, metadata["family"])
        if material_id not in selected and alias not in selected_aliases:
            selected.append(material_id)
            selected_aliases.add(alias)

    # The legacy baseline remains H0 only when it passes the same semantic gate
    # as every challenger. Otherwise the first safe ranked/fallback material is
    # the new effective H0 returned at index zero.
    add(baseline_material_id)

    raw_color_ranking = retrieval_group.get("color_ranking")
    if isinstance(raw_color_ranking, Sequence) and not isinstance(
        raw_color_ranking, (str, bytes)
    ):
        for row in sorted(
            (row for row in raw_color_ranking if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["rank"])
                if isinstance(row.get("rank"), int)
                and not isinstance(row.get("rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        ):
            add(row.get("material_id"))

    raw_shortlist = (
        visual_compatibility.get("shortlist")
        if isinstance(visual_compatibility, Mapping)
        else None
    )
    if isinstance(raw_shortlist, Sequence) and not isinstance(
        raw_shortlist, (str, bytes)
    ):
        for row in sorted(
            (row for row in raw_shortlist if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["compatibility_rank"])
                if isinstance(row.get("compatibility_rank"), int)
                and not isinstance(row.get("compatibility_rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        ):
            add(row.get("material_id"))

    raw_fused_ranking = retrieval_group.get("fused_ranking")
    if isinstance(raw_fused_ranking, Sequence) and not isinstance(
        raw_fused_ranking, (str, bytes)
    ):
        for row in raw_fused_ranking:
            if isinstance(row, Mapping):
                add(row.get("material_id"))

    def fallback_key(material_id: str) -> tuple[int, int, int, str]:
        metadata = compatible_metadata(material_id)
        if metadata is None:
            return (1, 1, 1_000_000, material_id)
        surface = metadata["surface_semantics"]
        finish = surface["finish"]
        return (
            0 if target_finish is not None and finish == target_finish else 1,
            0 if not material_id.rsplit("#", 1)[-1].casefold().endswith("_finish") else 1,
            _FALLBACK_FINISH_ORDER.get(finish, 1_000_000),
            material_id,
        )

    compatible_catalog_ids = [
        material_id
        for material_id in catalog_materials_by_id
        if isinstance(material_id, str) and compatible_metadata(material_id) is not None
    ]
    for material_id in sorted(compatible_catalog_ids, key=fallback_key):
        add(material_id)

    if len(selected) < 2:
        raise ComponentMdlTournamentError(
            "strict component semantics yielded fewer than two compatible MDL candidates"
        )
    return selected


def component_candidate_material_ids(
    *,
    baseline_material_id: str,
    retrieval_group: Mapping[str, Any],
    visual_compatibility: Mapping[str, Any] | None,
    maximum_candidates: int = 4,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    preferred_finish: str | None = None,
) -> list[str]:
    """Build a small, visually diverse fixed-MDL tournament shortlist.

    The retrieval-bank winner is always retained as the baseline.  The next
    two slots favour the independently computed colour ranking, since this is
    the cue that a library thumbnail often gets right even when its geometry
    or transmission is a poor proxy for the CAD part.  The final slot favours
    the compatibility gate's best different candidate.  The actual CAD render
    is the only final authority, so candidates need not already pass the
    pre-render physical-risk gate.
    """

    if not isinstance(baseline_material_id, str) or not baseline_material_id.startswith(
        "mdl:"
    ):
        raise ComponentMdlTournamentError("baseline_material_id must be an MDL ID")
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 2
    ):
        raise ComponentMdlTournamentError("maximum_candidates must be an integer >= 2")
    if not isinstance(retrieval_group, Mapping):
        raise ComponentMdlTournamentError("retrieval group is invalid")

    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        return _strict_component_candidate_material_ids(
            baseline_material_id=baseline_material_id,
            retrieval_group=retrieval_group,
            visual_compatibility=visual_compatibility,
            maximum_candidates=maximum_candidates,
            member_material_semantics=member_material_semantics,
            catalog_materials_by_id=catalog_materials_by_id,
            preferred_finish=preferred_finish,
        )
    if preferred_finish is not None:
        raise ComponentMdlTournamentError(
            "preferred_finish is available only with strict component semantics"
        )

    selected: list[str] = []

    def add(material_id: Any) -> None:
        if (
            isinstance(material_id, str)
            and material_id.startswith("mdl:")
            and material_id not in selected
            and len(selected) < maximum_candidates
        ):
            selected.append(material_id)

    add(baseline_material_id)
    raw_color_ranking = retrieval_group.get("color_ranking")
    if isinstance(raw_color_ranking, Sequence) and not isinstance(
        raw_color_ranking, (str, bytes)
    ):
        ranked_color_rows = sorted(
            (row for row in raw_color_ranking if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["rank"])
                if isinstance(row.get("rank"), int)
                and not isinstance(row.get("rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        )
        for row in ranked_color_rows[:2]:
            add(row.get("material_id"))

    raw_shortlist = (
        visual_compatibility.get("shortlist")
        if isinstance(visual_compatibility, Mapping)
        else None
    )
    if isinstance(raw_shortlist, Sequence) and not isinstance(
        raw_shortlist, (str, bytes)
    ):
        ranked_compatibility_rows = sorted(
            (row for row in raw_shortlist if isinstance(row, Mapping)),
            key=lambda row: (
                int(row["compatibility_rank"])
                if isinstance(row.get("compatibility_rank"), int)
                and not isinstance(row.get("compatibility_rank"), bool)
                else 1_000_000,
                str(row.get("material_id", "")),
            ),
        )
        for row in ranked_compatibility_rows:
            add(row.get("material_id"))
            if len(selected) >= maximum_candidates:
                break

    if len(selected) < 2:
        raw_fused_ranking = retrieval_group.get("fused_ranking")
        if isinstance(raw_fused_ranking, Sequence) and not isinstance(
            raw_fused_ranking, (str, bytes)
        ):
            for row in raw_fused_ranking:
                if isinstance(row, Mapping):
                    add(row.get("material_id"))
                if len(selected) >= maximum_candidates:
                    break
    if len(selected) < 2:
        raise ComponentMdlTournamentError(
            "component retrieval did not yield a second fixed-MDL candidate"
        )
    return selected


def build_component_candidate_plan(
    *,
    source_plan: Mapping[str, Any],
    component_id: str,
    member_part_ids: Sequence[str],
    material_id: str,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a candidate that changes one component to one fixed Base MDL."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
        raise ComponentMdlTournamentError("candidate material_id must be an MDL ID")
    members = _member_ids(member_part_ids)
    semantic_gate: dict[str, Any] | None = None
    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        normalized_members, target_family, _target_finish = _strict_member_semantics(
            member_material_semantics,
            expected_member_part_ids=members,
        )
        if (
            _catalog_candidate_semantics(
                material_id=material_id,
                catalog_materials_by_id=catalog_materials_by_id,
                member_material_semantics=normalized_members,
                target_family=target_family,
            )
            is None
        ):
            raise ComponentMdlTournamentError(
                "component candidate MDL is not compatible with every member"
            )
        semantic_gate = {
            "policy": "all_component_members_physical_semantics_compatible/v1",
            "member_material_semantics": copy.deepcopy(normalized_members),
            "member_material_semantics_sha256": _canonical_sha256(
                normalized_members
            ),
            "catalog_material_record": copy.deepcopy(
                dict(catalog_materials_by_id[material_id])
            ),
            "catalog_material_record_sha256": _canonical_sha256(
                catalog_materials_by_id[material_id]
            ),
            "target_family": target_family,
        }
    output = copy.deepcopy(dict(source_plan))
    assignments = _assignments(output)
    missing = sorted(set(members) - set(assignments))
    if missing:
        raise ComponentMdlTournamentError(
            f"component {component_id} has unknown Part-IDs: {missing}"
        )
    for part_id in members:
        assignment = assignments[part_id]
        assignment["material_id"] = material_id
        assignment.pop("parameters", None)
        provenance = assignment.get("provenance")
        updated_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
        updated_provenance["appearance_component_actual_mdl_candidate"] = {
            "component_id": component_id,
            "member_part_ids": members,
            "material_id": material_id,
            "source_plan_sha256": _canonical_sha256(source_plan),
            "mdl_parameter_mutation_allowed": False,
        }
        if semantic_gate is not None:
            updated_provenance["appearance_component_actual_mdl_candidate"][
                "semantic_compatibility_gate"
            ] = copy.deepcopy(semantic_gate)
        assignment["provenance"] = updated_provenance
    provenance = output.get("provenance")
    output_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    output_provenance["appearance_component_actual_mdl_candidate"] = {
        "component_id": component_id,
        "member_part_ids": members,
        "material_id": material_id,
        "source_plan_sha256": _canonical_sha256(source_plan),
        "mdl_parameter_mutation_allowed": False,
    }
    if semantic_gate is not None:
        output_provenance["appearance_component_actual_mdl_candidate"][
            "semantic_compatibility_gate"
        ] = copy.deepcopy(semantic_gate)
    output["provenance"] = output_provenance
    return output


def build_component_color_candidate_plan(
    *,
    source_plan: Mapping[str, Any],
    component_id: str,
    member_part_ids: Sequence[str],
    material_id: str,
    target_srgb: Sequence[float],
    member_material_semantics: Mapping[str, Mapping[str, Any]],
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one same-MDL, color-only H1 for every member of a component.

    The source component must still be native H0. Parameters already selected
    for other components are preserved, which lets callers combine independently
    render-approved component winners without rebuilding earlier assignments.
    """

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
        raise ComponentMdlTournamentError("color candidate material_id must be an MDL ID")
    members = _member_ids(member_part_ids)
    normalized_members, target_family, _target_finish = _strict_member_semantics(
        member_material_semantics,
        expected_member_part_ids=members,
    )
    catalog_metadata = _catalog_candidate_semantics(
        material_id=material_id,
        catalog_materials_by_id=catalog_materials_by_id,
        member_material_semantics=normalized_members,
        target_family=target_family,
    )
    if catalog_metadata is None:
        raise ComponentMdlTournamentError(
            "component color H1 material is not compatible with every member"
        )
    profile = tuning_profile_for_material(material_id)
    expected_surface_class = "metal" if target_family == "metal" else "dielectric"
    if profile is None or profile.surface_class != expected_surface_class:
        raise ComponentMdlTournamentError(
            "component color H1 lacks a reviewed same-material tuning profile"
        )
    try:
        parameters, authored = color_parameters_for_target_srgb(
            profile,
            target_srgb,
        )
    except (TypeError, ValueError) as exc:
        raise ComponentMdlTournamentError(
            f"component color H1 target is invalid: {exc}"
        ) from exc
    policy = parameter_policy_for_material(material_id)
    if set(parameters) != set(profile.color_parameters) or any(
        policy.get(name, (None, None, None))[0] != "color3f_linear"
        for name in parameters
    ):
        raise ComponentMdlTournamentError(
            "component color H1 attempted an unreviewed shader parameter"
        )

    output = copy.deepcopy(dict(source_plan))
    assignments = _assignments(output, allow_parameter_overrides=True)
    missing = sorted(set(members) - set(assignments))
    if missing:
        raise ComponentMdlTournamentError(
            f"component {component_id} has unknown Part-IDs: {missing}"
        )
    for part_id in members:
        assignment = assignments[part_id]
        if assignment.get("material_id") != material_id:
            raise ComponentMdlTournamentError(
                f"component color H1 would change the selected MDL for {part_id}"
            )
        existing_parameters = assignment.get("parameters")
        if isinstance(existing_parameters, Mapping) and existing_parameters:
            raise ComponentMdlTournamentError(
                f"component color H1 source is not native H0 for {part_id}"
            )

    semantic_hash = _canonical_sha256(normalized_members)
    catalog_hash = _canonical_sha256(catalog_materials_by_id[material_id])
    candidate_binding = {
        "schema_version": COLOR_SCHEMA_VERSION,
        "component_id": component_id,
        "member_part_ids": members,
        "candidate_id": "H1",
        "material_id": material_id,
        "same_material_id_as_h0": True,
        "source_plan_sha256": _canonical_sha256(source_plan),
        "member_material_semantics": copy.deepcopy(normalized_members),
        "member_material_semantics_sha256": semantic_hash,
        "catalog_material_record": copy.deepcopy(
            dict(catalog_materials_by_id[material_id])
        ),
        "catalog_material_record_sha256": catalog_hash,
        "target_family": target_family,
        "tuning_profile_id": profile.profile_id,
        "target_color_srgb": list(authored["base_color_srgb"]),
        "authored_parameter_names": sorted(parameters),
        "parameter_mutation_scope": "reviewed_color3f_linear_only",
        "minimum_score_improvement": COLOR_MINIMUM_SCORE_IMPROVEMENT,
        "maximum_member_regression": COLOR_MAXIMUM_MEMBER_REGRESSION,
    }
    for part_id in members:
        assignment = assignments[part_id]
        assignment["parameters"] = copy.deepcopy(parameters)
        provenance = assignment.get("provenance")
        updated_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
        updated_provenance["appearance_component_color_candidate"] = copy.deepcopy(
            candidate_binding
        )
        assignment["provenance"] = updated_provenance

    provenance = output.get("provenance")
    output_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    output_provenance["appearance_component_color_candidate"] = copy.deepcopy(
        candidate_binding
    )
    output["provenance"] = output_provenance
    return output


def score_component_render(
    *,
    component_id: str,
    member_part_ids: Sequence[str],
    evidence: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a pixel-weighted real-CAD similarity score for one component."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    members = _member_ids(member_part_ids)
    scores: list[dict[str, Any]] = []
    for part_id in members:
        try:
            score = score_part_id_render(
                part_id=part_id,
                evidence=evidence,
                spatial_mapping_report=spatial_mapping_report,
                rendered_registry=rendered_registry,
            )
        except PartIdParameterTournamentError as exc:
            raise ComponentMdlTournamentError(
                f"component {component_id} could not score {part_id}: {exc}"
            ) from exc
        scores.append(score)
    total_pixels = sum(int(row["comparison_pixel_count"]) for row in scores)
    if total_pixels <= 0:
        raise ComponentMdlTournamentError(
            f"component {component_id} has no registered comparison pixels"
        )

    def weighted(key: str) -> float:
        return sum(
            float(row[key]) * int(row["comparison_pixel_count"]) for row in scores
        ) / total_pixels

    return {
        "schema_version": SCHEMA_VERSION,
        "component_id": component_id,
        "member_part_ids": members,
        "member_score_count": len(scores),
        "comparison_pixel_count": total_pixels,
        "appearance_score": round(weighted("appearance_score"), 8),
        "color_score": round(weighted("color_score"), 8),
        "luma_score": round(weighted("luma_score"), 8),
        "lab_delta_e": round(weighted("lab_delta_e"), 8),
        "member_scores": scores,
    }


def _validated_component_render_score(
    value: Mapping[str, Any],
    *,
    component_id: str,
    label: str,
) -> tuple[float, list[str], dict[str, float]]:
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ComponentMdlTournamentError(
            f"{label} component render score has an invalid schema"
        )
    if value.get("component_id") != component_id:
        raise ComponentMdlTournamentError(
            f"{label} component render score belongs to a different component"
        )
    raw_members = value.get("member_part_ids")
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        raise ComponentMdlTournamentError(
            f"{label} component render score has invalid members"
        )
    members = _member_ids(raw_members)
    raw_scores = value.get("member_scores")
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        raise ComponentMdlTournamentError(
            f"{label} component render score has invalid member scores"
        )
    by_part: dict[str, float] = {}
    weighted_total = 0.0
    pixel_total = 0
    for raw_score in raw_scores:
        if not isinstance(raw_score, Mapping):
            raise ComponentMdlTournamentError(
                f"{label} component render score has a malformed member row"
            )
        part_id = raw_score.get("part_id")
        appearance = raw_score.get("appearance_score")
        pixels = raw_score.get("comparison_pixel_count")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in by_part
            or isinstance(appearance, bool)
            or not isinstance(appearance, (int, float))
            or not math.isfinite(float(appearance))
            or not 0.0 <= float(appearance) <= 1.0
            or isinstance(pixels, bool)
            or not isinstance(pixels, int)
            or pixels <= 0
        ):
            raise ComponentMdlTournamentError(
                f"{label} component render score has an invalid member row"
            )
        by_part[part_id] = float(appearance)
        weighted_total += float(appearance) * pixels
        pixel_total += pixels
    aggregate = value.get("appearance_score")
    if (
        set(by_part) != set(members)
        or value.get("member_score_count") != len(members)
        or value.get("comparison_pixel_count") != pixel_total
        or isinstance(aggregate, bool)
        or not isinstance(aggregate, (int, float))
        or not math.isfinite(float(aggregate))
        or not 0.0 <= float(aggregate) <= 1.0
        or abs(float(aggregate) - weighted_total / pixel_total) > 1e-7
    ):
        raise ComponentMdlTournamentError(
            f"{label} component render score is not internally consistent"
        )
    return float(aggregate), members, by_part


def select_component_color_winner(
    *,
    component_id: str,
    h0_score: Mapping[str, Any],
    h1_score: Mapping[str, Any],
    minimum_score_improvement: float = COLOR_MINIMUM_SCORE_IMPROVEMENT,
    maximum_member_regression: float = COLOR_MAXIMUM_MEMBER_REGRESSION,
) -> dict[str, Any]:
    """Select H1 only after aggregate improvement and every-member safety pass."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    for value, label in (
        (minimum_score_improvement, "minimum_score_improvement"),
        (maximum_member_regression, "maximum_member_regression"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ComponentMdlTournamentError(
                f"{label} must be a finite unit number"
            )
    h0_aggregate, h0_members, h0_by_part = _validated_component_render_score(
        h0_score,
        component_id=component_id,
        label="H0",
    )
    h1_aggregate, h1_members, h1_by_part = _validated_component_render_score(
        h1_score,
        component_id=component_id,
        label="H1",
    )
    if h0_members != h1_members:
        raise ComponentMdlTournamentError(
            "component color H0/H1 scores cover different members"
        )

    member_comparisons: list[dict[str, Any]] = []
    maximum_observed_regression = 0.0
    for part_id in h0_members:
        regression = h0_by_part[part_id] - h1_by_part[part_id]
        maximum_observed_regression = max(maximum_observed_regression, regression)
        member_comparisons.append(
            {
                "part_id": part_id,
                "h0_appearance_score": round(h0_by_part[part_id], 8),
                "h1_appearance_score": round(h1_by_part[part_id], 8),
                "h1_score_change": round(
                    h1_by_part[part_id] - h0_by_part[part_id], 8
                ),
            }
        )
    maximum_observed_regression = max(0.0, maximum_observed_regression)
    aggregate_improvement = h1_aggregate - h0_aggregate
    reason_codes: list[str] = []
    if aggregate_improvement < float(minimum_score_improvement):
        reason_codes.append("INSUFFICIENT_AGGREGATE_IMPROVEMENT")
    if maximum_observed_regression > float(maximum_member_regression):
        reason_codes.append("MEMBER_REGRESSION_ABOVE_MAXIMUM")
    h1_selected = not reason_codes
    return {
        "schema_version": COLOR_SCHEMA_VERSION,
        "component_id": component_id,
        "member_part_ids": h0_members,
        "selected_candidate_id": "H1" if h1_selected else "H0",
        "selection_status": (
            "COLOR_RENDER_WINNER" if h1_selected else "NATIVE_H0_RETAINED"
        ),
        "h0_appearance_score": round(h0_aggregate, 8),
        "h1_appearance_score": round(h1_aggregate, 8),
        "aggregate_score_improvement": round(aggregate_improvement, 8),
        "minimum_score_improvement": float(minimum_score_improvement),
        "maximum_observed_member_regression": round(
            maximum_observed_regression, 8
        ),
        "maximum_member_regression": float(maximum_member_regression),
        "reason_codes": reason_codes,
        "member_comparisons": member_comparisons,
        "h0_score_sha256": _canonical_sha256(h0_score),
        "h1_score_sha256": _canonical_sha256(h1_score),
    }


def select_component_mdl_winner(
    *,
    component_id: str,
    baseline_material_id: str,
    candidate_scores: Mapping[str, Mapping[str, Any]],
    minimum_score_improvement: float = 0.015,
    member_material_semantics: Mapping[str, Mapping[str, Any]] | None = None,
    catalog_materials_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    authorized_candidate_material_ids: Sequence[str] | None = None,
    lock_baseline_identity: bool = False,
) -> dict[str, Any]:
    """Choose a fixed MDL only when real-CAD evidence clearly improves it."""

    if not isinstance(component_id, str) or not component_id:
        raise ComponentMdlTournamentError("component_id must be non-empty")
    if not isinstance(baseline_material_id, str) or not baseline_material_id.startswith(
        "mdl:"
    ):
        raise ComponentMdlTournamentError("baseline_material_id must be an MDL ID")
    if (
        isinstance(minimum_score_improvement, bool)
        or not isinstance(minimum_score_improvement, (int, float))
        or not math.isfinite(float(minimum_score_improvement))
        or not 0.0 <= float(minimum_score_improvement) <= 1.0
    ):
        raise ComponentMdlTournamentError(
            "minimum_score_improvement must be a finite unit number"
        )
    if baseline_material_id not in candidate_scores:
        raise ComponentMdlTournamentError("candidate scores lack the baseline MDL")
    strict_semantics: tuple[dict[str, dict[str, Any]], str] | None = None
    if _semantic_gate_requested(
        member_material_semantics,
        catalog_materials_by_id,
    ):
        assert member_material_semantics is not None
        assert catalog_materials_by_id is not None
        normalized_members, target_family, _target_finish = _strict_member_semantics(
            member_material_semantics
        )
        strict_semantics = (normalized_members, target_family)
        if not 2 <= len(candidate_scores) <= 3:
            raise ComponentMdlTournamentError(
                "strict component winner needs two or three candidate scores"
            )
        if authorized_candidate_material_ids is not None:
            if isinstance(authorized_candidate_material_ids, (str, bytes)):
                raise ComponentMdlTournamentError(
                    "authorized component candidates are invalid"
                )
            authorized = list(authorized_candidate_material_ids)
            if (
                len(authorized) != len(set(authorized))
                or set(authorized) != set(candidate_scores)
                or not 2 <= len(authorized) <= 3
            ):
                raise ComponentMdlTournamentError(
                    "candidate scores do not exactly match the authorized shortlist"
                )
    elif authorized_candidate_material_ids is not None:
        raise ComponentMdlTournamentError(
            "authorized candidate binding requires strict component semantics"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for material_id, value in candidate_scores.items():
        score = value.get("appearance_score") if isinstance(value, Mapping) else None
        if (
            not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ComponentMdlTournamentError("candidate score is invalid")
        if strict_semantics is not None:
            assert catalog_materials_by_id is not None
            normalized_members, target_family = strict_semantics
            if (
                _catalog_candidate_semantics(
                    material_id=material_id,
                    catalog_materials_by_id=catalog_materials_by_id,
                    member_material_semantics=normalized_members,
                    target_family=target_family,
                )
                is None
            ):
                raise ComponentMdlTournamentError(
                    "candidate score map contains a semantically incompatible MDL"
                )
        normalized[material_id] = dict(value)
    winner_id = min(
        normalized,
        key=lambda material_id: (-float(normalized[material_id]["appearance_score"]), material_id),
    )
    baseline_score = float(normalized[baseline_material_id]["appearance_score"])
    winner_score = float(normalized[winner_id]["appearance_score"])
    accepted = not lock_baseline_identity and winner_id != baseline_material_id and (
        winner_score >= baseline_score + float(minimum_score_improvement)
    )
    selected_id = winner_id if accepted else baseline_material_id
    return {
        "schema_version": SCHEMA_VERSION,
        "component_id": component_id,
        "baseline_material_id": baseline_material_id,
        "selected_material_id": selected_id,
        "winning_candidate_material_id": winner_id,
        "baseline_appearance_score": round(baseline_score, 8),
        "winning_appearance_score": round(winner_score, 8),
        "score_improvement": round(winner_score - baseline_score, 8),
        "minimum_score_improvement": float(minimum_score_improvement),
        "selection_status": (
            "ACTUAL_CAD_RENDER_WINNER"
            if accepted
            else (
                "PREDICTED_MATERIAL_IDENTITY_LOCKED_COLOR_DEFERRED"
                if lock_baseline_identity
                else "BASELINE_RETAINED"
            )
        ),
        "predicted_material_identity_locked": bool(lock_baseline_identity),
        "mdl_parameter_mutation_allowed": False,
    }


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ComponentMdlTournamentError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_color_parameters(
    *,
    material_id: str,
    target_family: str,
    parameters: Any,
) -> dict[str, list[float]]:
    profile = tuning_profile_for_material(material_id)
    expected_surface_class = "metal" if target_family == "metal" else "dielectric"
    if profile is None or profile.surface_class != expected_surface_class:
        raise ComponentMdlTournamentError(
            "component color authorization lacks a reviewed same-material profile"
        )
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        profile.color_parameters
    ):
        raise ComponentMdlTournamentError(
            "component color authorization has an invalid parameter set"
        )
    policy = parameter_policy_for_material(material_id)
    normalized: dict[str, list[float]] = {}
    for name, raw_value in parameters.items():
        if policy.get(name, (None, None, None))[0] != "color3f_linear":
            raise ComponentMdlTournamentError(
                "component color authorization includes a non-color parameter"
            )
        if (
            not isinstance(raw_value, Sequence)
            or isinstance(raw_value, (str, bytes))
            or len(raw_value) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in raw_value
            )
        ):
            raise ComponentMdlTournamentError(
                "component color authorization has an invalid color3f value"
            )
        normalized[str(name)] = [float(value) for value in raw_value]
    return normalized


def _unit_tournament_threshold(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ComponentMdlTournamentError(f"{label} must be a finite unit number")
    return float(value)


def _identity_component_record(
    *,
    tournament_audit: Mapping[str, Any],
    component_id: str,
) -> Mapping[str, Any]:
    raw_components = tournament_audit.get("components")
    if not isinstance(raw_components, list):
        raise ComponentMdlTournamentError("component tournament audit has no components")
    matches = [
        row
        for row in raw_components
        if isinstance(row, Mapping) and row.get("component_id") == component_id
    ]
    if len(matches) != 1:
        raise ComponentMdlTournamentError(
            "component color record lacks one exact identity component contract"
        )
    return matches[0]


def _trusted_component_semantic_payload(
    *,
    assignments: Mapping[str, Mapping[str, Any]],
    tournament_audit: Mapping[str, Any],
    component_id: str,
    members: Sequence[str],
    material_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Replay the catalog/member semantic gate from sealed identity payloads."""

    normalized_members_list = _member_ids(members)
    first_gate: dict[str, Any] | None = None
    for part_id in normalized_members_list:
        assignment = assignments.get(part_id)
        provenance = (
            assignment.get("provenance")
            if isinstance(assignment, Mapping)
            else None
        )
        identity_binding = (
            provenance.get("appearance_component_actual_mdl_candidate")
            if isinstance(provenance, Mapping)
            else None
        )
        semantic_gate = (
            identity_binding.get("semantic_compatibility_gate")
            if isinstance(identity_binding, Mapping)
            else None
        )
        if (
            not isinstance(identity_binding, Mapping)
            or identity_binding.get("component_id") != component_id
            or identity_binding.get("member_part_ids") != normalized_members_list
            or identity_binding.get("material_id") != material_id
            or identity_binding.get("mdl_parameter_mutation_allowed") is not False
            or not isinstance(semantic_gate, Mapping)
            or semantic_gate.get("policy")
            != "all_component_members_physical_semantics_compatible/v1"
        ):
            raise ComponentMdlTournamentError(
                "final component assignment lacks its strict identity semantic binding"
            )
        gate = copy.deepcopy(dict(semantic_gate))
        if first_gate is None:
            first_gate = gate
        elif gate != first_gate:
            raise ComponentMdlTournamentError(
                "component members carry different semantic catalog payloads"
            )
    assert first_gate is not None

    raw_member_semantics = first_gate.get("member_material_semantics")
    raw_catalog_record = first_gate.get("catalog_material_record")
    if not isinstance(raw_member_semantics, Mapping) or not isinstance(
        raw_catalog_record, Mapping
    ):
        raise ComponentMdlTournamentError(
            "component identity semantic binding lacks replayable payloads"
        )
    normalized_members, target_family, _target_finish = _strict_member_semantics(
        raw_member_semantics,
        expected_member_part_ids=normalized_members_list,
    )
    catalog_record = copy.deepcopy(dict(raw_catalog_record))
    if (
        first_gate.get("member_material_semantics_sha256")
        != _canonical_sha256(normalized_members)
        or first_gate.get("catalog_material_record_sha256")
        != _canonical_sha256(catalog_record)
        or first_gate.get("target_family") != target_family
        or catalog_record.get("material_id") != material_id
        or _catalog_candidate_semantics(
            material_id=material_id,
            catalog_materials_by_id={material_id: catalog_record},
            member_material_semantics=normalized_members,
            target_family=target_family,
        )
        is None
    ):
        raise ComponentMdlTournamentError(
            "component identity semantic/catalog payload failed deterministic replay"
        )

    identity_record = _identity_component_record(
        tournament_audit=tournament_audit,
        component_id=component_id,
    )
    semantic_contract = identity_record.get("semantic_contract")
    if (
        identity_record.get("member_part_ids") != normalized_members_list
        or identity_record.get("selected_material_id") != material_id
        or not isinstance(semantic_contract, Mapping)
        or semantic_contract.get("component_id") != component_id
        or semantic_contract.get("member_part_ids") != normalized_members_list
        or semantic_contract.get("member_material_semantics") != normalized_members
        or semantic_contract.get("member_material_semantics_sha256")
        != _canonical_sha256(normalized_members)
    ):
        raise ComponentMdlTournamentError(
            "component identity audit and assignment semantics disagree"
        )
    return normalized_members, catalog_record, target_family


def _identity_plan_from_color_final(
    *,
    final_plan: Mapping[str, Any],
    color_components: Sequence[tuple[Mapping[str, Any], str, list[str], str, str]],
    expected_sha256: str,
) -> dict[str, Any]:
    """Reverse only authorized H1 deltas and recover the exact identity plan."""

    identity_plan = copy.deepcopy(dict(final_plan))
    assignments = _assignments(identity_plan, allow_parameter_overrides=True)
    for _record, _component_id, members, _material_id, selected_id in color_components:
        if selected_id != "H1":
            continue
        for part_id in members:
            assignment = assignments.get(part_id)
            if not isinstance(assignment, dict):
                raise ComponentMdlTournamentError(
                    "component color record references an unknown Part-ID"
                )
            assignment.pop("parameters", None)
            provenance = assignment.get("provenance")
            if not isinstance(provenance, dict) or not isinstance(
                provenance.pop("appearance_component_color_candidate", None),
                Mapping,
            ):
                raise ComponentMdlTournamentError(
                    "final component H1 cannot be reversed to its identity plan"
                )
    if _canonical_sha256(identity_plan) != expected_sha256:
        raise ComponentMdlTournamentError(
            "component color source identity plan cannot be reconstructed"
        )
    return identity_plan


def _copied_trusted_color_score_evidence(
    *,
    tournament_audit: Mapping[str, Any],
    trusted_color_score_evidence: ComponentColorScoreEvidence | None,
) -> _VerifiedComponentColorScoreEvidence | None:
    """Safely read render artifacts and require exact audit/capability cover."""

    raw_color = tournament_audit.get("component_color_tournament")
    if not isinstance(raw_color, Mapping):
        return None
    if not isinstance(trusted_color_score_evidence, ComponentColorScoreEvidence):
        raise ComponentMdlTournamentError(
            "component color authorization requires trusted score evidence"
        )
    raw_components = raw_color.get("components")
    if not isinstance(raw_components, list):
        raise ComponentMdlTournamentError(
            "component color tournament has no component records"
        )
    component_ids: list[str] = []
    components_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_component in raw_components:
        component_id = (
            raw_component.get("component_id")
            if isinstance(raw_component, Mapping)
            else None
        )
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_ids
        ):
            raise ComponentMdlTournamentError(
                "component color tournament has invalid component evidence cover"
            )
        component_ids.append(component_id)
        assert isinstance(raw_component, Mapping)
        components_by_id[component_id] = raw_component

    raw_evidence = trusted_color_score_evidence.evidence
    raw_spatial = trusted_color_score_evidence.spatial_mapping_report
    artifact_root = trusted_color_score_evidence.artifact_root
    h0_artifact = trusted_color_score_evidence.h0_artifact
    h1_artifacts = trusted_color_score_evidence.h1_artifacts_by_component
    if (
        not isinstance(raw_evidence, Mapping)
        or not isinstance(raw_spatial, Mapping)
        or not isinstance(h0_artifact, ComponentColorRenderArtifactPaths)
        or not isinstance(h1_artifacts, Mapping)
        or set(h1_artifacts) != set(component_ids)
        or any(
            not isinstance(
                h1_artifacts.get(component_id),
                ComponentColorRenderArtifactPaths,
            )
            for component_id in component_ids
        )
    ):
        raise ComponentMdlTournamentError(
            "trusted H1 render artifacts do not exactly cover color components"
        )

    source_identity_plan_sha256 = _sha256_text(
        raw_color.get("source_identity_plan_sha256"),
        "component color source identity plan hash",
    )
    h0_binding, h0_registry = _load_component_color_render_artifact_binding(
        artifact_root=artifact_root,
        artifact=h0_artifact,
        expected_plan_sha256=source_identity_plan_sha256,
    )
    h1_registries: dict[str, Mapping[str, Any]] = {}
    for component_id in component_ids:
        record = components_by_id[component_id]
        candidate_plan_sha256 = _sha256_text(
            record.get("color_candidate_plan_sha256"),
            "component color candidate plan hash",
        )
        h1_binding, h1_registry = _load_component_color_render_artifact_binding(
            artifact_root=artifact_root,
            artifact=h1_artifacts[component_id],
            expected_plan_sha256=candidate_plan_sha256,
        )
        expected_artifacts = {"h0": h0_binding, "h1": h1_binding}
        candidate_artifacts = record.get("candidate_artifacts")
        if (
            not isinstance(candidate_artifacts, Mapping)
            or dict(candidate_artifacts) != expected_artifacts
        ):
            raise ComponentMdlTournamentError(
                "component color audit artifacts do not match trusted files"
            )
        h1_registries[component_id] = copy.deepcopy(h1_registry)

    return _VerifiedComponentColorScoreEvidence(
        evidence=copy.deepcopy(dict(raw_evidence)),
        spatial_mapping_report=copy.deepcopy(dict(raw_spatial)),
        h0_rendered_registry=copy.deepcopy(h0_registry),
        h1_rendered_registries_by_component={
            component_id: h1_registries[component_id]
            for component_id in component_ids
        },
    )


def _component_color_authorizations(
    *,
    final_plan: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
    tournament_audit: Mapping[str, Any],
    trusted_color_score_evidence: _VerifiedComponentColorScoreEvidence | None,
) -> tuple[dict[str, Mapping[str, Any]], int]:
    """Validate the only contract that may authorize final color parameters."""

    raw_color = tournament_audit.get("component_color_tournament")
    if not isinstance(raw_color, Mapping):
        if any(
            isinstance(assignment.get("parameters"), Mapping)
            and bool(assignment.get("parameters"))
            for assignment in assignments.values()
        ):
            raise ComponentMdlTournamentError(
                "final plan has parameters without a component color authorization"
            )
        return {}, 0
    if raw_color.get("schema_version") != COLOR_SCHEMA_VERSION:
        raise ComponentMdlTournamentError(
            "component color tournament has an unsupported schema"
        )
    if trusted_color_score_evidence is None:
        raise ComponentMdlTournamentError(
            "component color authorization requires trusted score evidence"
        )
    source_identity_plan_sha256 = _sha256_text(
        raw_color.get("source_identity_plan_sha256"),
        "component color source identity plan hash",
    )
    if raw_color.get("final_plan_sha256") != _canonical_sha256(final_plan):
        raise ComponentMdlTournamentError(
            "component color tournament is not hash-bound to the final plan"
        )
    raw_components = raw_color.get("components")
    if not isinstance(raw_components, list):
        raise ComponentMdlTournamentError(
            "component color tournament has no component records"
        )

    minimum_score_improvement = _unit_tournament_threshold(
        raw_color.get("minimum_score_improvement"),
        "component color minimum score improvement",
    )
    maximum_member_regression = _unit_tournament_threshold(
        raw_color.get("maximum_member_regression"),
        "component color maximum member regression",
    )
    if (
        minimum_score_improvement != COLOR_MINIMUM_SCORE_IMPROVEMENT
        or maximum_member_regression != COLOR_MAXIMUM_MEMBER_REGRESSION
    ):
        raise ComponentMdlTournamentError(
            "component color tournament thresholds differ from the approved "
            "production contract"
        )
    if (
        tournament_audit.get("identity_final_plan_sha256")
        != source_identity_plan_sha256
    ):
        raise ComponentMdlTournamentError(
            "outer component tournament identity hash disagrees with its color contract"
        )

    color_components: list[
        tuple[Mapping[str, Any], str, list[str], str, str]
    ] = []
    component_ids: set[str] = set()
    preliminary_part_ids: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, Mapping):
            raise ComponentMdlTournamentError(
                f"component color tournament record {index} is invalid"
            )
        component_id = raw_component.get("component_id")
        material_id = raw_component.get("material_id")
        selected_candidate_id = raw_component.get("selected_candidate_id")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_ids
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
            or selected_candidate_id not in {"H0", "H1"}
        ):
            raise ComponentMdlTournamentError(
                "component color tournament record has invalid identity fields"
            )
        members = _member_ids(raw_component.get("member_part_ids", []))
        if preliminary_part_ids & set(members):
            raise ComponentMdlTournamentError(
                "component color authorizations overlap Part-IDs"
            )
        component_ids.add(component_id)
        preliminary_part_ids.update(members)
        color_components.append(
            (
                raw_component,
                component_id,
                members,
                material_id,
                selected_candidate_id,
            )
        )
    if (
        raw_color.get("candidate_count") != len(color_components)
        or raw_color.get("h1_winner_count")
        != sum(selected_id == "H1" for *_, selected_id in color_components)
    ):
        raise ComponentMdlTournamentError(
            "component color tournament summary does not match its records"
        )
    identity_plan = _identity_plan_from_color_final(
        final_plan=final_plan,
        color_components=color_components,
        expected_sha256=source_identity_plan_sha256,
    )

    authorized_by_part: dict[str, Mapping[str, Any]] = {}
    h1_component_count = 0
    for raw_component, component_id, members, material_id, selected_candidate_id in (
        color_components
    ):
        if raw_component.get("source_plan_sha256") != source_identity_plan_sha256:
            raise ComponentMdlTournamentError(
                "component color record is not bound to the identity plan"
            )
        candidate_plan_sha256 = _sha256_text(
            raw_component.get("color_candidate_plan_sha256"),
            "component color candidate plan hash",
        )
        h0_score = raw_component.get("h0_score")
        h1_score = raw_component.get("h1_score")
        if not isinstance(h0_score, Mapping) or not isinstance(h1_score, Mapping):
            raise ComponentMdlTournamentError(
                "component color record lacks replayable H0/H1 scores"
            )
        trusted_h0_score = score_component_render(
            component_id=component_id,
            member_part_ids=members,
            evidence=trusted_color_score_evidence.evidence,
            spatial_mapping_report=(
                trusted_color_score_evidence.spatial_mapping_report
            ),
            rendered_registry=trusted_color_score_evidence.h0_rendered_registry,
        )
        trusted_h1_score = score_component_render(
            component_id=component_id,
            member_part_ids=members,
            evidence=trusted_color_score_evidence.evidence,
            spatial_mapping_report=(
                trusted_color_score_evidence.spatial_mapping_report
            ),
            rendered_registry=(
                trusted_color_score_evidence.h1_rendered_registries_by_component[
                    component_id
                ]
            ),
        )
        if dict(h0_score) != trusted_h0_score or dict(h1_score) != trusted_h1_score:
            raise ComponentMdlTournamentError(
                "component color scores do not match trusted render evidence"
            )
        selection = raw_component.get("selection")
        if not isinstance(selection, Mapping):
            raise ComponentMdlTournamentError(
                "component color selection does not match its authorization record"
            )
        replayed_selection = select_component_color_winner(
            component_id=component_id,
            h0_score=h0_score,
            h1_score=h1_score,
            minimum_score_improvement=COLOR_MINIMUM_SCORE_IMPROVEMENT,
            maximum_member_regression=COLOR_MAXIMUM_MEMBER_REGRESSION,
        )
        if dict(selection) != replayed_selection or (
            replayed_selection["selected_candidate_id"] != selected_candidate_id
        ):
            raise ComponentMdlTournamentError(
                "component color selection failed deterministic score replay"
            )
        if (
            selection.get("h0_score_sha256") != _canonical_sha256(h0_score)
            or selection.get("h1_score_sha256") != _canonical_sha256(h1_score)
        ):
            raise ComponentMdlTournamentError(
                "component color score hashes do not match their payloads"
            )

        normalized_members, catalog_record, target_family = (
            _trusted_component_semantic_payload(
                assignments=assignments,
                tournament_audit=tournament_audit,
                component_id=component_id,
                members=members,
                material_id=material_id,
            )
        )
        raw_parameters = raw_component.get("parameters")
        if selected_candidate_id == "H0":
            if raw_parameters not in ({}, None):
                raise ComponentMdlTournamentError(
                    "native component H0 cannot authorize parameters"
                )
            normalized_parameters: dict[str, list[float]] = {}
        else:
            first_assignment = assignments.get(members[0])
            binding = (
                first_assignment.get("provenance", {}).get(
                    "appearance_component_color_candidate"
                )
                if isinstance(first_assignment, Mapping)
                and isinstance(first_assignment.get("provenance"), Mapping)
                else None
            )
            normalized_parameters = _validate_color_parameters(
                material_id=material_id,
                target_family=target_family,
                parameters=raw_parameters,
            )
            if not isinstance(binding, Mapping):
                raise ComponentMdlTournamentError(
                    "final component H1 lacks a replayable color binding"
                )
            target_srgb = binding.get("target_color_srgb")
            profile = tuning_profile_for_material(material_id)
            assert profile is not None
            try:
                replayed_parameters, authored = color_parameters_for_target_srgb(
                    profile,
                    target_srgb,
                )
            except (TypeError, ValueError) as exc:
                raise ComponentMdlTournamentError(
                    f"component H1 target color cannot be replayed: {exc}"
                ) from exc
            if (
                normalized_parameters != replayed_parameters
                or binding.get("target_color_srgb")
                != authored["base_color_srgb"]
                or binding.get("member_material_semantics") != normalized_members
                or binding.get("member_material_semantics_sha256")
                != _canonical_sha256(normalized_members)
                or binding.get("catalog_material_record") != catalog_record
                or binding.get("catalog_material_record_sha256")
                != _canonical_sha256(catalog_record)
                or binding.get("target_family") != target_family
            ):
                raise ComponentMdlTournamentError(
                    "component H1 target, parameters, or semantic payload failed replay"
                )
            rebuilt_candidate = build_component_color_candidate_plan(
                source_plan=identity_plan,
                component_id=component_id,
                member_part_ids=members,
                material_id=material_id,
                target_srgb=target_srgb,
                member_material_semantics=normalized_members,
                catalog_materials_by_id={material_id: catalog_record},
            )
            if _canonical_sha256(rebuilt_candidate) != candidate_plan_sha256:
                raise ComponentMdlTournamentError(
                    "component H1 candidate plan hash failed deterministic rebuild"
                )
            rebuilt_assignments = _assignments(
                rebuilt_candidate,
                allow_parameter_overrides=True,
            )
            if any(
                assignments[part_id] != rebuilt_assignments[part_id]
                for part_id in members
            ):
                raise ComponentMdlTournamentError(
                    "final component H1 lacks its exact color candidate binding; "
                    "assignments differ from the rebuilt candidate"
                )
            h1_component_count += 1

        for part_id in members:
            if part_id in authorized_by_part:
                raise ComponentMdlTournamentError(
                    "component color authorizations overlap Part-IDs"
                )
            assignment = assignments.get(part_id)
            if not isinstance(assignment, Mapping):
                raise ComponentMdlTournamentError(
                    "component color authorization references an unknown Part-ID"
                )
            assignment_parameters = assignment.get("parameters")
            actual_parameters = (
                dict(assignment_parameters)
                if isinstance(assignment_parameters, Mapping)
                else {}
            )
            if assignment.get("material_id") != material_id:
                raise ComponentMdlTournamentError(
                    "component color authorization would change MDL identity"
                )
            if actual_parameters != normalized_parameters:
                raise ComponentMdlTournamentError(
                    "final component parameters differ from their color authorization"
                )
            if selected_candidate_id == "H1":
                provenance = assignment.get("provenance")
                binding = (
                    provenance.get("appearance_component_color_candidate")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("schema_version") != COLOR_SCHEMA_VERSION
                    or binding.get("component_id") != component_id
                    or binding.get("member_part_ids") != members
                    or binding.get("candidate_id") != "H1"
                    or binding.get("material_id") != material_id
                    or binding.get("same_material_id_as_h0") is not True
                    or binding.get("source_plan_sha256")
                    != source_identity_plan_sha256
                    or binding.get("parameter_mutation_scope")
                    != "reviewed_color3f_linear_only"
                    or binding.get("minimum_score_improvement")
                    != COLOR_MINIMUM_SCORE_IMPROVEMENT
                    or binding.get("maximum_member_regression")
                    != COLOR_MAXIMUM_MEMBER_REGRESSION
                    or binding.get("authored_parameter_names")
                    != sorted(normalized_parameters)
                ):
                    raise ComponentMdlTournamentError(
                        "final component H1 lacks its exact color candidate binding"
                    )
                profile = tuning_profile_for_material(material_id)
                if (
                    profile is None
                    or binding.get("tuning_profile_id") != profile.profile_id
                ):
                    raise ComponentMdlTournamentError(
                        "final component H1 tuning profile binding is invalid"
                    )
            authorized_by_part[part_id] = raw_component

    parameterized_part_ids = {
        part_id
        for part_id, assignment in assignments.items()
        if isinstance(assignment.get("parameters"), Mapping)
        and bool(assignment.get("parameters"))
    }
    authorized_h1_part_ids = {
        part_id
        for part_id, authorization in authorized_by_part.items()
        if authorization.get("selected_candidate_id") == "H1"
    }
    if parameterized_part_ids != authorized_h1_part_ids:
        raise ComponentMdlTournamentError(
            "final parameterized Part-IDs do not exactly match H1 authorizations"
        )
    return authorized_by_part, h1_component_count


def rebind_part_id_material_audit_for_component_mdl_tournament(
    *,
    source_audit: Mapping[str, Any],
    source_plan: Mapping[str, Any] | None = None,
    final_plan: Mapping[str, Any],
    tournament_audit: Mapping[str, Any],
    trusted_color_score_evidence: ComponentColorScoreEvidence | None = None,
) -> dict[str, Any]:
    """Bind a Part-ID exact-cover audit to immutable component-MDL winners.

    The publication gate deliberately verifies each observed audit row against
    the final plan. A real-CAD component winner therefore refreshes its row
    material IDs and final-plan hash, while hidden Part-IDs remain unchanged.
    """

    trusted_color_score_evidence = _copied_trusted_color_score_evidence(
        tournament_audit=tournament_audit,
        trusted_color_score_evidence=trusted_color_score_evidence,
    )
    output = copy.deepcopy(dict(source_audit))
    raw_rows = output.get("parts")
    if not isinstance(raw_rows, list):
        raise ComponentMdlTournamentError("Part-ID material audit has no parts")
    color_contract_present = isinstance(
        tournament_audit.get("component_color_tournament"), Mapping
    )
    assignments = _assignments(
        final_plan,
        allow_parameter_overrides=color_contract_present,
    )
    source_assignments: dict[str, dict[str, Any]] | None = None
    if source_plan is not None:
        source_assignments = _assignments(
            source_plan,
            allow_parameter_overrides=True,
        )
        if (
            set(source_assignments) != set(assignments)
            or source_audit.get("output_plan_sha256")
            != _canonical_sha256(source_plan)
            or (
                tournament_audit.get("identity_source_plan_sha256") is not None
                and tournament_audit.get("identity_source_plan_sha256")
                != _canonical_sha256(source_plan)
            )
        ):
            raise ComponentMdlTournamentError(
                "Part-ID material audit is not hash-bound to the component "
                "tournament source plan"
            )
    color_authorization_by_part, color_h1_component_count = (
        _component_color_authorizations(
            final_plan=final_plan,
            assignments=assignments,
            tournament_audit=tournament_audit,
            trusted_color_score_evidence=trusted_color_score_evidence,
        )
    )
    rows_by_part: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise ComponentMdlTournamentError(
                f"Part-ID material audit row {index} is invalid"
            )
        part_id = raw_row.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in rows_by_part:
            raise ComponentMdlTournamentError(
                f"Part-ID material audit row {index} has an invalid Part-ID"
            )
        rows_by_part[part_id] = raw_row
    if set(rows_by_part) != set(assignments):
        raise ComponentMdlTournamentError(
            "Part-ID material audit does not exactly cover the final plan"
        )

    raw_components = tournament_audit.get("components")
    if not isinstance(raw_components, list):
        raise ComponentMdlTournamentError("component tournament audit has no components")
    winner_by_part: dict[str, Mapping[str, Any]] = {}
    for component in raw_components:
        if not isinstance(component, Mapping):
            raise ComponentMdlTournamentError("component tournament row is invalid")
        members = component.get("member_part_ids")
        selected_material_id = component.get("selected_material_id")
        baseline_material_id = component.get("baseline_material_id")
        if (
            not isinstance(members, list)
            or not isinstance(selected_material_id, str)
            or not selected_material_id.startswith("mdl:")
            or not isinstance(baseline_material_id, str)
            or not baseline_material_id.startswith("mdl:")
        ):
            raise ComponentMdlTournamentError("component tournament row is malformed")
        if selected_material_id == baseline_material_id:
            continue
        for part_id in members:
            if (
                not isinstance(part_id, str)
                or not part_id
                or part_id in winner_by_part
            ):
                raise ComponentMdlTournamentError(
                    "component tournament winner Part-IDs are invalid"
                )
            winner_by_part[part_id] = component

    for part_id, assignment in assignments.items():
        row = rows_by_part[part_id]
        row_status = row.get("status")
        if row_status == "unobserved_preserved":
            if row.get("material_id") != assignment.get("material_id"):
                raise ComponentMdlTournamentError(
                    f"unobserved Part-ID {part_id} changed in component tournament"
                )
            continue
        if row_status == "observed_low_confidence_baseline_retained":
            source_assignment = (
                source_assignments.get(part_id)
                if source_assignments is not None
                else None
            )
            provenance = assignment.get("provenance")
            rejected_material_id = row.get("rejected_qwen_material_id")
            rejected_confidence = row.get("rejected_qwen_confidence")
            confidence_floor = (
                provenance.get("applyable_review_confidence_floor")
                if isinstance(provenance, Mapping)
                else None
            )
            evidence_view_ids = row.get("evidence_view_ids")
            selected_view_id = (
                provenance.get("selected_reference_view_id_for_rejected_qwen")
                if isinstance(provenance, Mapping)
                else None
            )
            candidate_material_ids = (
                provenance.get("candidate_material_ids")
                if isinstance(provenance, Mapping)
                else None
            )
            parameter_candidates = (
                provenance.get("mdl_parameter_candidates")
                if isinstance(provenance, Mapping)
                else None
            )
            color_parameterization = (
                provenance.get("mdl_color_parameterization")
                if isinstance(provenance, Mapping)
                else None
            )
            raw_candidates = (
                parameter_candidates.get("candidates")
                if isinstance(parameter_candidates, Mapping)
                else None
            )
            if source_assignment is None:
                raise ComponentMdlTournamentError(
                    "observed low-confidence retained Part-ID requires its exact "
                    "component tournament source plan"
                )
            if (
                assignment != source_assignment
                or part_id in winner_by_part
                or part_id in color_authorization_by_part
                or assignment.get("status") != "policy_fallback"
                or assignment.get("material_id") != row.get("material_id")
                or bool(assignment.get("parameters"))
                or not isinstance(provenance, Mapping)
                or provenance.get("observed_part_id_qwen_selection_rejected")
                is not True
                or provenance.get("observed_part_id_qwen_rejection_reason")
                != "qwen_confidence_below_applyable_review_floor"
                or not isinstance(rejected_material_id, str)
                or not rejected_material_id.startswith("mdl:")
                or provenance.get("rejected_qwen_material_id")
                != rejected_material_id
                or isinstance(rejected_confidence, bool)
                or not isinstance(rejected_confidence, (int, float))
                or not math.isfinite(float(rejected_confidence))
                or provenance.get("rejected_qwen_confidence")
                != rejected_confidence
                or isinstance(confidence_floor, bool)
                or not isinstance(confidence_floor, (int, float))
                or not math.isfinite(float(confidence_floor))
                or float(confidence_floor)
                != MINIMUM_APPLYABLE_REVIEW_CONFIDENCE
                or not 0.0
                <= float(rejected_confidence)
                < MINIMUM_APPLYABLE_REVIEW_CONFIDENCE
                or not isinstance(evidence_view_ids, list)
                or len(evidence_view_ids) != 1
                or not isinstance(selected_view_id, str)
                or not selected_view_id
                or not isinstance(evidence_view_ids[0], str)
                or not evidence_view_ids[0]
                or evidence_view_ids != [selected_view_id]
                or not isinstance(candidate_material_ids, list)
                or rejected_material_id not in candidate_material_ids
                or not isinstance(parameter_candidates, Mapping)
                or parameter_candidates.get("part_id") != part_id
                or parameter_candidates.get("material_id")
                != assignment.get("material_id")
                or parameter_candidates.get("selected_candidate_id") != "H0"
                or parameter_candidates.get("parameters_applied_to_plan") is not False
                or not isinstance(raw_candidates, list)
                or len(raw_candidates) != 1
                or not isinstance(raw_candidates[0], Mapping)
                or raw_candidates[0].get("candidate_id") != "H0"
                or raw_candidates[0].get("material_id")
                != assignment.get("material_id")
                or raw_candidates[0].get("parameters") != {}
                or row.get("mdl_parameter_candidates") != parameter_candidates
                or not isinstance(color_parameterization, Mapping)
                or color_parameterization.get("selected_candidate_id") != "H0"
                or color_parameterization.get("parameters_applied") is not False
                or row.get("mdl_color_parameterization")
                != color_parameterization
            ):
                raise ComponentMdlTournamentError(
                    f"observed low-confidence Part-ID {part_id} did not retain "
                    "its exact audited policy baseline"
                )
            continue
        if row_status != "independently_selected":
            raise ComponentMdlTournamentError(
                f"Part-ID material audit has unsupported status for {part_id}"
            )
        material_id = assignment.get("material_id")
        if not isinstance(material_id, str) or not material_id.startswith("mdl:"):
            raise ComponentMdlTournamentError(
                f"final immutable MDL binding is invalid for {part_id}"
            )
        row["material_id"] = material_id
        color_authorization = color_authorization_by_part.get(part_id)
        if color_authorization is not None:
            parameters = assignment.get("parameters")
            selected_candidate_id = color_authorization.get("selected_candidate_id")
            row["parameters"] = (
                copy.deepcopy(dict(parameters))
                if isinstance(parameters, Mapping) and parameters
                else {}
            )
            row["appearance_component_color_tournament"] = {
                "schema_version": COLOR_SCHEMA_VERSION,
                "component_id": color_authorization.get("component_id"),
                "material_id": material_id,
                "selected_candidate_id": selected_candidate_id,
                "parameter_mutation_allowed": selected_candidate_id == "H1",
                "parameter_mutation_scope": (
                    "reviewed_color3f_linear_only"
                    if selected_candidate_id == "H1"
                    else "none"
                ),
            }
        winner = winner_by_part.get(part_id)
        if winner is not None:
            row["appearance_component_actual_mdl_tournament"] = {
                "component_id": winner.get("component_id"),
                "selected_material_id": material_id,
                "selection_status": winner.get("selection_status"),
                "mdl_parameter_mutation_allowed": False,
            }

    summary = output.get("summary")
    if not isinstance(summary, dict):
        raise ComponentMdlTournamentError("Part-ID material audit has no summary")
    independently_selected_count = sum(
        row.get("status") == "independently_selected" for row in rows_by_part.values()
    )
    unobserved_preserved_count = sum(
        row.get("status") == "unobserved_preserved" for row in rows_by_part.values()
    )
    low_confidence_retained_count = sum(
        row.get("status") == "observed_low_confidence_baseline_retained"
        for row in rows_by_part.values()
    )
    if (
        summary.get("part_count") != len(rows_by_part)
        or summary.get("independently_selected_count") != independently_selected_count
        or summary.get("unobserved_preserved_count") != unobserved_preserved_count
        or summary.get(
            "observed_low_confidence_baseline_retained_count",
            0,
        )
        != low_confidence_retained_count
        or (
            independently_selected_count
            + unobserved_preserved_count
            + low_confidence_retained_count
            != len(rows_by_part)
        )
        or summary.get("exact_cover") is not True
    ):
        raise ComponentMdlTournamentError(
            "Part-ID material audit summary does not match its rows"
        )
    output["output_plan_sha256"] = _canonical_sha256(final_plan)
    output["appearance_component_actual_mdl_tournament"] = {
        "schema_version": SCHEMA_VERSION,
        "audit_sha256": _canonical_sha256(tournament_audit),
        "winner_component_count": len(
            {str(component.get("component_id")) for component in winner_by_part.values()}
        ),
        "winner_part_count": len(winner_by_part),
        "mdl_parameter_mutation_allowed": bool(color_h1_component_count),
        "mdl_parameter_mutation_scope": (
            "reviewed_component_color3f_linear_only"
            if color_h1_component_count
            else "none"
        ),
        "color_h1_component_count": color_h1_component_count,
        "color_h1_part_count": sum(
            authorization.get("selected_candidate_id") == "H1"
            for authorization in color_authorization_by_part.values()
        ),
    }
    output.pop("integrity", None)
    output["integrity"] = {"document_sha256": _canonical_sha256(output)}
    return output


__all__ = [
    "COLOR_ARTIFACT_BINDING_SCHEMA_VERSION",
    "COLOR_MAXIMUM_MEMBER_REGRESSION",
    "COLOR_MINIMUM_SCORE_IMPROVEMENT",
    "COLOR_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ComponentColorRenderArtifactPaths",
    "ComponentColorScoreEvidence",
    "ComponentMdlTournamentError",
    "build_component_candidate_plan",
    "build_component_color_candidate_plan",
    "build_component_color_render_artifact_binding",
    "score_component_render",
    "select_component_color_winner",
    "select_component_mdl_winner",
]
