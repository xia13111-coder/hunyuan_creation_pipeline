#!/usr/bin/env python3
"""Run the conservative staged Qwen3-VL workflow on one or more reference views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from qwen_material_pipeline.core.progress import (
    emit_progress,
    emit_progress_event,
    format_progress_event,
    parse_progress_line,
)
from qwen_material_pipeline.core.material_stage_contract import (
    material_stage_contract_document,
)
from qwen_material_pipeline.core.staged_analysis import (
    AUTO_THRESHOLD,
    GROUP_MATERIAL_SCHEMA_VERSION,
    MaterialCollapseError,
    REVIEW_THRESHOLD,
    StagedAnalysisError,
    UNKNOWN_REASON_CODES,
    merge_staged_results,
    validate_palette,
)
from qwen_material_pipeline.evidence.confidence import (
    ConfidenceGateError,
    evaluate_confidence_gate,
)
from qwen_material_pipeline.evidence.color_semantics import fusion_color_label
from qwen_material_pipeline.evidence.face_recovery import (
    build_face_material_recovery,
)
from qwen_material_pipeline.evidence.geometry import (
    build_geometry_risk,
    validate_geometry_risk,
)
from qwen_material_pipeline.evidence.mapping import (
    apply_mapping_consensus_to_batches,
    build_view_group_id_maps,
    canonicalize_view_batch_mappings,
)
from qwen_material_pipeline.evidence.palette import filter_palette_by_image_evidence
from qwen_material_pipeline.evidence.palette_augmentation import (
    augment_palette_with_detected_accents,
)
from qwen_material_pipeline.evidence.palette_fusion import fuse_multiview_palettes
from qwen_material_pipeline.evidence.spatial import (
    apply_spatial_gate_to_batches,
    build_spatial_mapping_report,
)
from qwen_material_pipeline.materials.catalog import (
    DEFAULT_MATERIAL_ROOT,
    MaterialCatalog,
)
from qwen_material_pipeline.materials.disagreement_tournament import (
    RANKED_RETRIEVAL_CHALLENGER_BASIS,
    build_disagreement_tournament_contract,
)
from qwen_material_pipeline.materials.exact_mdl_tournament import (
    SELECTION_OBJECTIVES,
    SELECTION_OBJECTIVE_SEMANTIC,
    SELECTION_OBJECTIVE_VISUAL,
)
from qwen_material_pipeline.materials.mdl_similarity import (
    extract_mdl_appearance_profile,
    extract_thumbnail_appearance_profile,
    mvinverse_similarity_terms,
)
from qwen_material_pipeline.materials.tuning import tuning_profile_for_material
from qwen_material_pipeline.mvinverse.adapter import (
    DEFAULT_MAX_SIDE as MVINVERSE_DEFAULT_MAX_SIDE,
)
from qwen_material_pipeline.mvinverse.adapter import run_mvinverse_adapter
from qwen_material_pipeline.mvinverse.autonomy import (
    build_part_view_evidence,
    parameterize_auto_material_plan,
)
from qwen_material_pipeline.mvinverse.evidence import (
    build_mvinverse_evidence_from_manifest,
    validate_mvinverse_evidence,
)
from qwen_material_pipeline.qwen.client import QwenResponseError
from qwen_material_pipeline.qwen.local_vl import TransformersQwen3VLRunner
from qwen_material_pipeline.qwen.remote_vl import OpenAICompatibleVisionRunner
from qwen_material_pipeline.qwen.staged import LocalStagedQwenClient
from qwen_material_pipeline.retrieval.visual_materials import (
    BASE_BANK_FUSION_POLICY,
    BASE_BANK_RETRIEVAL_STRATEGY,
    LEGACY_FUSION_POLICY,
    LEGACY_RETRIEVAL_STRATEGY,
    _catalog_digest,
    _load_base_observation_bank,
    _load_catalog,
    _model_fingerprint,
    _verified_siglip2_model_identity,
)
from qwen_material_pipeline.segmentation.sam3_regions import (
    CROSS_GROUP_NEAR_DUPLICATE_IOU,
    DEFAULT_INFERENCE_SEED as SAM3_DEFAULT_INFERENCE_SEED,
    result_policy as sam3_result_policy,
)
from qwen_material_pipeline.segmentation.human_foreground import (
    ANNOTATION_SCHEMA_VERSION as SAM3_HUMAN_ANNOTATION_SCHEMA_VERSION,
    CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION,
    CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL,
    CONFIRMED_MASK_STRICT_MINIMUM_IOU,
    CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU,
    LEGACY_ANNOTATION_SCHEMA_VERSION as SAM3_LEGACY_HUMAN_ANNOTATION_SCHEMA_VERSION,
    load_annotations as load_sam3_foreground_annotations,
    materialize_annotation_bundle as materialize_sam3_foreground_bundle,
    require_replay_policy as require_sam3_foreground_replay_policy,
)
from qwen_material_pipeline.segmentation.foreground_seeds import (
    SEED_POLICY_SCHEMA_VERSION,
    build_automatic_foreground_seeds,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST = PACKAGE_DIR / "configs" / "materials" / "industrial_whitelist.json"
# Parts with 64..255 visible pixels remain useful review candidates.  The
# mapping normalizer deterministically prevents them from becoming automatic
# matches, while 256+ pixels are eligible for ``matched``.
DEFAULT_MIN_VISIBLE_PIXELS = 64
MIN_MATCH_VISIBLE_PIXELS = 256
ISOLATED_EVIDENCE_SCHEMA_VERSION = "qwen-isolated-part-evidence/v1"
PALETTE_SELECTION_SCHEMA_VERSION = "qwen-palette-reference-selection/v1"
PALETTE_MERGE_AUDIT_SCHEMA_VERSION = "qwen-palette-equivalent-merge/v1"
PALETTE_FAILURE_SCHEMA_VERSION = "qwen-material-palette-failure/v1"
INFERENCE_FAILURE_SCHEMA_VERSION = "qwen-material-inference-failure/v1"
INFERENCE_FAILURE_FILENAME = "inference_failure.json"
MVINVERSE_MODES = ("off", "run", "reuse")
MVINVERSE_OUTPUT_DIRECTORY = "mvinverse"
PROGRESS_SCOPE = "qwen_material_pipeline"
QWEN_LEDGER_SCHEMA_VERSION = "qwen-local-inference-ledger/v2"
QWEN35_CANONICAL_REPOSITORY = "Qwen/Qwen3.5-4B"
QWEN35_CANONICAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN35_CONTENT_MANIFEST_SHA256 = (
    "c47626862a1e5417b71e3faf9954282efd4dba913cedde0d119d30797c1f81ad"
)
SAM3_REQUEST_SCHEMA_VERSION = "qwen-sam3-region-request/v1"
SAM3_POINT_REQUEST_SCHEMA_VERSION = "qwen-sam3-region-request/v2"
SAM3_ORDERED_POINT_REQUEST_SCHEMA_VERSION = "qwen-sam3-region-request/v3"
SAM3_RESULT_SCHEMA_VERSION = "qwen-sam3-region-result/v1"
VISUAL_RETRIEVAL_REQUEST_SCHEMA_VERSION = "qwen-visual-material-retrieval-request/v1"
VISUAL_RETRIEVAL_RESULT_SCHEMA_VERSION = "qwen-visual-material-retrieval-result/v1"
MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION = (
    "qwen-derived-material-selection-confidence/v1"
)
MATERIAL_SELECTION_REVIEW_CONFIDENCE = 0.70
MATERIAL_SELECTION_AUTO_CONFIDENCE = 0.90
MATERIAL_SELECTION_AUTO_MINIMUM_RETRIEVAL_MARGIN = 0.10
_UNRESOLVED_SEMANTICS = frozenset({"other", "unknown"})
_INTRINSIC_METAL_COLOR_TOKENS = {
    "orange": frozenset({"copper", "bronze", "brass"}),
    "brown": frozenset({"copper", "bronze", "brass"}),
    "yellow": frozenset({"brass", "bronze", "gold"}),
}
_BASE_PAINT_EQUIVALENCE_RE = re.compile(
    r"^mdl:(?P<root>Base/)?Miscellaneous/(?P<name>Paint_(?:Gloss|Matte|Satin))"
    r"(?:_Finish)?\.mdl#Paint_(?:Gloss|Matte|Satin)(?:_Finish)?$"
)
_BASE_PAINT_PRIMARY_RE = re.compile(
    r"^mdl:(?:Base/)?Miscellaneous/Paint_(Gloss|Matte|Satin)" r"\.mdl#Paint_\1$"
)
_SURFACE_INTERPRETATIONS = (
    "conversion_coating",
    "applied_paint",
    "bare_metal",
)
_COATING_PHYSICS_TEMPLATES = (
    "painted_engineering_metal",
    "generic_applied_paint",
    "conversion_coating",
)
_DISAGREEMENT_MVINVERSE_NEIGHBOR_LIMIT = 6
_DISAGREEMENT_PRIMARY_TOURNAMENT_BUDGET = 10


def _foreground_maximum_image_fraction(configured: float) -> float:
    """Return the single frozen SAM3 whole-object coverage policy."""

    return max(float(configured), 0.90)


class DisagreementTournamentCandidateError(StagedAnalysisError):
    """A deterministic exact-MDL disagreement cannot form a safe render set."""

    stage_name = "material_disagreement_tournament"
    reason = "no_independent_exact_default_challenger"


_FORCED_NUMERIC_SOLID_SURFACE_FAMILIES = frozenset(
    {
        "ceramic",
        "composite",
        "glass",
        "metal",
        "paint",
        "plastic",
        "rubber",
        "wood",
    }
)
_APPLIED_PAINT_TOKENS = frozenset(
    {
        "coat",
        "coated",
        "coating",
        "enamel",
        "lacquer",
        "paint",
        "painted",
        "powder",
    }
)
_CONVERSION_COATING_TOKENS = frozenset(
    {
        "anodised",
        "anodized",
        "blackened",
        "blued",
        "galvanized",
        "oxide",
        "oxidized",
        "passivated",
        "phosphate",
        "plated",
    }
)
_DARK_SURFACE_TOKENS = frozenset(
    {
        "black",
        "blackened",
        "blued",
        "carbon",
        "charcoal",
        "dark",
        "graphite",
        "iron",
    }
)
_NEUTRAL_ENGINEERING_METAL_TOKENS = frozenset(
    {
        "aluminium",
        "aluminum",
        "carbon",
        "chrome",
        "iron",
        "stainless",
        "steel",
    }
)
_SEMANTIC_BARE_FINISHES = frozenset({"bare", "brushed", "polished"})
_ROUGHNESS_FINISH_TOKENS = {
    "glossy": frozenset({"gloss", "glossy", "polished", "smooth"}),
    "satin": frozenset({"brushed", "satin"}),
    "matte": frozenset({"matte", "matt", "rough"}),
}
_VISIBLE_WEAR_TOKENS = frozenset(
    {
        "corroded",
        "cracked",
        "damaged",
        "dirty",
        "oxidized",
        "rusted",
        "scratched",
        "weathered",
        "worn",
    }
)
_VISIBLE_SPECIAL_EFFECT_TOKENS = _VISIBLE_WEAR_TOKENS | frozenset(
    {
        "antique",
        "chameleon",
        "crackle",
        "crinkled",
        "flakes",
        "hammer",
        "hammered",
        "imperfection",
        "imperfections",
        "monsterflakes",
        "patina",
        "patinated",
        "pebble",
        "pebbles",
        "shifting",
        "splotch",
        "splotches",
        "stroke",
        "strokes",
        "bump",
        "bumped",
        "bumpy",
    }
)
_NICHE_DOMAIN_TOKEN_GROUPS = {
    "electronics_surface": {
        "material_tokens": frozenset(
            {"circuit", "pcb", "semiconductor", "solder", "soldermask"}
        ),
        "reference_tokens": frozenset(
            {
                "board",
                "circuit",
                "electronic",
                "electronics",
                "pcb",
                "semiconductor",
                "solder",
            }
        ),
    },
    "automotive_finish": {
        "material_tokens": frozenset({"automotive", "carpaint", "vehicle"}),
        "reference_tokens": frozenset({"automotive", "car", "vehicle"}),
    },
}
_MIN_RELIABLE_PBR_VIEWS = 2
_MAX_RELIABLE_ALBEDO_MAD = 0.15


def _unsupported_niche_domains(
    item: dict[str, Any],
    *,
    material_tokens: set[str],
    reference_tokens: set[str],
) -> list[str]:
    """Return genuinely domain-specific presets unsupported by the reference.

    Catalog keyword lists are intentionally broad.  A general-purpose
    anodized aluminium material may mention ``automotive`` alongside AEC,
    interior, and industrial uses; treating that one keyword as proof that the
    MDL is car-paint-only removes otherwise valid fixed-library candidates.
    Electronics entries remain token-specific, while automotive exclusion is
    restricted to the Carpaint catalog branch or an explicit carpaint/vehicle
    material identity.
    """

    unsupported: list[str] = []
    category_path = str(item.get("category_path") or "").casefold()
    for domain, tokens in _NICHE_DOMAIN_TOKEN_GROUPS.items():
        if reference_tokens & tokens["reference_tokens"]:
            continue
        if domain == "automotive_finish":
            domain_specific = "/carpaint" in f"/{category_path}" or bool(
                material_tokens & {"carpaint", "vehicle"}
            )
        else:
            domain_specific = bool(material_tokens & tokens["material_tokens"])
        if domain_specific:
            unsupported.append(domain)
    return unsupported


_MAX_RELIABLE_SCALAR_MAD = 0.12
_DARK_ALBEDO_LUMINANCE_MAX = 0.35
_DIELECTRIC_METALLIC_MAX = 0.35
_CONDUCTIVE_METALLIC_MIN = 0.65


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def _write_json(path: str | Path, value: Any) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resolved


def _write_inference_failure(
    destination: Path,
    *,
    error_code: str,
    failed_stage: str,
    detail: str,
    view_failures: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> Path:
    """Atomically publish a deterministic, non-retryable child-stage failure."""

    path = destination / INFERENCE_FAILURE_FILENAME
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    document = {
        "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
        "status": "FAILED",
        "error_code": error_code,
        "failed_stage": failed_stage,
        "retryable": False,
        "retry_scope": "none",
        "detail": detail,
        "view_failures": view_failures,
    }
    if context is not None:
        document["context"] = context
    try:
        _write_json(temporary, document)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve(strict=True)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _isolated_retrieval_runtime_identity(python_executable: Path) -> dict[str, Any]:
    script = (
        "import json,platform,sys;"
        "from pathlib import Path;"
        "import numpy,PIL,torch,transformers;"
        "print(json.dumps({"
        "'executable':str(Path(sys.executable).resolve()),"
        "'python':platform.python_version(),"
        "'torch':str(torch.__version__),"
        "'torch_cuda':getattr(torch.version,'cuda',None),"
        "'transformers':str(transformers.__version__),"
        "'numpy':str(numpy.__version__),"
        "'pillow':str(PIL.__version__)"
        "},sort_keys=True))"
    )
    try:
        process = subprocess.run(
            [
                str(python_executable.expanduser().resolve(strict=True)),
                "-c",
                script,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        identity = json.loads(process.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Unable to verify the configured retrieval Python runtime"
        ) from exc
    if not isinstance(identity, dict):
        raise ValueError("Retrieval Python returned an invalid runtime identity")
    return identity


def _quarantine_incomplete_stage(path: Path) -> Path:
    """Preserve a pipeline-owned partial stage so the next run can continue."""

    for index in range(1, 1000):
        candidate = path.with_name(f".{path.name}.abandoned-{index:03d}")
        if not candidate.exists():
            path.replace(candidate)
            return candidate
    raise ValueError(f"Too many abandoned stage directories next to {path}")


def _atomic_stage_directory(final_directory: Path) -> Path:
    staging = final_directory.with_name(f".{final_directory.name}.incomplete")
    if staging.exists():
        _quarantine_incomplete_stage(staging)
    return staging


def _parse_id_path(value: str, label: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{label} must use ID=IMAGE syntax")
    identifier, raw_path = value.split("=", 1)
    identifier = identifier.strip()
    if not identifier:
        raise argparse.ArgumentTypeError(f"{label} ID cannot be empty")
    path = Path(raw_path.strip()).expanduser().resolve(strict=True)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{label} image is not a file: {path}")
    return identifier, path


def _parse_force_unknown(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--force-unknown must use PART_ID=REASON syntax"
        )
    part_id, reason = (item.strip() for item in value.split("=", 1))
    if not part_id or reason not in UNKNOWN_REASON_CODES:
        raise argparse.ArgumentTypeError(
            "invalid forced-unknown reason; allowed: "
            + ", ".join(sorted(UNKNOWN_REASON_CODES))
        )
    return part_id, reason


def _artifact_slug(identifier: str) -> str:
    """Return a readable, path-safe fragment without trusting a reference ID."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._")
    return (slug or "reference")[:64]


def _merge_equivalent_palette_groups(
    palette: dict[str, Any], evidence_audit: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Collapse duplicate appearance signatures before scoring or mapping.

    Small vision models sometimes describe the same appearance twice using
    different evidence boxes.  That must add evidence, not create artificial
    palette diversity.  A representative is chosen by highest post-filter
    confidence and then original order; boxes are deduplicated and capped at
    the schema limit of four.
    """

    canonical = validate_palette(palette)
    signature_fields = ("family_hint", "base_color", "finish_hint")
    clusters: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, group in enumerate(canonical["groups"]):
        signature = tuple(group[field] for field in signature_fields)
        clusters.setdefault(signature, []).append((index, group))

    audit_by_group = {
        item.get("group_id"): item
        for item in evidence_audit.get("groups", [])
        if isinstance(item, dict) and isinstance(item.get("group_id"), str)
    }
    output_groups: list[dict[str, Any]] = []
    normalized_audit_groups: list[dict[str, Any]] = []
    cluster_audits: list[dict[str, Any]] = []
    ordered_clusters = sorted(clusters.items(), key=lambda item: item[1][0][0])
    for signature, members in ordered_clusters:
        representative_index, representative = max(
            members,
            key=lambda item: (float(item[1]["confidence"]), -item[0]),
        )
        ordered_members = [
            (representative_index, representative),
            *[item for item in members if item[0] != representative_index],
        ]
        unique_box_records: list[tuple[list[int], str, dict[str, Any] | None]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()
        input_box_count = 0
        for _member_index, member in ordered_members:
            member_id = member["group_id"]
            source_box_audits = audit_by_group.get(member_id, {}).get("boxes", [])
            for box in member["boxes"]:
                input_box_count += 1
                box_key = tuple(box)
                if box_key in seen_boxes:
                    continue
                seen_boxes.add(box_key)
                matching_audit = next(
                    (
                        record
                        for record in source_box_audits
                        if isinstance(record, dict)
                        and record.get("accepted") is True
                        and record.get("box") == box
                    ),
                    None,
                )
                unique_box_records.append((list(box), member_id, matching_audit))

        retained_box_records = unique_box_records[:4]
        output_group = dict(representative)
        output_group["boxes"] = [record[0] for record in retained_box_records]
        output_groups.append(output_group)

        normalized_boxes: list[dict[str, Any]] = []
        for box_index, (box, source_group_id, source_audit) in enumerate(
            retained_box_records
        ):
            record = dict(source_audit or {})
            record.update(
                {
                    "box_index": box_index,
                    "box": box,
                    "accepted": True,
                    "source_group_id": source_group_id,
                }
            )
            normalized_boxes.append(record)
        normalized_audit_groups.append(
            {
                "group_id": representative["group_id"],
                "base_color": representative["base_color"],
                "accepted": True,
                "source_group_ids": [member["group_id"] for _index, member in members],
                "boxes": normalized_boxes,
            }
        )
        cluster_audits.append(
            {
                "signature": dict(zip(signature_fields, signature, strict=True)),
                "member_group_ids": [member["group_id"] for _index, member in members],
                "representative_group_id": representative["group_id"],
                "representative_input_index": representative_index,
                "representative_confidence": representative["confidence"],
                "representative_visual_description": representative[
                    "visual_description"
                ],
                "representative_rule": (
                    "highest_filtered_confidence_then_first_input_group"
                ),
                "input_box_count": input_box_count,
                "unique_box_count": len(unique_box_records),
                "retained_box_count": len(retained_box_records),
                "duplicate_box_count": input_box_count - len(unique_box_records),
                "truncated_box_count": max(0, len(unique_box_records) - 4),
                "retained_boxes": [record[0] for record in retained_box_records],
            }
        )

    normalized = validate_palette(
        {
            "schema_version": canonical["schema_version"],
            "source_view_id": canonical["source_view_id"],
            "groups": output_groups,
        }
    )
    normalized_evidence_audit = dict(evidence_audit)
    normalized_evidence_audit.update(
        {
            "accepted_group_ids": [group["group_id"] for group in output_groups],
            "groups": normalized_audit_groups,
            "equivalent_group_merge_applied": any(
                len(members) > 1 for members in clusters.values()
            ),
        }
    )
    merge_audit = {
        "schema_version": PALETTE_MERGE_AUDIT_SCHEMA_VERSION,
        "source_view_id": canonical["source_view_id"],
        "signature_fields": list(signature_fields),
        "input_group_count": len(canonical["groups"]),
        "unique_signature_count": len(output_groups),
        "output_group_count": len(output_groups),
        "merged_group_count": len(canonical["groups"]) - len(output_groups),
        "clusters": cluster_audits,
    }
    return normalized, normalized_evidence_audit, merge_audit


def _palette_quality_metrics(
    palette: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Compute a conservative deterministic score for one filtered palette.

    The actual selection uses the documented lexicographic key.  Group count
    comes first because every group has already survived independent pixel
    checks; color diversity, normalized pixel support, and model confidence
    then break ties.  Input order is the final stable tie-break outside this
    function.
    """

    groups = palette.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("palette quality requires at least one filtered group")
    audit_groups = {
        item.get("group_id"): item
        for item in audit.get("groups", [])
        if isinstance(item, dict) and isinstance(item.get("group_id"), str)
    }
    group_pixel_scores: list[float] = []
    supporting_pixel_count = 0
    accepted_box_count = 0
    for group in groups:
        group_audit = audit_groups.get(group.get("group_id"), {})
        box_scores: list[float] = []
        box_supported_counts: list[int] = []
        for box in group_audit.get("boxes", []):
            if not isinstance(box, dict) or not box.get("accepted"):
                continue
            accepted_box_count += 1
            sampled = box.get("sampled_pixels")
            matching = box.get("matching_pixel_count")
            foreground = box.get("foreground_pixels")
            if isinstance(matching, int) and not isinstance(matching, bool):
                supported = max(0, matching)
            elif isinstance(foreground, int) and not isinstance(foreground, bool):
                supported = max(0, foreground)
            else:
                supported = 0
            box_supported_counts.append(supported)
            if (
                isinstance(sampled, int)
                and not isinstance(sampled, bool)
                and sampled > 0
            ):
                box_scores.append(min(1.0, supported / sampled))
        # Multiple citations increase robustness but must not inflate view
        # quality simply by repeating or overlapping the same surface box.
        supporting_pixel_count += max(box_supported_counts, default=0)
        group_pixel_scores.append(max(box_scores, default=0.0))

    group_count = len(groups)
    color_diversity_count = len(
        {
            group.get("base_color")
            for group in groups
            if isinstance(group, dict) and isinstance(group.get("base_color"), str)
        }
    )
    pixel_evidence_score = sum(group_pixel_scores) / group_count
    confidences = [
        float(group.get("confidence", 0.0))
        for group in groups
        if isinstance(group, dict)
        and isinstance(group.get("confidence"), (int, float))
        and not isinstance(group.get("confidence"), bool)
    ]
    mean_confidence = sum(confidences) / group_count
    minimum_confidence = min(confidences, default=0.0)
    # This packed score is human-readable audit data.  Selection uses the
    # unrounded key below so floating-point packing cannot change precedence.
    quality_score = (
        group_count * 1000.0
        + color_diversity_count * 10.0
        + pixel_evidence_score
        + minimum_confidence / 100.0
    )
    return {
        "group_count": group_count,
        "color_diversity_count": color_diversity_count,
        "accepted_box_count": accepted_box_count,
        "supporting_sampled_pixel_count": supporting_pixel_count,
        "pixel_evidence_score": round(pixel_evidence_score, 8),
        "mean_confidence": round(mean_confidence, 8),
        "minimum_confidence": round(minimum_confidence, 8),
        "quality_score": round(quality_score, 8),
        "selection_key": [
            group_count,
            color_diversity_count,
            round(pixel_evidence_score, 12),
            supporting_pixel_count,
            round(minimum_confidence, 12),
            round(mean_confidence, 12),
        ],
    }


def _select_palette_candidate(
    candidates: list[dict[str, Any]], requested: str
) -> dict[str, Any]:
    """Select one usable per-view palette, or fail closed with a clear reason."""

    if requested != "auto":
        for candidate in candidates:
            if candidate["reference_id"] == requested:
                if candidate["status"] != "usable":
                    detail = candidate.get("error") or "filtered palette is empty"
                    raise StagedAnalysisError(
                        f"Requested palette reference {requested!r} is unusable: {detail}"
                    )
                return candidate
        raise ValueError(
            f"--palette-reference does not match a --reference ID: {requested}"
        )

    usable = [candidate for candidate in candidates if candidate["status"] == "usable"]
    if not usable:
        details = "; ".join(
            f"{candidate['reference_id']}: {candidate.get('error', 'unusable')}"
            for candidate in candidates
        )
        raise StagedAnalysisError(
            "No reference view produced a non-empty pixel-supported palette"
            + (f" ({details})" if details else "")
        )
    # max() returns the first candidate on an exact tie, preserving CLI order.
    return max(
        usable, key=lambda candidate: tuple(candidate["quality"]["selection_key"])
    )


def _mapping_verification_candidates(
    candidates: list[dict[str, Any]],
    *,
    primary_reference_id: str,
    maximum_views: int,
    eligible_view_ids: set[str],
) -> list[dict[str, Any]]:
    """Choose the strongest independent views without weakening consensus.

    The primary mapping already receives a multi-view support sheet.  When a
    bounded verification pass is requested, prefer the strongest non-primary
    views so the two required votes add genuinely different visual evidence.
    Exhaustive mode preserves the original CLI order for compatibility.
    """

    if maximum_views < 0 or maximum_views == 1:
        raise ValueError("maximum_views must be 0 or at least 2")
    usable = [
        candidate
        for candidate in candidates
        if candidate.get("status") == "usable"
        and candidate.get("reference_id") in eligible_view_ids
    ]
    if maximum_views == 0 or len(usable) <= maximum_views:
        return usable

    indexed = list(enumerate(usable))
    ranked = sorted(
        indexed,
        key=lambda item: (
            tuple(item[1].get("quality", {}).get("selection_key", [])),
            -item[0],
        ),
        reverse=True,
    )
    non_primary = [
        candidate
        for _index, candidate in ranked
        if candidate.get("reference_id") != primary_reference_id
    ]
    selected = non_primary[:maximum_views]
    if len(selected) < maximum_views:
        selected_ids = {candidate.get("reference_id") for candidate in selected}
        selected.extend(
            candidate
            for _index, candidate in ranked
            if candidate.get("reference_id") not in selected_ids
        )
    return selected[:maximum_views]


def _required_usable_palette_view_count(
    *,
    reference_count: int,
    minimum_views: int,
    minimum_ratio: float,
) -> int:
    """Resolve the absolute and proportional unattended evidence gates."""

    if reference_count < 1 or minimum_views < 1:
        raise ValueError("palette view counts must be positive")
    if not math.isfinite(minimum_ratio) or not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum palette view ratio must be between zero and one")
    return max(minimum_views, math.ceil(reference_count * minimum_ratio))


def _palette_failure_document(
    *,
    reference_id: str,
    image: str,
    stage: str,
    error: str,
    generation_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an audit-only failure record that cannot validate as a palette."""

    result = {
        "schema_version": PALETTE_FAILURE_SCHEMA_VERSION,
        "source_view_id": reference_id,
        "image": image,
        "palette_status": "unusable",
        "failure_stage": stage,
        "error": error,
    }
    if generation_attempts:
        result["generation_attempts"] = generation_attempts
    return result


def _reference_manifest_source_view(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Persist exactly one of a normalized palette or a failure artifact."""

    status = candidate.get("status")
    if not isinstance(status, str) or status not in {"usable", "unusable"}:
        raise ValueError(
            f"Unknown palette status for {candidate.get('reference_id')!r}: {status!r}"
        )
    record: dict[str, Any] = {
        "id": candidate["reference_id"],
        "image": candidate["image"],
        "palette_mask": candidate.get("mask"),
        "palette_status": status,
    }
    model_image = candidate.get("model_image", candidate["image"])
    if model_image != candidate["image"]:
        record["model_image"] = model_image
    if candidate.get("mask_authority") is not None:
        record["palette_mask_authority"] = candidate["mask_authority"]
    artifact_fields = {
        "model": "model_palette_path",
        "evidence_audit": "evidence_audit_path",
        "pixel_filtered": "pixel_filtered_palette_path",
        "accent_augmentation_audit": "accent_augmentation_audit_path",
        "normalized": "normalized_palette_path",
        "normalized_evidence_audit": "normalized_evidence_audit_path",
        "merge_audit": "merge_audit_path",
    }
    if status == "unusable":
        # Never publish a failed or stale derived file under a palette-bearing
        # key.  A valid model response and its audit may still aid diagnosis
        # when the later pixel-evidence stage failed.
        artifact_fields = {
            "model": "model_palette_path",
            "evidence_audit": "evidence_audit_path",
        }
    artifacts = {
        artifact_name: candidate[candidate_field]
        for artifact_name, candidate_field in artifact_fields.items()
        if isinstance(candidate.get(candidate_field), str)
        and candidate[candidate_field]
    }
    if artifacts:
        record["palette_artifacts"] = artifacts

    if status == "usable":
        palette_path = candidate.get("normalized_palette_path")
        if not isinstance(palette_path, str) or not palette_path:
            raise ValueError(
                f"Usable palette view {candidate['reference_id']!r} has no "
                "normalized palette"
            )
        record["palette_path"] = palette_path
    else:
        failure_path = candidate.get("palette_failure_artifact_path")
        if not isinstance(failure_path, str) or not failure_path:
            raise ValueError(
                f"Unusable palette view {candidate['reference_id']!r} has no "
                "failure artifact"
            )
        record["palette_failure_artifact"] = failure_path
    return record


def _legacy_palette_from_fusion(
    fusion: dict[str, Any],
    *,
    primary_view_id: str,
) -> dict[str, Any]:
    """Project the lossless fusion contract onto the legacy palette schema.

    The legacy schema has only one top-level ``source_view_id``.  For groups
    visible in the primary view we therefore retain that view's exact local
    boxes so primary mapping citations remain valid.  Groups unique to other
    views retain their deterministic representative boxes and are consumed by
    the per-view mapping path; they are never cited as primary-view evidence.
    """

    raw_groups = fusion.get("canonical_palette", {}).get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise StagedAnalysisError("multiview palette fusion produced no groups")
    groups: list[dict[str, Any]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise StagedAnalysisError("multiview palette fusion group is invalid")
        sources = raw_group.get("sources")
        if not isinstance(sources, list) or not sources:
            raise StagedAnalysisError("multiview palette group has no source evidence")
        primary_sources = [
            source
            for source in sources
            if isinstance(source, dict) and source.get("view_id") == primary_view_id
        ]
        representative = raw_group.get("representative_ref")
        representative_sources = [
            source
            for source in sources
            if isinstance(source, dict)
            and isinstance(representative, dict)
            and source.get("view_id") == representative.get("view_id")
            and source.get("local_group_id") == representative.get("local_group_id")
        ]
        source = (
            primary_sources[0]
            if primary_sources
            else representative_sources[0]
            if representative_sources
            else sources[0]
        )
        groups.append(
            {
                "group_id": raw_group["group_id"],
                "family_hint": raw_group["family_hint"],
                "base_color": raw_group["base_color"],
                "finish_hint": raw_group["finish_hint"],
                "visual_description": raw_group["visual_description"],
                "boxes": copy.deepcopy(source["boxes"]),
                "confidence": raw_group["confidence"],
            }
        )
    return validate_palette(
        {
            "schema_version": "qwen-material-palette/v1",
            "source_view_id": primary_view_id,
            "groups": groups,
        }
    )


def _multiview_pre_filter_group_count(
    usable_candidates: list[dict[str, Any]],
    *,
    canonical_palette: dict[str, Any],
) -> int:
    """Return a cross-view diversity count without counting views as groups.

    Summing each view's model-group count makes four photos of a truly
    single-material object look like four distinct pre-filter appearances.
    Instead, count distinct model colour families across views and never
    report fewer groups than survived into the canonical union.  This keeps
    the legacy collapse diagnostic meaningful without making it depend on the
    number of input cameras.
    """

    signatures: set[str] = set()
    for candidate in usable_candidates:
        model_palette = candidate.get("model_palette")
        if not isinstance(model_palette, dict):
            continue
        for group in model_palette.get("groups", []):
            if not isinstance(group, dict):
                continue
            color = group.get("base_color")
            if isinstance(color, str):
                signatures.add(fusion_color_label(color))
    return max(len(canonical_palette["groups"]), len(signatures))


def _canonicalize_primary_batches(
    batches: list[dict[str, Any]],
    *,
    local_to_canonical: dict[str, str],
    canonical_palette: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate primary local group IDs before multiview consensus."""

    canonical_groups = {
        group["group_id"]: group for group in canonical_palette["groups"]
    }
    output: list[dict[str, Any]] = []
    for raw_batch in batches:
        batch = copy.deepcopy(raw_batch)
        for mapping in batch.get("mappings", []):
            if not isinstance(mapping, dict) or mapping.get("status") == "unknown":
                continue
            local_group_id = mapping.get("group_id")
            canonical_group_id = local_to_canonical.get(str(local_group_id))
            if canonical_group_id is None:
                mapping.update(
                    {
                        "group_id": None,
                        "mapping_confidence": 0.0,
                        "evidence_view_id": None,
                        "evidence_box_index": None,
                        "status": "unknown",
                        "reason_code": "unmapped_local_palette_group",
                    }
                )
                continue
            box_index = mapping.get("evidence_box_index")
            box_count = len(canonical_groups[canonical_group_id]["boxes"])
            if (
                isinstance(box_index, bool)
                or not isinstance(box_index, int)
                or not 0 <= box_index < box_count
            ):
                mapping.update(
                    {
                        "group_id": None,
                        "mapping_confidence": 0.0,
                        "evidence_view_id": None,
                        "evidence_box_index": None,
                        "status": "unknown",
                        "reason_code": "canonical_evidence_box_unavailable",
                    }
                )
                continue
            mapping["group_id"] = canonical_group_id
        output.append(batch)
    return output


def _render_view_order(render_set: dict[str, Any]) -> list[str]:
    views = render_set.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Registry has no render views")
    view_ids: list[str] = []
    for view in views:
        view_id = view.get("view_id") if isinstance(view, dict) else None
        if not isinstance(view_id, str) or not view_id:
            raise ValueError("Registry contains an invalid render view ID")
        view_ids.append(view_id)
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("Registry contains duplicate render view IDs")
    return view_ids


def _best_evidence_render(
    part: dict[str, Any],
    *,
    render_view_order: list[str],
    preferred_view_id: str | None,
) -> dict[str, Any] | None:
    """Choose the available render with the greatest visible-pixel count.

    ``--cad-view`` remains useful as a deterministic tie-breaker, but it no
    longer forces every part to use one global view.
    """

    view_rank = {view_id: index for index, view_id in enumerate(render_view_order)}
    candidates: list[tuple[int, bool, int, dict[str, Any]]] = []
    for render in part.get("renders", []):
        if not isinstance(render, dict):
            continue
        view_id = render.get("view_id")
        pixels = render.get("visible_pixels")
        if (
            not isinstance(view_id, str)
            or view_id not in view_rank
            or isinstance(pixels, bool)
            or not isinstance(pixels, int)
        ):
            continue
        candidates.append(
            (
                max(0, pixels),
                view_id == preferred_view_id,
                -view_rank[view_id],
                render,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda value: (value[0], value[1], value[2]))[3]


def _calibrated_visible_pixels_by_part(
    render_set: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Read current Part-ID visibility from continuous calibrated renders.

    A calibrated registry intentionally keeps the original per-part render
    records because they are the provenance for neutral isolated geometry
    crops.  Those records belong to the source pose bank, however, and must
    not be used as visibility measurements for the newly calibrated cameras.
    """

    visible_by_part: dict[str, dict[str, int]] = {}
    for view in render_set["views"]:
        view_id = view["view_id"]
        visible_parts = view.get("visible_parts")
        if not isinstance(visible_parts, list):
            raise ValueError(
                f"Continuous calibrated render {view_id!r} has no visible_parts list"
            )
        seen: set[str] = set()
        for item in visible_parts:
            part_id = item.get("part_id") if isinstance(item, dict) else None
            pixels = item.get("pixels") if isinstance(item, dict) else None
            if (
                not isinstance(part_id, str)
                or not part_id
                or part_id in seen
                or isinstance(pixels, bool)
                or not isinstance(pixels, int)
                or pixels < 0
            ):
                raise ValueError(
                    f"Continuous calibrated render {view_id!r} contains invalid "
                    "visible_parts"
                )
            seen.add(part_id)
            visible_by_part.setdefault(part_id, {})[view_id] = pixels
    return visible_by_part


def _isolated_evidence_summary(part: dict[str, Any]) -> dict[str, Any] | None:
    """Validate renderer-authored isolated evidence without inflating source pixels."""

    evidence = part.get("isolated_evidence")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError(
            f"Part {part.get('part_id')!r} isolated_evidence must be an object"
        )
    if evidence.get("schema_version") != ISOLATED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Part {part.get('part_id')!r} has unsupported isolated evidence"
        )
    selected = evidence.get("selected_view_ids")
    source_by_view = evidence.get("source_visible_pixels_by_view")
    normalized_by_view = evidence.get("normalized_visible_pixels_by_view")
    source_floor = evidence.get("source_pixel_floor")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not all(isinstance(value, str) and value for value in selected)
        or not isinstance(source_by_view, dict)
        or not isinstance(normalized_by_view, dict)
        or set(source_by_view) != set(selected)
        or set(normalized_by_view) != set(selected)
        or isinstance(source_floor, bool)
        or not isinstance(source_floor, int)
        or source_floor < 1
        or evidence.get("material_neutralized") is not True
        or evidence.get("background_removed") is not True
    ):
        raise ValueError(f"Part {part.get('part_id')!r} isolated evidence is malformed")
    for label, values in (
        ("source", source_by_view),
        ("normalized", normalized_by_view),
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError(
                f"Part {part.get('part_id')!r} isolated {label} pixels are invalid"
            )
    raw_by_view = {
        render["view_id"]: render["visible_pixels"]
        for render in part.get("renders", [])
        if isinstance(render, dict)
        and isinstance(render.get("view_id"), str)
        and isinstance(render.get("visible_pixels"), int)
        and not isinstance(render.get("visible_pixels"), bool)
        and render["visible_pixels"] >= 0
    }
    if any(
        view_id not in raw_by_view or source_by_view[view_id] != raw_by_view[view_id]
        for view_id in selected
    ):
        raise ValueError(
            f"Part {part.get('part_id')!r} isolated/source visibility differs"
        )
    source_max = max(source_by_view.values())
    normalized_max = max(normalized_by_view.values())
    eligible_views = sorted(
        view_id for view_id, pixels in source_by_view.items() if pixels >= source_floor
    )
    if (
        evidence.get("source_max_visible_pixels") != source_max
        or evidence.get("normalized_max_visible_pixels") != normalized_max
        or evidence.get("source_evidence_view_count") != len(eligible_views)
        or evidence.get("source_evidence_view_ids") != eligible_views
    ):
        raise ValueError(
            f"Part {part.get('part_id')!r} isolated evidence summary is inconsistent"
        )
    digest = evidence.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            f"Part {part.get('part_id')!r} isolated evidence SHA256 is invalid"
        )
    path = evidence.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"Part {part.get('part_id')!r} isolated evidence path is invalid"
        )
    return evidence


def _assign_best_evidence_views(
    parts: list[dict[str, Any]],
    *,
    render_set: dict[str, Any],
    preferred_view_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Annotate every part with its highest-visibility CAD evidence view."""

    view_order = _render_view_order(render_set)
    if preferred_view_id is not None and preferred_view_id not in view_order:
        raise ValueError(f"Registry has no render view: {preferred_view_id}")
    continuous_calibration = (
        render_set.get("continuous_camera_calibration") is True
    )
    calibrated_visibility = (
        _calibrated_visible_pixels_by_part(render_set)
        if continuous_calibration
        else {}
    )
    view_rank = {view_id: index for index, view_id in enumerate(view_order)}
    evidence_by_part: dict[str, dict[str, Any]] = {}
    for part in parts:
        part_id = part["part_id"]
        current_pixels_by_view: dict[str, int]
        if continuous_calibration:
            current_pixels_by_view = calibrated_visibility.get(part_id, {})
            view_id = max(
                view_order,
                key=lambda candidate: (
                    current_pixels_by_view.get(candidate, 0),
                    candidate == preferred_view_id,
                    -view_rank[candidate],
                ),
            )
            pixels = current_pixels_by_view.get(view_id, 0)
        else:
            best = _best_evidence_render(
                part,
                render_view_order=view_order,
                preferred_view_id=preferred_view_id,
            )
            view_id = best.get("view_id") if best is not None else None
            pixels = (
                max(0, best["visible_pixels"])
                if best is not None
                and isinstance(best.get("visible_pixels"), int)
                and not isinstance(best.get("visible_pixels"), bool)
                else 0
            )
            current_pixels_by_view = {
                render["view_id"]: max(0, render["visible_pixels"])
                for render in part.get("renders", [])
                if isinstance(render, dict)
                and render.get("view_id") in view_order
                and isinstance(render.get("visible_pixels"), int)
                and not isinstance(render.get("visible_pixels"), bool)
            }
        isolated = _isolated_evidence_summary(part)
        isolated_source_pixels = (
            int(isolated["source_max_visible_pixels"]) if isolated is not None else 0
        )
        if (
            not continuous_calibration
            and isolated is not None
            and isolated_source_pixels != pixels
        ):
            raise ValueError(
                f"Part {part_id} isolated/source visibility does not match "
                "its selected CAD render"
            )
        isolated_eligible = (
            isolated is not None
            and isolated_source_pixels >= int(isolated["source_pixel_floor"])
            and int(isolated["normalized_max_visible_pixels"])
            >= DEFAULT_MIN_VISIBLE_PIXELS
        )
        effective_pixels = (
            int(isolated["normalized_max_visible_pixels"])
            if isolated_eligible
            else pixels
        )
        part["evidence_cad_view_id"] = view_id
        part["evidence_source_visible_pixels"] = pixels
        part["evidence_visible_pixels"] = effective_pixels
        part["evidence_mode"] = (
            "isolated_mask_multiview" if isolated_eligible else "source_projection"
        )
        part["evidence_source_view_count"] = (
            sum(
                1
                for candidate in view_order
                if current_pixels_by_view.get(candidate, 0)
                >= DEFAULT_MIN_VISIBLE_PIXELS
            )
            if continuous_calibration
            else (
                int(isolated["source_evidence_view_count"])
                if isolated_eligible
                else sum(
                    1
                    for render in part.get("renders", [])
                    if isinstance(render, dict)
                    and render.get("view_id") in view_order
                    and isinstance(render.get("visible_pixels"), int)
                    and not isinstance(render.get("visible_pixels"), bool)
                    and render["visible_pixels"] >= DEFAULT_MIN_VISIBLE_PIXELS
                )
            )
        )
        evidence_by_part[part_id] = {
            "cad_view_id": view_id,
            # Legacy field now names the effective model-evidence size.  The
            # original projection count is always retained separately.
            "visible_pixels": effective_pixels,
            "source_visible_pixels": pixels,
            "effective_visible_pixels": effective_pixels,
            "evidence_mode": part["evidence_mode"],
            "source_evidence_view_count": part["evidence_source_view_count"],
            "isolated_evidence_sha256": (
                isolated.get("sha256") if isolated_eligible else None
            ),
        }
    return evidence_by_part


def _dot(left: list[float], right: list[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _spatial_part_batches(
    parts: list[dict[str, Any]],
    *,
    render_set: dict[str, Any],
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    if not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between 1 and 4")
    basis = render_set.get("analysis_basis_world", {})
    up = basis.get("up", [0.0, 0.0, 1.0])
    right = basis.get("right", [1.0, 0.0, 0.0])

    def key(part: dict[str, Any]) -> tuple[float, float, str]:
        bbox = part.get("world_bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 2
            or any(not isinstance(point, list) or len(point) != 3 for point in bbox)
        ):
            return (0.0, 0.0, part["part_id"])
        center = [
            (float(bbox[0][index]) + float(bbox[1][index])) * 0.5 for index in range(3)
        ]
        return (-_dot(center, up), _dot(center, right), part["part_id"])

    ordered = sorted(parts, key=key)
    return [
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    ]


def _view_grouped_part_batches(
    parts: list[dict[str, Any]],
    *,
    render_set: dict[str, Any],
    batch_size: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Batch spatially without ever mixing different evidence CAD views."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for part in parts:
        view_id = part.get("evidence_cad_view_id")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(
                f"Target part {part.get('part_id')!r} has no evidence CAD view"
            )
        grouped.setdefault(view_id, []).append(part)

    view_order = _render_view_order(render_set)
    unknown_views = set(grouped) - set(view_order)
    if unknown_views:
        raise ValueError(
            "Target parts reference unknown CAD views: "
            + ", ".join(sorted(unknown_views))
        )
    result: list[tuple[str, list[dict[str, Any]]]] = []
    for view_id in view_order:
        view_parts = grouped.get(view_id, [])
        for batch in _spatial_part_batches(
            view_parts, render_set=render_set, batch_size=batch_size
        ):
            result.append((view_id, batch))
    return result


def _make_batch_sheet(
    part_ids: list[str],
    best_highlights: dict[str, str],
    output_path: Path,
    *,
    cell_size: int = 360,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    columns = 2
    rows = 2
    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), (236, 236, 236))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except OSError:
        font = ImageFont.load_default()
    for index, part_id in enumerate(part_ids):
        source = Path(best_highlights[part_id]).expanduser().resolve(strict=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        target = cell_size - 20
        scale = min(target / image.width, target / image.height)
        resized = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        col = index % columns
        row = index // columns
        x = col * cell_size + (cell_size - resized.width) // 2
        y = row * cell_size + (cell_size - resized.height) // 2
        canvas.paste(resized, (x, y))
        draw.rectangle(
            (
                col * cell_size + 4,
                row * cell_size + 4,
                col * cell_size + 116,
                row * cell_size + 38,
            ),
            fill=(20, 20, 20),
        )
        draw.text(
            (col * cell_size + 12, row * cell_size + 20),
            part_id,
            fill=(255, 255, 255),
            font=font,
            anchor="lm",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path.resolve()


def _make_reference_sheet(
    references: list[tuple[str, Path]],
    output_path: Path,
    *,
    cell_size: int = 512,
) -> Path:
    """Pack 2..4 unordered views of the same asset into one bounded image."""

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if not 2 <= len(references) <= 4:
        raise ValueError("reference sheet requires 2..4 views")
    columns = 2
    rows = math.ceil(len(references) / columns)
    header_height = 44
    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()
    for index, (view_id, source) in enumerate(references):
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        available_width = cell_size - 16
        available_height = cell_size - header_height - 12
        scale = min(
            available_width / image.width,
            available_height / image.height,
        )
        resized = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        cell_left = column * cell_size
        cell_top = row * cell_size
        x = cell_left + (cell_size - resized.width) // 2
        y = cell_top + header_height + (available_height - resized.height) // 2
        canvas.paste(resized, (x, y))
        draw.rectangle(
            (cell_left, cell_top, cell_left + cell_size - 1, cell_top + header_height),
            fill=(28, 28, 28),
        )
        draw.text(
            (cell_left + 12, cell_top + header_height // 2),
            f"SAME ASSET VIEW {index + 1}: {view_id}",
            fill=(255, 255, 255),
            font=font,
            anchor="lm",
        )
        draw.rectangle(
            (
                cell_left,
                cell_top,
                cell_left + cell_size - 1,
                cell_top + cell_size - 1,
            ),
            outline=(100, 100, 100),
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path.resolve()


def _reference_group_crop(
    reference_path: Path,
    group: dict[str, Any],
    output_path: Path,
    *,
    mask_path: Path | None = None,
) -> Path:
    from PIL import Image, ImageOps

    with Image.open(reference_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    mask = None
    if mask_path is not None:
        with Image.open(mask_path) as opened:
            mask = ImageOps.exif_transpose(opened).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        bbox = mask.point(lambda value: 255 if value >= 128 else 0).getbbox()
        if bbox is None:
            mask.close()
            image.close()
            raise StagedAnalysisError(f"accepted group mask is empty: {mask_path}")
        left, top, right, bottom = bbox
    else:
        boxes = group.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            image.close()
            raise StagedAnalysisError("group crop requires boxes or an accepted mask")
        box = max(
            boxes,
            key=lambda value: (value[2] - value[0]) * (value[3] - value[1]),
        )
        x0, y0, x1, y1 = box
        left = int(math.floor(x0 * image.width / 1000))
        top = int(math.floor(y0 * image.height / 1000))
        right = int(math.ceil(x1 * image.width / 1000))
        bottom = int(math.ceil(y1 * image.height / 1000))
    margin = max(6, int(max(right - left, bottom - top) * 0.08))
    crop_box = (
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    )
    crop = image.crop(crop_box)
    if mask is not None:
        mask_crop = mask.crop(crop_box)
        neutral = Image.new("RGB", crop.size, (127, 127, 127))
        neutral.paste(crop, mask=mask_crop)
        crop.close()
        mask_crop.close()
        mask.close()
        crop = neutral
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    crop.close()
    image.close()
    return output_path.resolve()


def _reference_group_multiview_evidence(
    *,
    group_id: str,
    fusion_group: dict[str, Any],
    reference_paths: dict[str, Path],
    output_path: Path,
    masks: dict[tuple[str, str], Path] | None = None,
    require_masks: bool = False,
) -> Path:
    """Build one order-stable sheet from independently observed group crops.

    Material identity is frequently ambiguous in a single tight crop (for
    example a copper tube versus orange paint).  Palette fusion already proves
    which local regions represent the same canonical appearance across views;
    reuse only those cited regions here.  This adds no semantic guess and keeps
    the material decision generic and unattended.
    """

    raw_sources = fusion_group.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise StagedAnalysisError(
            f"fusion group {group_id} has no material-evidence sources"
        )
    representative = fusion_group.get("representative_ref")
    representative_key = (
        (
            representative.get("view_id"),
            representative.get("local_group_id"),
        )
        if isinstance(representative, dict)
        else (None, None)
    )
    candidates: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
    seen_view_ids: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        view_id = source.get("view_id")
        local_group_id = source.get("local_group_id")
        boxes = source.get("boxes")
        confidence = source.get("confidence")
        if (
            not isinstance(view_id, str)
            or view_id not in reference_paths
            or not isinstance(local_group_id, str)
            or not isinstance(boxes, list)
            or not boxes
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
        ):
            continue
        if require_masks and (view_id, group_id) not in (masks or {}):
            continue
        maximum_box_area = max(
            (
                int(box[2] - box[0]) * int(box[3] - box[1])
                for box in boxes
                if isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, int) for value in box)
            ),
            default=0,
        )
        if maximum_box_area <= 0:
            continue
        candidates.append(
            (
                (
                    0 if (view_id, local_group_id) == representative_key else 1,
                    -float(confidence),
                    -maximum_box_area,
                    view_id,
                    local_group_id,
                ),
                view_id,
                source,
            )
        )
    selected: list[tuple[str, dict[str, Any]]] = []
    for _rank, view_id, source in sorted(candidates, key=lambda item: item[0]):
        if view_id in seen_view_ids:
            continue
        seen_view_ids.add(view_id)
        selected.append((view_id, source))
        if len(selected) == 4:
            break
    if not selected:
        raise StagedAnalysisError(
            f"fusion group {group_id} has no valid material-evidence crop"
        )

    crop_dir = output_path.parent / f"{group_id}_views"
    crops: list[tuple[str, Path]] = []
    for index, (view_id, source) in enumerate(selected, start=1):
        safe_view_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", view_id).strip("._")
        crop_path = _reference_group_crop(
            reference_paths[view_id],
            {"boxes": source["boxes"]},
            crop_dir / f"{index:02d}_{safe_view_id or 'view'}.png",
            mask_path=(masks or {}).get((view_id, group_id)),
        )
        crops.append((view_id, crop_path))
    if len(crops) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(crops[0][1].read_bytes())
        return output_path.resolve()
    return _make_reference_sheet(crops, output_path)


def _annotate_reference_palette(
    reference_path: Path,
    palette: dict[str, Any],
    output_path: Path,
) -> Path:
    """Draw deterministic group labels so localization does not rely on JSON coordinates."""

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    with Image.open(reference_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, min(image.width, image.height) // 28),
        )
    except OSError:
        font = ImageFont.load_default()
    colors = (
        (255, 48, 48),
        (45, 150, 255),
        (255, 190, 35),
        (210, 55, 255),
        (30, 210, 120),
        (255, 105, 180),
    )
    for group_index, group in enumerate(palette["groups"]):
        color = colors[group_index % len(colors)]
        label = f"{group['group_id']} {group['base_color']} {group['finish_hint']}"
        for box_index, box in enumerate(group["boxes"]):
            x0, y0, x1, y1 = box
            rectangle = (
                int(round(x0 * image.width / 1000)),
                int(round(y0 * image.height / 1000)),
                int(round(x1 * image.width / 1000)),
                int(round(y1 * image.height / 1000)),
            )
            draw.rectangle(rectangle, outline=color, width=max(2, image.width // 180))
            text = label if box_index == 0 else f"{group['group_id']} box {box_index}"
            text_box = draw.textbbox((rectangle[0], rectangle[1]), text, font=font)
            draw.rectangle(text_box, fill=(0, 0, 0))
            draw.text((rectangle[0], rectangle[1]), text, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path.resolve()


def _material_tokens(record: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for key in (
        "display_name",
        "description",
        "family",
        "material_id",
        "mdl_path",
        "sub_identifier",
        "category_path",
    ):
        value = record.get(key)
        if isinstance(value, str):
            values.append(value)
    for key in ("keywords", "colors", "finishes"):
        value = record.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return {
        token.casefold()
        for value in values
        for token in value.replace("-", " ").replace("_", " ").split()
        if token
    }


def _surface_interpretation(record: dict[str, Any]) -> str:
    """Classify one catalog entry by library semantics, never by part identity."""

    tokens = _material_tokens(record)
    family = record.get("family")
    # A conversion treatment can also be described as "coated".  Test the
    # physically specific metal treatment before the generic coating token so
    # anodized/plated metals cannot masquerade as ordinary paint.
    if family == "metal" and tokens & _CONVERSION_COATING_TOKENS:
        return "conversion_coating"
    if family == "paint" or tokens & _APPLIED_PAINT_TOKENS:
        return "applied_paint"
    if family == "metal":
        return "bare_metal"
    return "other"


def _coating_physics_template(record: dict[str, Any]) -> str | None:
    """Return a substrate-aware coating hypothesis for bounded retrieval."""

    tokens = _material_tokens(record)
    family = record.get("family")
    if family == "metal" and tokens & _CONVERSION_COATING_TOKENS:
        return "conversion_coating"
    if family == "metal" and "steel" in tokens and tokens & _APPLIED_PAINT_TOKENS:
        return "painted_engineering_metal"
    if family == "paint":
        return "generic_applied_paint"
    return None


def _unit_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    return float(value)


def _reliable_scalar_stat(
    evidence: dict[str, Any] | None,
    channel: str,
) -> tuple[float | None, bool]:
    if not isinstance(evidence, dict):
        return None, False
    stats = evidence.get(channel)
    distinct_views = evidence.get("distinct_view_count")
    if (
        not isinstance(stats, dict)
        or isinstance(distinct_views, bool)
        or not isinstance(distinct_views, int)
        or distinct_views < _MIN_RELIABLE_PBR_VIEWS
        or stats.get("sample_count") != distinct_views
    ):
        return None, False
    median = _unit_number(stats.get("median"))
    mad = _unit_number(stats.get("mad"))
    iqr = _unit_number(stats.get("iqr"))
    reliable = (
        median is not None
        and mad is not None
        and iqr is not None
        and mad <= _MAX_RELIABLE_SCALAR_MAD
        and iqr <= 0.20
    )
    return median, reliable


def _reliable_albedo_stat(
    evidence: dict[str, Any] | None,
) -> tuple[list[float] | None, bool]:
    if not isinstance(evidence, dict):
        return None, False
    stats = evidence.get("albedo")
    distinct_views = evidence.get("distinct_view_count")
    if (
        not isinstance(stats, dict)
        or isinstance(distinct_views, bool)
        or not isinstance(distinct_views, int)
        or distinct_views < _MIN_RELIABLE_PBR_VIEWS
        or stats.get("sample_count") != distinct_views
    ):
        return None, False
    raw_median = stats.get("median")
    raw_mad = stats.get("mad")
    if (
        not isinstance(raw_median, list)
        or len(raw_median) != 3
        or not isinstance(raw_mad, list)
        or len(raw_mad) != 3
    ):
        return None, False
    median = [_unit_number(value) for value in raw_median]
    mad = [_unit_number(value) for value in raw_mad]
    if None in median or None in mad:
        return None, False
    values = [float(value) for value in median]
    reliable = max(float(value) for value in mad) <= _MAX_RELIABLE_ALBEDO_MAD
    return values, reliable


def _roughness_class(value: float | None, *, reliable: bool) -> str:
    if not reliable or value is None:
        return "unknown"
    if value <= 0.30:
        return "glossy"
    if value >= 0.58:
        return "matte"
    return "satin"


def _metallicity_class(value: float | None, *, reliable: bool) -> str:
    if not reliable or value is None:
        return "unknown"
    if value <= _DIELECTRIC_METALLIC_MAX:
        return "dielectric"
    if value >= _CONDUCTIVE_METALLIC_MIN:
        return "conductive"
    return "ambiguous"


def _semantic_surface_class(
    group: dict[str, Any],
    *,
    canonical_finish: str,
    finish_reliable: bool,
    description_reliable: bool,
) -> str:
    description_tokens = (
        {
            token.casefold()
            for token in str(group.get("visual_description") or "")
            .replace("-", " ")
            .replace("_", " ")
            .split()
        }
        if description_reliable
        else set()
    )
    if (
        finish_reliable
        and canonical_finish == "painted"
        or description_tokens & (_APPLIED_PAINT_TOKENS | _CONVERSION_COATING_TOKENS)
    ):
        return "coating"
    if (
        finish_reliable
        and canonical_finish in _SEMANTIC_BARE_FINISHES
        or {"bare", "unpainted", "exposed"} & description_tokens
    ):
        return "bare"
    return "ambiguous"


def _surface_candidate_context(
    group: dict[str, Any],
    *,
    canonical_finish: str,
    finish_reliable: bool,
    description_reliable: bool,
    family_reliable: bool,
    mvinverse_pbr_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    albedo, albedo_reliable = _reliable_albedo_stat(mvinverse_pbr_evidence)
    metallic, metallic_reliable = _reliable_scalar_stat(
        mvinverse_pbr_evidence, "metallic"
    )
    roughness, roughness_reliable = _reliable_scalar_stat(
        mvinverse_pbr_evidence, "roughness"
    )
    luminance = (
        0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]
        if albedo_reliable and albedo is not None
        else None
    )
    semantic_class = _semantic_surface_class(
        group,
        canonical_finish=canonical_finish,
        finish_reliable=finish_reliable,
        description_reliable=description_reliable,
    )
    metallicity = _metallicity_class(metallic, reliable=metallic_reliable)
    dark_multiview_color = (
        luminance is not None and luminance <= _DARK_ALBEDO_LUMINANCE_MAX
    )
    semantic_numeric_conflict = (
        semantic_class == "coating"
        and metallicity == "conductive"
        or semantic_class == "bare"
        and metallicity == "dielectric"
    )
    return {
        "family_reliable": family_reliable,
        "semantic_surface_class": semantic_class,
        "mvinverse_surface_class": (
            mvinverse_pbr_evidence.get("surface_class")
            if isinstance(mvinverse_pbr_evidence, dict)
            else None
        ),
        "multi_view_albedo_reliable": albedo_reliable,
        "albedo_median": albedo if albedo_reliable else None,
        "albedo_luminance": luminance,
        "dark_multiview_color": dark_multiview_color,
        "metallic_reliable": metallic_reliable,
        "observed_metallic": metallic,
        "metallicity_class": metallicity,
        "roughness_reliable": roughness_reliable,
        "observed_roughness": roughness,
        "roughness_class": _roughness_class(roughness, reliable=roughness_reliable),
        "semantic_numeric_conflict": semantic_numeric_conflict,
    }


def _shortlist_materials(
    group: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    candidates, _audit = _shortlist_materials_with_audit(
        group,
        pool,
        limit=limit,
    )
    return candidates


def _material_selection_context(
    group: dict[str, Any],
    fusion_group: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a selector-only group without promoting weak finish semantics.

    The canonical palette remains the lossless evidence record.  This helper
    creates a separate context for catalog retrieval and Qwen selection so a
    single review-confidence finish label cannot outweigh several unresolved
    views.  A resolved finish is trusted when two independent views support it
    or one supporting source reaches the automatic-confidence threshold.
    """

    raw_confidence = group.get("confidence")
    if (
        isinstance(raw_confidence, bool)
        or not isinstance(raw_confidence, (int, float))
        or not math.isfinite(float(raw_confidence))
        or not 0.0 <= float(raw_confidence) <= 1.0
    ):
        raise ValueError("material selection group confidence must be in [0, 1]")
    confidence = float(raw_confidence)
    finish = str(group["finish_hint"])

    supporting_view_ids: set[str] = set()
    conflicting_view_ids: set[str] = set()
    conflicting_values: set[str] = set()
    maximum_support_confidence = 0.0
    sources = fusion_group.get("sources") if isinstance(fusion_group, dict) else None
    if isinstance(sources, list) and sources:
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValueError(f"fusion material source {index} must be an object")
            view_id = source.get("view_id")
            source_finish = source.get("finish_hint")
            source_confidence = source.get("confidence")
            if (
                not isinstance(view_id, str)
                or not view_id
                or not isinstance(source_finish, str)
                or isinstance(source_confidence, bool)
                or not isinstance(source_confidence, (int, float))
                or not math.isfinite(float(source_confidence))
                or not 0.0 <= float(source_confidence) <= 1.0
            ):
                raise ValueError(
                    f"fusion material source {index} has invalid finish evidence"
                )
            if source_finish in _UNRESOLVED_SEMANTICS:
                continue
            if source_finish == finish:
                supporting_view_ids.add(view_id)
                maximum_support_confidence = max(
                    maximum_support_confidence,
                    float(source_confidence),
                )
            else:
                conflicting_view_ids.add(view_id)
                conflicting_values.add(source_finish)
    elif finish not in _UNRESOLVED_SEMANTICS:
        supporting_view_ids.add("canonical")
        maximum_support_confidence = confidence

    multiview_confirmed = len(supporting_view_ids) >= 2
    high_confidence_confirmed = maximum_support_confidence >= AUTO_THRESHOLD
    finish_reliable = (
        finish not in _UNRESOLVED_SEMANTICS
        and not conflicting_view_ids
        and (multiview_confirmed or high_confidence_confirmed)
    )
    description_reliable = finish_reliable and confidence >= AUTO_THRESHOLD

    reason_codes: list[str] = []
    if finish in _UNRESOLVED_SEMANTICS:
        reason_codes.append("finish_hint_already_unresolved")
    if conflicting_view_ids:
        reason_codes.append("resolved_finish_conflict")
    if (
        finish not in _UNRESOLVED_SEMANTICS
        and not conflicting_view_ids
        and not supporting_view_ids
    ):
        reason_codes.append("no_resolved_finish_support")
    if (
        len(supporting_view_ids) == 1
        and not high_confidence_confirmed
        and not conflicting_view_ids
    ):
        reason_codes.append("single_review_confidence_finish_source")
    if multiview_confirmed and not conflicting_view_ids:
        reason_codes.append("finish_confirmed_by_independent_views")
    elif high_confidence_confirmed and not conflicting_view_ids:
        reason_codes.append("finish_confirmed_by_high_confidence_source")
    if confidence < AUTO_THRESHOLD:
        reason_codes.append("visual_description_below_auto_confidence")
    if not finish_reliable:
        reason_codes.append("visual_description_depends_on_unreliable_finish")

    selection_group = copy.deepcopy(group)
    if not finish_reliable:
        selection_group["finish_hint"] = "other"
    if not description_reliable:
        reliable_terms = [str(group["base_color"])]
        family = str(group["family_hint"])
        if family not in _UNRESOLVED_SEMANTICS:
            reliable_terms.append(family)
        if finish_reliable:
            reliable_terms.append(finish)
        selection_group["visual_description"] = (
            " ".join(reliable_terms)
            + " appearance region; finer physical material identity is "
            "unresolved in palette metadata"
        )

    reliability = {
        "policy": {
            "automatic_confidence_threshold": AUTO_THRESHOLD,
            "review_confidence_threshold": REVIEW_THRESHOLD,
            "minimum_independent_support_views": 2,
            "unresolved_values": sorted(_UNRESOLVED_SEMANTICS),
        },
        "finish_hint": {
            "canonical_value": finish,
            "selection_value": selection_group["finish_hint"],
            "reliable": finish_reliable,
            "supporting_view_ids": sorted(supporting_view_ids),
            "conflicting_view_ids": sorted(conflicting_view_ids),
            "conflicting_values": sorted(conflicting_values),
            "maximum_support_confidence": maximum_support_confidence,
            "multiview_confirmed": multiview_confirmed,
            "high_confidence_confirmed": high_confidence_confirmed,
        },
        "visual_description": {
            "canonical_value": str(group["visual_description"]),
            "selection_value": selection_group["visual_description"],
            "reliable": description_reliable,
            "canonical_confidence": confidence,
            "requires_reliable_finish": True,
        },
        "selection_context_modified": selection_group != group,
        "canonical_group_preserved": True,
        "reason_codes": reason_codes,
    }
    return selection_group, reliability


def _family_hint_is_reliable(
    group: dict[str, Any],
    fusion_group: dict[str, Any] | None,
) -> bool:
    """Return whether independent views establish a dominant material family.

    A single contradictory view must not erase a consistent multi-view
    majority.  Perspective, occlusion, and dielectric coatings commonly make
    one view of painted metal look like plastic.  We therefore require the
    canonical family to have at least two independent supporting views and to
    beat every alternative by both view count and summed confidence.  Ties and
    one-view margins remain unresolved and retain the broader catalog pool.
    """

    family = group.get("family_hint")
    if not isinstance(family, str) or family in _UNRESOLVED_SEMANTICS:
        return False
    raw_confidence = group.get("confidence")
    confidence = (
        float(raw_confidence)
        if isinstance(raw_confidence, (int, float))
        and not isinstance(raw_confidence, bool)
        and math.isfinite(float(raw_confidence))
        else 0.0
    )
    sources = fusion_group.get("sources") if isinstance(fusion_group, dict) else None
    if not isinstance(sources, list) or not sources:
        return confidence >= AUTO_THRESHOLD
    votes: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            return False
        view_id = source.get("view_id")
        source_family = source.get("family_hint")
        source_confidence = source.get("confidence")
        if (
            not isinstance(view_id, str)
            or not view_id
            or not isinstance(source_family, str)
            or isinstance(source_confidence, bool)
            or not isinstance(source_confidence, (int, float))
            or not math.isfinite(float(source_confidence))
        ):
            return False
        if source_family in _UNRESOLVED_SEMANTICS:
            continue
        vote = votes.setdefault(
            source_family,
            {
                "view_ids": set(),
                "confidence_sum": 0.0,
                "maximum_confidence": 0.0,
            },
        )
        if view_id in vote["view_ids"]:
            return False
        vote["view_ids"].add(view_id)
        vote["confidence_sum"] += float(source_confidence)
        vote["maximum_confidence"] = max(
            float(vote["maximum_confidence"]),
            float(source_confidence),
        )
    canonical_vote = votes.get(family)
    if canonical_vote is None:
        return False
    canonical_count = len(canonical_vote["view_ids"])
    canonical_sum = float(canonical_vote["confidence_sum"])
    if canonical_count < 2:
        return (
            canonical_count == 1
            and not any(value != family for value in votes)
            and float(canonical_vote["maximum_confidence"]) >= AUTO_THRESHOLD
        )
    alternatives = [
        (
            len(vote["view_ids"]),
            float(vote["confidence_sum"]),
        )
        for value, vote in votes.items()
        if value != family
    ]
    if not alternatives:
        return True
    strongest_count, strongest_sum = max(alternatives)
    return (
        canonical_count >= strongest_count + 2
        and canonical_sum >= strongest_sum + REVIEW_THRESHOLD
    )


def _shortlist_materials_with_audit(
    group: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    limit: int = 4,
    semantic_reliability: dict[str, Any] | None = None,
    family_reliable: bool = False,
    mvinverse_pbr_evidence: dict[str, Any] | None = None,
    allow_parameter_writes: bool = True,
    visual_similarity_first: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank a bounded material shortlist and expose the deterministic margin.

    The vision model's confidence is not a candidate margin.  Persisting this
    independent retrieval evidence lets downstream gates distinguish a clear
    catalog match from a near tie without asking the model to invent a score.
    The returned candidate records remain backward-compatible: model payloads
    deliberately hide the added ``retrieval_*`` audit fields.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(family_reliable, bool):
        raise ValueError("family_reliable must be a boolean")
    if not isinstance(allow_parameter_writes, bool):
        raise ValueError("allow_parameter_writes must be a boolean")
    if not isinstance(visual_similarity_first, bool):
        raise ValueError("visual_similarity_first must be a boolean")
    if not pool:
        raise ValueError("material pool cannot be empty")
    finish_reliable = True
    description_reliable = True
    if semantic_reliability is not None:
        finish_audit = semantic_reliability.get("finish_hint")
        description_audit = semantic_reliability.get("visual_description")
        if (
            not isinstance(finish_audit, dict)
            or not isinstance(finish_audit.get("reliable"), bool)
            or not isinstance(description_audit, dict)
            or not isinstance(description_audit.get("reliable"), bool)
        ):
            raise ValueError("semantic_reliability has invalid field decisions")
        finish_reliable = finish_audit["reliable"]
        description_reliable = description_audit["reliable"]
    description_tokens = (
        {
            token.casefold()
            for token in group["visual_description"]
            .replace("-", " ")
            .replace("_", " ")
            .split()
        }
        if description_reliable
        else set()
    )
    substance_tokens = {
        "aluminium",
        "aluminum",
        "brass",
        "copper",
        "galvanized",
        "plastic",
        "rubber",
        "silicone",
        "stainless",
        "steel",
    }
    family = group["family_hint"]
    color = group["base_color"]
    finish = group["finish_hint"]
    mvinverse_surface_class = (
        mvinverse_pbr_evidence.get("surface_class")
        if isinstance(mvinverse_pbr_evidence, dict)
        else None
    )
    canonical_finish = finish
    if semantic_reliability is not None:
        finish_audit = semantic_reliability["finish_hint"]
        raw_canonical_finish = finish_audit.get("canonical_value")
        if isinstance(raw_canonical_finish, str):
            canonical_finish = raw_canonical_finish
    surface_context = _surface_candidate_context(
        group,
        canonical_finish=canonical_finish,
        finish_reliable=finish_reliable,
        description_reliable=description_reliable,
        family_reliable=family_reliable,
        mvinverse_pbr_evidence=mvinverse_pbr_evidence,
    )
    observed_metallic = surface_context["observed_metallic"]
    applied_coating_confirmed = (
        family == "metal"
        and surface_context["semantic_surface_class"] == "coating"
        and surface_context["metallicity_class"] == "dielectric"
        and not surface_context["semantic_numeric_conflict"]
    )
    applied_coating_plausible = (
        family == "metal"
        and not applied_coating_confirmed
        and surface_context["metallicity_class"] != "unknown"
        and (
            surface_context["semantic_surface_class"] == "coating"
            or surface_context["metallicity_class"] == "dielectric"
        )
    )
    intrinsic_surface_ambiguity = (
        family == "metal"
        and color in _INTRINSIC_METAL_COLOR_TOKENS
        and not finish_reliable
    )

    compatible_families = {family}
    if family == "metal":
        # A painted metal remains a metal part in visual semantics, while the
        # closest NVIDIA Base preset may correctly live in the paint family.
        compatible_families.add("paint")
    family_pool = [item for item in pool if item.get("family") in compatible_families]
    paint_pool = [item for item in pool if item.get("family") == "paint"]
    # A single family hint is fallible, so legacy/single-view groups retain the
    # global pool.  Independently confirmed metal evidence restricts retrieval
    # to metal plus paint, but never to paint alone: conversion coatings and a
    # bare-metal counter-hypothesis must remain available to Qwen.
    candidates = (
        pool
        if visual_similarity_first
        else (family_pool if family_reliable and family_pool else pool)
    )
    family_pool_used = candidates is family_pool
    paint_pool_used = False
    pre_dedup_candidate_count = len(candidates)
    candidates_by_equivalence: dict[str, dict[str, Any]] = {}
    for item in candidates:
        material_id = str(item.get("material_id") or "")
        equivalence_key = _base_paint_equivalence_key(material_id)
        current = candidates_by_equivalence.get(equivalence_key)
        if current is None or material_id == equivalence_key:
            candidates_by_equivalence[equivalence_key] = item
    candidates = list(candidates_by_equivalence.values())
    available_interpretations = {
        interpretation: sum(
            _surface_interpretation(item) == interpretation for item in candidates
        )
        for interpretation in _SURFACE_INTERPRETATIONS
    }
    balanced_surface_shortlist = (
        not visual_similarity_first
        and family == "metal"
        and family_reliable
        and surface_context["dark_multiview_color"]
        and limit >= len(_SURFACE_INTERPRETATIONS)
        and all(available_interpretations.values())
    )
    intrinsic_identity_tokens = sorted(
        _INTRINSIC_METAL_COLOR_TOKENS.get(color, frozenset())
    )
    # Hue alone cannot distinguish a clean intrinsic metal from paint.
    # Thumbnail similarity may otherwise fill a bounded shortlist with many
    # adjacent bronze/brass variants and omit literal clean Copper before Qwen
    # sees the crop.  Preserve one clean representative for every plausible
    # intrinsic identity plus an applied-paint counter-hypothesis.  This
    # diversifies candidates; it does not decide the material.
    intrinsic_identity_shortlist = (
        not visual_similarity_first
        and intrinsic_surface_ambiguity
        and bool(intrinsic_identity_tokens)
        and limit >= len(intrinsic_identity_tokens) + 1
        and all(
            any(identity in _material_tokens(item) for item in candidates)
            for identity in intrinsic_identity_tokens
        )
        and any(_surface_interpretation(item) == "applied_paint" for item in candidates)
    )
    available_coating_templates = {
        template: sum(
            _coating_physics_template(item) == template for item in candidates
        )
        for template in _COATING_PHYSICS_TEMPLATES
    }
    # When independent semantics and MVInverse agree that a metal substrate is
    # covered by a dielectric painted layer, thumbnail colour alone must not
    # fill the whole shortlist with decorative paint or anodized metal.  Keep
    # one clean engineering paint, one generic paint, and one conversion
    # treatment so Qwen can decide the physical identity from the reference
    # crop.  This remains retrieval diversity; it never hard-codes a winner.
    coating_physics_shortlist = (
        not visual_similarity_first
        and not allow_parameter_writes
        and applied_coating_confirmed
        and finish_reliable
        and canonical_finish == "painted"
        and limit >= len(_COATING_PHYSICS_TEMPLATES)
        and all(available_coating_templates.values())
    )

    def score(item: dict[str, Any]) -> tuple[float, str, tuple[str, ...]]:
        colors = set(item.get("colors") or [])
        finishes = set(item.get("finishes") or [])
        material_tokens = _material_tokens(item)
        interpretation = _surface_interpretation(item)
        value = 0.0
        matched_fields: list[str] = []

        def matched(name: str) -> None:
            if name not in matched_fields:
                matched_fields.append(name)

        item_family = item.get("family")
        if not visual_similarity_first and item_family == family:
            value += 120
            matched("family")
        elif (
            not visual_similarity_first
            and family == "metal"
            and item_family == "paint"
            and applied_coating_confirmed
        ):
            # A visible paint layer is dielectric even when its substrate is
            # metal.  Treat a paint MDL as a first-class physical match rather
            # than allowing a fixed-color anodized preset to win merely
            # because its catalog family is ``metal``.
            value += 120
            matched("coating_surface_family")
        if not visual_similarity_first and family == "metal":
            semantic_surface = surface_context["semantic_surface_class"]
            if applied_coating_confirmed:
                if interpretation == "applied_paint":
                    value += 125
                    matched("confirmed_applied_coating")
                elif interpretation == "conversion_coating":
                    value += 105
                    matched("confirmed_applied_coating")
                elif interpretation == "bare_metal":
                    # Preserve one independently visible counter-hypothesis
                    # without letting it outrank corroborated coating evidence.
                    value += 20
            elif applied_coating_plausible:
                if interpretation in {"applied_paint", "conversion_coating"}:
                    value += 75
                    matched("plausible_applied_coating")
                elif interpretation == "bare_metal":
                    value += 35
            elif semantic_surface == "bare":
                if interpretation == "bare_metal":
                    value += 105
                    matched("semantic_surface_interpretation")
                elif interpretation == "conversion_coating":
                    value += 35

            metallicity = surface_context["metallicity_class"]
            if metallicity == "dielectric":
                if interpretation == "applied_paint":
                    value += 70
                    matched("mvinverse_metallicity_class")
                elif interpretation == "conversion_coating":
                    value += 40
                    matched("mvinverse_metallicity_class")
            elif metallicity == "conductive":
                if interpretation in {"conversion_coating", "bare_metal"}:
                    value += 70
                    matched("mvinverse_metallicity_class")
            elif metallicity == "ambiguous" and interpretation in (
                _SURFACE_INTERPRETATIONS
            ):
                value += 25
                matched("mvinverse_metallicity_class")

        intrinsic_color_match = bool(
            _INTRINSIC_METAL_COLOR_TOKENS.get(color, frozenset()) & material_tokens
        )
        if color in colors:
            value += 90
            matched("color")
        elif intrinsic_color_match:
            # When the finish is unconfirmed, pixel hue supports intrinsic
            # base metal and applied coating equally.  This keeps both physical
            # interpretations in a low-margin shortlist instead of encoding
            # an orange/brown/yellow => specific-metal rule.
            value += 90 if intrinsic_surface_ambiguity else 60
            matched("color")
        if surface_context["dark_multiview_color"] and (
            visual_similarity_first or family == "metal"
        ):
            if material_tokens & _DARK_SURFACE_TOKENS:
                value += 65
                matched("multiview_albedo_color")
            elif interpretation == "applied_paint":
                # Base paint is colour-tunable only after a separate parameter
                # gate, so keep it in the contest without pretending its white
                # library preview already matches the dark observation.
                value += 30
                matched("multiview_albedo_color")
            elif (
                interpretation == "bare_metal"
                and material_tokens & _NEUTRAL_ENGINEERING_METAL_TOKENS
            ):
                value += 25
                matched("multiview_albedo_color")
        if finish_reliable and finish in finishes:
            value += 45
            matched("finish")
        if finish_reliable and finish in {"painted", "matte"}:
            if "opaque" in material_tokens:
                value += 35
                matched("optical_mode")
            elif "translucent" in material_tokens or "transparent" in material_tokens:
                value -= 25
        expected_finish_tokens = _ROUGHNESS_FINISH_TOKENS.get(
            surface_context["roughness_class"], frozenset()
        )
        if expected_finish_tokens and material_tokens & expected_finish_tokens:
            value += 45
            matched("mvinverse_roughness_class")
        token_matches = description_tokens & material_tokens
        material_matches = token_matches & substance_tokens
        # An explicit visible substance cue ("copper tube", "rubber hose")
        # must be able to outweigh one fallible family label.
        value += 3 * len(token_matches)
        if not visual_similarity_first:
            value += 240 * len(material_matches)
        if token_matches:
            matched("description_tokens")
        if not allow_parameter_writes:
            unsupported_domains = _unsupported_niche_domains(
                item,
                material_tokens=material_tokens,
                reference_tokens=description_tokens,
            )
            # The complete catalog remains searchable, but a niche-use preset
            # needs positive reference semantics for that domain.  A shared
            # word such as "painted" is not evidence that an ordinary plastic
            # plate is PCB solder mask or that an industrial enclosure is
            # automotive car paint.
            value -= 320.0 * len(unsupported_domains)
            observed_effects = description_tokens & _VISIBLE_SPECIAL_EFFECT_TOKENS
            unobserved_effects = (
                material_tokens & _VISIBLE_SPECIAL_EFFECT_TOKENS
            ) - observed_effects
            # A fixed MDL's cracks, wear, patina, flakes and other conspicuous
            # effects cannot be disabled after selection.  Such presets need
            # positive reliable reference semantics; semantic uncertainty is
            # not permission to invent irreversible surface detail.
            value -= min(240.0, 90.0 * len(unobserved_effects))
        pbr_bonus, pbr_matches = mvinverse_similarity_terms(
            item.get("appearance_profile"),
            mvinverse_pbr_evidence,
            fixed_defaults_required=not allow_parameter_writes,
            thumbnail_profile=item.get("thumbnail_appearance_profile"),
        )
        value += pbr_bonus
        for pbr_match in pbr_matches:
            matched(pbr_match)
        tuning_profile = tuning_profile_for_material(str(item["material_id"]))
        if (
            allow_parameter_writes
            and tuning_profile is not None
            and isinstance(mvinverse_pbr_evidence, dict)
            and mvinverse_pbr_evidence.get("surface_class")
            == tuning_profile.surface_class
            and isinstance(mvinverse_pbr_evidence.get("suggestion"), dict)
            and mvinverse_pbr_evidence["suggestion"].get("decision") == "auto"
            and mvinverse_pbr_evidence["suggestion"].get("auto_parameter_eligible")
            is True
        ):
            # Prefer an MDL whose exposed inputs can reproduce the measured
            # appearance.  Fixed named colors remain candidates, but cannot
            # beat an equally plausible, MVInverse-tunable template.
            value += 90
            matched("mvinverse_tunable_template")
        tie = hashlib.sha256(
            f"{group['group_id']}\0{item['material_id']}".encode("utf-8")
        ).hexdigest()
        return value, tie, tuple(matched_fields)

    ranked_all_before_tunable_dedup = sorted(
        ((item, score(item)) for item in candidates),
        key=lambda entry: (entry[1][0], entry[1][1]),
        reverse=True,
    )
    ranked_all: list[tuple[dict[str, Any], tuple[float, str, tuple[str, ...]]]] = []
    seen_tunable_equivalence_keys: set[str] = set()
    for entry in ranked_all_before_tunable_dedup:
        equivalence_key = _mvinverse_tunable_equivalence_key(
            str(entry[0]["material_id"]),
            mvinverse_pbr_evidence,
            allow_parameter_writes=allow_parameter_writes,
        )
        if equivalence_key in seen_tunable_equivalence_keys:
            continue
        seen_tunable_equivalence_keys.add(equivalence_key)
        ranked_all.append(entry)
    if intrinsic_identity_shortlist:
        required_ids: set[str] = set()
        for identity_token in intrinsic_identity_tokens:
            identity_entries = [
                entry
                for entry in ranked_all
                if identity_token in _material_tokens(entry[0])
            ]
            if not identity_entries:
                continue

            def identity_key(
                entry: tuple[
                    dict[str, Any],
                    tuple[float, str, tuple[str, ...]],
                ],
            ) -> tuple[int, int, float, str]:
                item, scored = entry
                exact_clean_name = int(
                    str(item.get("sub_identifier") or "").casefold() == identity_token
                    or str(item.get("display_name") or "").casefold() == identity_token
                )
                visible_effect_count = len(
                    _material_tokens(item) & _VISIBLE_SPECIAL_EFFECT_TOKENS
                )
                return (
                    exact_clean_name,
                    -visible_effect_count,
                    scored[0],
                    str(item.get("material_id") or ""),
                )

            required_ids.add(
                str(max(identity_entries, key=identity_key)[0]["material_id"])
            )
        paint_entry = next(
            (
                entry
                for entry in ranked_all
                if _surface_interpretation(entry[0]) == "applied_paint"
            ),
            None,
        )
        if paint_entry is not None:
            required_ids.add(str(paint_entry[0]["material_id"]))
        for item, _scored in ranked_all:
            if len(required_ids) >= limit:
                break
            required_ids.add(str(item["material_id"]))
        ranked = [
            entry
            for entry in ranked_all
            if str(entry[0]["material_id"]) in required_ids
        ]
    elif coating_physics_shortlist:
        required_ids: set[str] = set()
        for template in _COATING_PHYSICS_TEMPLATES:
            matching_entries = [
                entry
                for entry in ranked_all
                if _coating_physics_template(entry[0]) == template
            ]

            def coating_template_key(
                entry: tuple[
                    dict[str, Any],
                    tuple[float, str, tuple[str, ...]],
                ],
            ) -> tuple[int, float, str]:
                item, scored = entry
                visible_effect_count = len(
                    _material_tokens(item) & _VISIBLE_SPECIAL_EFFECT_TOKENS
                )
                return (
                    -visible_effect_count,
                    scored[0],
                    str(item.get("material_id") or ""),
                )

            required_ids.add(
                str(max(matching_entries, key=coating_template_key)[0]["material_id"])
            )
        for item, _scored in ranked_all:
            if len(required_ids) >= limit:
                break
            required_ids.add(str(item["material_id"]))
        ranked = [
            entry
            for entry in ranked_all
            if str(entry[0]["material_id"]) in required_ids
        ]
    elif balanced_surface_shortlist:
        required_ids: set[str] = set()
        for interpretation in _SURFACE_INTERPRETATIONS:
            representative = next(
                entry
                for entry in ranked_all
                if _surface_interpretation(entry[0]) == interpretation
            )
            required_ids.add(str(representative[0]["material_id"]))
        for item, _scored in ranked_all:
            if len(required_ids) >= limit:
                break
            required_ids.add(str(item["material_id"]))
        ranked = [
            entry
            for entry in ranked_all
            if str(entry[0]["material_id"]) in required_ids
        ]
    else:
        ranked = ranked_all[:limit]
    selected: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    for rank, (item, (value, _tie, matched_fields)) in enumerate(
        ranked[:limit], start=1
    ):
        candidate = dict(item)
        candidate["surface_interpretation"] = _surface_interpretation(item)
        candidate["retrieval_rank"] = rank
        candidate["retrieval_score"] = value
        candidate["retrieval_matched_fields"] = list(matched_fields)
        selected.append(candidate)
        ranking.append(
            {
                "rank": rank,
                "material_id": item["material_id"],
                "score": value,
                "matched_fields": list(matched_fields),
            }
        )

    selected_interpretations = {
        interpretation: [
            candidate["material_id"]
            for candidate in selected
            if candidate["surface_interpretation"] == interpretation
        ]
        for interpretation in _SURFACE_INTERPRETATIONS
    }
    selected_coating_templates = {
        template: [
            candidate["material_id"]
            for candidate in selected
            if _coating_physics_template(candidate) == template
        ]
        for template in _COATING_PHYSICS_TEMPLATES
    }

    top_score = ranking[0]["score"]
    runner_up_score = ranking[1]["score"] if len(ranking) > 1 else None
    score_margin = top_score - runner_up_score if runner_up_score is not None else None
    normalized_margin = (
        score_margin / max(abs(top_score), 1) if score_margin is not None else None
    )
    audit = {
        "strategy": (
            "visual_mvinverse_similarity_score/v1"
            if visual_similarity_first
            else (
                "family_gated_semantic_mvinverse_similarity_score/v7"
                if allow_parameter_writes
                else "family_gated_semantic_mvinverse_similarity_score/v12"
            )
        ),
        "pool_count": len(pool),
        "eligible_pool_count": len(candidates),
        "pre_duplicate_alias_dedup_count": pre_dedup_candidate_count,
        "duplicate_alias_dedup_count": pre_dedup_candidate_count - len(candidates),
        "mvinverse_tunable_equivalence_dedup_count": (
            len(ranked_all_before_tunable_dedup) - len(ranked_all)
        ),
        "family_pool_available": bool(family_pool),
        "family_pool_used": family_pool_used,
        "paint_pool_available": bool(paint_pool),
        "paint_pool_used": paint_pool_used,
        "semantic_reliability": copy.deepcopy(semantic_reliability),
        "finish_evidence_used": finish_reliable,
        "description_evidence_used": description_reliable,
        "intrinsic_surface_ambiguity": intrinsic_surface_ambiguity,
        "mvinverse_surface_class": mvinverse_surface_class,
        "observed_metallic": observed_metallic,
        "applied_coating_confirmed": applied_coating_confirmed,
        "applied_coating_plausible": applied_coating_plausible,
        "niche_domain_policy": {
            "mode": "positive_reference_semantics_required",
            "domains": sorted(_NICHE_DOMAIN_TOKEN_GROUPS),
        },
        "surface_interpretation_policy": {
            "mode": (
                "balanced_intrinsic_metal_identities"
                if intrinsic_identity_shortlist
                else (
                    "balanced_confirmed_applied_coating_physics"
                    if coating_physics_shortlist
                    else (
                        "balanced_dark_metal_surface_interpretations"
                        if balanced_surface_shortlist
                        else "score_ranked"
                    )
                )
            ),
            "active": (
                balanced_surface_shortlist
                or intrinsic_identity_shortlist
                or coating_physics_shortlist
            ),
            "family_reliable": family_reliable,
            "semantic_surface_class": surface_context["semantic_surface_class"],
            "semantic_numeric_conflict": surface_context["semantic_numeric_conflict"],
            "multi_view_albedo_reliable": surface_context["multi_view_albedo_reliable"],
            "albedo_median": surface_context["albedo_median"],
            "albedo_luminance": surface_context["albedo_luminance"],
            "dark_multiview_color": surface_context["dark_multiview_color"],
            "metallic_reliable": surface_context["metallic_reliable"],
            "metallicity_class": surface_context["metallicity_class"],
            "roughness_reliable": surface_context["roughness_reliable"],
            "observed_roughness": surface_context["observed_roughness"],
            "roughness_class": surface_context["roughness_class"],
            "required_interpretations": (
                list(_SURFACE_INTERPRETATIONS) if balanced_surface_shortlist else []
            ),
            "required_intrinsic_material_identities": (
                intrinsic_identity_tokens if intrinsic_identity_shortlist else []
            ),
            "required_coating_physics_templates": (
                list(_COATING_PHYSICS_TEMPLATES) if coating_physics_shortlist else []
            ),
            "available_interpretation_counts": available_interpretations,
            "selected_material_ids_by_interpretation": selected_interpretations,
            "available_coating_physics_template_counts": (available_coating_templates),
            "selected_material_ids_by_coating_physics_template": (
                selected_coating_templates
            ),
            "complete_required_coverage": (
                all(
                    selected_interpretations[value]
                    for value in _SURFACE_INTERPRETATIONS
                )
                if balanced_surface_shortlist
                else (
                    all(
                        selected_coating_templates[template]
                        for template in _COATING_PHYSICS_TEMPLATES
                    )
                    if coating_physics_shortlist
                    else (
                        all(
                            any(
                                identity in _material_tokens(candidate)
                                for candidate in selected
                            )
                            for identity in intrinsic_identity_tokens
                        )
                        and any(
                            candidate["surface_interpretation"] == "applied_paint"
                            for candidate in selected
                        )
                        if intrinsic_identity_shortlist
                        else True
                    )
                )
            ),
        },
        "limit": limit,
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "score_margin": score_margin,
        "normalized_margin": normalized_margin,
        "margin_available": normalized_margin is not None,
        "ranking": ranking,
    }
    if not allow_parameter_writes:
        audit["fixed_library_defaults_required"] = True
        audit["thumbnail_default_evidence_count"] = sum(
            isinstance(item.get("thumbnail_appearance_profile"), dict)
            for item in candidates
        )
        audit[
            "unobserved_fixed_effect_policy"
        ] = "positive_reliable_semantics_required/v1"
    return selected, audit


def _base_paint_equivalence_key(material_id: str) -> str:
    """Collapse NVIDIA Base's byte-equivalent ``*_Finish`` paint aliases."""

    match = _BASE_PAINT_EQUIVALENCE_RE.fullmatch(material_id)
    return (
        "mdl:"
        + (match.group("root") or "")
        + f"Miscellaneous/{match.group('name')}.mdl#{match.group('name')}"
        if match is not None
        else material_id
    )


def _mvinverse_tunable_equivalence_key(
    material_id: str,
    mvinverse_pbr_evidence: dict[str, Any] | None,
    *,
    allow_parameter_writes: bool = True,
) -> str:
    """Collapse preset exports that MVInverse will parameterize identically.

    NVIDIA modules such as ``Steel_Painted.mdl`` expose many named colour
    presets.  Once verified MVInverse evidence is allowed to replace their
    colour/PBR controls, those exports are not independent material
    hypotheses.  Treating them as separate candidates creates an artificial
    near-tie and can fill the entire bounded shortlist with one effective
    material.
    """

    if not isinstance(allow_parameter_writes, bool):
        raise ValueError("allow_parameter_writes must be a boolean")
    if not allow_parameter_writes:
        return material_id
    suggestion = (
        mvinverse_pbr_evidence.get("suggestion")
        if isinstance(mvinverse_pbr_evidence, dict)
        else None
    )
    profile = tuning_profile_for_material(material_id)
    if not (
        isinstance(suggestion, dict)
        and suggestion.get("decision") == "auto"
        and suggestion.get("auto_parameter_eligible") is True
        and profile is not None
        and mvinverse_pbr_evidence.get("surface_class") == profile.surface_class
    ):
        return material_id
    module_id = material_id.split("#", 1)[0]
    return f"mvinverse-tunable:{module_id}:{profile.profile_id}"


def _mvinverse_exact_default_candidates(
    pool: list[dict[str, Any]],
    mvinverse_pbr_evidence: dict[str, Any] | None,
    *,
    limit: int,
    selection_group: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return exact immutable MDL defaults nearest to MVInverse observations."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("MVInverse exact-default candidate limit must be positive")
    if not isinstance(mvinverse_pbr_evidence, dict):
        return []
    reference_tokens = (
        {
            token.casefold()
            for token in str(selection_group.get("visual_description") or "")
            .replace("-", " ")
            .replace("_", " ")
            .split()
        }
        if isinstance(selection_group, dict)
        else set()
    )
    semantic_family = (
        selection_group.get("family_hint")
        if isinstance(selection_group, dict)
        else None
    )
    ranked: list[tuple[float, str, list[str], dict[str, Any]]] = []
    seen: set[str] = set()
    for item in pool:
        material_id = item.get("material_id")
        if not isinstance(material_id, str) or not material_id or material_id in seen:
            continue
        seen.add(material_id)
        material_tokens = _material_tokens(item)
        if _unsupported_niche_domains(
            item,
            material_tokens=material_tokens,
            reference_tokens=reference_tokens,
        ):
            continue
        # An inverse-rendered RGB/PBR triplet can numerically match a light,
        # liquid, or food preset even though the observed palette group is an
        # ordinary solid surface.  Those domains remain searchable by the
        # wider visual tournament, but they are not trustworthy *forced*
        # numeric neighbours without corresponding group semantics.
        candidate_family = item.get("family")
        allowed_numeric_families = set(_FORCED_NUMERIC_SOLID_SURFACE_FAMILIES)
        if isinstance(semantic_family, str):
            allowed_numeric_families.add(semantic_family)
        if (
            candidate_family not in allowed_numeric_families
            or (material_tokens & _VISIBLE_SPECIAL_EFFECT_TOKENS) - reference_tokens
        ):
            continue
        score, matched_fields = mvinverse_similarity_terms(
            item.get("appearance_profile"),
            mvinverse_pbr_evidence,
            fixed_defaults_required=True,
            thumbnail_profile=item.get("thumbnail_appearance_profile"),
        )
        # Colour is the only numeric channel available for most NVIDIA named
        # exports.  Requiring it prevents a roughness-only preset from entering
        # a colour disagreement tournament merely because its scalar happens
        # to be close.
        if "mvinverse_color" not in matched_fields or score <= 0.0:
            continue
        ranked.append((float(score), material_id, matched_fields, item))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    output: list[dict[str, Any]] = []
    for rank, (score, _material_id, matched_fields, item) in enumerate(
        ranked[:limit],
        start=1,
    ):
        candidate = copy.deepcopy(item)
        candidate["retrieval_rank"] = rank
        candidate["retrieval_score"] = score
        candidate["retrieval_matched_fields"] = list(matched_fields)
        candidate[
            "tournament_candidate_basis"
        ] = "mvinverse_nearest_exact_library_default"
        output.append(candidate)
    return output


def _build_disagreement_tournament_candidates(
    *,
    forward: dict[str, Any],
    reverse: dict[str, Any],
    provisional_seed: dict[str, Any],
    qwen_candidates: list[dict[str, Any]],
    tournament_candidates: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    mvinverse_pbr_evidence: dict[str, Any] | None,
    maximum_candidates: int,
    selection_group: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Force unresolved Qwen choices and numeric neighbours into render QA."""

    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 3
    ):
        raise ValueError(
            "forward/reverse disagreement requires at least three exact-MDL "
            "tournament candidates"
        )
    forward_id = str(forward["material_id"])
    reverse_id = str(reverse["material_id"])
    if forward_id == reverse_id:
        raise ValueError("disagreement tournament received agreeing Qwen choices")
    neighbor_limit = max(
        1,
        min(
            _DISAGREEMENT_MVINVERSE_NEIGHBOR_LIMIT,
            maximum_candidates - 2,
        ),
    )
    numeric_neighbors = _mvinverse_exact_default_candidates(
        pool,
        mvinverse_pbr_evidence,
        limit=neighbor_limit,
        selection_group=selection_group,
    )

    all_records: dict[str, dict[str, Any]] = {}
    for candidate in [
        *qwen_candidates,
        *tournament_candidates,
        *numeric_neighbors,
    ]:
        material_id = candidate.get("material_id")
        if (
            isinstance(material_id, str)
            and material_id
            and material_id not in all_records
        ):
            all_records[material_id] = copy.deepcopy(candidate)
    if forward_id not in all_records or reverse_id not in all_records:
        raise ValueError("Qwen disagreement references a material outside the catalog")

    numeric_ids = [
        str(candidate["material_id"])
        for candidate in numeric_neighbors
        if candidate["material_id"] not in {forward_id, reverse_id}
    ]
    if not numeric_ids:
        # The closest numeric preset may legitimately equal a disputed choice,
        # but at least one independently ranked library default must remain in
        # the visual contest.
        for candidate in _mvinverse_exact_default_candidates(
            pool,
            mvinverse_pbr_evidence,
            limit=min(len(pool), maximum_candidates),
            selection_group=selection_group,
        ):
            material_id = str(candidate["material_id"])
            all_records[material_id] = copy.deepcopy(candidate)
            if material_id not in {forward_id, reverse_id}:
                numeric_ids.append(material_id)
                break
    numeric_ids = numeric_ids[: max(1, maximum_candidates - 2)]
    retrieval_ids: list[str] = []
    if not numeric_ids:
        # Small, occluded, or single-view parts can legitimately have no
        # accepted MVInverse albedo/PBR estimate.  Preserve the fail-closed
        # requirement for an independent exact-MDL challenger without
        # pretending rejected numeric evidence exists: use the already
        # bounded SigLIP2/DINO (or deterministic retrieval) ranking and leave
        # the final decision to exact rendered comparison.
        for candidate in tournament_candidates:
            material_id = candidate.get("material_id")
            if (
                not isinstance(material_id, str)
                or not material_id
                or material_id in {forward_id, reverse_id}
                or material_id in retrieval_ids
            ):
                continue
            fallback = copy.deepcopy(candidate)
            fallback["tournament_candidate_basis"] = RANKED_RETRIEVAL_CHALLENGER_BASIS
            all_records[material_id] = fallback
            retrieval_ids.append(material_id)
            break
    if not numeric_ids and not retrieval_ids:
        raise DisagreementTournamentCandidateError(
            "forward/reverse disagreement has neither an accepted "
            "MVInverse-nearest exact library-default candidate nor an "
            "independent ranked visual-retrieval exact-default challenger"
        )
    required_ids = list(
        dict.fromkeys(
            [
                forward_id,
                reverse_id,
                *numeric_ids,
                *retrieval_ids,
            ]
        )
    )
    if len(required_ids) > maximum_candidates:
        required_ids = required_ids[:maximum_candidates]
        numeric_ids = [
            material_id
            for material_id in required_ids
            if material_id not in {forward_id, reverse_id}
        ]
        retrieval_ids = [
            material_id for material_id in retrieval_ids if material_id in required_ids
        ]

    primary_budget = min(
        maximum_candidates,
        max(_DISAGREEMENT_PRIMARY_TOURNAMENT_BUDGET, len(required_ids)),
    )
    primary_ids = list(required_ids)
    for candidate in qwen_candidates:
        material_id = str(candidate["material_id"])
        if material_id not in primary_ids and len(primary_ids) < primary_budget:
            primary_ids.append(material_id)
    primary = [copy.deepcopy(all_records[material_id]) for material_id in primary_ids]

    wider_ids = list(primary_ids)
    for candidate in tournament_candidates:
        material_id = str(candidate["material_id"])
        if material_id not in wider_ids and len(wider_ids) < maximum_candidates:
            wider_ids.append(material_id)
    wider = [copy.deepcopy(all_records[material_id]) for material_id in wider_ids]
    contract = build_disagreement_tournament_contract(
        forward_material_id=forward_id,
        reverse_material_id=reverse_id,
        provisional_seed_material_id=str(provisional_seed["material_id"]),
        mvinverse_exact_default_material_ids=numeric_ids,
        tournament_candidate_material_ids=wider_ids,
        retrieval_exact_default_material_ids=retrieval_ids,
    )
    return primary, wider, contract


def _confirm_material_choices(
    first: dict[str, Any],
    second: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    mvinverse_pbr_evidence: dict[str, Any] | None = None,
    allow_parameter_writes: bool = True,
) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(allow_parameter_writes, bool):
        raise ValueError("allow_parameter_writes must be a boolean")
    first_id = first["material_id"]
    second_id = second["material_id"]
    if first_id == second_id:
        return first, True, "exact_forward_reverse_agreement"
    candidate_by_id = {
        str(candidate.get("material_id")): candidate for candidate in candidates
    }
    visual_retrieval_candidates = any(
        isinstance(candidate.get("retrieval_matched_fields"), list)
        and "siglip2_catalog_wide_visual" in candidate["retrieval_matched_fields"]
        for candidate in candidates
    )
    first_candidate = candidate_by_id.get(first_id)
    second_candidate = candidate_by_id.get(second_id)
    intrinsic_identity_tokens = set().union(*_INTRINSIC_METAL_COLOR_TOKENS.values())
    if (
        not allow_parameter_writes
        and not visual_retrieval_candidates
        and isinstance(first_candidate, dict)
        and isinstance(second_candidate, dict)
        and _surface_interpretation(first_candidate) == "bare_metal"
        and _surface_interpretation(second_candidate) == "bare_metal"
        and _material_tokens(first_candidate) & intrinsic_identity_tokens
        and _material_tokens(second_candidate) & intrinsic_identity_tokens
    ):
        first_score = first_candidate.get("retrieval_score")
        second_score = second_candidate.get("retrieval_score")
        if (
            isinstance(first_score, (int, float))
            and not isinstance(first_score, bool)
            and math.isfinite(float(first_score))
            and isinstance(second_score, (int, float))
            and not isinstance(second_score, bool)
            and math.isfinite(float(second_score))
            and float(first_score) != float(second_score)
        ):
            winner = first if float(first_score) > float(second_score) else second
            return (
                dict(winner),
                True,
                "immutable_intrinsic_metal_class_agreement",
            )
    if (
        not allow_parameter_writes
        and not visual_retrieval_candidates
        and isinstance(first_candidate, dict)
        and isinstance(second_candidate, dict)
        and _surface_interpretation(first_candidate) == "applied_paint"
        and _surface_interpretation(second_candidate) == "applied_paint"
        and {
            str(value).casefold()
            for value in first_candidate.get("colors") or []
            if isinstance(value, str)
        }
        & {
            str(value).casefold()
            for value in second_candidate.get("colors") or []
            if isinstance(value, str)
        }
        and isinstance(mvinverse_pbr_evidence, dict)
        and mvinverse_pbr_evidence.get("surface_class") == "dielectric"
    ):
        first_score = first_candidate.get("retrieval_score")
        second_score = second_candidate.get("retrieval_score")
        if (
            isinstance(first_score, (int, float))
            and not isinstance(first_score, bool)
            and math.isfinite(float(first_score))
            and isinstance(second_score, (int, float))
            and not isinstance(second_score, bool)
            and math.isfinite(float(second_score))
            and float(first_score) != float(second_score)
        ):
            winner = first if float(first_score) > float(second_score) else second
            return (
                dict(winner),
                True,
                "immutable_applied_paint_appearance_agreement",
            )
    first_key = _base_paint_equivalence_key(first_id)
    second_key = _base_paint_equivalence_key(second_id)
    candidate_ids = {candidate["material_id"] for candidate in candidates}
    if first_key == second_key and first_key in candidate_ids:
        chosen = dict(first)
        chosen["material_id"] = first_key
        return chosen, True, "nvidia_base_duplicate_paint_alias_agreement"
    first_profile = tuning_profile_for_material(first_id)
    second_profile = tuning_profile_for_material(second_id)
    first_module = first_id.split("#", 1)[0]
    second_module = second_id.split("#", 1)[0]
    evidence_suggestion = (
        mvinverse_pbr_evidence.get("suggestion")
        if isinstance(mvinverse_pbr_evidence, dict)
        else None
    )
    if (
        allow_parameter_writes
        and first_profile is not None
        and second_profile is not None
        and first_profile.profile_id == second_profile.profile_id
        and first_module == second_module
        and isinstance(evidence_suggestion, dict)
        and evidence_suggestion.get("decision") == "auto"
        and evidence_suggestion.get("auto_parameter_eligible") is True
        and mvinverse_pbr_evidence.get("surface_class") == first_profile.surface_class
    ):
        # Export variants from one tunable MDL module differ mainly in preset
        # defaults that the verified MVInverse delta will replace.  Resolve
        # Qwen's variant disagreement to the deterministic retrieval leader.
        chosen = dict(first)
        chosen["material_id"] = str(candidates[0]["material_id"])
        return chosen, True, "mvinverse_tunable_module_agreement"
    first_paint = _BASE_PAINT_PRIMARY_RE.fullmatch(first_id)
    second_paint = _BASE_PAINT_PRIMARY_RE.fullmatch(second_id)
    roughness_stats = (
        mvinverse_pbr_evidence.get("roughness")
        if isinstance(mvinverse_pbr_evidence, dict)
        else None
    )
    raw_roughness = (
        roughness_stats.get("median") if isinstance(roughness_stats, dict) else None
    )
    if (
        first_paint is not None
        and second_paint is not None
        and isinstance(raw_roughness, (int, float))
        and not isinstance(raw_roughness, bool)
        and math.isfinite(float(raw_roughness))
        and 0.0 <= float(raw_roughness) <= 1.0
    ):
        roughness = float(raw_roughness)
        finish = (
            "Gloss" if roughness <= 0.3 else "Matte" if roughness >= 0.6 else "Satin"
        )
        preferred_ids = (
            f"mdl:Miscellaneous/Paint_{finish}.mdl#Paint_{finish}",
            f"mdl:Base/Miscellaneous/Paint_{finish}.mdl#Paint_{finish}",
        )
        preferred_id = next(
            (value for value in preferred_ids if value in candidate_ids),
            None,
        )
        if preferred_id is not None:
            chosen = dict(first)
            chosen["material_id"] = preferred_id
            return chosen, True, "mvinverse_resolved_base_paint_finish"
    return first, False, "forward_reverse_disagreement"


def _resolve_immutable_coating_physics_choice(
    chosen: dict[str, Any],
    *,
    confirmed: bool,
    confirmation_basis: str,
    candidates: list[dict[str, Any]],
    retrieval_audit: dict[str, Any],
) -> tuple[dict[str, Any], bool, str, dict[str, Any]]:
    """Seal an immutable choice without replacing the selected MDL.

    Physical evidence is used while retrieving and confirming candidates.
    Once Qwen and the deterministic confirmation rule have selected an exact
    NVIDIA MDL export, a later substrate/physics heuristic must not exchange
    it for a different export.  That would change appearance after selection
    and makes the immutable contract meaningless.
    """

    del candidates, retrieval_audit
    resolution = {
        "applied": False,
        "mode": "immutable_selected_mdl_preserved",
        "original_material_id": chosen["material_id"],
        "resolved_material_id": chosen["material_id"],
        "selected_mdl_parameters_mutable": False,
    }
    return chosen, confirmed, confirmation_basis, resolution


def _catalog_pool(
    catalog: MaterialCatalog,
    whitelist_path: Path,
) -> list[dict[str, Any]]:
    document = _read_json(whitelist_path)
    ids = document.get("material_ids")
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Whitelist material_ids must be a string array")
    if len(set(ids)) != len(ids):
        raise ValueError("Whitelist material_ids must be unique")
    if document.get("scope") == "catalog_exact":
        catalog_ids = {record.material_id for record in catalog.materials}
        allowlist_ids = set(ids)
        if allowlist_ids != catalog_ids:
            missing = sorted(catalog_ids - allowlist_ids)
            unexpected = sorted(allowlist_ids - catalog_ids)
            raise ValueError(
                "Catalog-exact allowlist does not equal the complete material "
                f"catalog; missing={missing}, unexpected={unexpected}"
            )
        if document.get("material_count") != len(catalog_ids):
            raise ValueError(
                "Catalog-exact allowlist material_count does not match the catalog"
            )
    records = [catalog.get(material_id).to_dict() for material_id in ids]
    for record in records:
        mdl_file, _sub_identifier = catalog.resolve_material(record["material_id"])
        profile = extract_mdl_appearance_profile(
            mdl_file,
            record["sub_identifier"],
        )
        if profile is not None:
            record["appearance_profile"] = profile
        thumbnail = record.get("thumbnail_path")
        if isinstance(thumbnail, str):
            path = (catalog.root / thumbnail).resolve(strict=True)
            path.relative_to(catalog.root)
            record["thumbnail_image"] = str(path)
            thumbnail_profile = extract_thumbnail_appearance_profile(path)
            if thumbnail_profile is not None:
                record["thumbnail_appearance_profile"] = thumbnail_profile
    return records


def _find_render_view(render_set: dict[str, Any], view_id: str) -> dict[str, Any]:
    for record in render_set.get("views", []):
        if isinstance(record, dict) and record.get("view_id") == view_id:
            return record
    raise ValueError(f"Registry has no render view: {view_id}")


def _batch_geometry_views(
    render_set: dict[str, Any], view_id: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the matching CAD-overview and part-ID inputs for one batch."""

    render_view = _find_render_view(render_set, view_id)
    rgb = render_view.get("rgb")
    part_ids = render_view.get("part_ids_annotated") or render_view.get("part_ids")
    if not isinstance(rgb, str) or not rgb:
        raise ValueError(f"Render view {view_id!r} has no RGB overview")
    if not isinstance(part_ids, str) or not part_ids:
        raise ValueError(f"Render view {view_id!r} has no part-ID overview")
    return (
        {"id": f"cad_{view_id}", "image": rgb},
        {"id": f"part_ids_{view_id}", "image": part_ids},
    )


def _evidence_highlight(part: dict[str, Any], best_evidence: dict[str, str]) -> str:
    """Prefer mask-isolated multiview evidence, with legacy highlight fallback."""

    part_id = part["part_id"]
    isolated = _isolated_evidence_summary(part)
    if isolated is not None:
        path = isolated.get("path")
        if isinstance(path, str) and path:
            return path
    evidence_view_id = part.get("evidence_cad_view_id")
    for render in part.get("renders", []):
        if (
            isinstance(render, dict)
            and render.get("view_id") == evidence_view_id
            and isinstance(render.get("highlight_path"), str)
            and render["highlight_path"]
        ):
            return render["highlight_path"]
    fallback = best_evidence.get(part_id)
    if not isinstance(fallback, str) or not fallback:
        raise ValueError(
            f"Registry has no isolated evidence or highlight for target part "
            f"{part_id} in view "
            f"{evidence_view_id!r}; rerun render_part_views.py"
        )
    return fallback


def _materialize_calibrated_fallback_highlights(
    parts: list[dict[str, Any]],
    *,
    render_set: dict[str, Any],
    best_evidence: dict[str, str],
    output_dir: Path,
) -> dict[str, str]:
    """Create identity crops for parts revealed only by calibrated cameras.

    Continuous registration can reveal a part that was hidden in every source
    pose-bank view.  Such a part legitimately has no old isolated crop.  The
    calibrated RGB and lossless Part-ID render are sufficient to create the
    same geometry-only highlight deterministically, without rerendering or
    treating CAD appearance as photographic material evidence.
    """

    if render_set.get("continuous_camera_calibration") is not True:
        return {}

    import numpy as np
    from PIL import Image

    from qwen_material_pipeline.usd.render import (
        _highlighted_context_crop,
        _part_color,
    )

    visibility = _calibrated_visible_pixels_by_part(render_set)
    generated: dict[str, str] = {}
    loaded_views: dict[str, tuple[Any, Any]] = {}
    for part in parts:
        part_id = part["part_id"]
        if _isolated_evidence_summary(part) is not None:
            continue
        existing = best_evidence.get(part_id)
        if (
            isinstance(existing, str)
            and existing
            and Path(existing).expanduser().is_file()
        ):
            continue
        view_id = part.get("evidence_cad_view_id")
        if not isinstance(view_id, str) or not view_id:
            continue
        expected_pixels = visibility.get(part_id, {}).get(view_id, 0)
        if expected_pixels <= 0:
            continue
        if view_id not in loaded_views:
            view = _find_render_view(render_set, view_id)
            rgb_path = view.get("rgb")
            part_ids_path = view.get("part_ids_raw") or view.get("part_ids")
            if (
                not isinstance(rgb_path, str)
                or not rgb_path
                or not isinstance(part_ids_path, str)
                or not part_ids_path
            ):
                raise ValueError(
                    f"Calibrated render {view_id!r} lacks RGB/Part-ID inputs"
                )
            with Image.open(Path(rgb_path).expanduser().resolve(strict=True)) as opened:
                rgb_image = opened.convert("RGB")
            with Image.open(
                Path(part_ids_path).expanduser().resolve(strict=True)
            ) as opened:
                part_ids = np.asarray(opened.convert("RGB"))
            if part_ids.shape[:2] != (rgb_image.height, rgb_image.width):
                raise ValueError(
                    f"Calibrated render {view_id!r} RGB/Part-ID dimensions differ"
                )
            loaded_views[view_id] = (rgb_image, part_ids)
        rgb_image, part_ids = loaded_views[view_id]
        target_color = np.asarray(_part_color(part_id), dtype=np.uint8)
        mask = np.all(part_ids == target_color, axis=2)
        actual_pixels = int(mask.sum())
        if actual_pixels != expected_pixels:
            raise ValueError(
                f"Calibrated Part-ID mask for {part_id} in {view_id!r} has "
                f"{actual_pixels} pixels; registry records {expected_pixels}"
            )
        highlighted = _highlighted_context_crop(rgb_image, mask, part_id)
        if highlighted is None:
            raise ValueError(
                f"Calibrated Part-ID mask could not highlight {part_id} "
                f"in {view_id!r}"
            )
        output_path = output_dir / f"{part_id}_{view_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        highlighted.save(output_path)
        generated[part_id] = str(output_path.resolve(strict=True))
    return generated


def _mvinverse_frame_indices(ledger: dict[str, Any]) -> dict[str, int]:
    """Recover the only permitted frame order from the adapter ledger."""

    inputs = ledger.get("inputs")
    source_views = inputs.get("source_views") if isinstance(inputs, dict) else None
    if not isinstance(source_views, list) or not source_views:
        raise ValueError("MVInverse ledger has no source-view frame mapping")
    mapping: dict[str, int] = {}
    seen_indices: set[int] = set()
    for record in source_views:
        if not isinstance(record, dict):
            raise ValueError("MVInverse ledger source-view record is invalid")
        view_id = record.get("view_id")
        index = record.get("index")
        if (
            not isinstance(view_id, str)
            or not view_id
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or view_id in mapping
            or index in seen_indices
        ):
            raise ValueError("MVInverse ledger frame mapping is invalid or duplicated")
        mapping[view_id] = index
        seen_indices.add(index)
    if seen_indices != set(range(len(source_views))):
        raise ValueError("MVInverse ledger frame indices must be contiguous from zero")
    return mapping


def _mvinverse_groups_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = report.get("groups")
    if not isinstance(groups, list):
        raise ValueError("MVInverse evidence report has no fused groups")
    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_id"), str):
            raise ValueError("MVInverse fused group is invalid")
        if group["group_id"] in result:
            raise ValueError(f"Duplicate MVInverse fused group: {group['group_id']}")
        result[group["group_id"]] = group
    return result


def _mvinverse_view_groups(
    report: dict[str, Any], group_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Return accepted, independently imaged regions for one canonical group."""

    records: list[tuple[str, dict[str, Any]]] = []
    views = report.get("views")
    if not isinstance(views, list):
        raise ValueError("MVInverse evidence report has no view records")
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("view_id"), str):
            raise ValueError("MVInverse evidence view record is invalid")
        groups = view.get("groups")
        if not isinstance(groups, list):
            raise ValueError("MVInverse evidence view has no group records")
        matching = [
            item
            for item in groups
            if isinstance(item, dict) and item.get("group_id") == group_id
        ]
        if len(matching) != 1:
            raise ValueError(
                f"MVInverse evidence must contain one {group_id} record per view"
            )
        record = matching[0]
        if record.get("accepted") is True:
            records.append((view["view_id"], record))
    return records


def _group_with_pbr_context(
    group: dict[str, Any],
    *,
    fused: dict[str, Any] | None,
    view_record: dict[str, Any] | None = None,
    allow_parameter_writes: bool = True,
) -> dict[str, Any]:
    """Attach bounded numeric evidence to the Qwen choice prompt."""

    if not isinstance(allow_parameter_writes, bool):
        raise ValueError("allow_parameter_writes must be a boolean")
    if fused is None and view_record is None and allow_parameter_writes:
        return group
    context: dict[str, Any] = {
        "source": "MVInverse image-space prediction; supporting evidence only",
        "selected_mdl_parameters_mutable": allow_parameter_writes,
        "library_defaults_must_match_reference": not allow_parameter_writes,
    }
    if fused is not None:
        suggestion = fused.get("suggestion")
        context.update(
            {
                "surface_class": fused.get("surface_class"),
                "distinct_view_count": fused.get("distinct_view_count"),
                "fused_albedo": (
                    fused.get("albedo", {}).get("median")
                    if isinstance(fused.get("albedo"), dict)
                    else None
                ),
                "fused_metallic": (
                    fused.get("metallic", {}).get("median")
                    if isinstance(fused.get("metallic"), dict)
                    else None
                ),
                "fused_roughness": (
                    fused.get("roughness", {}).get("median")
                    if isinstance(fused.get("roughness"), dict)
                    else None
                ),
                "parameter_decision": (
                    suggestion.get("decision") if isinstance(suggestion, dict) else None
                ),
            }
        )
    if view_record is not None:
        context["current_view_albedo"] = (
            view_record.get("albedo", {}).get("median")
            if isinstance(view_record.get("albedo"), dict)
            else None
        )
        context["current_view_metallic"] = (
            view_record.get("metallic", {}).get("median")
            if isinstance(view_record.get("metallic"), dict)
            else None
        )
        context["current_view_roughness"] = (
            view_record.get("roughness", {}).get("median")
            if isinstance(view_record.get("roughness"), dict)
            else None
        )
    return {**group, "mvinverse_pbr_context": context}


def _retrieval_margin_for_choice(
    choice: dict[str, Any], retrieval_audit: dict[str, Any]
) -> float | None:
    ranking = retrieval_audit.get("ranking")
    if (
        retrieval_audit.get("margin_available") is True
        and isinstance(ranking, list)
        and ranking
        and isinstance(ranking[0], dict)
        and ranking[0].get("material_id") == choice.get("material_id")
        and isinstance(retrieval_audit.get("normalized_margin"), (int, float))
        and not isinstance(retrieval_audit.get("normalized_margin"), bool)
    ):
        return float(retrieval_audit["normalized_margin"])
    return None


def _derive_material_selection_confidence(
    *,
    first: dict[str, Any],
    second: dict[str, Any],
    chosen: dict[str, Any],
    confirmed: bool,
    confirmation_basis: str,
    retrieval_audit: dict[str, Any],
    independent_choices: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Derive authoring confidence from independent evidence, not self-report.

    Qwen's numeric confidence is retained in ``first``, ``second`` and each
    per-view record for audit only.  It is not a calibrated authoring signal:
    a numeric example in the prompt can anchor every response to the same
    value.  Forward/reverse agreement therefore grants only review confidence.
    Automatic confidence additionally requires two distinct reference views,
    the same exact resolved MDL, and an independently computed retrieval-top
    margin.
    """

    if not isinstance(confirmed, bool):
        raise ValueError("confirmed must be a boolean")
    chosen_id = str(chosen.get("material_id") or "")
    if not chosen_id:
        raise ValueError("chosen material has no material_id")
    reported_forward = float(first["confidence"])
    reported_reverse = float(second["confidence"])
    supporting_view_ids = sorted(
        {
            str(record["view_id"])
            for record in independent_choices
            if isinstance(record, dict)
            and isinstance(record.get("view_id"), str)
            and record["view_id"]
            and record.get("material_id") == chosen_id
        }
    )
    ranking = retrieval_audit.get("ranking")
    chosen_rank = (
        next(
            (
                int(record["rank"])
                for record in ranking
                if isinstance(record, dict)
                and record.get("material_id") == chosen_id
                and isinstance(record.get("rank"), int)
                and not isinstance(record.get("rank"), bool)
            ),
            None,
        )
        if isinstance(ranking, list)
        else None
    )
    margin = (
        float(retrieval_audit["normalized_margin"])
        if retrieval_audit.get("margin_available") is True
        and isinstance(retrieval_audit.get("normalized_margin"), (int, float))
        and not isinstance(retrieval_audit.get("normalized_margin"), bool)
        else None
    )
    retrieval_top_supported = chosen_rank == 1
    strong_independent_support = (
        confirmed
        and len(supporting_view_ids) >= 2
        and retrieval_top_supported
        and margin is not None
        and margin >= MATERIAL_SELECTION_AUTO_MINIMUM_RETRIEVAL_MARGIN
    )
    if not confirmed:
        derived = 0.0
        decision = "unconfirmed_preserve"
        reason_codes = ["material_choice_not_confirmed"]
    elif strong_independent_support:
        derived = MATERIAL_SELECTION_AUTO_CONFIDENCE
        decision = "independent_multimodel_auto"
        reason_codes = [
            "forward_reverse_resolution_confirmed",
            "two_independent_reference_views_agree",
            "retrieval_top_agrees",
            "retrieval_margin_meets_auto_floor",
        ]
    else:
        derived = MATERIAL_SELECTION_REVIEW_CONFIDENCE
        decision = "order_stable_review"
        reason_codes = ["forward_reverse_resolution_confirmed"]
        if len(supporting_view_ids) < 2:
            reason_codes.append("insufficient_independent_reference_views")
        if not retrieval_top_supported:
            reason_codes.append("retrieval_top_disagrees")
        if margin is None:
            reason_codes.append("retrieval_margin_unavailable")
        elif margin < MATERIAL_SELECTION_AUTO_MINIMUM_RETRIEVAL_MARGIN:
            reason_codes.append("retrieval_margin_below_auto_floor")
    audit = {
        "schema_version": MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION,
        "derived_confidence": derived,
        "decision": decision,
        "confirmation_basis": confirmation_basis,
        "reported_forward_confidence": reported_forward,
        "reported_reverse_confidence": reported_reverse,
        "reported_confidence_is_authoritative": False,
        "supporting_independent_view_ids": supporting_view_ids,
        "minimum_independent_view_count": 2,
        "chosen_retrieval_rank": chosen_rank,
        "retrieval_top_supported": retrieval_top_supported,
        "normalized_retrieval_margin": margin,
        "minimum_auto_retrieval_margin": (
            MATERIAL_SELECTION_AUTO_MINIMUM_RETRIEVAL_MARGIN
        ),
        "reason_codes": reason_codes,
    }
    return derived, audit


def _validate_qwen35_runtime_manifest(
    model_root: Path,
    checkpoint_identity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate the exact Qwen3.5 runtime file set against its trust anchor."""

    runtime_files = checkpoint_identity.get("runtime_files")
    if (
        not isinstance(runtime_files, list)
        or _canonical_sha256(runtime_files) != QWEN35_CONTENT_MANIFEST_SHA256
    ):
        raise ValueError(
            "Qwen3.5 checkpoint runtime manifest does not match the pinned digest"
        )
    records: dict[str, dict[str, Any]] = {}
    for raw in runtime_files:
        path_text = raw.get("path") if isinstance(raw, dict) else None
        size = raw.get("bytes") if isinstance(raw, dict) else None
        sha256 = raw.get("sha256") if isinstance(raw, dict) else None
        if (
            not isinstance(path_text, str)
            or not path_text
            or path_text in records
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError(
                "Qwen3.5 checkpoint contains an invalid runtime file record"
            )
        relative = Path(path_text)
        source_path = model_root / relative
        try:
            resolved = source_path.resolve(strict=True)
            resolved.relative_to(model_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Qwen3.5 runtime file escapes the checkpoint: {path_text!r}"
            ) from exc
        if (
            relative.is_absolute()
            or source_path.is_symlink()
            or not resolved.is_file()
            or relative.as_posix() != path_text
        ):
            raise ValueError(
                f"Qwen3.5 runtime file is not a regular relative file: {path_text!r}"
            )
        records[path_text] = raw

    identity_path = model_root / "checkpoint_identity.json"
    actual_files = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and path != identity_path
    }
    if actual_files != set(records):
        raise ValueError(
            "Qwen3.5 checkpoint runtime file set differs from the pinned manifest"
        )
    for path_text, record in records.items():
        path = model_root / path_text
        if (
            path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError(
                f"Qwen3.5 runtime file failed size/SHA-256 validation: {path_text}"
            )
    if checkpoint_identity.get("config_sha256") != records.get("config.json", {}).get(
        "sha256"
    ):
        raise ValueError(
            "Qwen3.5 checkpoint config identity differs from the runtime manifest"
        )
    return records


def _qwen_inference_ledger(
    *,
    model_identity: dict[str, Any],
    requested_family: str,
    requested_revision: str | None,
    palette_max_new_tokens_ceiling: int | None = None,
    minimum_usable_palette_views: int = 1,
    minimum_usable_palette_view_ratio: float = 0.0,
) -> dict[str, Any]:
    model_type = model_identity.get("model_type")
    expected_model_type = {
        "auto": model_type,
        "qwen3_vl": "qwen3_vl",
        "qwen3_5": "qwen3_5",
        "openai_compatible": "openai_compatible",
    }.get(requested_family)
    if expected_model_type is None:
        raise ValueError(f"Unsupported Qwen model family: {requested_family}")
    if model_type != expected_model_type:
        raise ValueError(
            "Qwen model family does not match the local checkpoint: "
            f"requested={requested_family!r}, model_type={model_type!r}"
        )
    if model_type == "qwen3_5" and requested_revision != QWEN35_CANONICAL_REVISION:
        raise ValueError(
            "Qwen3.5 requires the pinned --qwen-model-revision "
            f"{QWEN35_CANONICAL_REVISION}; found {requested_revision!r}"
        )
    frozen_identity = copy.deepcopy(model_identity)
    model_path_raw = frozen_identity.get("model_path")
    weights = frozen_identity.get("weights")
    if isinstance(model_path_raw, str) and isinstance(weights, dict):
        model_root = Path(model_path_raw).expanduser().resolve(strict=True)
        files = weights.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Qwen model identity contains no weight files")
        for record in files:
            relative = record.get("path") if isinstance(record, dict) else None
            if not isinstance(relative, str) or not relative:
                raise ValueError("Qwen model identity has an invalid weight path")
            weight_path = (model_root / relative).resolve(strict=True)
            try:
                weight_path.relative_to(model_root)
            except ValueError as exc:
                raise ValueError("Qwen weight identity escapes model root") from exc
            record["sha256"] = _sha256_file(weight_path)
        weights["full_file_sha256_complete"] = True
        weights["sample_fingerprint"] = weights.get("fingerprint")
        weights["full_fingerprint"] = _canonical_sha256(
            {
                "files": files,
                "indexes": weights.get("indexes", []),
                "full_file_sha256_complete": True,
            }
        )
        weights["fingerprint"] = weights["full_fingerprint"]
        if requested_revision is not None:
            checkpoint_identity_path = model_root / "checkpoint_identity.json"
            if (
                not checkpoint_identity_path.is_file()
                or checkpoint_identity_path.is_symlink()
            ):
                raise ValueError(
                    "Qwen checkpoint has no regular checkpoint_identity.json; "
                    "install the pinned model with scripts/setup_qwen35_runtime.sh"
                )
            checkpoint_identity_path = checkpoint_identity_path.resolve(strict=True)
            try:
                checkpoint_identity_path.relative_to(model_root)
            except ValueError as exc:
                raise ValueError(
                    "Qwen checkpoint identity escapes the model root"
                ) from exc
            checkpoint_identity = _read_json(checkpoint_identity_path)
            expected_repository = (
                QWEN35_CANONICAL_REPOSITORY
                if model_type == "qwen3_5"
                else checkpoint_identity.get("repository")
            )
            expected_manifest_sha256 = (
                QWEN35_CONTENT_MANIFEST_SHA256
                if model_type == "qwen3_5"
                else checkpoint_identity.get("content_manifest_sha256")
            )
            if (
                checkpoint_identity.get("schema_version") != "qwen-local-checkpoint/v1"
                or checkpoint_identity.get("repository") != expected_repository
                or checkpoint_identity.get("revision") != requested_revision
                or checkpoint_identity.get("config_sha256")
                != _sha256_file(model_root / "config.json")
                or checkpoint_identity.get("content_manifest_sha256")
                != expected_manifest_sha256
                or (
                    model_type == "qwen3_5"
                    and _canonical_sha256(checkpoint_identity.get("runtime_files"))
                    != expected_manifest_sha256
                )
            ):
                raise ValueError(
                    "Qwen local checkpoint identity does not match the requested "
                    "revision/config"
                )
            if model_type == "qwen3_5":
                expected_files = _validate_qwen35_runtime_manifest(
                    model_root,
                    checkpoint_identity,
                )
                for record in files:
                    expected = expected_files.get(record["path"])
                    if (
                        not isinstance(expected, dict)
                        or expected.get("bytes") != record.get("bytes")
                        or expected.get("sha256") != record.get("sha256")
                    ):
                        raise ValueError(
                            "Qwen local weight does not match the pinned "
                            f"checkpoint manifest: {record['path']}"
                        )
            frozen_identity["checkpoint_identity"] = checkpoint_identity
        frozen_identity["sample_fingerprint"] = frozen_identity.get("fingerprint")
        full_identity_payload = {
            key: value
            for key, value in frozen_identity.items()
            if key not in {"fingerprint", "full_fingerprint"}
        }
        frozen_identity["full_fingerprint"] = _canonical_sha256(full_identity_payload)
        frozen_identity["fingerprint"] = frozen_identity["full_fingerprint"]
    unsigned = {
        "schema_version": QWEN_LEDGER_SCHEMA_VERSION,
        "requested_model_family": requested_family,
        "requested_model_revision": requested_revision,
        "model_identity": frozen_identity,
        "palette_generation_policy": {
            "initial_max_new_tokens": frozen_identity.get("generation", {}).get(
                "max_new_tokens"
            ),
            "max_new_tokens_ceiling": (
                palette_max_new_tokens_ceiling
                if palette_max_new_tokens_ceiling is not None
                else frozen_identity.get("generation", {}).get("max_new_tokens")
            ),
            "truncation_growth_factor": 2,
            "retry_condition": "token_limit_reached_without_eos",
            "minimum_usable_views": minimum_usable_palette_views,
            "minimum_usable_view_ratio": minimum_usable_palette_view_ratio,
        },
        "backend_lock_policy": (
            "preflight_once_no_remote_code_no_midrun_backend_switch"
        ),
    }
    return {
        **unsigned,
        "integrity": {"ledger_sha256": _canonical_sha256(unsigned)},
    }


def _write_or_validate_qwen_ledger(
    path: Path,
    *,
    ledger: dict[str, Any],
    resume: bool,
) -> Path:
    if path.is_file():
        persisted = _read_json(path)
        if persisted != ledger:
            raise ValueError(
                "Qwen checkpoint/resume rejected: model, processor, weights, "
                "runtime, or generation contract changed"
            )
        return path.resolve(strict=True)
    if resume:
        raise ValueError(
            "Qwen checkpoint/resume rejected: qwen_inference_ledger.json is missing"
        )
    return _write_json(path, ledger)


def _build_sam3_region_request(
    *,
    parsed_references: list[tuple[str, Path]],
    palette_candidates: list[dict[str, Any]],
    palette: dict[str, Any],
    palette_fusion: dict[str, Any] | None,
    selected_reference_id: str,
) -> dict[str, Any]:
    source_paths = dict(parsed_references)
    region_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if palette_fusion is not None:
        raw_groups = palette_fusion.get("canonical_palette", {}).get("groups")
        if not isinstance(raw_groups, list):
            raise ValueError("Palette fusion has no canonical groups for SAM3")
        for canonical_group in raw_groups:
            if not isinstance(canonical_group, dict):
                raise ValueError("Palette fusion canonical group is invalid")
            group_id = canonical_group.get("group_id")
            sources = canonical_group.get("sources")
            if not isinstance(group_id, str) or not isinstance(sources, list):
                raise ValueError("Palette fusion group lacks sources for SAM3")
            prompt = " ".join(
                str(canonical_group.get(key) or "")
                for key in (
                    "visual_description",
                    "family_hint",
                    "base_color",
                    "finish_hint",
                )
            ).strip()
            for source in sources:
                if not isinstance(source, dict):
                    continue
                view_id = source.get("view_id")
                local_group_id = source.get("local_group_id")
                boxes = source.get("boxes")
                if (
                    not isinstance(view_id, str)
                    or view_id not in source_paths
                    or not isinstance(local_group_id, str)
                    or not isinstance(boxes, list)
                    or not boxes
                ):
                    continue
                key = (view_id, group_id)
                region = region_by_key.setdefault(
                    key,
                    {
                        "view_id": view_id,
                        "group_id": group_id,
                        "local_group_id": local_group_id,
                        "prompt": prompt or "visual material surface",
                        "boxes": [],
                    },
                )
                for box in boxes:
                    if box not in region["boxes"]:
                        region["boxes"].append(copy.deepcopy(box))
    else:
        selected = next(
            (
                candidate
                for candidate in palette_candidates
                if candidate.get("reference_id") == selected_reference_id
                and candidate.get("status") == "usable"
            ),
            None,
        )
        if selected is None:
            raise ValueError("Selected palette view is unavailable for SAM3")
        canonical_by_id = {
            group["group_id"]: group for group in palette.get("groups", [])
        }
        for local_group in selected["palette"]["groups"]:
            group_id = local_group["group_id"]
            canonical = canonical_by_id.get(group_id, local_group)
            prompt = " ".join(
                str(canonical.get(key) or "")
                for key in (
                    "visual_description",
                    "family_hint",
                    "base_color",
                    "finish_hint",
                )
            ).strip()
            region_by_key[(selected_reference_id, group_id)] = {
                "view_id": selected_reference_id,
                "group_id": group_id,
                "local_group_id": group_id,
                "prompt": prompt or "visual material surface",
                "boxes": copy.deepcopy(local_group["boxes"]),
            }
    regions = [
        region_by_key[key]
        for key in sorted(region_by_key)
        if region_by_key[key]["boxes"]
    ]
    if not regions:
        raise ValueError("No bounded material regions are available for SAM3")
    return {
        "schema_version": SAM3_REQUEST_SCHEMA_VERSION,
        "source_views": [
            {"id": view_id, "image": str(path)} for view_id, path in parsed_references
        ],
        "regions": regions,
        "prompt_authority": (
            "qwen_palette_boxes_and_visual_description_geometry_prompt"
        ),
    }


def _build_sam3_foreground_request(
    parsed_references: list[tuple[str, Path]],
    *,
    annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model-independent whole-object request before any VLM call."""

    if annotations is not None:
        annotation_schema = annotations.get("schema_version")
        if annotation_schema == SAM3_HUMAN_ANNOTATION_SCHEMA_VERSION:
            request_schema = SAM3_ORDERED_POINT_REQUEST_SCHEMA_VERSION
        elif annotation_schema == SAM3_LEGACY_HUMAN_ANNOTATION_SCHEMA_VERSION:
            request_schema = SAM3_POINT_REQUEST_SCHEMA_VERSION
        else:
            raise ValueError(
                "Human SAM3 foreground annotations use an unsupported schema"
            )
        views = annotations.get("source_views")
        if not isinstance(views, list):
            raise ValueError("Human SAM3 foreground annotations have no source_views")
        by_id = {str(view.get("id")): view for view in views if isinstance(view, dict)}
        expected_ids = [view_id for view_id, _path in parsed_references]
        if set(by_id) != set(expected_ids):
            raise ValueError(
                "Human SAM3 foreground annotations do not exactly cover references"
            )
        return {
            "schema_version": request_schema,
            "source_views": [
                {"id": view_id, "image": str(path)}
                for view_id, path in parsed_references
            ],
            "regions": [
                {
                    "view_id": view_id,
                    "group_id": "__foreground__",
                    "local_group_id": None,
                    "click_sets": by_id[view_id]["click_sets"],
                    "confirmed_mask": by_id[view_id]["confirmed_mask"],
                }
                for view_id in expected_ids
            ],
            "prompt_authority": "human_confirmed_sam3_interactive_points",
            "human_annotation": {
                "schema_version": annotation_schema,
                "document_sha256": annotations["integrity"]["document_sha256"],
                "all_views_confirmed": True,
                "human_mask_is_authoritative": True,
                "formal_rerun_minimum_iou": 0.995,
            },
        }

    seed_records = [
        {
            "view_id": view_id,
            **build_automatic_foreground_seeds(path),
        }
        for view_id, path in parsed_references
    ]
    seeds_by_view = {str(record["view_id"]): record for record in seed_records}
    return {
        "schema_version": SAM3_REQUEST_SCHEMA_VERSION,
        "source_views": [
            {"id": view_id, "image": str(path)} for view_id, path in parsed_references
        ],
        "regions": [
            {
                "view_id": view_id,
                "group_id": "__foreground__",
                "local_group_id": None,
                "prompt": ("manufactured object"),
                # Geometry is deliberately model-independent.  Tight proposals
                # suppress viewport backgrounds before SAM3 performs the final
                # semantic pixel segmentation.
                "boxes": seeds_by_view[view_id]["boxes"],
            }
            for view_id, _path in parsed_references
        ],
        "prompt_authority": (
            "sam3_text_plus_automatic_image_geometry_no_vlm_no_material_assumption"
        ),
        "foreground_seed_policy": {
            "schema_version": SEED_POLICY_SCHEMA_VERSION,
            "method": (
                "border_median_color_distance_otsu_morphology_connected_components"
            ),
            "views": seed_records,
        },
    }


def _write_foreground_masked_references(
    *,
    parsed_references: list[tuple[str, Path]],
    foreground_masks: dict[str, Path],
    output_dir: Path,
) -> tuple[list[tuple[str, Path]], Path]:
    """Replace non-object pixels with neutral gray for all model inference."""

    from PIL import Image, ImageOps

    output_dir.mkdir(parents=True, exist_ok=True)
    masked_references: list[tuple[str, Path]] = []
    audit_views: list[dict[str, Any]] = []
    for view_id, source_path in parsed_references:
        mask_path = foreground_masks.get(view_id)
        if mask_path is None:
            raise ValueError(
                f"SAM3 foreground segmentation produced no accepted mask for {view_id}"
            )
        with Image.open(source_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(mask_path) as opened_mask:
            mask = ImageOps.exif_transpose(opened_mask).convert("L")
        if mask.size != source.size:
            raise ValueError(
                "SAM3 foreground mask dimensions differ from source image: "
                f"view={view_id}, source={source.size}, mask={mask.size}"
            )
        neutral = Image.new("RGB", source.size, (127, 127, 127))
        masked = Image.composite(source, neutral, mask)
        masked_path = output_dir / f"{_artifact_slug(view_id)}.foreground.png"
        masked.save(masked_path)
        resolved_masked = masked_path.resolve(strict=True)
        masked_references.append((view_id, resolved_masked))
        audit_views.append(
            {
                "id": view_id,
                "source_image": str(source_path),
                "source_image_sha256": _sha256_file(source_path),
                "foreground_mask": str(mask_path),
                "foreground_mask_sha256": _sha256_file(mask_path),
                "model_image": str(resolved_masked),
                "model_image_sha256": _sha256_file(resolved_masked),
                "background_fill_rgb": [127, 127, 127],
            }
        )
    audit_path = _write_json(
        output_dir.parent / "foreground_inference_manifest.json",
        {
            "schema_version": "sam3-foreground-inference-manifest/v1",
            "policy": ("all_vlm_and_mvinverse_inputs_are_sam3_foreground_only"),
            "source_views": audit_views,
        },
    )
    return masked_references, audit_path


def _validate_sam3_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    request_path: Path,
    repository: Path,
    checkpoint: Path,
    device: str,
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    inference_seed: int = SAM3_DEFAULT_INFERENCE_SEED,
) -> dict[tuple[str, str], Path]:
    if manifest.get("schema_version") != SAM3_RESULT_SCHEMA_VERSION:
        raise ValueError("SAM3 result uses an unsupported schema")
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("SAM3 result has no integrity record")
    unsigned = {key: value for key, value in manifest.items() if key != "integrity"}
    if integrity.get("result_sha256") != _canonical_sha256(unsigned):
        raise ValueError("SAM3 result integrity hash mismatch")
    request_document = _read_json(request_path)
    raw_request_views = request_document.get("source_views")
    if not isinstance(raw_request_views, list) or not raw_request_views:
        raise ValueError("SAM3 request contains no source views")
    expected_sources: dict[str, Path] = {}
    for raw_view in raw_request_views:
        if not isinstance(raw_view, dict) or not isinstance(raw_view.get("id"), str):
            raise ValueError("SAM3 request contains an invalid source view")
        view_id = raw_view["id"]
        image_raw = raw_view.get("image")
        if not isinstance(image_raw, str) or not image_raw:
            raise ValueError("SAM3 request source view has no image")
        image_path = Path(image_raw).expanduser()
        if not image_path.is_absolute():
            image_path = request_path.parent / image_path
        image_path = image_path.resolve(strict=True)
        if view_id in expected_sources:
            raise ValueError("SAM3 request duplicates a source view")
        expected_sources[view_id] = image_path
    raw_request_regions = request_document.get("regions")
    if not isinstance(raw_request_regions, list) or not raw_request_regions:
        raise ValueError("SAM3 request contains no regions")
    expected_region_records: dict[tuple[str, str], dict[str, Any]] = {}
    for region in raw_request_regions:
        if (
            not isinstance(region, dict)
            or not isinstance(region.get("view_id"), str)
            or not isinstance(region.get("group_id"), str)
            or region["view_id"] not in expected_sources
        ):
            raise ValueError("SAM3 request contains an invalid region")
        identity = (region["view_id"], region["group_id"])
        if identity in expected_region_records:
            raise ValueError("SAM3 request contains duplicate regions")
        expected_region_records[identity] = region
    expected_regions = set(expected_region_records)
    if len(expected_regions) != len(raw_request_regions):
        raise ValueError("SAM3 request contains invalid or duplicate regions")
    request_record = manifest.get("request")
    if (
        not isinstance(request_record, dict)
        or request_record.get("sha256")
        != hashlib.sha256(request_path.read_bytes()).hexdigest()
        or request_record.get("document_sha256") != _canonical_sha256(request_document)
    ):
        raise ValueError("SAM3 result is stale for the current region request")
    backend = manifest.get("backend")
    repository = repository.expanduser().resolve(strict=True)
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    request_schema = request_document.get("schema_version")
    expected_human_interactivity = any(
        bool(region.get("click_sets")) for region in raw_request_regions
    )
    if expected_human_interactivity and request_schema not in {
        SAM3_POINT_REQUEST_SCHEMA_VERSION,
        SAM3_ORDERED_POINT_REQUEST_SCHEMA_VERSION,
    }:
        raise ValueError("SAM3 interactive regions use an unsupported request schema")
    expected_automatic_shape_interactivity = any(
        isinstance(region.get("cad_projection_seed"), dict)
        or isinstance(region.get("cad_amodal_template"), dict)
        for region in raw_request_regions
    )
    expected_instance_interactivity = (
        expected_human_interactivity or expected_automatic_shape_interactivity
    )
    expected_ordered_interactivity = (
        request_schema == SAM3_ORDERED_POINT_REQUEST_SCHEMA_VERSION
    )
    if (
        not isinstance(backend, dict)
        or Path(str(backend.get("repository", ""))).resolve() != repository
        or backend.get("repository_revision") != _git_revision(repository)
        or Path(str(backend.get("checkpoint", ""))).resolve() != checkpoint
        or backend.get("checkpoint_sha256") != _sha256_file(checkpoint)
        or backend.get("device") != device
        or backend.get("instance_interactivity_enabled", False)
        is not expected_instance_interactivity
    ):
        raise ValueError("SAM3 result backend does not match the current frozen model")
    expected_policy = sam3_result_policy(
        minimum_model_score=minimum_model_score,
        minimum_prompt_overlap=minimum_prompt_overlap,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
        human_interactive_requested=expected_human_interactivity,
        automatic_shape_interactive_requested=(
            expected_automatic_shape_interactivity
        ),
        ordered_interaction_requested=expected_ordered_interactivity,
        inference_seed=inference_seed,
    )
    if manifest.get("policy") != expected_policy:
        raise ValueError("SAM3 result policy does not match the current request")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("SAM3 result records are invalid")
    accepted: dict[tuple[str, str], Path] = {}
    accepted_arrays: dict[tuple[str, str], Any] = {}
    seen_regions: set[tuple[str, str]] = set()
    source_hashes: dict[Path, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("SAM3 result contains a malformed record")
        source_raw = record.get("source_image")
        source_digest = record.get("source_image_sha256")
        if not isinstance(source_raw, str) or not isinstance(source_digest, str):
            raise ValueError("SAM3 result record has no source image identity")
        source_path = Path(source_raw).expanduser().resolve(strict=True)
        actual_source_digest = source_hashes.setdefault(
            source_path, _sha256_file(source_path)
        )
        if source_digest != actual_source_digest:
            raise ValueError(
                f"SAM3 source image changed after inference: {source_path}"
            )
        view_id = record.get("view_id")
        group_id = record.get("group_id")
        if not isinstance(view_id, str) or not isinstance(group_id, str):
            raise ValueError("SAM3 result record has an invalid region identity")
        identity = (view_id, group_id)
        if identity in seen_regions:
            raise ValueError(f"SAM3 result duplicates a region: {identity}")
        seen_regions.add(identity)
        expected_source = expected_sources.get(view_id)
        if expected_source is None or source_path != expected_source:
            raise ValueError(
                "SAM3 result source image does not match the current request"
            )
        expected_region = expected_region_records.get(identity)
        if expected_region is None:
            raise ValueError("SAM3 result contains a region not present in the request")
        expected_confirmed_array = None
        if expected_human_interactivity:
            expected_click_sets = expected_region.get("click_sets")
            expected_mask = expected_region.get("confirmed_mask")
            expected_prompt_mode = (
                "human_ordered_incremental_points"
                if expected_ordered_interactivity
                else "human_interactive_points"
            )
            expected_replay_mode = (
                "ordered_events_previous_logits"
                if expected_ordered_interactivity
                else "unordered_single_call"
            )
            expected_event_count = (
                sum(len(click_set["events"]) for click_set in expected_click_sets)
                if expected_ordered_interactivity
                and isinstance(expected_click_sets, list)
                else 0
            )
            replay_metadata_valid = (
                record.get("interaction_replay_mode") == expected_replay_mode
                and record.get("event_count") == expected_event_count
                if expected_ordered_interactivity
                else record.get("interaction_replay_mode")
                in (None, expected_replay_mode)
                and record.get("event_count") in (None, 0)
            )
            if (
                not isinstance(expected_click_sets, list)
                or not expected_click_sets
                or not isinstance(expected_mask, dict)
                or record.get("prompt_mode") != expected_prompt_mode
                or not replay_metadata_valid
                or record.get("click_sets") != expected_click_sets
                or record.get("point_set_count") != len(expected_click_sets)
                or record.get("accepted_point_set_count") != len(expected_click_sets)
                or record.get("accepted") is not True
                or record.get("reason_codes") != []
            ):
                raise ValueError(
                    "Interactive SAM3 result does not reproduce every requested click set"
                )
            if expected_ordered_interactivity:
                point_set_audits = record.get("point_set_audits")
                if not isinstance(point_set_audits, list) or len(
                    point_set_audits
                ) != len(expected_click_sets):
                    raise ValueError(
                        "Ordered SAM3 result lacks one replay audit per click set"
                    )
                for click_set_index, (click_set, point_set_audit) in enumerate(
                    zip(expected_click_sets, point_set_audits)
                ):
                    event_audits = (
                        point_set_audit.get("event_audits")
                        if isinstance(point_set_audit, dict)
                        else None
                    )
                    if (
                        not isinstance(point_set_audit, dict)
                        or point_set_audit.get("click_set_index") != click_set_index
                        or point_set_audit.get("events") != click_set["events"]
                        or point_set_audit.get("initial_candidate_index")
                        != click_set["initial_candidate_index"]
                        or point_set_audit.get("event_count")
                        != len(click_set["events"])
                        or point_set_audit.get("accepted") is not True
                        or not isinstance(event_audits, list)
                        or len(event_audits) != len(click_set["events"])
                        or not isinstance(event_audits[-1], dict)
                        or event_audits[-1].get("accepted") is not True
                    ):
                        raise ValueError(
                            "Ordered SAM3 click-set replay audit is invalid"
                        )
                    for event_index, event_audit in enumerate(event_audits):
                        if not isinstance(event_audit, dict):
                            raise ValueError(
                                "Ordered SAM3 event replay audit is invalid"
                            )
                        first_event = event_index == 0
                        selected_candidate_is_valid = (
                            event_audit.get("selected_candidate_index")
                            == click_set["initial_candidate_index"]
                            and event_audit.get("candidate_selection")
                            == "persisted_initial_candidate"
                            if first_event
                            else event_audit.get("selected_candidate_index") == 0
                            and event_audit.get("candidate_selection")
                            == "single_mask_refinement"
                        )
                        candidates = event_audit.get("candidates")
                        if (
                            event_audit.get("event_index") != event_index
                            or event_audit.get("event")
                            != click_set["events"][event_index]
                            or event_audit.get("event_count") != event_index + 1
                            or event_audit.get("multimask_output")
                            is not (event_index == 0)
                            or event_audit.get("used_previous_logits")
                            is not (event_index > 0)
                            or not selected_candidate_is_valid
                            or not isinstance(candidates, list)
                            or len(candidates) != (3 if first_event else 1)
                        ):
                            raise ValueError(
                                "Ordered SAM3 event replay audit is invalid"
                            )
            confirmed_raw = expected_mask.get("path")
            confirmed_sha256 = expected_mask.get("sha256")
            if not isinstance(confirmed_raw, str) or not isinstance(
                confirmed_sha256, str
            ):
                raise ValueError(
                    "Interactive SAM3 request has an invalid confirmed mask"
                )
            confirmed_path = Path(confirmed_raw).expanduser()
            if not confirmed_path.is_absolute():
                confirmed_path = request_path.parent / confirmed_path
            confirmed_path = confirmed_path.resolve(strict=True)
            if _sha256_file(confirmed_path) != confirmed_sha256:
                raise ValueError("Human-confirmed SAM3 mask changed after the request")
            import numpy as np
            from PIL import Image

            with Image.open(confirmed_path) as opened:
                expected_confirmed_array = (
                    np.asarray(opened.convert("L"), dtype=np.uint8) > 0
                )
            confirmed_pixels = int(np.count_nonzero(expected_confirmed_array))
            replay_audit = record.get("confirmed_mask_audit")
            strict_reproduction_valid = (
                isinstance(replay_audit, dict)
                and isinstance(replay_audit.get("reproduction_iou"), (int, float))
                and not isinstance(replay_audit.get("reproduction_iou"), bool)
                and float(replay_audit["reproduction_iou"])
                >= CONFIRMED_MASK_STRICT_MINIMUM_IOU
            )
            bounded_reproduction_valid = (
                isinstance(replay_audit, dict)
                and replay_audit.get("acceptance_mode")
                == "bounded_human_confirmed"
                and replay_audit.get("bounded_minimum_precision")
                == CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                and replay_audit.get("bounded_minimum_recall")
                == CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
                and isinstance(
                    replay_audit.get("reproduction_precision"), (int, float)
                )
                and not isinstance(
                    replay_audit.get("reproduction_precision"), bool
                )
                and float(replay_audit["reproduction_precision"])
                >= CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                and isinstance(replay_audit.get("reproduction_recall"), (int, float))
                and not isinstance(replay_audit.get("reproduction_recall"), bool)
                and float(replay_audit["reproduction_recall"])
                >= CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
            )
            symmetric_reproduction_valid = (
                isinstance(replay_audit, dict)
                and replay_audit.get("acceptance_mode")
                == "symmetric_boundary_drift"
                and replay_audit.get("symmetric_minimum_reproduction_iou")
                == CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU
                and isinstance(
                    replay_audit.get("reproduction_iou"), (int, float)
                )
                and not isinstance(replay_audit.get("reproduction_iou"), bool)
                and float(replay_audit["reproduction_iou"])
                >= CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU
            )
            if (
                not isinstance(replay_audit, dict)
                or replay_audit.get("sha256") != confirmed_sha256
                or replay_audit.get("confirmed_mask_pixels") != confirmed_pixels
                or replay_audit.get("minimum_reproduction_iou")
                != CONFIRMED_MASK_STRICT_MINIMUM_IOU
                or not (
                    strict_reproduction_valid
                    or symmetric_reproduction_valid
                    or bounded_reproduction_valid
                )
                or replay_audit.get("accepted") is not True
                or replay_audit.get("authoritative_output") != "human_confirmed_mask"
            ):
                raise ValueError(
                    "Interactive SAM3 confirmed-mask reproduction audit is invalid"
                )
        if record.get("accepted") is not True:
            continue
        mask = record.get("mask")
        if (
            not isinstance(view_id, str)
            or not isinstance(group_id, str)
            or not isinstance(mask, dict)
            or not isinstance(mask.get("path"), str)
            or not isinstance(mask.get("sha256"), str)
        ):
            raise ValueError("Accepted SAM3 record is malformed")
        mask_path = Path(mask["path"]).expanduser()
        if not mask_path.is_absolute():
            mask_path = manifest_path.parent / mask_path
        mask_path = mask_path.resolve(strict=True)
        if _sha256_file(mask_path) != mask["sha256"]:
            raise ValueError(f"SAM3 accepted mask hash mismatch: {mask_path}")
        import numpy as np
        from PIL import Image

        with Image.open(mask_path) as opened:
            mask_array = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        if expected_confirmed_array is not None and not np.array_equal(
            mask_array, expected_confirmed_array
        ):
            raise ValueError(
                "Interactive SAM3 output differs from the human-confirmed mask"
            )
        mask_pixels = int(np.count_nonzero(mask_array))
        if (
            mask_pixels < minimum_mask_pixels
            or record.get("mask_pixels") != mask_pixels
        ):
            raise ValueError(f"SAM3 accepted mask pixel audit mismatch: {mask_path}")
        accepted[identity] = mask_path
        if group_id != "__foreground__":
            accepted_arrays[identity] = mask_array
    if seen_regions != expected_regions:
        raise ValueError(
            "SAM3 result does not exactly cover the current region request"
        )
    summary = manifest.get("summary")
    accepted_count = len(accepted)
    if summary != {
        "region_count": len(records),
        "accepted_region_count": accepted_count,
        "rejected_region_count": len(records) - accepted_count,
    }:
        raise ValueError("SAM3 result summary counts are invalid")
    accepted_identities = sorted(accepted_arrays)
    for offset, left_identity in enumerate(accepted_identities):
        left = accepted_arrays[left_identity]
        left_pixels = int(np.count_nonzero(left))
        for right_identity in accepted_identities[offset + 1 :]:
            if left_identity[0] != right_identity[0]:
                continue
            right = accepted_arrays[right_identity]
            if left.shape != right.shape:
                raise ValueError(
                    "SAM3 accepted masks from one view have different shapes"
                )
            intersection = int(np.count_nonzero(left & right))
            if intersection < minimum_mask_pixels:
                continue
            right_pixels = int(np.count_nonzero(right))
            iou = intersection / max(
                1,
                left_pixels + right_pixels - intersection,
            )
            if iou >= CROSS_GROUP_NEAR_DUPLICATE_IOU:
                raise ValueError(
                    "SAM3 accepted masks contain an unresolved cross-group "
                    "near duplicate: "
                    f"{left_identity!r} vs {right_identity!r}, iou={iou:.6f}"
                )
    return accepted


def _build_visual_retrieval_request(
    *,
    catalog: Path,
    material_root: Path,
    palette: dict[str, Any],
    pbr_groups: dict[str, dict[str, Any]],
    parsed_references: list[tuple[str, Path]],
    accepted_masks: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    reference_paths = dict(parsed_references)
    groups: list[dict[str, Any]] = []
    for group in palette["groups"]:
        group_id = group["group_id"]
        pbr = pbr_groups.get(group_id)
        descriptor: dict[str, Any] = {
            key: group.get(key)
            for key in (
                "visual_description",
                "family_hint",
                "base_color",
                "finish_hint",
            )
        }
        if isinstance(pbr, dict):
            descriptor["surface_class"] = pbr.get("surface_class")
            roughness = pbr.get("roughness")
            metallic = pbr.get("metallic")
            descriptor["roughness_hint"] = (
                roughness.get("median") if isinstance(roughness, dict) else None
            )
            descriptor["metallicity_hint"] = (
                metallic.get("median") if isinstance(metallic, dict) else None
            )
        observations = [
            {
                "view_id": view_id,
                "image": str(reference_paths[view_id]),
                "mask": str(mask_path),
            }
            for (view_id, candidate_group_id), mask_path in sorted(
                accepted_masks.items()
            )
            if candidate_group_id == group_id and view_id in reference_paths
        ]
        groups.append(
            {
                "group_id": group_id,
                "descriptor": descriptor,
                "observations": observations,
            }
        )
    return {
        "schema_version": VISUAL_RETRIEVAL_REQUEST_SCHEMA_VERSION,
        "catalog": str(catalog),
        "material_root": str(material_root),
        "groups": groups,
        "candidate_scope": "complete_catalog_exact_allowlist",
        "final_authority": "exact_mdl_render_tournament",
    }


def _validate_visual_retrieval_result(
    result: dict[str, Any],
    *,
    request_path: Path,
    catalog_path: Path,
    material_root: Path,
    siglip2_model_path: Path,
    dinov2_model_path: Path,
    retrieval_python: Path,
    siglip_top_k: int,
    final_top_k: int,
    batch_size: int,
    device: str,
    allowed_material_ids: set[str],
    observation_bank_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if result.get("schema_version") != VISUAL_RETRIEVAL_RESULT_SCHEMA_VERSION:
        raise ValueError("Visual retrieval result uses an unsupported schema")
    integrity = result.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Visual retrieval result has no integrity record")
    unsigned = {key: value for key, value in result.items() if key != "integrity"}
    if integrity.get("result_sha256") != _canonical_sha256(unsigned):
        raise ValueError("Visual retrieval result integrity hash mismatch")
    request_record = result.get("request")
    if (
        not isinstance(request_record, dict)
        or request_record.get("sha256")
        != hashlib.sha256(request_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Visual retrieval result is stale for the current request")
    catalog_record = result.get("catalog")
    catalog_path = catalog_path.expanduser().resolve(strict=True)
    material_root = material_root.expanduser().resolve(strict=True)
    if (
        not isinstance(catalog_record, dict)
        or Path(str(catalog_record.get("path", ""))).resolve() != catalog_path
        or catalog_record.get("sha256") != _sha256_file(catalog_path)
        or Path(str(catalog_record.get("material_root", ""))).resolve() != material_root
        or catalog_record.get("all_catalog_materials_indexed") is not True
        or catalog_record.get("material_count") != len(allowed_material_ids)
    ):
        raise ValueError("Visual retrieval did not index the exact complete catalog")
    backends = result.get("backends")
    if not isinstance(backends, dict):
        raise ValueError("Visual retrieval result has no backend identities")
    if backends.get("runtime") != _isolated_retrieval_runtime_identity(
        retrieval_python
    ):
        raise ValueError("Visual retrieval Python/runtime identity changed")
    siglip_backend = backends.get("siglip2")
    dino_backend = backends.get("dinov2")
    current_materials = _load_catalog(catalog_path, material_root)
    siglip_identity = _verified_siglip2_model_identity(
        siglip2_model_path.expanduser().resolve(strict=True)
    )
    dino_identity = _model_fingerprint(
        dinov2_model_path.expanduser().resolve(strict=True)
    )
    if (
        not isinstance(siglip_backend, dict)
        or siglip_backend.get("model") != siglip_identity
        or not isinstance(dino_backend, dict)
        or dino_backend.get("model") != dino_identity
    ):
        raise ValueError(
            "Visual retrieval result model identity changed; cached ranking rejected"
        )
    expected_fusion = LEGACY_FUSION_POLICY
    expected_gallery_source = "catalog_thumbnails_and_text"
    if observation_bank_path is not None:
        bank = _load_base_observation_bank(
            bank_dir=observation_bank_path,
            material_root=material_root,
            material_ids=[str(item["material_id"]) for item in current_materials],
            siglip_model_identity=siglip_identity,
            dino_model_identity=dino_identity,
        )
        if (
            siglip_backend.get("index_source")
            != "nvidia_base_observation_bank"
            or siglip_backend.get("observation_bank") != bank["identity"]
            or dino_backend.get("index_source")
            != "nvidia_base_observation_bank"
            or dino_backend.get("observation_bank") != bank["identity"]
        ):
            raise ValueError(
                "Visual retrieval result does not use the configured Base "
                "observation bank"
            )
        expected_fusion = BASE_BANK_FUSION_POLICY
        expected_gallery_source = "nvidia_base_observation_bank"
    else:
        current_catalog_digest, _catalog_records = _catalog_digest(
            current_materials
        )
        if siglip_backend.get("catalog_digest") != current_catalog_digest:
            raise ValueError(
                "Visual retrieval catalog-thumbnail index changed"
            )
    policy = result.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("siglip_top_k") != min(siglip_top_k, len(allowed_material_ids))
        or policy.get("final_top_k") != final_top_k
        or policy.get("batch_size") != batch_size
        or policy.get("device") != device
        or policy.get("fusion") != expected_fusion
        or policy.get("gallery_source") != expected_gallery_source
        or policy.get("final_authority") != "exact_mdl_render_tournament"
        or policy.get("missing_mask_policy") != "fail_closed_to_legacy_retrieval"
    ):
        raise ValueError("Visual retrieval result policy changed")
    groups = result.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Visual retrieval group results are invalid")
    request_document = _read_json(request_path)
    expected_group_ids = [
        str(group.get("group_id"))
        for group in request_document.get("groups", [])
        if isinstance(group, dict)
    ]
    if len(expected_group_ids) != len(request_document.get("groups", [])) or len(
        expected_group_ids
    ) != len(set(expected_group_ids)):
        raise ValueError("Visual retrieval request contains invalid groups")
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_id"), str):
            raise ValueError("Visual retrieval contains an invalid group")
        group_id = group["group_id"]
        if group_id in by_id:
            raise ValueError(f"Visual retrieval duplicates group {group_id}")
        ranking = group.get("fused_ranking")
        expected_strategy = (
            BASE_BANK_RETRIEVAL_STRATEGY
            if observation_bank_path is not None
            else LEGACY_RETRIEVAL_STRATEGY
        )
        if group.get("retrieval_strategy") != expected_strategy:
            raise ValueError(
                f"Visual retrieval strategy is invalid for {group_id}"
            )
        if not isinstance(ranking, list):
            raise ValueError(f"Visual retrieval ranking is invalid for {group_id}")
        ranked_ids = [
            row.get("material_id") if isinstance(row, dict) else None for row in ranking
        ]
        if any(
            not isinstance(material_id, str) or material_id not in allowed_material_ids
            for material_id in ranked_ids
        ) or len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"Visual retrieval ranking escapes catalog for {group_id}")
        by_id[group_id] = group
    if set(by_id) != set(expected_group_ids):
        raise ValueError(
            "Visual retrieval result does not exactly cover the current groups"
        )
    return by_id


def _visual_candidates_with_audit(
    *,
    group_id: str,
    visual_group: dict[str, Any],
    pool: list[dict[str, Any]],
    limit: int,
    fallback_candidates: list[dict[str, Any]],
    fallback_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranking = visual_group.get("fused_ranking")
    if visual_group.get("accepted") is not True or not isinstance(ranking, list):
        return fallback_candidates, fallback_audit
    pool_by_id = {str(item["material_id"]): item for item in pool}
    selected: list[dict[str, Any]] = []
    audit_ranking: list[dict[str, Any]] = []
    for row in ranking[:limit]:
        if not isinstance(row, dict):
            continue
        material_id = row.get("material_id")
        item = pool_by_id.get(str(material_id))
        if item is None:
            continue
        candidate = dict(item)
        candidate["surface_interpretation"] = _surface_interpretation(item)
        candidate["retrieval_rank"] = len(selected) + 1
        candidate["retrieval_score"] = float(row["score"])
        retrieval_strategy = visual_group.get("retrieval_strategy")
        if retrieval_strategy == BASE_BANK_RETRIEVAL_STRATEGY:
            candidate["retrieval_matched_fields"] = [
                "siglip2_base_bank_rig_visual",
                *(
                    ["dinov2_base_bank_surface_texture"]
                    if row.get("dino_rank") is not None
                    else []
                ),
                "masked_color_appearance",
                *(
                    ["mvinverse_authored_pbr_prior"]
                    if row.get("mvinverse_rank") is not None
                    else []
                ),
            ]
        else:
            candidate["retrieval_matched_fields"] = [
                "siglip2_catalog_wide_visual",
                *(
                    ["dinov2_masked_dense_texture"]
                    if row.get("dino_rank") is not None
                    else []
                ),
            ]
        selected.append(candidate)
        audit_ranking.append(
            {
                "rank": len(selected),
                "material_id": material_id,
                "score": float(row["score"]),
                "matched_fields": list(candidate["retrieval_matched_fields"]),
                "siglip2_rank": row.get("siglip2_rank"),
                "siglip2_score": row.get("siglip2_score"),
                "dino_rank": row.get("dino_rank"),
                "dino_score": row.get("dino_score"),
                **(
                    {
                        "color_rank": row.get("color_rank"),
                        "color_score": row.get("color_score"),
                        "mvinverse_rank": row.get("mvinverse_rank"),
                        "mvinverse_score": row.get("mvinverse_score"),
                    }
                    if retrieval_strategy == BASE_BANK_RETRIEVAL_STRATEGY
                    else {}
                ),
            }
        )
    if not selected:
        return fallback_candidates, fallback_audit
    top_score = audit_ranking[0]["score"]
    runner_up_score = audit_ranking[1]["score"] if len(audit_ranking) > 1 else None
    score_margin = top_score - runner_up_score if runner_up_score is not None else None
    normalized_margin = (
        score_margin / max(abs(top_score), 1e-12) if score_margin is not None else None
    )
    return selected, {
        "strategy": retrieval_strategy,
        "group_id": group_id,
        "pool_count": len(pool),
        "eligible_pool_count": len(pool),
        "full_catalog_indexed": True,
        "final_authority": "exact_mdl_render_tournament",
        "fallback_audit": fallback_audit,
        "limit": limit,
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "score_margin": score_margin,
        "normalized_margin": normalized_margin,
        "margin_available": normalized_margin is not None,
        "ranking": audit_ranking,
        "fixed_library_defaults_required": True,
    }


def _drain_face_region_stream(
    stream: Any,
    chunks: list[str],
    *,
    forward_progress: bool,
) -> None:
    """Drain one child pipe without allowing diagnostics onto parent stdout."""

    try:
        for line in iter(stream.readline, ""):
            chunks.append(line)
            if not forward_progress:
                continue
            event = parse_progress_line(line)
            if event is None:
                continue
            sys.stderr.write(format_progress_event(event) + "\n")
            sys.stderr.flush()
    finally:
        stream.close()


def _run_isolated_tool(
    *,
    command: list[str],
    output_dir: Path,
    label: str,
    timeout_seconds: int,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        environment.pop(name, None)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError(f"{label} process pipes were not created")
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    threads = [
        threading.Thread(
            target=_drain_face_region_stream,
            args=(process.stdout, stdout_chunks),
            kwargs={"forward_progress": False},
            name=f"{label}-stdout",
        ),
        threading.Thread(
            target=_drain_face_region_stream,
            args=(process.stderr, stderr_chunks),
            kwargs={"forward_progress": True},
            name=f"{label}-stderr",
        ),
    ]
    for thread in threads:
        thread.start()
    timeout_error: subprocess.TimeoutExpired | None = None
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timeout_error = exc
        process.kill()
        process.wait()
    finally:
        for thread in threads:
            thread.join()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{label}.stdout.log").write_text(
        "".join(stdout_chunks), encoding="utf-8"
    )
    (output_dir / f"{label}.stderr.log").write_text(
        "".join(stderr_chunks), encoding="utf-8"
    )
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        ) from timeout_error
    if process.returncode != 0:
        raise RuntimeError(
            f"{label} failed; inspect {output_dir / f'{label}.stderr.log'}"
        )


def _preferred_face_region_projection_views(registry: Path) -> list[str]:
    """Use the renderer's bounded evidence bank for CPU face projections.

    Pose-bank renders may contain many near-duplicate camera poses.  The
    renderer already records the canonical view subset used to choose each
    part's best evidence.  Projecting face labels only into that complete
    canonical subset avoids repeating expensive CPU rasterization without
    changing mesh topology analysis or image resolution.
    """

    document = _read_json(registry.expanduser().resolve(strict=True))
    render_set = document.get("render_set")
    if not isinstance(render_set, dict):
        return []
    raw_views = render_set.get("part_evidence_view_ids")
    if raw_views is None:
        return []
    if (
        not isinstance(raw_views, list)
        or not raw_views
        or any(not isinstance(view_id, str) or not view_id for view_id in raw_views)
        or len(set(raw_views)) != len(raw_views)
    ):
        raise ValueError(
            "render_set.part_evidence_view_ids must be a non-empty unique string array"
        )
    available = {
        view.get("view_id")
        for view in render_set.get("views", [])
        if isinstance(view, dict) and isinstance(view.get("view_id"), str)
    }
    # Camera calibration deliberately replaces the pose-bank render set with
    # one final registered view per reference image.  Its registry retains the
    # source pose-bank evidence IDs for provenance, so that historical list can
    # contain views that are no longer present in the calibrated render set.
    # Face-region projection must use only images that the current registry can
    # actually supply.  Preserve the renderer's preferred order where possible
    # and fall back to the current view order when the two sets are disjoint.
    selected = [view_id for view_id in raw_views if view_id in available]
    if selected:
        return selected
    return [
        view["view_id"]
        for view in render_set.get("views", [])
        if isinstance(view, dict)
        and isinstance(view.get("view_id"), str)
        and view["view_id"]
    ]


def _run_or_reuse_face_region_evidence(
    *,
    registry: Path,
    output_dir: Path,
    python_executable: Path,
    reuse_existing: bool,
    timeout_seconds: int,
) -> Path:
    """Generate hash-bound topology evidence in an isolated Isaac process."""

    destination = output_dir.expanduser().resolve()
    manifest = destination / "manifest.json"
    if reuse_existing:
        if not manifest.is_file():
            raise ValueError(
                f"geometry-risk reuse requires an existing manifest: {manifest}"
            )
        return manifest.resolve(strict=True)
    if destination.exists():
        raise ValueError(
            f"face-region output already exists; use MVInverse reuse mode: {destination}"
        )
    executable = python_executable.expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"--face-region-python is not executable: {executable}")
    command = [
        str(executable),
        str(PACKAGE_DIR / "evidence" / "face_regions.py"),
        "--registry",
        str(registry.expanduser().resolve(strict=True)),
        "--rendered-registry",
        str(registry.expanduser().resolve(strict=True)),
        "--output-dir",
        str(destination),
    ]
    projection_views = _preferred_face_region_projection_views(registry)
    if projection_views:
        command.extend(["--views", ",".join(projection_views)])
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for name in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        environment.pop(name, None)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("face-region process pipes were not created")
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    drain_threads = [
        threading.Thread(
            target=_drain_face_region_stream,
            args=(process.stdout, stdout_chunks),
            kwargs={"forward_progress": True},
            name="face-region-stdout",
        ),
        threading.Thread(
            target=_drain_face_region_stream,
            args=(process.stderr, stderr_chunks),
            kwargs={"forward_progress": True},
            name="face-region-stderr",
        ),
    ]
    for thread in drain_threads:
        thread.start()
    timeout_error: subprocess.TimeoutExpired | None = None
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timeout_error = exc
        process.kill()
        process.wait()
    finally:
        for thread in drain_threads:
            thread.join()
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "face_region.stdout.log").write_text(stdout, encoding="utf-8")
    (output_dir / "face_region.stderr.log").write_text(stderr, encoding="utf-8")
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=stdout,
            stderr=stderr,
        ) from timeout_error
    if process.returncode != 0:
        raise RuntimeError(
            "face-region evidence generation failed; inspect "
            f"{output_dir / 'face_region.stderr.log'}"
        )
    if not manifest.is_file():
        raise RuntimeError("face-region process succeeded without a manifest")
    return manifest.resolve(strict=True)


def _resume_unattended_from_materials(
    *,
    args: argparse.Namespace,
    destination: Path,
    registry: dict[str, Any],
    geometry_risk_report: dict[str, Any],
    face_region_manifest_path: Path,
) -> int:
    """Resume only deterministic stages after verified material inference.

    This checkpoint is intentionally unavailable for palette or mapping-only
    runs.  Every persisted input is revalidated against the current rendered
    registry, MVInverse evidence, spatial evidence, and material catalog before
    any output plan is accepted.
    """

    required_paths = {
        "qwen_ledger": destination / "qwen_inference_ledger.json",
        "palette": destination / "palette.json",
        "mvinverse_evidence": destination / "mvinverse_pbr_evidence.json",
        "staged_result": destination / "staged_result.json",
        "material_plan": destination / "material_plan.json",
        "group_materials": destination / "group_materials.json",
        "material_choice_audit": destination / "material_choice_audit.json",
        "view_evidence": destination / "view_evidence.json",
        "mapping_votes": destination / "part_mapping_multiview_votes.json",
        "mapping_audit": destination / "part_mapping_multiview_audit.json",
        "spatial_report": destination / "spatial_mapping_report.json",
        "spatial_audit": destination / "spatial_mapping_audit.json",
        "material_stage_contract": destination / "material_stage_contract.json",
    }
    multimodel_enabled = (
        args.mvinverse_mode != "off"
        and args.qwen_model_family in {"qwen3_5", "openai_compatible"}
    )
    palette_group_multimodel_enabled = (
        multimodel_enabled and args.material_assignment_unit == "palette_group"
    )
    if multimodel_enabled:
        required_paths.update(
            {
                "sam3_foreground_manifest": (
                    destination / "sam3_foreground" / "manifest.json"
                ),
                "sam3_foreground_request": (
                    destination / "sam3_foreground_request.json"
                ),
                "foreground_inference_manifest": (
                    destination
                    / "foreground_inference"
                    / "foreground_inference_manifest.json"
                ),
            }
        )
    if palette_group_multimodel_enabled:
        required_paths.update(
            {
                "sam3_manifest": destination / "sam3_regions" / "manifest.json",
                "sam3_request": destination / "sam3_region_request.json",
                "visual_retrieval_request": (
                    destination / "visual_retrieval_request.json"
                ),
                "visual_retrieval": (
                    destination / "visual_retrieval" / "visual_retrieval.json"
                ),
            }
        )
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    batch_paths = sorted((destination / "batches").glob("*.json"))
    if missing or not batch_paths:
        raise ValueError(
            "material-stage resume checkpoint is incomplete; missing="
            f"{missing}, batch_count={len(batch_paths)}"
        )
    if _read_json(required_paths["material_stage_contract"]) != (
        material_stage_contract_document()
    ):
        raise ValueError(
            "material-stage resume rejected: pipeline revision changed"
        )

    palette = _read_json(required_paths["palette"])
    validate_palette(palette)
    mvinverse_evidence = _read_json(required_paths["mvinverse_evidence"])
    validate_mvinverse_evidence(mvinverse_evidence)
    staged_result = _read_json(required_paths["staged_result"])
    persisted_material_plan = _read_json(required_paths["material_plan"])
    if staged_result.get("material_plan") != persisted_material_plan:
        raise ValueError(
            "material-stage resume rejected: staged result/material plan mismatch"
        )
    group_materials = _read_json(required_paths["group_materials"])
    material_audit = _read_json(required_paths["material_choice_audit"])
    view_evidence = _read_json(required_paths["view_evidence"])
    batch_results = [_read_json(path) for path in batch_paths]

    votes_document = _read_json(required_paths["mapping_votes"])
    mapping_votes = votes_document.get("votes")
    if not isinstance(mapping_votes, list):
        raise ValueError("material-stage resume mapping votes are invalid")
    mapping_consensus = apply_mapping_consensus_to_batches(batch_results, mapping_votes)
    if mapping_consensus["audit"] != _read_json(required_paths["mapping_audit"]):
        raise ValueError(
            "material-stage resume rejected: mapping consensus audit changed"
        )
    spatial_mapping_report = _read_json(required_paths["spatial_report"])
    spatial_mapping_gate = apply_spatial_gate_to_batches(
        mapping_consensus["gate_batches"],
        spatial_mapping_report,
    )
    if spatial_mapping_gate["audit"] != _read_json(required_paths["spatial_audit"]):
        raise ValueError(
            "material-stage resume rejected: spatial mapping audit changed"
        )
    gate_batches = spatial_mapping_gate["gate_batches"]

    catalog = MaterialCatalog.load(args.catalog, material_root=args.material_root)
    pool = _catalog_pool(catalog, args.whitelist)
    allowed_material_ids = {item["material_id"] for item in pool}
    if multimodel_enabled:
        if args.sam3_repo is None or args.sam3_checkpoint is None:
            raise ValueError("multimodel resume requires SAM3 paths")
        _validate_sam3_manifest(
            _read_json(required_paths["sam3_foreground_manifest"]),
            manifest_path=required_paths["sam3_foreground_manifest"],
            request_path=required_paths["sam3_foreground_request"],
            repository=args.sam3_repo,
            checkpoint=args.sam3_checkpoint,
            device=args.sam3_device,
            minimum_model_score=args.sam3_minimum_model_score,
            minimum_prompt_overlap=args.sam3_minimum_prompt_overlap,
            maximum_image_fraction=_foreground_maximum_image_fraction(
                args.sam3_maximum_image_fraction
            ),
            minimum_mask_pixels=args.sam3_minimum_mask_pixels,
        )
    if palette_group_multimodel_enabled:
        _validate_sam3_manifest(
            _read_json(required_paths["sam3_manifest"]),
            manifest_path=required_paths["sam3_manifest"],
            request_path=required_paths["sam3_request"],
            repository=args.sam3_repo,
            checkpoint=args.sam3_checkpoint,
            device=args.sam3_device,
            minimum_model_score=args.sam3_minimum_model_score,
            minimum_prompt_overlap=args.sam3_minimum_prompt_overlap,
            maximum_image_fraction=args.sam3_maximum_image_fraction,
            minimum_mask_pixels=args.sam3_minimum_mask_pixels,
        )
        _validate_visual_retrieval_result(
            _read_json(required_paths["visual_retrieval"]),
            request_path=required_paths["visual_retrieval_request"],
            catalog_path=args.catalog,
            material_root=args.material_root,
            siglip2_model_path=args.siglip2_model,
            dinov2_model_path=args.dinov2_model,
            retrieval_python=args.retrieval_python,
            siglip_top_k=args.siglip_top_k,
            final_top_k=args.retrieval_final_top_k,
            batch_size=args.retrieval_batch_size,
            device=args.retrieval_device,
            allowed_material_ids={str(value) for value in allowed_material_ids},
            observation_bank_path=args.retrieval_observation_bank,
        )
    gate_report = evaluate_confidence_gate(
        staged_result,
        registry,
        batches=gate_batches,
        material_choice_audit=material_audit,
        view_evidence=view_evidence,
        geometry_risk_report=geometry_risk_report,
        independent_validation_audit=spatial_mapping_gate["audit"],
    )
    _write_json(destination / "confidence_gate.json", gate_report)
    _write_json(
        destination / "auto_material_plan.json",
        gate_report["auto_material_plan"],
    )
    autonomous = parameterize_auto_material_plan(
        auto_material_plan=gate_report["auto_material_plan"],
        batches=gate_batches,
        palette=palette,
        mvinverse_evidence=mvinverse_evidence,
        allowed_material_ids=allowed_material_ids,
        allow_parameter_writes=not args.immutable_mdl_after_selection,
    )
    _write_json(destination / "mvinverse_autonomy.json", autonomous)
    _write_json(
        destination / "autonomous_uniform_material_plan.json",
        autonomous["material_plan"],
    )
    face_recovery = build_face_material_recovery(
        base_material_plan=autonomous["material_plan"],
        confidence_gate=gate_report,
        face_region_manifest=face_region_manifest_path,
        spatial_mapping_report=spatial_mapping_report,
        canonical_palette=palette,
        mvinverse_evidence=mvinverse_evidence,
        batches=gate_batches,
        allowed_material_ids=allowed_material_ids,
        group_materials=group_materials,
        material_choice_audit=material_audit,
        allow_parameter_writes=not args.immutable_mdl_after_selection,
    )
    _write_json(destination / "face_material_recovery.json", face_recovery)
    _write_json(
        destination / "autonomous_material_plan.json",
        face_recovery["material_plan"],
    )
    final_assignment_count = len(face_recovery["material_plan"]["assignments"])
    unattended_summary = {
        "state": (
            "READY_TO_APPLY" if final_assignment_count else "COMPLETED_SAFE_NOOP"
        ),
        "resume_mode": "verified_material_stage_checkpoint",
        "confidence_gate": gate_report["summary"],
        "geometry_risk": geometry_risk_report["summary"],
        "mapping_consensus": mapping_consensus["audit"]["summary"],
        "spatial_mapping": spatial_mapping_report["summary"],
        "spatial_gate": spatial_mapping_gate["audit"]["summary"],
        "parameterization": autonomous["summary"],
        "face_material_recovery": face_recovery["summary"],
        "artifacts": {
            "qwen_ledger": str(destination / "qwen_inference_ledger.json"),
            "mvinverse_ledger": str(
                destination
                / MVINVERSE_OUTPUT_DIRECTORY
                / "mvinverse_inference_ledger.json"
            ),
            "pbr_evidence": str(destination / "mvinverse_pbr_evidence.json"),
            "geometry_risk": str(destination / "geometry_uniform_material_risk.json"),
            "mapping_consensus": str(destination / "part_mapping_multiview_audit.json"),
            "spatial_mapping_report": str(destination / "spatial_mapping_report.json"),
            "spatial_mapping_audit": str(destination / "spatial_mapping_audit.json"),
            "view_evidence": str(destination / "view_evidence.json"),
            "confidence_gate": str(destination / "confidence_gate.json"),
            "face_material_recovery": str(destination / "face_material_recovery.json"),
            "material_plan": str(destination / "autonomous_material_plan.json"),
            **(
                {
                    "sam3_foreground_manifest": str(
                        destination / "sam3_foreground" / "manifest.json"
                    ),
                    "foreground_inference_manifest": str(
                        destination
                        / "foreground_inference"
                        / "foreground_inference_manifest.json"
                    ),
                    "sam3_manifest": str(
                        destination / "sam3_regions" / "manifest.json"
                    ),
                    "visual_retrieval": str(
                        destination / "visual_retrieval" / "visual_retrieval.json"
                    ),
                }
                if multimodel_enabled
                else {}
            ),
        },
    }
    if not palette_group_multimodel_enabled:
        unattended_summary["artifacts"].pop("sam3_manifest", None)
        unattended_summary["artifacts"].pop("visual_retrieval", None)
    _write_json(destination / "unattended_result.json", unattended_summary)
    print(
        json.dumps(
            {
                "output": str(destination),
                **staged_result["audit"],
                "unattended": unattended_summary,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="ID=IMAGE; repeat 2..4 times for unordered same-asset views",
    )
    parser.add_argument(
        "--palette-reference",
        default="auto",
        help=(
            "auto selects the strongest independently verified original view; "
            "otherwise provide one exact --reference ID"
        ),
    )
    parser.add_argument(
        "--palette-mask",
        action="append",
        default=[],
        help=(
            "optional ID=IMAGE foreground mask for the matching original "
            "--reference; repeat per view as needed"
        ),
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="local Qwen checkpoint; omitted for openai_compatible inference",
    )
    parser.add_argument(
        "--qwen-model-family",
        choices=("auto", "qwen3_vl", "qwen3_5", "openai_compatible"),
        default="auto",
        help=(
            "vision-language backend; openai_compatible replaces local Qwen "
            "with a remote GPT-compatible endpoint"
        ),
    )
    parser.add_argument(
        "--qwen-model-revision",
        help="audited upstream snapshot revision for the local Qwen checkpoint",
    )
    parser.add_argument("--openai-base-url")
    parser.add_argument("--openai-model", default="gpt-5.6")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--openai-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--remote-parallel-requests",
        type=int,
        default=1,
        help=(
            "bounded concurrency for independent mapping requests when using "
            "the openai_compatible backend; local checkpoints remain serial"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cad-view",
        default="iso",
        help=(
            "preferred CAD view when visible-pixel counts tie; each part otherwise "
            "uses its highest-visibility rendered view"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mapping-verification-views",
        type=int,
        default=0,
        help=(
            "maximum usable reference views used for independent mapping "
            "verification; 0 preserves exhaustive verification, otherwise "
            "the value must be at least 2"
        ),
    )
    parser.add_argument(
        "--min-visible-pixels",
        type=int,
        default=DEFAULT_MIN_VISIBLE_PIXELS,
        help=(
            "minimum best-view visible pixels to send for mapping/review; "
            f"automatic matched status additionally requires {MIN_MATCH_VISIBLE_PIXELS}"
        ),
    )
    parser.add_argument("--force-unknown", action="append", default=[])
    parser.add_argument(
        "--stop-after", choices=("palette", "mapping", "materials"), default="materials"
    )
    parser.add_argument(
        "--material-assignment-unit",
        choices=("palette_group", "part_id"),
        default="palette_group",
        help=(
            "part_id keeps SAM3 as whole-workpiece foreground only and skips "
            "palette-group region segmentation; final material ownership is "
            "resolved independently by CAD Part ID in the outer pipeline"
        ),
    )
    parser.add_argument("--orientation-confidence", type=float, default=0.95)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-new-tokens-ceiling",
        type=int,
        help=(
            "palette-only hard ceiling for automatic truncation retries; "
            "omitting it preserves the single-budget legacy behavior"
        ),
    )
    parser.add_argument(
        "--minimum-usable-palette-views",
        type=int,
        default=1,
        help="minimum independently validated reference palettes required",
    )
    parser.add_argument(
        "--minimum-usable-palette-view-ratio",
        type=float,
        default=0.0,
        help="minimum usable/reference palette ratio required from 0 to 1",
    )
    parser.add_argument("--max-image-pixels", type=int, default=768 * 768)
    parser.add_argument("--max-total-pixels", type=int, default=4 * 768 * 768)
    parser.add_argument(
        "--mvinverse-mode",
        choices=MVINVERSE_MODES,
        default="off",
        help=(
            "off keeps the legacy staged workflow; run performs one isolated "
            "MVInverse inference; reuse verifies and reuses an existing hash-bound run"
        ),
    )
    parser.add_argument("--mvinverse-repo", type=Path)
    parser.add_argument("--mvinverse-python", type=Path)
    parser.add_argument("--mvinverse-checkpoint", type=Path)
    parser.add_argument("--mvinverse-model-revision")
    parser.add_argument("--mvinverse-device", default="cuda")
    parser.add_argument(
        "--mvinverse-max-side", type=int, default=MVINVERSE_DEFAULT_MAX_SIDE
    )
    parser.add_argument(
        "--mvinverse-oom-retry-max-side",
        action="append",
        type=int,
        default=None,
        help="fixed lower retry resolution; repeat in strictly descending order",
    )
    parser.add_argument("--mvinverse-no-oom-retry", action="store_true")
    parser.add_argument("--mvinverse-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--acknowledge-mvinverse-noncommercial",
        action="store_true",
        help="explicitly acknowledge the upstream MVInverse non-commercial license",
    )
    parser.add_argument(
        "--immutable-mdl-after-selection",
        action="store_true",
        help=(
            "use selected NVIDIA MDL library defaults and forbid MVInverse "
            "parameter writes"
        ),
    )
    parser.add_argument(
        "--exact-mdl-tournament-max-candidates",
        type=int,
        default=12,
        help=(
            "maximum number of catalog-ranked exact MDL exports retained for "
            "the immutable post-QA tournament; Qwen still chooses from the "
            "bounded four-item semantic shortlist"
        ),
    )
    parser.add_argument(
        "--material-selection-objective",
        choices=sorted(SELECTION_OBJECTIVES),
        default=SELECTION_OBJECTIVE_SEMANTIC,
        help=(
            "semantic_compatible_visual keeps the legacy material-family gate; "
            "visual_similarity searches across families and uses semantics for "
            "audit only"
        ),
    )
    parser.add_argument(
        "--face-region-python",
        type=Path,
        help=(
            "Isaac Sim python.sh used to generate deterministic topology risk "
            "evidence before unattended material application"
        ),
    )
    parser.add_argument("--sam3-python", type=Path)
    parser.add_argument("--sam3-repo", type=Path)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument(
        "--sam3-foreground-annotations",
        type=Path,
        help=(
            "human-confirmed interactive foreground JSON; replaces only the "
            "automatic whole-workpiece SAM3 seed stage"
        ),
    )
    parser.add_argument("--sam3-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sam3-minimum-model-score", type=float, default=0.45)
    parser.add_argument("--sam3-minimum-prompt-overlap", type=float, default=0.25)
    parser.add_argument("--sam3-maximum-image-fraction", type=float, default=0.80)
    parser.add_argument("--sam3-minimum-mask-pixels", type=int, default=32)
    parser.add_argument("--retrieval-python", type=Path)
    parser.add_argument("--siglip2-model", type=Path)
    parser.add_argument("--dinov2-model", type=Path)
    parser.add_argument("--retrieval-cache-dir", type=Path)
    parser.add_argument("--retrieval-observation-bank", type=Path)
    parser.add_argument("--retrieval-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--siglip-top-k", type=int, default=64)
    parser.add_argument("--retrieval-final-top-k", type=int, default=32)
    parser.add_argument("--retrieval-batch-size", type=int, default=24)
    parser.add_argument(
        "--resume-from-materials",
        action="store_true",
        help=(
            "internal fail-closed recovery: verify persisted material inference "
            "and rerun only deterministic unattended gates"
        ),
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parsed_references = [
        _parse_id_path(value, "--reference") for value in args.reference
    ]
    if len(parsed_references) > 4:
        raise ValueError("Provide at most four --reference views")
    reference_ids = [view_id for view_id, _path in parsed_references]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Reference IDs must be unique")
    parsed_masks = [
        _parse_id_path(value, "--palette-mask") for value in args.palette_mask
    ]
    mask_ids = [view_id for view_id, _path in parsed_masks]
    if len(set(mask_ids)) != len(mask_ids):
        raise ValueError("Palette mask IDs must be unique")
    unknown_mask_ids = set(mask_ids) - set(reference_ids)
    if unknown_mask_ids:
        raise ValueError(
            "--palette-mask ID has no matching --reference: "
            + ", ".join(sorted(unknown_mask_ids))
        )
    if args.palette_reference != "auto" and args.palette_reference not in reference_ids:
        raise ValueError(
            "--palette-reference does not match a --reference ID: "
            f"{args.palette_reference}"
        )
    if args.max_new_tokens_ceiling is None:
        args.max_new_tokens_ceiling = args.max_new_tokens
    for label, value in (
        ("--max-new-tokens-ceiling", args.max_new_tokens_ceiling),
        ("--minimum-usable-palette-views", args.minimum_usable_palette_views),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if args.max_new_tokens_ceiling < args.max_new_tokens:
        raise ValueError(
            "--max-new-tokens-ceiling cannot be smaller than --max-new-tokens"
        )
    if (
        isinstance(args.minimum_usable_palette_view_ratio, bool)
        or not isinstance(args.minimum_usable_palette_view_ratio, (int, float))
        or not math.isfinite(float(args.minimum_usable_palette_view_ratio))
        or not 0.0 <= float(args.minimum_usable_palette_view_ratio) <= 1.0
    ):
        raise ValueError(
            "--minimum-usable-palette-view-ratio must be between zero and one"
        )
    required_usable_palette_views = _required_usable_palette_view_count(
        reference_count=len(parsed_references),
        minimum_views=args.minimum_usable_palette_views,
        minimum_ratio=float(args.minimum_usable_palette_view_ratio),
    )
    if required_usable_palette_views > len(parsed_references):
        raise ValueError(
            "Usable palette evidence gate requires "
            f"{required_usable_palette_views} view(s), but only "
            f"{len(parsed_references)} reference view(s) were supplied"
        )
    palette_masks = dict(parsed_masks)
    reserved_prefixes = (
        "cad_",
        "part_ids_",
        "part_contact_",
        "part_highlight_",
        "batch_parts_",
    )
    if any(view_id.startswith(reserved_prefixes) for view_id in reference_ids):
        raise ValueError("Reference ID must not use a reserved geometry prefix")
    if not 1 <= args.batch_size <= 4:
        raise ValueError("--batch-size must be between 1 and 4")
    if args.mapping_verification_views < 0 or args.mapping_verification_views == 1:
        raise ValueError("--mapping-verification-views must be 0 or at least 2")
    if not 1 <= args.remote_parallel_requests <= 8:
        raise ValueError("--remote-parallel-requests must be between 1 and 8")
    if args.exact_mdl_tournament_max_candidates < 2:
        raise ValueError("--exact-mdl-tournament-max-candidates must be at least two")
    if args.min_visible_pixels < 1:
        raise ValueError("--min-visible-pixels must be positive")
    for label, value in (
        ("--max-new-tokens", args.max_new_tokens),
        ("--max-image-pixels", args.max_image_pixels),
        ("--max-total-pixels", args.max_total_pixels),
        ("--sam3-minimum-mask-pixels", args.sam3_minimum_mask_pixels),
        ("--siglip-top-k", args.siglip_top_k),
        ("--retrieval-final-top-k", args.retrieval_final_top_k),
        ("--retrieval-batch-size", args.retrieval_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if args.retrieval_final_top_k > args.siglip_top_k:
        raise ValueError("--retrieval-final-top-k cannot exceed --siglip-top-k")
    for label, value in (
        ("--sam3-minimum-model-score", args.sam3_minimum_model_score),
        ("--sam3-minimum-prompt-overlap", args.sam3_minimum_prompt_overlap),
        ("--sam3-maximum-image-fraction", args.sam3_maximum_image_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{label} must be between zero and one")
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / INFERENCE_FAILURE_FILENAME).unlink(missing_ok=True)
    mvinverse_enabled = args.mvinverse_mode != "off"
    multimodel_required = (
        mvinverse_enabled
        and args.stop_after == "materials"
        and args.qwen_model_family in {"qwen3_5", "openai_compatible"}
    )
    if multimodel_required:
        missing_multimodel = [
            flag
            for flag, value in (
                ("--sam3-python", args.sam3_python),
                ("--sam3-repo", args.sam3_repo),
                ("--sam3-checkpoint", args.sam3_checkpoint),
                ("--retrieval-python", args.retrieval_python),
                ("--siglip2-model", args.siglip2_model),
                ("--dinov2-model", args.dinov2_model),
                ("--retrieval-cache-dir", args.retrieval_cache_dir),
            )
            if value is None
        ]
        if missing_multimodel:
            raise ValueError(
                "unattended material mode requires " + ", ".join(missing_multimodel)
            )

    human_foreground_annotations: dict[str, Any] | None = None
    stored_foreground_annotation = destination / "sam3_foreground_annotations.json"
    stored_foreground_source = destination / "sam3_foreground_annotation_source.json"
    if args.sam3_foreground_annotations is None and (
        stored_foreground_annotation.exists() or stored_foreground_source.exists()
    ):
        raise ValueError(
            "this resumable run was created with human SAM3 foreground "
            "annotations; refusing to fall back to automatic foreground"
        )
    if args.sam3_foreground_annotations is not None:
        if not multimodel_required:
            raise ValueError(
                "--sam3-foreground-annotations requires the staged multimodel "
                "materials workflow"
            )
        if args.sam3_repo is None or args.sam3_checkpoint is None:
            raise AssertionError("SAM3 annotation validation lacks model paths")
        annotation_path = args.sam3_foreground_annotations.expanduser().resolve(
            strict=True
        )
        (
            human_foreground_annotations,
            _confirmed_masks,
        ) = load_sam3_foreground_annotations(
            annotation_path,
            references=parsed_references,
            repository=args.sam3_repo,
            checkpoint=args.sam3_checkpoint,
        )
        require_sam3_foreground_replay_policy(
            human_foreground_annotations,
            minimum_prompt_agreement=args.sam3_minimum_prompt_overlap,
            maximum_image_fraction=_foreground_maximum_image_fraction(
                args.sam3_maximum_image_fraction
            ),
            minimum_mask_pixels=args.sam3_minimum_mask_pixels,
        )
        source_identity = {
            "schema_version": "sam3-human-foreground-source/v1",
            "source_json_sha256": _sha256_file(annotation_path),
            "source_document_sha256": human_foreground_annotations["integrity"][
                "document_sha256"
            ],
            "confirmed_mask_sha256_by_view": {
                str(view["id"]): str(view["confirmed_mask"]["sha256"])
                for view in human_foreground_annotations["source_views"]
            },
        }
        source_identity_path = stored_foreground_source
        if source_identity_path.is_file():
            if _read_json(source_identity_path) != source_identity:
                raise ValueError(
                    "SAM3 foreground annotation source changed inside a resumable run"
                )
        else:
            _write_json(source_identity_path, source_identity)
        human_foreground_annotations = materialize_sam3_foreground_bundle(
            human_foreground_annotations,
            destination=stored_foreground_annotation,
            references=parsed_references,
            repository=args.sam3_repo,
            checkpoint=args.sam3_checkpoint,
        )

    foreground_masks: dict[str, Path] = {}
    inference_references = list(parsed_references)
    if multimodel_required:
        if (
            args.sam3_python is None
            or args.sam3_repo is None
            or args.sam3_checkpoint is None
        ):
            raise AssertionError("SAM3 foreground validation was skipped")
        foreground_request_path = _write_json(
            destination / "sam3_foreground_request.json",
            _build_sam3_foreground_request(
                parsed_references,
                annotations=human_foreground_annotations,
            ),
        )
        foreground_output_dir = destination / "sam3_foreground"
        foreground_manifest_path = foreground_output_dir / "manifest.json"
        foreground_maximum_image_fraction = _foreground_maximum_image_fraction(
            args.sam3_maximum_image_fraction
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="sam3_foreground",
            state="start",
            current=0,
            total=len(parsed_references),
            unit="views",
            detail=(
                (
                    "Human-confirmed SAM3 point prompts are being reproduced "
                    "before all material inference"
                )
                if human_foreground_annotations is not None
                else (
                    "SAM3 foreground segmentation started before all material inference"
                )
            ),
        )
        if not (args.mvinverse_mode == "reuse" and foreground_manifest_path.is_file()):
            if foreground_output_dir.exists():
                if foreground_manifest_path.is_file():
                    raise ValueError(
                        "SAM3 foreground output already exists but reuse was "
                        "not requested"
                    )
                _quarantine_incomplete_stage(foreground_output_dir)
            foreground_staging_dir = _atomic_stage_directory(foreground_output_dir)
            _run_isolated_tool(
                command=[
                    str(args.sam3_python.expanduser().resolve(strict=True)),
                    str(PACKAGE_DIR / "segmentation" / "sam3_regions.py"),
                    "--request",
                    str(foreground_request_path),
                    "--repository",
                    str(args.sam3_repo.expanduser().resolve(strict=True)),
                    "--checkpoint",
                    str(args.sam3_checkpoint.expanduser().resolve(strict=True)),
                    "--output-dir",
                    str(foreground_staging_dir),
                    "--device",
                    args.sam3_device,
                    "--minimum-model-score",
                    str(args.sam3_minimum_model_score),
                    "--minimum-prompt-overlap",
                    str(args.sam3_minimum_prompt_overlap),
                    "--maximum-image-fraction",
                    str(foreground_maximum_image_fraction),
                    "--minimum-mask-pixels",
                    str(args.sam3_minimum_mask_pixels),
                    "--seed",
                    "0",
                ],
                output_dir=foreground_staging_dir,
                label="sam3_foreground",
                timeout_seconds=args.mvinverse_timeout_seconds,
            )
            if not (foreground_staging_dir / "manifest.json").is_file():
                raise RuntimeError(
                    "SAM3 foreground stage completed without a reusable manifest"
                )
            foreground_staging_dir.replace(foreground_output_dir)
        foreground_manifest = _read_json(foreground_manifest_path)
        try:
            accepted_foreground = _validate_sam3_manifest(
                foreground_manifest,
                manifest_path=foreground_manifest_path,
                request_path=foreground_request_path,
                repository=args.sam3_repo,
                checkpoint=args.sam3_checkpoint,
                device=args.sam3_device,
                minimum_model_score=args.sam3_minimum_model_score,
                minimum_prompt_overlap=args.sam3_minimum_prompt_overlap,
                maximum_image_fraction=foreground_maximum_image_fraction,
                minimum_mask_pixels=args.sam3_minimum_mask_pixels,
            )
        except Exception:
            if foreground_output_dir.exists():
                _quarantine_incomplete_stage(foreground_output_dir)
            raise
        foreground_masks = {
            view_id: mask_path
            for (view_id, group_id), mask_path in accepted_foreground.items()
            if group_id == "__foreground__"
        }
        expected_foreground_views = set(reference_ids)
        if set(foreground_masks) != expected_foreground_views:
            missing_views = sorted(expected_foreground_views - set(foreground_masks))
            raise ValueError(
                "SAM3 foreground stage is fail-closed; every reference view "
                "must have one accepted foreground mask. Missing: "
                + ", ".join(missing_views)
            )
        (
            inference_references,
            _foreground_inference_manifest,
        ) = _write_foreground_masked_references(
            parsed_references=parsed_references,
            foreground_masks=foreground_masks,
            output_dir=destination / "foreground_inference" / "masked_views",
        )
        # SAM3 foreground is authoritative in the unattended route. Optional
        # legacy palette masks are retained only for non-multimodel runs.
        palette_masks = dict(foreground_masks)
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="sam3_foreground",
            state="complete",
            current=len(foreground_masks),
            total=len(parsed_references),
            unit="views",
            detail=(
                "SAM3 foreground segmentation completed; all downstream "
                "model inputs are background-neutralized"
                + (
                    " from human-confirmed masks"
                    if human_foreground_annotations is not None
                    else ""
                )
            ),
        )
    remote_vlm = args.qwen_model_family == "openai_compatible"
    if remote_vlm:
        if args.model_path is not None:
            raise ValueError(
                "--model-path must be omitted for openai_compatible inference"
            )
        if not isinstance(args.openai_base_url, str) or not args.openai_base_url:
            raise ValueError("openai_compatible inference requires --openai-base-url")
    elif args.remote_parallel_requests != 1:
        raise ValueError(
            "--remote-parallel-requests greater than one is restricted to "
            "openai_compatible inference"
        )
    elif args.model_path is None:
        raise ValueError("local Qwen inference requires --model-path")
    if args.qwen_model_family == "qwen3_5" and args.device_map == "auto":
        # Never hide 24 GB capacity problems behind CPU offload.  The
        # production Qwen3.5 runtime owns one GPU at a time and fails clearly
        # if the requested bounded image/token budget cannot fit.
        args.device_map = "cuda:0"
    runner: TransformersQwen3VLRunner | OpenAICompatibleVisionRunner | None
    if remote_vlm:
        runner = OpenAICompatibleVisionRunner(
            base_url=args.openai_base_url,
            model=args.openai_model,
            api_key_env=args.openai_api_key_env,
            reasoning_effort=args.openai_reasoning_effort,
            timeout_seconds=args.openai_timeout_seconds,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        assert args.model_path is not None
        runner = TransformersQwen3VLRunner(
            args.model_path,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            max_total_pixels=args.max_total_pixels,
        )
    # Validate and freeze the exact local inference backend before any costly
    # subprocess.  Resume performs the same dependency-free identity check so
    # stale Qwen outputs cannot survive a checkpoint or processor change.
    runner.preflight()
    model_identity = getattr(runner, "model_identity", None)
    if model_identity is None and args.qwen_model_family == "auto":
        # Dependency-injected unit runners from the legacy direct API predate
        # the production identity contract.  Production configs always select
        # an explicit family and can never enter this compatibility branch.
        model_identity = {
            "backend": "dependency_injected_test_runner",
            "model_type": "qwen3_vl",
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "enable_thinking": False,
            },
            "fingerprint": "dependency-injected",
        }
    if not isinstance(model_identity, dict):
        raise ValueError("Qwen runner did not expose a valid frozen model identity")
    qwen_ledger = _qwen_inference_ledger(
        model_identity=model_identity,
        requested_family=args.qwen_model_family,
        requested_revision=args.qwen_model_revision,
        palette_max_new_tokens_ceiling=args.max_new_tokens_ceiling,
        minimum_usable_palette_views=args.minimum_usable_palette_views,
        minimum_usable_palette_view_ratio=float(args.minimum_usable_palette_view_ratio),
    )
    _write_or_validate_qwen_ledger(
        destination / "qwen_inference_ledger.json",
        ledger=qwen_ledger,
        resume=args.resume_from_materials,
    )

    mvinverse_ledger: dict[str, Any] | None = None
    mvinverse_manifest_path: Path | None = None
    face_region_manifest_path: Path | None = None
    if mvinverse_enabled:
        missing = [
            flag
            for flag, value in (
                ("--mvinverse-repo", args.mvinverse_repo),
                ("--mvinverse-python", args.mvinverse_python),
                ("--mvinverse-checkpoint", args.mvinverse_checkpoint),
            )
            if value is None
        ]
        if missing:
            raise ValueError("MVInverse mode requires " + ", ".join(missing))
        if not args.acknowledge_mvinverse_noncommercial:
            raise ValueError(
                "MVInverse mode requires explicit --acknowledge-mvinverse-noncommercial"
            )
        if args.stop_after == "materials" and args.face_region_python is None:
            raise ValueError(
                "unattended material mode requires --face-region-python so "
                "complex or multi-surface meshes cannot be uniformly overwritten"
            )
        if args.mvinverse_no_oom_retry and args.mvinverse_oom_retry_max_side:
            raise ValueError(
                "--mvinverse-no-oom-retry cannot be combined with "
                "--mvinverse-oom-retry-max-side"
            )
        mvinverse_manifest_path = _write_json(
            destination / "mvinverse_reference_manifest.json",
            {
                "source_views": [
                    {
                        "id": view_id,
                        "image": str(path),
                        "original_image": str(dict(parsed_references)[view_id]),
                    }
                    for view_id, path in inference_references
                ],
                "view_order_semantics": (
                    "explicit_cli_order_same_asset_views_sam3_foreground_only"
                ),
            },
        )
        retry_sides = (
            [] if args.mvinverse_no_oom_retry else args.mvinverse_oom_retry_max_side
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="mvinverse",
            state="start",
            detail=f"MVInverse {args.mvinverse_mode} started",
        )
        mvinverse_ledger = run_mvinverse_adapter(
            reference_manifest=mvinverse_manifest_path,
            repo=args.mvinverse_repo,
            python_executable=args.mvinverse_python,
            checkpoint=args.mvinverse_checkpoint,
            output_dir=destination / MVINVERSE_OUTPUT_DIRECTORY,
            acknowledge_noncommercial=True,
            model_revision=args.mvinverse_model_revision,
            device=args.mvinverse_device,
            max_side=args.mvinverse_max_side,
            oom_retry_max_sides=retry_sides,
            reuse_existing=args.mvinverse_mode == "reuse",
            timeout_seconds=args.mvinverse_timeout_seconds,
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="mvinverse",
            state="complete",
            detail=f"MVInverse {args.mvinverse_mode} completed",
        )
        if mvinverse_ledger.get("status") not in {"SUCCESS", "REUSED"}:
            raise ValueError(
                "MVInverse adapter did not produce a verified successful ledger"
            )
        if args.stop_after == "materials":
            if args.face_region_python is None:
                raise AssertionError("face-region Python validation was skipped")
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="face_regions",
                state="start",
                detail="face-region evidence generation started",
            )
            face_region_manifest_path = _run_or_reuse_face_region_evidence(
                registry=args.registry,
                output_dir=destination / "face_regions",
                python_executable=args.face_region_python,
                reuse_existing=args.mvinverse_mode == "reuse",
                timeout_seconds=args.mvinverse_timeout_seconds,
            )
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="face_regions",
                state="complete",
                detail="face-region evidence generation completed",
            )

    registry = _read_json(args.registry)
    parts = registry.get("parts")
    render_set = registry.get("render_set")
    if not isinstance(parts, list) or not parts or not isinstance(render_set, dict):
        raise ValueError("Registry must contain rendered parts and render_set")
    part_by_id = {
        part["part_id"]: part
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("part_id"), str)
    }
    if len(part_by_id) != len(parts):
        raise ValueError("Registry contains invalid or duplicate part IDs")
    geometry_risk_report: dict[str, Any] | None = None
    if mvinverse_enabled and args.stop_after == "materials":
        if face_region_manifest_path is None:
            raise AssertionError("enabled unattended run has no face-region manifest")
        geometry_risk_report = build_geometry_risk(
            face_region_manifest_path,
            args.registry,
        )
        validate_geometry_risk(geometry_risk_report)
        _write_json(
            destination / "geometry_uniform_material_risk.json",
            geometry_risk_report,
        )
    if args.resume_from_materials:
        if (
            not mvinverse_enabled
            or args.stop_after != "materials"
            or geometry_risk_report is None
            or face_region_manifest_path is None
        ):
            raise ValueError(
                "--resume-from-materials requires verified MVInverse unattended mode"
            )
        return _resume_unattended_from_materials(
            args=args,
            destination=destination,
            registry=registry,
            geometry_risk_report=geometry_risk_report,
            face_region_manifest_path=face_region_manifest_path,
        )
    evidence_by_part = _assign_best_evidence_views(
        parts,
        render_set=render_set,
        preferred_view_id=args.cad_view,
    )
    best_highlights = render_set.get("best_highlights")
    if not isinstance(best_highlights, dict):
        raise ValueError("Registry has no best_highlights; rerun render_part_views.py")
    best_evidence = render_set.get("best_evidence", best_highlights)
    if not isinstance(best_evidence, dict):
        raise ValueError("Registry best_evidence must be an object")

    forced: dict[str, str] = {}
    for part_id, part in part_by_id.items():
        evidence_pixels = part["evidence_visible_pixels"]
        if evidence_pixels == 0:
            forced[part_id] = "no_cad_render"
        elif evidence_pixels < args.min_visible_pixels:
            forced[part_id] = "too_small"
    for raw in args.force_unknown:
        part_id, reason = _parse_force_unknown(raw)
        if part_id not in part_by_id:
            raise ValueError(f"Unknown --force-unknown part ID: {part_id}")
        forced[part_id] = reason

    target_parts = [part for part in parts if part["part_id"] not in forced]
    calibrated_highlights = _materialize_calibrated_fallback_highlights(
        target_parts,
        render_set=render_set,
        best_evidence=best_evidence,
        output_dir=destination / "calibrated_part_highlights",
    )
    if calibrated_highlights:
        best_evidence = {**best_evidence, **calibrated_highlights}
    batches = _view_grouped_part_batches(
        target_parts, render_set=render_set, batch_size=args.batch_size
    )
    batch_targets = {
        f"B{index:02d}": [part["part_id"] for part in batch]
        for index, (_view_id, batch) in enumerate(batches, start=1)
    }
    batch_cad_views = {
        f"B{index:02d}": view_id
        for index, (view_id, _batch) in enumerate(batches, start=1)
    }
    for part_id, evidence in evidence_by_part.items():
        evidence["mapping_eligible"] = part_id not in forced
        evidence["automatic_match_eligible"] = (
            part_id not in forced
            and evidence["visible_pixels"] >= MIN_MATCH_VISIBLE_PIXELS
        )
    if runner is None:
        raise AssertionError("live staged inference runner was not initialized")
    model_loaded = False

    def _mark_model_load_start() -> None:
        nonlocal model_loaded
        if not model_loaded:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="model_load",
                state="start",
                detail="Qwen model loading started with first inference",
            )

    def _mark_model_load_complete() -> None:
        nonlocal model_loaded
        if not model_loaded:
            model_loaded = True
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="model_load",
                state="complete",
                detail="Qwen model loading completed",
            )

    def runner_with_progress(payload: Any) -> str:
        _mark_model_load_start()
        result = runner(payload)
        _mark_model_load_complete()
        return result

    metadata_generator = getattr(runner, "generate_with_metadata", None)

    def generation_runner_with_progress(
        payload: Any,
        *,
        max_new_tokens: int | None = None,
    ) -> Any:
        if not callable(metadata_generator):
            raise AssertionError("Qwen metadata generator is unavailable")
        _mark_model_load_start()
        result = (
            metadata_generator(payload)
            if max_new_tokens is None
            else metadata_generator(payload, max_new_tokens=max_new_tokens)
        )
        _mark_model_load_complete()
        return result

    active_palette_context: dict[str, Any] = {
        "view_id": None,
        "index": None,
        "total": None,
    }

    def generation_event_progress(event: Any) -> None:
        if not isinstance(event, dict) or not str(
            event.get("stage_name", "")
        ).startswith("01_palette"):
            return
        status = event.get("status")
        state = {
            "truncated_retry": "retry",
            "schema_retry": "retry",
            "valid": "complete",
            "truncated_exhausted": "failed",
            "invalid": "failed",
        }.get(status, "update")
        budget = event.get("budget")
        detail = (
            f"view={active_palette_context['view_id']} "
            f"reference={active_palette_context['index']}/"
            f"{active_palette_context['total']} "
            f"attempt={event.get('attempt')} budget={budget} "
            f"generated_tokens={event.get('generated_tokens')} status={status}"
        )
        if status == "truncated_retry" and isinstance(budget, int):
            detail += " next_budget=" + str(
                min(budget * 2, args.max_new_tokens_ceiling)
            )
        if event.get("error_reason"):
            detail += f" reason={event['error_reason']}"
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="palette_view",
            state=state,
            detail=detail,
        )

    client = LocalStagedQwenClient(
        model=(args.openai_model if remote_vlm else args.model_path.expanduser().name),
        runner=runner_with_progress,
        raw_output_dir=destination / "raw",
        generation_runner=(
            generation_runner_with_progress if callable(metadata_generator) else None
        ),
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_ceiling=args.max_new_tokens_ceiling,
        generation_event_callback=generation_event_progress,
        checkpoint_dir=destination / "qwen_stage_checkpoints",
        checkpoint_identity_sha256=_sha256_file(
            destination / "qwen_inference_ledger.json"
        ),
        reuse_checkpoints=args.mvinverse_mode == "reuse",
    )
    palette_candidates: list[dict[str, Any]] = []
    inference_reference_paths = dict(inference_references)
    palette_reference_total = len(parsed_references)
    emit_progress(
        scope=PROGRESS_SCOPE,
        stage="palette_references",
        state="start",
        current=0,
        total=palette_reference_total,
        unit="references",
        detail="palette reference processing started",
    )
    for index, (candidate_id, candidate_path) in enumerate(parsed_references, start=1):
        artifact_dir = (
            destination
            / "palette_views"
            / f"{index:02d}_{_artifact_slug(candidate_id)}"
        )
        model_path = artifact_dir / "palette.model.json"
        audit_path = artifact_dir / "palette_evidence_audit.json"
        raw_filtered_path = artifact_dir / "palette.filtered.json"
        accent_audit_path = artifact_dir / "palette_accent_augmentation.json"
        normalized_path = artifact_dir / "palette.json"
        normalized_audit_path = artifact_dir / "palette.normalized_evidence_audit.json"
        merge_audit_path = artifact_dir / "palette_merge_audit.json"
        failure_path = artifact_dir / "palette_failure.json"
        mask_path = palette_masks.get(candidate_id)
        candidate: dict[str, Any] = {
            "reference_id": candidate_id,
            "image": str(candidate_path),
            "model_image": str(inference_reference_paths[candidate_id]),
            "mask": str(mask_path) if mask_path is not None else None,
            "mask_authority": (
                "sam3_foreground_before_material_inference"
                if candidate_id in foreground_masks
                else ("user_supplied_palette_mask" if mask_path is not None else None)
            ),
            "input_order": index,
            "palette_failure_artifact_path": str(failure_path.resolve()),
            "status": "unusable",
        }
        reference_candidate_view = {
            "id": candidate_id,
            "image": candidate["model_image"],
        }
        active_palette_context.update(
            {
                "view_id": candidate_id,
                "index": index,
                "total": palette_reference_total,
            }
        )
        client_generation_events = getattr(client, "generation_events", None)
        generation_event_start = (
            len(client_generation_events)
            if isinstance(client_generation_events, list)
            else 0
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="palette_view",
            state="start",
            detail=(
                f"view={candidate_id} reference={index}/{palette_reference_total} "
                f"attempt=1 budget={args.max_new_tokens}"
            ),
        )
        try:
            candidate_model_palette = client.extract_palette(
                reference_candidate_view,
                run_label=f"{index:02d}_{_artifact_slug(candidate_id)}",
            )
        except ValueError as exc:
            latest_generation_events = getattr(client, "generation_events", None)
            candidate["generation_attempts"] = (
                [
                    dict(event)
                    for event in latest_generation_events[generation_event_start:]
                ]
                if isinstance(latest_generation_events, list)
                else []
            )
            if not candidate["generation_attempts"]:
                emit_progress(
                    scope=PROGRESS_SCOPE,
                    stage="palette_view",
                    state="failed",
                    detail=(
                        f"view={candidate_id} reference={index}/"
                        f"{palette_reference_total} reason={type(exc).__name__}"
                    ),
                )
            candidate["error"] = f"palette extraction failed: {exc}"
            for stale_path in (
                model_path,
                raw_filtered_path,
                accent_audit_path,
                normalized_path,
                normalized_audit_path,
                merge_audit_path,
            ):
                stale_path.unlink(missing_ok=True)
            _write_json(
                audit_path,
                {
                    "image": str(candidate_path),
                    "mask": candidate["mask"],
                    "usable": False,
                    "error": candidate["error"],
                },
            )
            candidate["evidence_audit_path"] = str(audit_path.resolve())
            _write_json(
                failure_path,
                _palette_failure_document(
                    reference_id=candidate_id,
                    image=str(candidate_path),
                    stage="palette_extraction",
                    error=candidate["error"],
                    generation_attempts=candidate["generation_attempts"],
                ),
            )
            palette_candidates.append(candidate)
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="palette_references",
                state="update",
                current=index,
                total=palette_reference_total,
                unit="references",
                detail=f"palette reference {candidate_id} is unusable",
            )
            continue
        latest_generation_events = getattr(client, "generation_events", None)
        candidate["generation_attempts"] = (
            [dict(event) for event in latest_generation_events[generation_event_start:]]
            if isinstance(latest_generation_events, list)
            else []
        )
        if not candidate["generation_attempts"]:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="palette_view",
                state="complete",
                detail=(
                    f"view={candidate_id} reference={index}/"
                    f"{palette_reference_total} status=valid"
                ),
            )
        _write_json(model_path, candidate_model_palette)
        candidate["model_palette_path"] = str(model_path.resolve())
        candidate["model_palette"] = candidate_model_palette
        candidate["model_group_count"] = len(candidate_model_palette["groups"])
        filter_kwargs = {"mask_path": mask_path} if mask_path is not None else {}
        try:
            candidate_palette, candidate_audit = filter_palette_by_image_evidence(
                candidate_model_palette,
                candidate_path,
                **filter_kwargs,
            )
        except ValueError as exc:
            candidate["error"] = f"pixel evidence filtering failed: {exc}"
            for stale_path in (
                raw_filtered_path,
                accent_audit_path,
                normalized_path,
                normalized_audit_path,
                merge_audit_path,
            ):
                stale_path.unlink(missing_ok=True)
            _write_json(
                audit_path,
                {
                    "image": str(candidate_path),
                    "mask": candidate["mask"],
                    "usable": False,
                    "error": candidate["error"],
                },
            )
            candidate["evidence_audit_path"] = str(audit_path.resolve())
            _write_json(
                failure_path,
                _palette_failure_document(
                    reference_id=candidate_id,
                    image=str(candidate_path),
                    stage="pixel_evidence_filter",
                    error=candidate["error"],
                    generation_attempts=candidate.get("generation_attempts"),
                ),
            )
            palette_candidates.append(candidate)
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="palette_references",
                state="update",
                current=index,
                total=palette_reference_total,
                unit="references",
                detail=f"palette reference {candidate_id} is unusable",
            )
            continue
        failure_path.unlink(missing_ok=True)
        augmented_palette, accent_audit = augment_palette_with_detected_accents(
            candidate_palette,
            candidate_path,
            mask_path=mask_path,
        )
        _write_json(accent_audit_path, accent_audit)
        candidate["accent_augmentation_audit"] = accent_audit
        candidate["accent_augmentation_audit_path"] = str(accent_audit_path.resolve())
        if accent_audit["added_group_ids"]:
            candidate_palette, candidate_audit = filter_palette_by_image_evidence(
                augmented_palette,
                candidate_path,
                **filter_kwargs,
            )
        _write_json(audit_path, candidate_audit)
        _write_json(raw_filtered_path, candidate_palette)
        candidate["evidence_audit_path"] = str(audit_path.resolve())
        candidate["pixel_filtered_palette_path"] = str(raw_filtered_path.resolve())
        (
            normalized_palette,
            normalized_audit,
            candidate_merge_audit,
        ) = _merge_equivalent_palette_groups(candidate_palette, candidate_audit)
        _write_json(normalized_path, normalized_palette)
        _write_json(normalized_audit_path, normalized_audit)
        _write_json(merge_audit_path, candidate_merge_audit)
        candidate["normalized_palette_path"] = str(normalized_path.resolve())
        candidate["normalized_evidence_audit_path"] = str(
            normalized_audit_path.resolve()
        )
        candidate["merge_audit_path"] = str(merge_audit_path.resolve())
        candidate["pixel_filtered_palette"] = candidate_palette
        candidate["evidence_audit"] = candidate_audit
        candidate["palette"] = normalized_palette
        candidate["normalized_audit"] = normalized_audit
        candidate["merge_audit"] = candidate_merge_audit
        candidate["quality"] = _palette_quality_metrics(
            normalized_palette, normalized_audit
        )
        candidate["filtered_group_count"] = len(candidate_palette["groups"])
        candidate["unique_appearance_signature_count"] = len(
            normalized_palette["groups"]
        )
        candidate["merged_equivalent_group_count"] = candidate_merge_audit[
            "merged_group_count"
        ]
        candidate["status"] = "usable"
        palette_candidates.append(candidate)
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="palette_references",
            state="update",
            current=index,
            total=palette_reference_total,
            unit="references",
            detail=f"palette reference {candidate_id} processed",
        )

    selection_strategy = (
        "explicit_reference_id"
        if args.palette_reference != "auto"
        else "deterministic_multiview_union_with_best_view_primary_mapping"
    )

    def selection_views(selected: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in candidate.items()
                if key
                not in {
                    "model_palette",
                    "pixel_filtered_palette",
                    "evidence_audit",
                    "palette",
                    "normalized_audit",
                    "merge_audit",
                    "accent_augmentation_audit",
                }
            }
            | {"selected": candidate is selected}
            for candidate in palette_candidates
        ]

    usable_candidate_count = sum(
        candidate.get("status") == "usable" for candidate in palette_candidates
    )
    if usable_candidate_count < required_usable_palette_views:
        detail = (
            "Usable palette evidence gate rejected the run: "
            f"{usable_candidate_count}/{palette_reference_total} view(s) usable, "
            f"required={required_usable_palette_views}"
        )
        view_failures = [
            {
                "view_id": candidate["reference_id"],
                "error": candidate.get("error", "unusable"),
                "failure_artifact": candidate.get("palette_failure_artifact_path"),
                "generation_attempts": candidate.get("generation_attempts", []),
            }
            for candidate in palette_candidates
            if candidate.get("status") != "usable"
        ]
        _write_json(
            destination / "palette_selection.json",
            {
                "schema_version": PALETTE_SELECTION_SCHEMA_VERSION,
                "requested_palette_reference": args.palette_reference,
                "selection_strategy": selection_strategy,
                "exact_tie_break": "first_cli_reference",
                "selected_reference_id": None,
                "selection_error": detail,
                "minimum_usable_views": required_usable_palette_views,
                "usable_view_count": usable_candidate_count,
                "views": selection_views(None),
            },
        )
        _write_inference_failure(
            destination,
            error_code="insufficient_usable_palette_views",
            failed_stage="palette",
            detail=detail,
            view_failures=view_failures,
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="palette_references",
            state="failed",
            current=palette_reference_total,
            total=palette_reference_total,
            unit="references",
            detail=detail,
        )
        raise StagedAnalysisError(detail)

    emit_progress(
        scope=PROGRESS_SCOPE,
        stage="palette_references",
        state="complete",
        current=palette_reference_total,
        total=palette_reference_total,
        unit="references",
        detail=(
            "palette reference processing completed; "
            f"usable={usable_candidate_count}/{palette_reference_total}, "
            f"required={required_usable_palette_views}"
        ),
    )

    try:
        selected_candidate = _select_palette_candidate(
            palette_candidates, args.palette_reference
        )
    except ValueError as exc:
        _write_json(
            destination / "palette_selection.json",
            {
                "schema_version": PALETTE_SELECTION_SCHEMA_VERSION,
                "requested_palette_reference": args.palette_reference,
                "selection_strategy": selection_strategy,
                "exact_tie_break": "first_cli_reference",
                "selected_reference_id": None,
                "selection_error": str(exc),
                "views": selection_views(None),
            },
        )
        selection_detail = f"Palette reference selection failed: {exc}"
        _write_inference_failure(
            destination,
            error_code="palette_reference_selection_failed",
            failed_stage="palette_selection",
            detail=selection_detail,
            view_failures=[
                {
                    "view_id": candidate["reference_id"],
                    "error": candidate.get("error", "unusable"),
                    "failure_artifact": candidate.get("palette_failure_artifact_path"),
                    "generation_attempts": candidate.get("generation_attempts", []),
                }
                for candidate in palette_candidates
                if candidate.get("status") != "usable"
            ],
        )
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="palette_selection",
            state="failed",
            detail=selection_detail,
        )
        raise
    reference_id = selected_candidate["reference_id"]
    reference_path = Path(selected_candidate["image"])
    reference_view = {"id": reference_id, "image": str(reference_path)}
    model_palette = selected_candidate["model_palette"]
    pixel_filtered_palette = selected_candidate["pixel_filtered_palette"]
    palette_evidence_audit = selected_candidate["evidence_audit"]
    normalized_palette_evidence_audit = selected_candidate["normalized_audit"]
    palette_merge_audit = selected_candidate["merge_audit"]
    palette_fusion: dict[str, Any] | None = None
    usable_candidates = [
        candidate
        for candidate in palette_candidates
        if candidate.get("status") == "usable"
    ]
    if args.palette_reference == "auto" and len(usable_candidates) > 1:
        palette_fusion = fuse_multiview_palettes(
            [candidate["palette"] for candidate in usable_candidates],
            augmentation_audits=[
                candidate["accent_augmentation_audit"]
                for candidate in usable_candidates
            ],
        )
        palette = _legacy_palette_from_fusion(
            palette_fusion,
            primary_view_id=reference_id,
        )
        _write_json(destination / "palette_fusion.json", palette_fusion)
    else:
        palette = selected_candidate["palette"]
    _write_json(destination / "palette.model.json", model_palette)
    _write_json(destination / "palette.filtered.json", pixel_filtered_palette)
    _write_json(destination / "palette_evidence_audit.json", palette_evidence_audit)
    _write_json(
        destination / "palette.normalized_evidence_audit.json",
        normalized_palette_evidence_audit,
    )
    _write_json(destination / "palette_merge_audit.json", palette_merge_audit)
    _write_json(destination / "palette.json", palette)

    annotated_reference_paths: dict[str, Path] = {}
    if mvinverse_enabled:
        for candidate in palette_candidates:
            if candidate.get("status") != "usable":
                continue
            candidate_id = candidate["reference_id"]
            annotated_reference_paths[candidate_id] = _annotate_reference_palette(
                Path(candidate["model_image"]),
                candidate["palette"],
                destination
                / "annotated_references"
                / f"{_artifact_slug(candidate_id)}.png",
            )
        reference_view = {
            "id": reference_id,
            "image": str(annotated_reference_paths[reference_id]),
        }

    remaining_references = [
        (view_id, path)
        for view_id, path in inference_references
        if view_id != reference_id
    ]
    support_reference_view: dict[str, str] | None = None
    support_manifest: dict[str, Any] | None = None
    if len(remaining_references) == 1:
        support_id, support_path = remaining_references[0]
        support_reference_view = {"id": support_id, "image": str(support_path)}
        support_manifest = {
            **support_reference_view,
            "is_contact_sheet": False,
            "source_view_ids": [support_id],
        }
    elif len(remaining_references) >= 2:
        support_path = _make_reference_sheet(
            remaining_references,
            destination / "reference_support_multiview.png",
        )
        support_reference_view = {
            "id": "ref_support_multiview",
            "image": str(support_path),
        }
        support_manifest = {
            **support_reference_view,
            "is_contact_sheet": True,
            "source_view_ids": [view_id for view_id, _path in remaining_references],
        }

    selection_audit_path = _write_json(
        destination / "palette_selection.json",
        {
            "schema_version": PALETTE_SELECTION_SCHEMA_VERSION,
            "requested_palette_reference": args.palette_reference,
            "selection_strategy": selection_strategy,
            "exact_tie_break": "first_cli_reference",
            "selected_reference_id": reference_id,
            "canonical_palette_mode": (
                "multiview_union" if palette_fusion is not None else "single_view"
            ),
            "palette_fusion": (
                str((destination / "palette_fusion.json").resolve())
                if palette_fusion is not None
                else None
            ),
            "views": selection_views(selected_candidate),
        },
    )
    reference_manifest_path = _write_json(
        destination / "reference_manifest.json",
        {
            "source_views": [
                _reference_manifest_source_view(candidate)
                for candidate in palette_candidates
            ],
            "palette_reference_request": args.palette_reference,
            "palette_selection_audit": str(selection_audit_path),
            "model_reference_view": {
                "id": reference_id,
                "image": str(
                    inference_reference_paths.get(reference_id, reference_path)
                ),
                "is_contact_sheet": False,
                **(
                    {"original_image": str(reference_path)}
                    if inference_reference_paths.get(reference_id, reference_path)
                    != reference_path
                    else {}
                ),
            },
            "mapping_support_view": support_manifest,
            "view_order_semantics": "unordered_same_asset_views",
        },
    )
    multimodel_enabled = (
        mvinverse_enabled
        and args.stop_after == "materials"
        and all(
            value is not None
            for value in (
                args.sam3_python,
                args.sam3_repo,
                args.sam3_checkpoint,
                args.retrieval_python,
                args.siglip2_model,
                args.dinov2_model,
                args.retrieval_cache_dir,
            )
        )
    )
    material_region_segmentation_enabled = (
        multimodel_enabled and args.material_assignment_unit == "palette_group"
    )
    accepted_sam3_masks: dict[tuple[str, str], Path] = {}
    sam3_manifest: dict[str, Any] | None = None
    if material_region_segmentation_enabled:
        sam3_request_path = _write_json(
            destination / "sam3_region_request.json",
            _build_sam3_region_request(
                parsed_references=parsed_references,
                palette_candidates=palette_candidates,
                palette=palette,
                palette_fusion=palette_fusion,
                selected_reference_id=reference_id,
            ),
        )
        # Qwen has completed palette inference.  Release its weights before
        # SAM3 acquires the same GPU; the frozen runner will lazy-load the same
        # backend again for mapping.
        runner.unload()
        model_loaded = False
        sam3_output_dir = destination / "sam3_regions"
        sam3_manifest_path = sam3_output_dir / "manifest.json"
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="sam3_regions",
            state="start",
            detail="SAM3 material-region segmentation started",
        )
        if not (args.mvinverse_mode == "reuse" and sam3_manifest_path.is_file()):
            if sam3_output_dir.exists():
                if sam3_manifest_path.is_file():
                    raise ValueError(
                        "SAM3 output already exists but reuse was not requested"
                    )
                _quarantine_incomplete_stage(sam3_output_dir)
            sam3_staging_dir = _atomic_stage_directory(sam3_output_dir)
            _run_isolated_tool(
                command=[
                    str(args.sam3_python.expanduser().resolve(strict=True)),
                    str(PACKAGE_DIR / "segmentation" / "sam3_regions.py"),
                    "--request",
                    str(sam3_request_path),
                    "--repository",
                    str(args.sam3_repo.expanduser().resolve(strict=True)),
                    "--checkpoint",
                    str(args.sam3_checkpoint.expanduser().resolve(strict=True)),
                    "--output-dir",
                    str(sam3_staging_dir),
                    "--device",
                    args.sam3_device,
                    "--minimum-model-score",
                    str(args.sam3_minimum_model_score),
                    "--minimum-prompt-overlap",
                    str(args.sam3_minimum_prompt_overlap),
                    "--maximum-image-fraction",
                    str(args.sam3_maximum_image_fraction),
                    "--minimum-mask-pixels",
                    str(args.sam3_minimum_mask_pixels),
                    "--seed",
                    "0",
                ],
                output_dir=sam3_staging_dir,
                label="sam3_regions",
                timeout_seconds=args.mvinverse_timeout_seconds,
            )
            if not (sam3_staging_dir / "manifest.json").is_file():
                raise RuntimeError("SAM3 stage completed without a reusable manifest")
            sam3_staging_dir.replace(sam3_output_dir)
        sam3_manifest = _read_json(sam3_manifest_path)
        try:
            accepted_all_sam3_masks = _validate_sam3_manifest(
                sam3_manifest,
                manifest_path=sam3_manifest_path,
                request_path=sam3_request_path,
                repository=args.sam3_repo,
                checkpoint=args.sam3_checkpoint,
                device=args.sam3_device,
                minimum_model_score=args.sam3_minimum_model_score,
                minimum_prompt_overlap=args.sam3_minimum_prompt_overlap,
                maximum_image_fraction=args.sam3_maximum_image_fraction,
                minimum_mask_pixels=args.sam3_minimum_mask_pixels,
            )
        except Exception:
            if sam3_output_dir.exists():
                _quarantine_incomplete_stage(sam3_output_dir)
            raise
        canonical_group_ids = {str(group["group_id"]) for group in palette["groups"]}
        unexpected_group_ids = {
            group_id
            for _view_id, group_id in accepted_all_sam3_masks
            if group_id not in canonical_group_ids
        }
        if unexpected_group_ids:
            raise ValueError(
                "SAM3 result contains unexpected group IDs: "
                + ", ".join(sorted(unexpected_group_ids))
            )
        accepted_sam3_masks = {
            identity: path
            for identity, path in accepted_all_sam3_masks.items()
            if identity[1] in canonical_group_ids
        }
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="sam3_regions",
            state="complete",
            current=len(sam3_manifest["records"]),
            total=len(sam3_manifest["records"]),
            unit="regions",
            detail=(
                "SAM3 segmentation completed with "
                f"{len(accepted_sam3_masks)} accepted material masks and "
                f"{len(foreground_masks)} already-validated foreground masks; "
                "rejected material masks remain non-authoritative"
            ),
        )
    mvinverse_evidence: dict[str, Any] | None = None
    if mvinverse_enabled:
        if mvinverse_ledger is None:
            raise AssertionError("enabled MVInverse run has no adapter ledger")
        if mvinverse_manifest_path is None:
            raise AssertionError(
                "enabled MVInverse run has no bound reference manifest"
            )
        mvinverse_evidence = build_mvinverse_evidence_from_manifest(
            destination / MVINVERSE_OUTPUT_DIRECTORY / "maps",
            mvinverse_manifest_path,
            destination / "palette.json",
            masks=accepted_sam3_masks,
            require_explicit_masks=material_region_segmentation_enabled,
            frame_indices=_mvinverse_frame_indices(mvinverse_ledger),
            inference_ledger=(
                destination
                / MVINVERSE_OUTPUT_DIRECTORY
                / "mvinverse_inference_ledger.json"
            ),
        )
        validate_mvinverse_evidence(mvinverse_evidence)
        _write_json(destination / "mvinverse_pbr_evidence.json", mvinverse_evidence)
    batch_plan = {
        "reference_view_id": reference_id,
        # Retain the legacy field while documenting its narrower role.
        "cad_view": args.cad_view,
        "cad_view_strategy": "per_part_max_visible_pixels",
        "preferred_cad_view": args.cad_view,
        "requested_min_visible_pixels": args.min_visible_pixels,
        "min_visible_pixels": args.min_visible_pixels,
        "minimum_matched_visible_pixels": MIN_MATCH_VISIBLE_PIXELS,
        "evidence_pixel_view": "per_part_best",
        "part_evidence": evidence_by_part,
        "forced_unknown_parts": forced,
        "batch_cad_views": batch_cad_views,
        "batch_targets": batch_targets,
    }
    _write_json(destination / "batch_plan.json", batch_plan)
    if args.stop_after == "palette":
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "palette_reference_id": reference_id,
                    "palette_groups": len(palette["groups"]),
                }
            )
        )
        return 0

    batch_results = []
    primary_batch_total = len(batches)
    if primary_batch_total:
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="primary_mapping",
            state="start",
            current=0,
            total=primary_batch_total,
            unit="batches",
            detail="primary mapping started",
        )
    else:
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="primary_mapping",
            state="start",
            detail="primary mapping has no eligible batches",
        )

    def run_primary_mapping_batch(
        index: int,
        batch_view_id: str,
        batch: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        batch_id = f"B{index:02d}"
        cad_view, part_id_view = _batch_geometry_views(render_set, batch_view_id)
        batch_highlights = {
            part["part_id"]: _evidence_highlight(part, best_evidence) for part in batch
        }
        sheet = _make_batch_sheet(
            [part["part_id"] for part in batch],
            batch_highlights,
            destination / "batch_sheets" / f"{batch_id}.png",
        )
        result = client.map_part_batch(
            reference_view=reference_view,
            support_reference_view=support_reference_view,
            cad_view=cad_view,
            part_id_view=part_id_view,
            batch_sheet_view={"id": f"batch_parts_{batch_id}", "image": str(sheet)},
            palette=selected_candidate["palette"],
            target_parts=batch,
            batch_id=batch_id,
        )
        _write_json(destination / "batches" / f"{batch_id}.json", result)
        return index, result

    primary_results_by_index: dict[int, dict[str, Any]] = {}
    primary_jobs = [
        (index, batch_view_id, batch)
        for index, (batch_view_id, batch) in enumerate(batches, start=1)
    ]
    if remote_vlm and args.remote_parallel_requests > 1 and primary_jobs:
        with ThreadPoolExecutor(
            max_workers=min(args.remote_parallel_requests, len(primary_jobs)),
            thread_name_prefix="qwen-map-primary",
        ) as executor:
            futures = {
                executor.submit(run_primary_mapping_batch, *job): job[0]
                for job in primary_jobs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, result = future.result()
                primary_results_by_index[index] = result
                emit_progress(
                    scope=PROGRESS_SCOPE,
                    stage="primary_mapping",
                    state="update",
                    current=completed,
                    total=primary_batch_total,
                    unit="batches",
                    detail=(
                        f"primary mapping batch B{index:02d} completed "
                        f"(parallel={args.remote_parallel_requests})"
                    ),
                )
    else:
        for completed, job in enumerate(primary_jobs, start=1):
            index, result = run_primary_mapping_batch(*job)
            primary_results_by_index[index] = result
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="primary_mapping",
                state="update",
                current=completed,
                total=primary_batch_total,
                unit="batches",
                detail=f"primary mapping batch B{index:02d} completed",
            )
    batch_results = [
        primary_results_by_index[index] for index in sorted(primary_results_by_index)
    ]
    if client.repair_events:
        _write_json(
            destination / "model_repair_events.json",
            {"events": client.repair_events},
        )
    if primary_batch_total:
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="primary_mapping",
            state="complete",
            current=primary_batch_total,
            total=primary_batch_total,
            unit="batches",
            detail="primary mapping completed",
        )
    else:
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="primary_mapping",
            state="complete",
            detail="primary mapping completed with no eligible batches",
        )
    if palette_fusion is not None:
        primary_map = palette_fusion["view_group_id_maps"].get(reference_id, {})
        batch_results = _canonicalize_primary_batches(
            batch_results,
            local_to_canonical=primary_map,
            canonical_palette=palette,
        )
        for result in batch_results:
            _write_json(
                destination / "batches" / f"{result['batch_id']}.json",
                result,
            )
    if args.stop_after == "mapping":
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "palette_groups": len(palette["groups"]),
                    "batch_count": len(batch_results),
                    "forced_unknown_count": len(forced),
                }
            )
        )
        return 0

    gate_batches = batch_results
    mapping_consensus: dict[str, Any] | None = None
    spatial_mapping_report: dict[str, Any] | None = None
    spatial_mapping_gate: dict[str, Any] | None = None
    if mvinverse_evidence is not None:
        view_group_id_maps = (
            palette_fusion["view_group_id_maps"]
            if palette_fusion is not None
            else build_view_group_id_maps(mvinverse_evidence)
        )
        # Unknown primary rows cannot be promoted by either mapping consensus
        # or the downstream spatial gate.  Sending those rows through every
        # reference view consumes remote inference without changing any gate
        # decision.  Review rows remain eligible because two independent
        # agreeing views may safely promote their already-cited group.
        primary_status_by_part = {
            row["part_id"]: row["status"]
            for batch in batch_results
            for row in batch["mappings"]
        }
        consensus_candidate_ids = {
            part["part_id"]
            for part in target_parts
            if primary_status_by_part.get(part["part_id"]) != "unknown"
        }
        excluded_primary_unknown_ids = {
            part["part_id"]
            for part in target_parts
            if primary_status_by_part.get(part["part_id"]) == "unknown"
        }
        consensus_parts = [
            part_by_id[part_id] for part_id in sorted(consensus_candidate_ids)
        ]
        view_batches_by_id: dict[str, list[dict[str, Any]]] = {}
        consensus_batch_number = 1000
        multiview_batch_plans: list[
            tuple[dict[str, Any], list[tuple[str, list[dict[str, Any]]]]]
        ] = []
        verification_candidates = _mapping_verification_candidates(
            palette_candidates,
            primary_reference_id=reference_id,
            maximum_views=args.mapping_verification_views,
            eligible_view_ids={
                view_id
                for view_id, group_map in view_group_id_maps.items()
                if group_map
            },
        )
        for candidate in verification_candidates:
            view_id = candidate["reference_id"]
            candidate_batches = _view_grouped_part_batches(
                consensus_parts,
                render_set=render_set,
                batch_size=args.batch_size,
            )
            multiview_batch_plans.append((candidate, candidate_batches))
            view_batches_by_id[view_id] = []
        multiview_batch_total = sum(
            len(candidate_batches)
            for _candidate, candidate_batches in multiview_batch_plans
        )
        multiview_batch_current = 0
        if multiview_batch_total:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="multiview_mapping",
                state="start",
                current=0,
                total=multiview_batch_total,
                unit="batches",
                detail="multiview mapping started",
            )
        else:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="multiview_mapping",
                state="start",
                detail="multiview mapping has no eligible batches",
            )
        multiview_jobs: list[
            tuple[
                int,
                dict[str, Any],
                str,
                str,
                list[dict[str, Any]],
                str,
            ]
        ] = []
        for candidate, candidate_batches in multiview_batch_plans:
            view_id = candidate["reference_id"]
            for view_batch_index, (
                cad_view_id,
                candidate_batch,
            ) in enumerate(candidate_batches):
                batch_id = f"B{consensus_batch_number:04d}"
                consensus_batch_number += 1
                multiview_jobs.append(
                    (
                        view_batch_index,
                        candidate,
                        view_id,
                        cad_view_id,
                        candidate_batch,
                        batch_id,
                    )
                )

        def run_multiview_mapping_batch(
            view_batch_index: int,
            candidate: dict[str, Any],
            view_id: str,
            cad_view_id: str,
            candidate_batch: list[dict[str, Any]],
            batch_id: str,
        ) -> tuple[str, int, str, dict[str, Any]]:
            cad_view, part_id_view = _batch_geometry_views(render_set, cad_view_id)
            highlights = {
                part["part_id"]: _evidence_highlight(part, best_evidence)
                for part in candidate_batch
            }
            sheet = _make_batch_sheet(
                [part["part_id"] for part in candidate_batch],
                highlights,
                destination
                / "mapping_view_sheets"
                / _artifact_slug(view_id)
                / f"{batch_id}.png",
            )
            view_batch = client.map_part_batch(
                reference_view={
                    "id": view_id,
                    "image": str(annotated_reference_paths[view_id]),
                },
                support_reference_view=None,
                cad_view=cad_view,
                part_id_view=part_id_view,
                batch_sheet_view={
                    "id": f"batch_parts_{batch_id}",
                    "image": str(sheet),
                },
                palette=candidate["palette"],
                target_parts=candidate_batch,
                batch_id=batch_id,
            )
            _write_json(
                destination
                / "mapping_view_batches"
                / _artifact_slug(view_id)
                / f"{batch_id}.json",
                view_batch,
            )
            return view_id, view_batch_index, batch_id, view_batch

        multiview_results: dict[str, dict[int, dict[str, Any]]] = {
            view_id: {} for view_id in view_batches_by_id
        }
        if remote_vlm and args.remote_parallel_requests > 1 and multiview_jobs:
            with ThreadPoolExecutor(
                max_workers=min(
                    args.remote_parallel_requests,
                    len(multiview_jobs),
                ),
                thread_name_prefix="qwen-map-multiview",
            ) as executor:
                futures = {
                    executor.submit(run_multiview_mapping_batch, *job): job[5]
                    for job in multiview_jobs
                }
                for future in as_completed(futures):
                    view_id, view_batch_index, batch_id, view_batch = future.result()
                    multiview_results[view_id][view_batch_index] = view_batch
                    multiview_batch_current += 1
                    emit_progress(
                        scope=PROGRESS_SCOPE,
                        stage="multiview_mapping",
                        state="update",
                        current=multiview_batch_current,
                        total=multiview_batch_total,
                        unit="batches",
                        detail=(
                            f"multiview mapping {view_id}/{batch_id} completed "
                            f"(parallel={args.remote_parallel_requests})"
                        ),
                    )
        else:
            for job in multiview_jobs:
                (
                    view_id,
                    view_batch_index,
                    batch_id,
                    view_batch,
                ) = run_multiview_mapping_batch(*job)
                multiview_results[view_id][view_batch_index] = view_batch
                multiview_batch_current += 1
                emit_progress(
                    scope=PROGRESS_SCOPE,
                    stage="multiview_mapping",
                    state="update",
                    current=multiview_batch_current,
                    total=multiview_batch_total,
                    unit="batches",
                    detail=f"multiview mapping {view_id}/{batch_id} completed",
                )
        for view_id in view_batches_by_id:
            view_batches_by_id[view_id] = [
                multiview_results[view_id][index]
                for index in sorted(multiview_results[view_id])
            ]
        if multiview_batch_total:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="multiview_mapping",
                state="complete",
                current=multiview_batch_total,
                total=multiview_batch_total,
                unit="batches",
                detail="multiview mapping completed",
            )
        else:
            emit_progress(
                scope=PROGRESS_SCOPE,
                stage="multiview_mapping",
                state="complete",
                detail="multiview mapping completed with no eligible batches",
            )
        mapping_votes = canonicalize_view_batch_mappings(
            view_batches_by_id,
            view_group_id_maps,
        )
        _write_json(
            destination / "part_mapping_multiview_votes.json",
            {
                "view_group_id_maps": view_group_id_maps,
                "candidate_part_ids": sorted(consensus_candidate_ids),
                "verification_policy": {
                    "maximum_views": args.mapping_verification_views,
                    "exhaustive": args.mapping_verification_views == 0,
                    "primary_reference_id": reference_id,
                    "selected_view_ids": [
                        candidate["reference_id"]
                        for candidate in verification_candidates
                    ],
                    "excluded_primary_unknown_part_ids": sorted(
                        excluded_primary_unknown_ids
                    ),
                },
                "votes": mapping_votes,
            },
        )
        mapping_consensus = apply_mapping_consensus_to_batches(
            batch_results,
            mapping_votes,
        )
        gate_batches = mapping_consensus["gate_batches"]
        _write_json(
            destination / "part_mapping_multiview_audit.json",
            mapping_consensus["audit"],
        )
        usable_palettes = {
            candidate["reference_id"]: candidate["palette"]
            for candidate in palette_candidates
            if candidate.get("status") == "usable"
        }
        usable_palette_audits = {
            candidate["reference_id"]: candidate["normalized_audit"]
            for candidate in palette_candidates
            if candidate.get("status") == "usable"
        }
        spatial_mapping_report = build_spatial_mapping_report(
            reference_manifest_path,
            args.registry,
            view_group_id_maps,
            mapping_votes,
            normalized_palettes_by_view=usable_palettes,
            palette_audits_by_view=usable_palette_audits,
            include_all_parts=True,
            progress_callback=emit_progress_event,
        )
        _write_json(
            destination / "spatial_mapping_report.json",
            spatial_mapping_report,
        )
        spatial_mapping_gate = apply_spatial_gate_to_batches(
            gate_batches,
            spatial_mapping_report,
        )
        gate_batches = spatial_mapping_gate["gate_batches"]
        _write_json(
            destination / "spatial_mapping_audit.json",
            spatial_mapping_gate["audit"],
        )

    catalog = MaterialCatalog.load(args.catalog, material_root=args.material_root)
    pool = _catalog_pool(catalog, args.whitelist)
    group_materials = {
        "schema_version": GROUP_MATERIAL_SCHEMA_VERSION,
        "selections": [],
    }
    material_audit: dict[str, Any] = {}
    group_view_choices: dict[str, list[dict[str, Any]]] = {}
    reference_paths = dict(parsed_references)
    pbr_groups = (
        _mvinverse_groups_by_id(mvinverse_evidence)
        if mvinverse_evidence is not None
        else {}
    )
    fusion_groups_by_id = (
        {
            group["group_id"]: group
            for group in palette_fusion["canonical_palette"]["groups"]
        }
        if palette_fusion is not None
        else {}
    )
    visual_retrieval_by_group: dict[str, dict[str, Any]] = {}
    if material_region_segmentation_enabled:
        retrieval_request_path = _write_json(
            destination / "visual_retrieval_request.json",
            _build_visual_retrieval_request(
                catalog=args.catalog.expanduser().resolve(strict=True),
                material_root=args.material_root.expanduser().resolve(strict=True),
                palette=palette,
                pbr_groups=pbr_groups,
                parsed_references=parsed_references,
                accepted_masks=accepted_sam3_masks,
            ),
        )
        # Mapping is complete.  Give SigLIP2 and DINOv2 exclusive GPU
        # ownership; Qwen reloads only for the four-candidate decision calls.
        runner.unload()
        model_loaded = False
        retrieval_output_dir = destination / "visual_retrieval"
        retrieval_result_path = retrieval_output_dir / "visual_retrieval.json"
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="visual_retrieval",
            state="start",
            detail=(
                "SigLIP2 full NVIDIA catalog retrieval and DINOv2 masked "
                "texture reranking started"
            ),
        )
        retrieval_checkpoint_reused = False
        if args.mvinverse_mode == "reuse" and retrieval_result_path.is_file():
            try:
                visual_retrieval_by_group = _validate_visual_retrieval_result(
                    _read_json(retrieval_result_path),
                    request_path=retrieval_request_path,
                    catalog_path=args.catalog,
                    material_root=args.material_root,
                    siglip2_model_path=args.siglip2_model,
                    dinov2_model_path=args.dinov2_model,
                    retrieval_python=args.retrieval_python,
                    siglip_top_k=args.siglip_top_k,
                    final_top_k=args.retrieval_final_top_k,
                    batch_size=args.retrieval_batch_size,
                    device=args.retrieval_device,
                    allowed_material_ids={
                        str(item["material_id"]) for item in pool
                    },
                    observation_bank_path=args.retrieval_observation_bank,
                )
                retrieval_checkpoint_reused = True
            except (OSError, RuntimeError, ValueError) as exc:
                abandoned = _quarantine_incomplete_stage(retrieval_output_dir)
                emit_progress(
                    scope=PROGRESS_SCOPE,
                    stage="visual_retrieval",
                    state="update",
                    detail=(
                        "Cached retrieval is stale for the current material "
                        "regions and was archived; recomputing only SigLIP2/"
                        f"DINOv2 retrieval (diagnostics: {abandoned}; {exc})"
                    ),
                )
        if not retrieval_checkpoint_reused:
            if retrieval_output_dir.exists():
                if retrieval_result_path.is_file():
                    raise ValueError(
                        "Visual retrieval output already exists but reuse was "
                        "not requested"
                    )
                _quarantine_incomplete_stage(retrieval_output_dir)
            retrieval_command_prefix = [
                str(args.retrieval_python.expanduser().resolve(strict=True)),
                str(PACKAGE_DIR / "retrieval" / "visual_materials.py"),
                "--request",
                str(retrieval_request_path),
                "--siglip2-model",
                str(args.siglip2_model.expanduser().resolve(strict=True)),
                "--dinov2-model",
                str(args.dinov2_model.expanduser().resolve(strict=True)),
                "--cache-dir",
                str(args.retrieval_cache_dir.expanduser().resolve()),
            ]
            if args.retrieval_observation_bank is not None:
                retrieval_command_prefix.extend(
                    [
                        "--observation-bank",
                        str(
                            args.retrieval_observation_bank.expanduser().resolve(
                                strict=True
                            )
                        ),
                    ]
                )
            retrieval_attempts = 3
            for retrieval_attempt in range(1, retrieval_attempts + 1):
                retrieval_staging_dir = _atomic_stage_directory(
                    retrieval_output_dir
                )
                try:
                    _run_isolated_tool(
                        command=[
                            *retrieval_command_prefix,
                            "--output-dir",
                            str(retrieval_staging_dir),
                            "--device",
                            args.retrieval_device,
                            "--siglip-top-k",
                            str(args.siglip_top_k),
                            "--final-top-k",
                            str(args.retrieval_final_top_k),
                            "--batch-size",
                            str(args.retrieval_batch_size),
                        ],
                        output_dir=retrieval_staging_dir,
                        label="visual_retrieval",
                        timeout_seconds=args.mvinverse_timeout_seconds,
                    )
                    if not (
                        retrieval_staging_dir / "visual_retrieval.json"
                    ).is_file():
                        raise RuntimeError(
                            "Visual retrieval completed without a reusable result"
                        )
                except (RuntimeError, subprocess.TimeoutExpired):
                    if retrieval_attempt >= retrieval_attempts:
                        raise
                    abandoned = (
                        _quarantine_incomplete_stage(retrieval_staging_dir)
                        if retrieval_staging_dir.exists()
                        else None
                    )
                    emit_progress(
                        scope=PROGRESS_SCOPE,
                        stage="visual_retrieval",
                        state="update",
                        current=retrieval_attempt,
                        total=retrieval_attempts,
                        unit="attempts",
                        detail=(
                            "Retrieval runtime failed; retrying SigLIP2 and "
                            "DINOv2 in a fresh process"
                            + (
                                f" (diagnostics: {abandoned})"
                                if abandoned is not None
                                else ""
                            )
                        ),
                    )
                    continue
                retrieval_staging_dir.replace(retrieval_output_dir)
                break
            visual_retrieval = _read_json(retrieval_result_path)
            try:
                visual_retrieval_by_group = _validate_visual_retrieval_result(
                    visual_retrieval,
                    request_path=retrieval_request_path,
                    catalog_path=args.catalog,
                    material_root=args.material_root,
                    siglip2_model_path=args.siglip2_model,
                    dinov2_model_path=args.dinov2_model,
                    retrieval_python=args.retrieval_python,
                    siglip_top_k=args.siglip_top_k,
                    final_top_k=args.retrieval_final_top_k,
                    batch_size=args.retrieval_batch_size,
                    device=args.retrieval_device,
                    allowed_material_ids={
                        str(item["material_id"]) for item in pool
                    },
                    observation_bank_path=args.retrieval_observation_bank,
                )
            except Exception:
                if retrieval_output_dir.exists():
                    _quarantine_incomplete_stage(retrieval_output_dir)
                raise
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="visual_retrieval",
            state="complete",
            current=len(visual_retrieval_by_group),
            total=len(palette["groups"]),
            unit="groups",
            detail=(
                "SigLIP2/DINOv2 retrieval completed; exact MDL rendering "
                "remains the final authority"
            ),
        )
    material_group_total = len(palette["groups"])
    emit_progress(
        scope=PROGRESS_SCOPE,
        stage="material_groups",
        state="start",
        current=0,
        total=material_group_total,
        unit="groups",
        detail="material group selection started",
    )
    for group_index, group in enumerate(palette["groups"], start=1):
        group_id = group["group_id"]
        fused_pbr = pbr_groups.get(group_id)
        fusion_group = fusion_groups_by_id.get(group_id)
        selection_group, semantic_reliability = _material_selection_context(
            group,
            fusion_group,
        )
        selection_group[
            "material_selection_objective"
        ] = args.material_selection_objective
        candidates, retrieval_audit = _shortlist_materials_with_audit(
            selection_group,
            pool,
            semantic_reliability=semantic_reliability,
            family_reliable=_family_hint_is_reliable(group, fusion_group),
            mvinverse_pbr_evidence=fused_pbr,
            allow_parameter_writes=not args.immutable_mdl_after_selection,
        )
        tournament_candidates = candidates
        tournament_retrieval_audit = retrieval_audit
        if args.immutable_mdl_after_selection and (
            args.material_selection_objective == SELECTION_OBJECTIVE_VISUAL
            or args.exact_mdl_tournament_max_candidates > len(candidates)
        ):
            (
                tournament_candidates,
                tournament_retrieval_audit,
            ) = _shortlist_materials_with_audit(
                selection_group,
                pool,
                limit=args.exact_mdl_tournament_max_candidates,
                semantic_reliability=semantic_reliability,
                family_reliable=_family_hint_is_reliable(group, fusion_group),
                mvinverse_pbr_evidence=fused_pbr,
                allow_parameter_writes=False,
                visual_similarity_first=(
                    args.material_selection_objective == SELECTION_OBJECTIVE_VISUAL
                ),
            )
        visual_group = visual_retrieval_by_group.get(group_id)
        if visual_group is not None:
            candidates, retrieval_audit = _visual_candidates_with_audit(
                group_id=group_id,
                visual_group=visual_group,
                pool=pool,
                limit=4,
                fallback_candidates=candidates,
                fallback_audit=retrieval_audit,
            )
            (
                tournament_candidates,
                tournament_retrieval_audit,
            ) = _visual_candidates_with_audit(
                group_id=group_id,
                visual_group=visual_group,
                pool=pool,
                limit=args.exact_mdl_tournament_max_candidates,
                fallback_candidates=tournament_candidates,
                fallback_audit=tournament_retrieval_audit,
            )
        has_authoritative_group_mask = any(
            candidate_group_id == group_id
            for _view_id, candidate_group_id in accepted_sam3_masks
        )
        if material_region_segmentation_enabled and not has_authoritative_group_mask:
            if not candidates:
                raise ValueError(f"No safe fallback material exists for {group_id}")
            fallback_material_id = str(candidates[0]["material_id"])
            first = {
                "material_id": fallback_material_id,
                "confidence": 0.0,
                "rationale": (
                    "SAM3 rejected every region; Qwen crop inference was skipped "
                    "and the bounded legacy retrieval seed is unconfirmed"
                ),
            }
            second = dict(first)
            chosen = dict(first)
            confirmed = False
            confirmation_basis = "sam3_mask_unavailable_fail_closed"
        else:
            crop_reference_path = reference_path
            crop_group = group
            if fusion_group is not None:
                crop = _reference_group_multiview_evidence(
                    group_id=group_id,
                    fusion_group=fusion_group,
                    reference_paths=reference_paths,
                    output_path=destination / "reference_crops" / f"{group_id}.png",
                    masks=accepted_sam3_masks,
                    require_masks=material_region_segmentation_enabled,
                )
            else:
                crop = _reference_group_crop(
                    crop_reference_path,
                    crop_group,
                    destination / "reference_crops" / f"{group_id}.png",
                    mask_path=accepted_sam3_masks.get((reference_id, group_id)),
                )
            fused_choice_group = _group_with_pbr_context(
                selection_group,
                fused=fused_pbr,
                allow_parameter_writes=not args.immutable_mdl_after_selection,
            )
            first = client.choose_group_material(
                reference_crop_view={
                    "id": f"ref_group_{group_id}",
                    "image": str(crop),
                },
                group=fused_choice_group,
                candidate_materials=candidates,
                run_label="forward",
            )
            reversed_candidates = list(reversed(candidates))
            second = client.choose_group_material(
                reference_crop_view={
                    "id": f"ref_group_{group_id}",
                    "image": str(crop),
                },
                group=fused_choice_group,
                candidate_materials=reversed_candidates,
                run_label="reverse",
            )
            chosen, confirmed, confirmation_basis = _confirm_material_choices(
                first,
                second,
                candidates,
                mvinverse_pbr_evidence=fused_pbr,
                allow_parameter_writes=not args.immutable_mdl_after_selection,
            )
        physics_resolution = {
            "applied": False,
            "mode": "mutable_selected_mdl",
            "original_material_id": chosen["material_id"],
            "resolved_material_id": chosen["material_id"],
        }
        if args.immutable_mdl_after_selection:
            (
                chosen,
                confirmed,
                confirmation_basis,
                physics_resolution,
            ) = _resolve_immutable_coating_physics_choice(
                chosen,
                confirmed=confirmed,
                confirmation_basis=confirmation_basis,
                candidates=candidates,
                retrieval_audit=retrieval_audit,
            )
        disagreement_tournament: dict[str, Any] | None = None
        persisted_candidates = candidates
        persisted_tournament_candidates = tournament_candidates
        if (
            args.immutable_mdl_after_selection
            and not confirmed
            and confirmation_basis == "forward_reverse_disagreement"
        ):
            (
                persisted_candidates,
                persisted_tournament_candidates,
                disagreement_tournament,
            ) = _build_disagreement_tournament_candidates(
                forward=first,
                reverse=second,
                provisional_seed=chosen,
                qwen_candidates=candidates,
                tournament_candidates=tournament_candidates,
                pool=pool,
                mvinverse_pbr_evidence=fused_pbr,
                maximum_candidates=args.exact_mdl_tournament_max_candidates,
                selection_group=selection_group,
            )
        _write_json(
            destination / "material_candidates" / f"{group_id}.json",
            {
                "group": group,
                "selection_group": selection_group,
                "semantic_reliability": semantic_reliability,
                "mvinverse_pbr_evidence": fused_pbr,
                "retrieval_audit": retrieval_audit,
                "candidates": persisted_candidates,
                "tournament_selection_objective": (args.material_selection_objective),
                "tournament_retrieval_audit": tournament_retrieval_audit,
                "tournament_candidates": persisted_tournament_candidates,
                **(
                    {"disagreement_tournament": disagreement_tournament}
                    if disagreement_tournament is not None
                    else {}
                ),
            },
        )
        chosen_retrieval = next(
            (
                record
                for record in retrieval_audit["ranking"]
                if record["material_id"] == chosen["material_id"]
            ),
            None,
        )
        independent_choices: list[dict[str, Any]] = []
        if mvinverse_evidence is not None:
            for view_id, view_pbr in _mvinverse_view_groups(
                mvinverse_evidence, group_id
            ):
                boxes = view_pbr.get("boxes")
                accepted_mask = accepted_sam3_masks.get((view_id, group_id))
                if accepted_mask is None and (not isinstance(boxes, list) or not boxes):
                    continue
                if view_id not in reference_paths:
                    raise ValueError(
                        f"MVInverse evidence references unknown source view: {view_id}"
                    )
                view_group = {
                    **group,
                    "boxes": boxes if isinstance(boxes, list) else [],
                }
                view_crop = _reference_group_crop(
                    reference_paths[view_id],
                    view_group,
                    destination
                    / "reference_crops_multiview"
                    / group_id
                    / f"{_artifact_slug(view_id)}.png",
                    mask_path=accepted_mask,
                )
                view_choice = client.choose_group_material(
                    reference_crop_view={
                        "id": f"ref_group_{group_id}_{view_id}",
                        "image": str(view_crop),
                    },
                    group=_group_with_pbr_context(
                        selection_group,
                        fused=None,
                        view_record=view_pbr,
                        allow_parameter_writes=not args.immutable_mdl_after_selection,
                    ),
                    candidate_materials=candidates,
                    run_label=f"evidence_{_artifact_slug(view_id)}",
                )
                independent_choices.append(
                    {
                        "view_id": view_id,
                        "canonical_group_id": group_id,
                        "source_local_group_id": view_pbr["association"].get(
                            "matched_group_id"
                        ),
                        "model_material_id": view_choice["material_id"],
                        "material_id": (
                            chosen["material_id"]
                            if _mvinverse_tunable_equivalence_key(
                                view_choice["material_id"],
                                fused_pbr,
                                allow_parameter_writes=not (
                                    args.immutable_mdl_after_selection
                                ),
                            )
                            == _mvinverse_tunable_equivalence_key(
                                chosen["material_id"],
                                fused_pbr,
                                allow_parameter_writes=not (
                                    args.immutable_mdl_after_selection
                                ),
                            )
                            else view_choice["material_id"]
                        ),
                        "confidence": view_choice["confidence"],
                        "candidate_margin": _retrieval_margin_for_choice(
                            (
                                {
                                    **view_choice,
                                    "material_id": chosen["material_id"],
                                }
                                if _mvinverse_tunable_equivalence_key(
                                    view_choice["material_id"],
                                    fused_pbr,
                                    allow_parameter_writes=not (
                                        args.immutable_mdl_after_selection
                                    ),
                                )
                                == _mvinverse_tunable_equivalence_key(
                                    chosen["material_id"],
                                    fused_pbr,
                                    allow_parameter_writes=not (
                                        args.immutable_mdl_after_selection
                                    ),
                                )
                                else view_choice
                            ),
                            retrieval_audit,
                        ),
                        "resolution_basis": (
                            "exact_material"
                            if view_choice["material_id"] == chosen["material_id"]
                            else (
                                "mvinverse_tunable_module_equivalence"
                                if _mvinverse_tunable_equivalence_key(
                                    view_choice["material_id"],
                                    fused_pbr,
                                    allow_parameter_writes=not (
                                        args.immutable_mdl_after_selection
                                    ),
                                )
                                == _mvinverse_tunable_equivalence_key(
                                    chosen["material_id"],
                                    fused_pbr,
                                    allow_parameter_writes=not (
                                        args.immutable_mdl_after_selection
                                    ),
                                )
                                else "unresolved_material_difference"
                            )
                        ),
                        "crop": str(view_crop),
                        "mvinverse_association": view_pbr["association"],
                    }
                )
        group_view_choices[group_id] = independent_choices
        (
            selection_confidence,
            confidence_derivation,
        ) = _derive_material_selection_confidence(
            first=first,
            second=second,
            chosen=chosen,
            confirmed=confirmed,
            confirmation_basis=confirmation_basis,
            retrieval_audit=retrieval_audit,
            independent_choices=independent_choices,
        )
        group_materials["selections"].append(
            {
                "group_id": group_id,
                "material_id": chosen["material_id"],
                "confidence": selection_confidence,
                "confirmed": confirmed,
            }
        )
        material_audit[group_id] = {
            "selection_group": selection_group,
            "semantic_reliability": semantic_reliability,
            "retrieval_audit": retrieval_audit,
            "chosen_retrieval_rank": (
                chosen_retrieval["rank"] if chosen_retrieval is not None else None
            ),
            "model_choice_matches_retrieval_top": (
                chosen_retrieval is not None and chosen_retrieval["rank"] == 1
            ),
            "forward": first,
            "reverse": second,
            "confirmed": confirmed,
            "confirmation_basis": confirmation_basis,
            "confirmed_material_id": (chosen["material_id"] if confirmed else None),
            "selection_confidence": selection_confidence,
            "confidence_derivation": confidence_derivation,
            "physics_consistency_resolution": physics_resolution,
            **(
                {"disagreement_tournament": disagreement_tournament}
                if disagreement_tournament is not None
                else {}
            ),
            "independent_view_choices": independent_choices,
        }
        emit_progress(
            scope=PROGRESS_SCOPE,
            stage="material_groups",
            state="update",
            current=group_index,
            total=material_group_total,
            unit="groups",
            detail=f"material group {group_id} completed",
        )
        if client.repair_events:
            _write_json(
                destination / "model_repair_events.json",
                {"events": client.repair_events},
            )
    emit_progress(
        scope=PROGRESS_SCOPE,
        stage="material_groups",
        state="complete",
        current=material_group_total,
        total=material_group_total,
        unit="groups",
        detail="material group selection completed",
    )
    _write_json(destination / "group_materials.json", group_materials)
    _write_json(destination / "material_choice_audit.json", material_audit)

    allowed_material_ids = {item["material_id"] for item in pool}
    result = merge_staged_results(
        palette=palette,
        pre_filter_palette_group_count=_multiview_pre_filter_group_count(
            usable_candidates,
            canonical_palette=palette,
        ),
        batches=gate_batches,
        batch_targets=batch_targets,
        material_selections=group_materials,
        allowed_material_ids=allowed_material_ids,
        all_part_ids=set(part_by_id),
        forced_unknown_parts=forced,
        orientation_confidence=args.orientation_confidence,
    )
    _write_json(destination / "staged_result.json", result)
    _write_json(destination / "material_plan.json", result["material_plan"])
    unattended_summary: dict[str, Any] | None = None
    if mvinverse_evidence is not None:
        view_evidence = build_part_view_evidence(
            batches=gate_batches,
            group_view_choices=group_view_choices,
            mapping_votes=mapping_votes,
        )
        _write_json(destination / "view_evidence.json", view_evidence)
        gate_report = evaluate_confidence_gate(
            result,
            registry,
            batches=gate_batches,
            material_choice_audit=material_audit,
            view_evidence=view_evidence,
            geometry_risk_report=geometry_risk_report,
            independent_validation_audit=spatial_mapping_gate["audit"],
        )
        _write_json(destination / "confidence_gate.json", gate_report)
        _write_json(
            destination / "auto_material_plan.json",
            gate_report["auto_material_plan"],
        )
        validate_mvinverse_evidence(mvinverse_evidence)
        autonomous = parameterize_auto_material_plan(
            auto_material_plan=gate_report["auto_material_plan"],
            batches=gate_batches,
            palette=palette,
            mvinverse_evidence=mvinverse_evidence,
            allowed_material_ids=allowed_material_ids,
            allow_parameter_writes=not args.immutable_mdl_after_selection,
        )
        _write_json(destination / "mvinverse_autonomy.json", autonomous)
        _write_json(
            destination / "autonomous_uniform_material_plan.json",
            autonomous["material_plan"],
        )
        if face_region_manifest_path is None:
            raise AssertionError("enabled unattended run has no face-region manifest")
        face_recovery = build_face_material_recovery(
            base_material_plan=autonomous["material_plan"],
            confidence_gate=gate_report,
            face_region_manifest=face_region_manifest_path,
            spatial_mapping_report=spatial_mapping_report,
            canonical_palette=palette,
            mvinverse_evidence=mvinverse_evidence,
            batches=gate_batches,
            allowed_material_ids=allowed_material_ids,
            group_materials=group_materials,
            material_choice_audit=material_audit,
            allow_parameter_writes=not args.immutable_mdl_after_selection,
        )
        _write_json(destination / "face_material_recovery.json", face_recovery)
        _write_json(
            destination / "autonomous_material_plan.json",
            face_recovery["material_plan"],
        )
        final_assignment_count = len(face_recovery["material_plan"]["assignments"])
        unattended_summary = {
            "state": (
                "READY_TO_APPLY" if final_assignment_count else "COMPLETED_SAFE_NOOP"
            ),
            "confidence_gate": gate_report["summary"],
            "geometry_risk": geometry_risk_report["summary"],
            "mapping_consensus": mapping_consensus["audit"]["summary"],
            "spatial_mapping": spatial_mapping_report["summary"],
            "spatial_gate": spatial_mapping_gate["audit"]["summary"],
            "parameterization": autonomous["summary"],
            "face_material_recovery": face_recovery["summary"],
            "artifacts": {
                "qwen_ledger": str(destination / "qwen_inference_ledger.json"),
                "mvinverse_ledger": str(
                    destination
                    / MVINVERSE_OUTPUT_DIRECTORY
                    / "mvinverse_inference_ledger.json"
                ),
                "pbr_evidence": str(destination / "mvinverse_pbr_evidence.json"),
                "geometry_risk": str(
                    destination / "geometry_uniform_material_risk.json"
                ),
                "mapping_consensus": str(
                    destination / "part_mapping_multiview_audit.json"
                ),
                "spatial_mapping_report": str(
                    destination / "spatial_mapping_report.json"
                ),
                "spatial_mapping_audit": str(
                    destination / "spatial_mapping_audit.json"
                ),
                "view_evidence": str(destination / "view_evidence.json"),
                "confidence_gate": str(destination / "confidence_gate.json"),
                "face_material_recovery": str(
                    destination / "face_material_recovery.json"
                ),
                "material_plan": str(destination / "autonomous_material_plan.json"),
                **(
                    {
                        "sam3_foreground_manifest": str(
                            destination / "sam3_foreground" / "manifest.json"
                        ),
                        "foreground_inference_manifest": str(
                            destination
                            / "foreground_inference"
                            / "foreground_inference_manifest.json"
                        ),
                        "sam3_manifest": str(
                            destination / "sam3_regions" / "manifest.json"
                        ),
                        "visual_retrieval": str(
                            destination / "visual_retrieval" / "visual_retrieval.json"
                        ),
                    }
                    if material_region_segmentation_enabled
                    else {}
                ),
            },
        }
        _write_json(destination / "unattended_result.json", unattended_summary)
    console_result: dict[str, Any] = {
        "output": str(destination),
        **result["audit"],
    }
    if unattended_summary is not None:
        console_result["unattended"] = unattended_summary
    _write_json(
        destination / "material_stage_contract.json",
        material_stage_contract_document(),
    )
    print(json.dumps(console_result, ensure_ascii=False))
    return 0


def _requested_output_directory(argv: list[str] | None) -> Path | None:
    """Read ``--output-dir`` without reparsing or mutating the full CLI."""

    values = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(values):
        if value == "--output-dir":
            if index + 1 >= len(values):
                return None
            return Path(values[index + 1]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the staged workflow and classify deterministic inference failures.

    Palette-specific failures are published at their precise evidence
    boundary.  This outer guard covers every other structured Qwen call and
    fail-closed material-selection contract so deterministic failures cannot
    trigger an identical fresh-process retry of the expensive mapping and
    retrieval stages.
    """

    try:
        return _main(argv)
    except (QwenResponseError, StagedAnalysisError, ConfidenceGateError) as exc:
        destination = _requested_output_directory(argv)
        stage_name = (
            "confidence_gate"
            if isinstance(exc, ConfidenceGateError)
            else getattr(exc, "stage_name", None)
        )
        is_deterministic_stage_failure = (
            isinstance(exc, (QwenResponseError, ConfidenceGateError))
        ) or (isinstance(stage_name, str) and bool(stage_name))
        if (
            is_deterministic_stage_failure
            and destination is not None
            and destination.is_dir()
        ):
            failure_path = destination / INFERENCE_FAILURE_FILENAME
            if not failure_path.exists():
                reason = getattr(exc, "reason", None)
                raw_output_path = getattr(exc, "raw_output_path", None)
                checkpoint_path = getattr(exc, "checkpoint_path", None)
                checkpoint_context = getattr(exc, "checkpoint_context", None)
                context = {
                    "exception_type": type(exc).__name__,
                    **({"reason": reason} if isinstance(reason, str) else {}),
                    **(
                        {"raw_output": str(raw_output_path)}
                        if isinstance(raw_output_path, (str, Path))
                        else {}
                    ),
                    **(
                        {"checkpoint": str(checkpoint_path)}
                        if isinstance(checkpoint_path, (str, Path))
                        else {}
                    ),
                    **(
                        {"checkpoint_context": checkpoint_context}
                        if isinstance(checkpoint_context, dict)
                        else {}
                    ),
                    **(
                        {"collapse_diagnostic": exc.diagnostic}
                        if isinstance(exc, MaterialCollapseError)
                        else {}
                    ),
                }
                _write_inference_failure(
                    destination,
                    error_code=(
                        "material_disagreement_tournament_unbuildable"
                        if stage_name == "material_disagreement_tournament"
                        else "material_collapse_detected"
                        if stage_name == "material_collapse_gate"
                        else "confidence_gate_contract_invalid"
                        if isinstance(exc, ConfidenceGateError)
                        else "qwen_stage_checkpoint_invalid"
                        if isinstance(checkpoint_path, (str, Path))
                        else "qwen_structured_output_invalid"
                        if isinstance(exc, QwenResponseError)
                        else "qwen_structured_output_schema_invalid"
                    ),
                    failed_stage=(
                        str(stage_name)
                        if isinstance(stage_name, str) and stage_name
                        else "structured_qwen"
                    ),
                    detail=str(exc),
                    view_failures=[],
                    context=context,
                )
                emit_progress(
                    scope=PROGRESS_SCOPE,
                    stage=(
                        str(stage_name)
                        if isinstance(stage_name, str) and stage_name
                        else "structured_qwen"
                    ),
                    state="failed",
                    detail=str(exc),
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
