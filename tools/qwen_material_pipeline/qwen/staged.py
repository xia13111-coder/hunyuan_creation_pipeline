"""Qwen payloads for the small, rotation-invariant staged material workflow.

The model never receives USD/MDL filesystem paths.  Each live call has one
bounded job: extract a visible palette, map at most a few highlighted parts to
that palette, or choose among at most four already-whitelisted materials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_material_pipeline.core.staged_analysis import (
    BASE_COLORS,
    FAMILY_HINTS,
    FINISH_HINTS,
    PALETTE_SCHEMA_VERSION,
    StagedAnalysisError,
    normalize_part_palette_batch,
    validate_palette,
)
from qwen_material_pipeline.qwen.client import (
    QwenContentParseError,
    QwenResponseError,
    load_image_url,
    parse_plan_content_with_audit,
)
from qwen_material_pipeline.qwen.local_vl import LocalGenerationResult


MATERIAL_CHOICE_SCHEMA_VERSION = "qwen-palette-material-choice/v1"
VALIDATED_STAGE_CHECKPOINT_SCHEMA_VERSION = (
    "qwen-validated-stage-checkpoint/v1"
)
GEOMETRY_VIEW_PREFIXES = (
    "cad_",
    "part_ids_",
    "part_contact_",
    "part_highlight_",
    "batch_parts_",
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise StagedAnalysisError(
            f"value is not canonical JSON: {error}"
        ) from error
    return rendered.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Durably publish one complete JSON object without a partial-file window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StagedAnalysisError(f"{label} must be a non-empty string")
    return value.strip()


def _view(view: Mapping[str, Any], label: str, *, geometry: bool) -> dict[str, str]:
    if not isinstance(view, Mapping):
        raise StagedAnalysisError(f"{label} must be an object")
    view_id = _nonempty(view.get("id"), f"{label}.id")
    image = view.get("image")
    if not isinstance(image, (str, Path)):
        raise StagedAnalysisError(f"{label}.image must be a path or URL")
    is_geometry = view_id.startswith(GEOMETRY_VIEW_PREFIXES)
    if geometry and not is_geometry:
        raise StagedAnalysisError(f"{label}.id must use a reserved geometry prefix")
    if not geometry and is_geometry:
        raise StagedAnalysisError(f"{label}.id must identify a user reference")
    return {"id": view_id, "image": str(image)}


def _image_blocks(view: Mapping[str, str], label: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": f"{label} view_id: {view['id']}"},
        {
            "type": "image_url",
            "image_url": {"url": load_image_url(view["image"])},
        },
    ]


def _payload(model: str, content: list[dict[str, Any]], system: str) -> dict[str, Any]:
    model_name = _nonempty(model, "model")
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
        "enable_thinking": False,
    }


def build_palette_payload(
    model: str,
    reference_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a one-image request that extracts only visibly present appearances."""

    reference = _view(reference_view, "reference_view", geometry=False)
    output_shape = {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": reference["id"],
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "one allowed family",
                "base_color": "one allowed color",
                "finish_hint": "one allowed finish",
                "visual_description": "short visible surface description",
                "boxes": [[0, 0, 1000, 1000]],
                "confidence": 0.0,
            }
        ],
    }
    prompt = "\n".join(
        (
            "Extract only object-surface appearances visibly present in the supplied reference image.",
            "The input is one original camera view, not a contact sheet; reason only from this image.",
            "Group by material family plus base color plus finish, not by object part.",
            "In visual_description, name the visible object type and any directly visible substance cue (for example copper tube, rubber hose, plastic cap, galvanized fastener); do not describe it only as an orange/blue/gray part.",
            "Use specular response, surface texture, and industrial shape to distinguish a bare copper/brass tube from orange paint and a flexible hose from a painted rigid rod, but leave uncertain substances generic.",
            "Actively inspect every foreground quadrant for secondary appearances; do not stop after finding the largest housing or panel.",
            "Keep black or dark painted structural parts, white or light-gray modules, and bare silver metal in separate groups whenever each is visibly present.",
            "A black object on a black background is still an object when its silhouette, edge highlights, occlusion, or connected industrial geometry is visible; cite only a small illuminated interior patch and never cite the background.",
            "Do not merge dark painted arms, brackets, bases, or covers into bare silver metal merely because they have bright edge highlights.",
            "Ignore the background, grid lines, axes, UI overlays, lighting, shadows, and reflections.",
            "Do not infer hidden or rear surfaces and do not choose any MDL material in this step.",
            "Every group must cite 1 to 4 tight [x0,y0,x1,y1] boxes using integer 0..1000 image coordinates.",
            "Each box must lie tightly inside one single, visibly homogeneous object surface showing only that group's color and finish.",
            "Exclude background, silhouette boundaries, seams, neighboring colors/materials, cast shadows, glare, and specular highlights from every box.",
            "If a part contains multiple colors or materials, use separate small interior boxes; never wrap one box around the mixed part.",
            "Never use the whole image or a broad multi-surface region as a placeholder.",
            "No single box may cover 85 percent or more of the image area.",
            "Use sequential unique group IDs G01, G02, ... and return at most 12 groups.",
            "Allowed family_hint values: " + json.dumps(sorted(FAMILY_HINTS)),
            "Allowed base_color values: " + json.dumps(sorted(BASE_COLORS)),
            "Allowed finish_hint values: " + json.dumps(sorted(FINISH_HINTS)),
            "Return exactly one strict JSON object with no Markdown or prose.",
            "The top-level and group fields must exactly match this shape: "
            + json.dumps(output_shape, ensure_ascii=False),
        )
    )
    content = _image_blocks(reference, "USER REFERENCE")
    content.append({"type": "text", "text": prompt})
    return _payload(
        model,
        content,
        "You are a conservative visible-surface palette extractor. Obey the exact JSON contract.",
    )


def _build_palette_schema_retry_payload(
    original_payload: Mapping[str, Any],
    *,
    source_view_id: str,
    rejected_content: str,
    schema_error: Exception,
) -> dict[str, Any]:
    """Build one causal, schema-only correction request.

    The first response is retained verbatim in the conversation so the model
    can correct the actual rejected document.  This is deliberately not a
    schema repair: the second model response must independently pass the same
    strict parser and palette validator as the first.
    """

    payload = deepcopy(dict(original_payload))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise StagedAnalysisError(
            "palette schema retry requires an original messages array"
        )
    expected_group_fields = [
        "group_id",
        "family_hint",
        "base_color",
        "finish_hint",
        "visual_description",
        "boxes",
        "confidence",
    ]
    correction = "\n".join(
        (
            "Your immediately preceding response was valid JSON but failed the "
            "required palette schema.",
            f"Source view_id (must be preserved exactly): {source_view_id}",
            f"Schema validation error: {schema_error}",
            "Correct that rejected response for the same supplied source image.",
            "Return exactly one strict JSON object only: no Markdown fences, "
            "no prose, no explanation, and no diagnostic wrapper.",
            "The top-level keys must be exactly "
            '["schema_version","source_view_id","groups"].',
            f"schema_version must be exactly {PALETTE_SCHEMA_VERSION!r}.",
            "Each groups item must have exactly these keys: "
            + json.dumps(expected_group_fields, separators=(",", ":"))
            + ".",
            "Do not return error, usable, status, or any other extra field.",
            "All original enum, box, confidence, sequential-ID, and visible-"
            "evidence constraints still apply without relaxation.",
        )
    )
    messages.extend(
        (
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": rejected_content,
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": correction}],
            },
        )
    )
    return payload


def _build_strict_json_retry_payload(
    original_payload: Mapping[str, Any],
    *,
    rejected_content: str,
    parse_error: Exception,
    contract_context: str,
) -> dict[str, Any]:
    """Request one fresh strict-JSON response without accepting malformed text."""

    payload = deepcopy(dict(original_payload))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise StagedAnalysisError(
            "strict JSON retry requires an original messages array"
        )
    messages.extend(
        (
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rejected_content}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(
                            (
                                "FORMAT REPAIR: your immediately preceding "
                                "response was rejected by the strict JSON parser.",
                                "Parser error: " + str(parse_error),
                                contract_context,
                                "Regenerate the complete answer for the same "
                                "images and task.",
                                "Return exactly one strict JSON object only: no "
                                "second object, trailing text, Markdown fences, "
                                "prose, comments, or diagnostic wrapper.",
                                "Do not quote or embed the rejected response.",
                            )
                        ),
                    }
                ],
            },
        )
    )
    return payload


def _part_summary(part: Mapping[str, Any]) -> dict[str, Any]:
    part_id = _nonempty(part.get("part_id"), "part.part_id")
    renders = part.get("renders")
    visible_pixels = []
    if isinstance(renders, Sequence) and not isinstance(renders, (str, bytes)):
        for render in renders:
            if isinstance(render, Mapping):
                pixels = render.get("visible_pixels")
                if isinstance(pixels, int) and not isinstance(pixels, bool):
                    visible_pixels.append(pixels)
    result: dict[str, Any] = {
        "part_id": part_id,
        "max_visible_pixels": max(visible_pixels, default=0),
    }
    evidence_pixels = part.get("evidence_visible_pixels")
    if isinstance(evidence_pixels, int) and not isinstance(evidence_pixels, bool):
        result["evidence_visible_pixels"] = evidence_pixels
    source_pixels = part.get("evidence_source_visible_pixels")
    if isinstance(source_pixels, int) and not isinstance(source_pixels, bool):
        result["evidence_source_visible_pixels"] = source_pixels
    evidence_mode = part.get("evidence_mode")
    if isinstance(evidence_mode, str) and evidence_mode:
        result["evidence_mode"] = evidence_mode
    source_view_count = part.get("evidence_source_view_count")
    if isinstance(source_view_count, int) and not isinstance(source_view_count, bool):
        result["evidence_source_view_count"] = source_view_count
    for key in ("point_count", "face_count", "world_bbox"):
        value = part.get(key)
        if value is not None:
            result[key] = value
    return result


def build_part_palette_payload(
    model: str,
    *,
    reference_view: Mapping[str, Any],
    cad_view: Mapping[str, Any],
    part_id_view: Mapping[str, Any],
    batch_sheet_view: Mapping[str, Any],
    palette: Mapping[str, Any],
    target_parts: Sequence[Mapping[str, Any]],
    batch_id: str,
    support_reference_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a four/five-image, exact-cover part-to-palette request."""

    reference = _view(reference_view, "reference_view", geometry=False)
    cad = _view(cad_view, "cad_view", geometry=True)
    part_ids = _view(part_id_view, "part_id_view", geometry=True)
    batch_sheet = _view(batch_sheet_view, "batch_sheet_view", geometry=True)
    support_reference = (
        _view(support_reference_view, "support_reference_view", geometry=False)
        if support_reference_view is not None
        else None
    )
    canonical_palette = validate_palette(
        palette, allowed_reference_view_ids={reference["id"]}
    )
    batch_name = _nonempty(batch_id, "batch_id")
    if not re.fullmatch(r"B[0-9]{2,4}", batch_name):
        raise StagedAnalysisError("batch_id must use B followed by 2..4 digits")
    if (
        isinstance(target_parts, (str, bytes))
        or not isinstance(target_parts, Sequence)
        or not 1 <= len(target_parts) <= 4
    ):
        raise StagedAnalysisError("target_parts must contain 1..4 part objects")
    summaries = [_part_summary(part) for part in target_parts]
    target_ids = [part["part_id"] for part in summaries]
    if len(set(target_ids)) != len(target_ids):
        raise StagedAnalysisError("target_parts contains duplicate part_id values")
    valid_box_indices = {
        group["group_id"]: list(range(len(group["boxes"])))
        for group in canonical_palette["groups"]
    }

    # Keep only semantic claims in the model contract.  The caller derives the
    # schema version, request batch ID, status, and evidence view deterministically
    # before passing the result through the unchanged strict validator.  Legacy
    # full-schema answers remain accepted by that normalization boundary.
    output_shape = {
        "mappings": [
            {
                "part_id": "one target part ID",
                "group_id": "one supplied Gxx or null",
                "mapping_confidence": 0.0,
                "evidence_box_index": "valid zero-based box index or null",
                "reason_code": "allowed reason code",
            }
        ],
    }
    prompt = "\n".join(
        (
            "Map every highlighted target CAD part to one visible reference-image palette group.",
            "The primary user reference is one original camera view and is the only material-evidence image.",
            "If SAME-ASSET SUPPORT VIEWS are supplied, use them only to confirm invariant shape, attachment, and localization; the primary USER REFERENCE palette box remains the required material/color citation.",
            "The photo and CAD show the same asset after a supplied global rigid rotation; do not reject identity because one representation was originally horizontal and the other upright.",
            "Use invariant shape, attachment, and neighboring-part relationships. Ignore absolute CAD world axes and view-label semantics.",
            "CAD, part-ID, and red-outline highlight images prove geometry and identity only; their neutral gray or red pixels never prove material or color.",
            "Only the user reference and the cited palette box prove a real surface appearance.",
            "Return exactly one mapping for every target part and no mapping for any other part.",
            "Choose a group only when the highlighted CAD part can be localized to that group's cited reference box.",
            "Do not propagate a dominant body color to neighboring fasteners, rods, hoses, arms, or panels.",
            "Do not infer from typical machine construction. If a part is occluded, absent, or cannot be localized, do not cite a palette group.",
            "For evidence_mode=isolated_mask_multiview, evidence_visible_pixels is the normalized geometry-only target size and evidence_source_visible_pixels is the unchanged source projection size. The isolated target is background-removed, grayscale, and enlarged only for shape identity; it never proves material or color.",
            "A target with evidence_visible_pixels below 256 may cite a group only for human review; use reason_code too_small. If evidence_visible_pixels is absent, use max_visible_pixels.",
            "Use reason_code shape_and_location or direct_visual_match for a positive localization; partial_visibility, ambiguous, or too_small for uncertain localization; and occluded, not_in_reference, or no_cad_render when no citation is possible.",
            "When group_id is non-null, evidence_box_index must exist in that chosen group.",
            "When no group can be cited, set both group_id and evidence_box_index to null and keep mapping_confidence below 0.60.",
            "evidence_box_index is the zero-based index inside the chosen palette group's boxes array; it is NOT the target-part number, target-list position, or 2x2 tile number.",
            "The only valid evidence_box_index values for this request are: "
            + json.dumps(valid_box_indices, ensure_ascii=False),
            "Do not return schema_version, batch_id, status, or evidence_view_id; the caller derives them from this request.",
            "Return exactly one strict JSON object with no Markdown or prose and exactly these semantic fields: "
            + json.dumps(output_shape, ensure_ascii=False),
            "Target parts: " + json.dumps(summaries, ensure_ascii=False),
            "Palette groups: " + json.dumps(canonical_palette, ensure_ascii=False),
        )
    )
    content: list[dict[str, Any]] = []
    content.extend(_image_blocks(reference, "USER REFERENCE - MATERIAL EVIDENCE"))
    if support_reference is not None:
        content.extend(
            _image_blocks(
                support_reference,
                "SAME-ASSET SUPPORT VIEWS - IDENTITY/LOCALIZATION ONLY",
            )
        )
    content.extend(_image_blocks(cad, "ORIENTED CAD OVERVIEW - GEOMETRY ONLY"))
    content.extend(_image_blocks(part_ids, "PART-ID OVERVIEW - GEOMETRY ONLY"))
    content.extend(_image_blocks(batch_sheet, "TARGET HIGHLIGHTS - GEOMETRY ONLY"))
    content.append({"type": "text", "text": prompt})
    return _payload(
        model,
        content,
        "You are a conservative rotation-invariant part correspondence classifier. Obey the exact JSON contract.",
    )


def _visible_material(candidate: Mapping[str, Any]) -> dict[str, Any]:
    material_id = _nonempty(candidate.get("material_id"), "candidate.material_id")
    visible = {"material_id": material_id}
    for key in (
        "display_name",
        "family",
        "colors",
        "finishes",
        "description",
        "appearance_profile",
        "surface_interpretation",
    ):
        value = candidate.get(key)
        if value is not None:
            visible[key] = value
    return visible


def build_group_material_payload(
    model: str,
    *,
    reference_crop_view: Mapping[str, Any],
    group: Mapping[str, Any],
    candidate_materials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one appearance-group choice among at most four local candidates."""

    reference = _view(reference_crop_view, "reference_crop_view", geometry=False)
    group_id = _nonempty(group.get("group_id"), "group.group_id")
    if (
        isinstance(candidate_materials, (str, bytes))
        or not isinstance(candidate_materials, Sequence)
        or not 1 <= len(candidate_materials) <= 4
    ):
        raise StagedAnalysisError("candidate_materials must contain 1..4 objects")
    visible_candidates = [_visible_material(item) for item in candidate_materials]
    material_ids = [item["material_id"] for item in visible_candidates]
    if len(set(material_ids)) != len(material_ids):
        raise StagedAnalysisError("candidate_materials contains duplicate IDs")
    content = _image_blocks(reference, "REFERENCE CROP - MATERIAL EVIDENCE")
    for candidate, visible in zip(candidate_materials, visible_candidates, strict=True):
        preview = candidate.get("thumbnail_image") or candidate.get("preview_image")
        if preview:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "CANDIDATE PREVIEW - SELECTABLE, NOT EVIDENCE; "
                        f"material_id: {visible['material_id']}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": load_image_url(preview)},
                }
            )
    output_shape = {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": group_id,
        "material_id": "one supplied exact material_id",
        "confidence": 0.75,
    }
    pbr_context = group.get("mvinverse_pbr_context")
    immutable_defaults = (
        isinstance(pbr_context, Mapping)
        and pbr_context.get("selected_mdl_parameters_mutable") is False
        and pbr_context.get("library_defaults_must_match_reference") is True
    )
    visual_only = group.get("material_selection_objective") == "visual_similarity"
    selection_instruction = (
        "Optimize visible similarity only. Physical material name, family, "
        "category, and engineering plausibility are not constraints. A "
        "candidate from another physical family is correct when its immutable "
        "preview more closely matches the reference crop's color, value, "
        "highlight shape, reflectance, roughness, and surface texture. Never "
        "prefer semantic plausibility over the visible pixels."
        if visual_only
        else (
            "Match physical family first, then base color, finish, roughness, "
            "and wear. Do not choose the first candidate by default."
        )
    )
    default_appearance_instruction = (
        "The selected MDL and all of its parameters will be immutable. "
        "Choose the candidate whose supplied appearance_profile and preview "
        "already match the reference; do not assume any later recoloring, "
        "roughness edit, metallic edit, texture disablement, or wear removal."
        if immutable_defaults
        else "Use the supplied appearance_profile and preview as bounded catalog evidence."
    )
    prompt = "\n".join(
        (
            "Choose the closest material-library appearance for exactly one reference palette group.",
            "Select only an exact supplied material_id, including the literal '.mdl#ExportName' suffix. Candidate previews are alternatives, never reference evidence.",
            selection_instruction,
            "Treat the palette finish_hint and visual_description as fallible upstream observations; the reference crop is primary evidence for visible construction and surface response.",
            "The reference may be a labeled sheet of 2 to 4 tight crops from different camera views of the same canonical appearance group; use their shared physical cues and do not treat the tiles as different materials.",
            "Choose an applied-paint or conversion-coating candidate only when the crop visibly supports that treatment. For reddish, orange, brown, or yellow metal, compare continuous specular highlights and curved tube/fitting geometry against clean copper, brass, and bronze candidates instead of assuming colored paint.",
            default_appearance_instruction,
            "confidence is your calibrated confidence that the selected exact material_id is the closest visual match among only the supplied candidates. Use 0.85..1.0 for a clear crop-and-preview match, 0.60..0.849999 for a plausible but ambiguous match, and below 0.60 when the supplied candidates or crop do not distinguish the choice. Do not copy the illustrative confidence value from the output shape.",
            "Return exactly one strict JSON object with no Markdown or prose and exactly this shape: "
            + json.dumps(output_shape, ensure_ascii=False),
            "Palette group: " + json.dumps(dict(group), ensure_ascii=False),
            "Candidate materials: "
            + json.dumps(visible_candidates, ensure_ascii=False),
        )
    )
    content.append({"type": "text", "text": prompt})
    return _payload(
        model,
        content,
        "You are a bounded material-library classifier. Obey the exact JSON contract.",
    )


def validate_group_material_choice(
    choice: Mapping[str, Any],
    *,
    group_id: str,
    allowed_material_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(choice, Mapping):
        raise StagedAnalysisError("material choice must be an object")
    expected = {"schema_version", "group_id", "material_id", "confidence"}
    if set(choice) != expected:
        raise StagedAnalysisError(
            "material choice fields are invalid; "
            f"unexpected={sorted(set(choice) - expected)}, missing={sorted(expected - set(choice))}"
        )
    if choice["schema_version"] != MATERIAL_CHOICE_SCHEMA_VERSION:
        raise StagedAnalysisError("unsupported material choice schema_version")
    expected_group = _nonempty(group_id, "group_id")
    if choice["group_id"] != expected_group:
        raise StagedAnalysisError("material choice contains the wrong group_id")
    allowed = set(allowed_material_ids)
    material_id = choice["material_id"]
    if isinstance(material_id, str) and material_id not in allowed:
        # Qwen occasionally converts an exact
        # ``mdl:dir/module.mdl#Export`` candidate to ``mdl:dir/Export``.
        # Accept that formatting-only omission only when it maps back to one
        # and only one supplied candidate. No fuzzy or semantic guessing is
        # permitted here.
        aliases: dict[str, set[str]] = {}
        for candidate_id in allowed:
            if ".mdl#" not in candidate_id:
                continue
            module_id, export_name = candidate_id.split("#", 1)
            module_without_extension = module_id[: -len(".mdl")]
            module_parent = module_without_extension.rsplit("/", 1)[0]
            for alias in (
                module_id,
                module_without_extension,
                f"{module_without_extension}#{export_name}",
                f"{module_parent}/{export_name}",
            ):
                aliases.setdefault(alias, set()).add(candidate_id)
        matches = aliases.get(material_id, set())
        if len(matches) == 1:
            material_id = next(iter(matches))
    if not isinstance(material_id, str) or material_id not in allowed:
        raise StagedAnalysisError(
            f"material choice contains unknown material_id: {material_id!r}"
        )
    confidence = choice["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise StagedAnalysisError("material choice confidence must be from 0 to 1")
    return {
        "schema_version": MATERIAL_CHOICE_SCHEMA_VERSION,
        "group_id": expected_group,
        "material_id": material_id,
        "confidence": float(confidence),
    }


class LocalStagedQwenClient:
    """Run staged payloads through one shared, lazily loaded local Qwen runner."""

    def __init__(
        self,
        *,
        model: str,
        runner: Callable[[Mapping[str, Any]], str],
        raw_output_dir: str | Path,
        generation_runner: Callable[..., LocalGenerationResult] | None = None,
        max_new_tokens: int | None = None,
        max_new_tokens_ceiling: int | None = None,
        generation_event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_identity_sha256: str | None = None,
        reuse_checkpoints: bool = False,
    ) -> None:
        self.model = _nonempty(model, "model")
        self.runner = runner
        self.raw_output_dir = Path(raw_output_dir).expanduser().resolve()
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.repair_events: list[dict[str, Any]] = []
        discovered_generation_runner = getattr(runner, "generate_with_metadata", None)
        self.generation_runner = generation_runner or (
            discovered_generation_runner
            if callable(discovered_generation_runner)
            else None
        )
        discovered_budget = getattr(runner, "max_new_tokens", None)
        if discovered_budget is None and self.generation_runner is not None:
            discovered_budget = getattr(
                getattr(self.generation_runner, "__self__", None),
                "max_new_tokens",
                None,
            )
        self.max_new_tokens = self._optional_positive_int(
            max_new_tokens if max_new_tokens is not None else discovered_budget,
            "max_new_tokens",
        )
        self.max_new_tokens_ceiling = self._optional_positive_int(
            max_new_tokens_ceiling,
            "max_new_tokens_ceiling",
        )
        if (
            self.max_new_tokens is not None
            and self.max_new_tokens_ceiling is not None
            and self.max_new_tokens_ceiling < self.max_new_tokens
        ):
            raise ValueError(
                "max_new_tokens_ceiling must be greater than or equal to "
                "max_new_tokens"
            )
        if generation_event_callback is not None and not callable(
            generation_event_callback
        ):
            raise TypeError("generation_event_callback must be callable or None")
        self.generation_event_callback = generation_event_callback
        self.generation_events: list[dict[str, Any]] = []
        if not isinstance(reuse_checkpoints, bool):
            raise TypeError("reuse_checkpoints must be a boolean")
        if checkpoint_dir is None:
            if checkpoint_identity_sha256 is not None:
                raise ValueError(
                    "checkpoint_identity_sha256 requires checkpoint_dir"
                )
            if reuse_checkpoints:
                raise ValueError("reuse_checkpoints requires checkpoint_dir")
            self.checkpoint_dir: Path | None = None
            self.checkpoint_identity_sha256: str | None = None
        else:
            if (
                not isinstance(checkpoint_identity_sha256, str)
                or _SHA256_RE.fullmatch(checkpoint_identity_sha256) is None
            ):
                raise ValueError(
                    "checkpoint_identity_sha256 must be a lowercase SHA-256 "
                    "when checkpoint_dir is configured"
                )
            self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.reuse_checkpoints = reuse_checkpoints

    @staticmethod
    def _optional_positive_int(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer or None")
        return value

    @staticmethod
    def _safe_stage_name(name: str) -> str:
        return _SAFE_NAME_RE.sub("_", name).strip("._") or "stage"

    def _checkpoint_path(self, stage_name: str) -> Path:
        if self.checkpoint_dir is None:
            raise AssertionError("checkpoint directory is not configured")
        return (
            self.checkpoint_dir
            / f"{self._safe_stage_name(stage_name)}.checkpoint.json"
        )

    def _checkpoint_error(
        self,
        stage_name: str,
        checkpoint_path: Path,
        detail: str,
        *,
        source_stage_name: str | None = None,
    ) -> StagedAnalysisError:
        raw_stage_name = source_stage_name or stage_name
        raw_output_path = (
            self.raw_output_dir
            / f"{self._safe_stage_name(raw_stage_name)}.raw.txt"
        )
        error = StagedAnalysisError(
            f"validated Qwen checkpoint for {stage_name!r} is unsafe: {detail}; "
            f"checkpoint={checkpoint_path}; raw={raw_output_path}"
        )
        error.stage_name = stage_name
        error.raw_output_path = raw_output_path
        error.checkpoint_path = checkpoint_path
        error.checkpoint_context = {
            "stage_name": stage_name,
            "source_stage_name": raw_stage_name,
            "checkpoint_path": str(checkpoint_path),
            "raw_output_path": str(raw_output_path),
        }
        return error

    def _artifact_record(
        self,
        path: Path,
        *,
        checkpoint_path: Path,
    ) -> dict[str, str]:
        if not path.is_file():
            raise StagedAnalysisError(
                f"validated Qwen checkpoint artifact is missing: {path}"
            )
        return {
            "relative_path": os.path.relpath(
                path, start=checkpoint_path.parent
            ),
            "sha256": _file_sha256(path),
        }

    def _write_validated_checkpoint(
        self,
        *,
        stage_name: str,
        stage_kind: str,
        source_stage_name: str,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        if self.checkpoint_dir is None:
            return
        if self.checkpoint_identity_sha256 is None:
            raise AssertionError("checkpoint identity is not configured")
        if source_stage_name != stage_name and re.fullmatch(
            re.escape(stage_name) + r"_retry[1-9][0-9]*",
            source_stage_name,
        ) is None:
            raise self._checkpoint_error(
                stage_name,
                self._checkpoint_path(stage_name),
                "source stage is not the original stage or one of its retries",
                source_stage_name=source_stage_name,
            )
        checkpoint_path = self._checkpoint_path(stage_name)
        safe_source = self._safe_stage_name(source_stage_name)
        raw_path = self.raw_output_dir / f"{safe_source}.raw.txt"
        parse_path = self.raw_output_dir / f"{safe_source}.parse.json"
        try:
            audit = json.loads(parse_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"final parse audit cannot be read: {error}",
                source_stage_name=source_stage_name,
            ) from error
        if (
            not isinstance(audit, Mapping)
            or audit.get("strict_json_valid") is not True
            or audit.get("schema_validation_status") != "valid"
            or audit.get("schema_valid") is not True
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "final parse audit does not prove strict JSON and schema "
                "validation",
                source_stage_name=source_stage_name,
            )
        try:
            raw_record = self._artifact_record(
                raw_path, checkpoint_path=checkpoint_path
            )
            parse_record = self._artifact_record(
                parse_path, checkpoint_path=checkpoint_path
            )
        except (OSError, StagedAnalysisError) as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                str(error),
                source_stage_name=source_stage_name,
            ) from error
        if audit.get("raw_sha256") != raw_record["sha256"]:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "parse audit raw_sha256 does not match the raw output",
                source_stage_name=source_stage_name,
            )
        canonical_result = dict(result)
        unsigned: dict[str, Any] = {
            "schema_version": VALIDATED_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "stage_name": stage_name,
            "stage_kind": stage_kind,
            "input_payload_sha256": _canonical_sha256(payload),
            "checkpoint_identity_sha256": self.checkpoint_identity_sha256,
            "provenance": {
                "source_stage_name": source_stage_name,
                "raw_output": raw_record,
                "parse_audit": parse_record,
            },
            "result": canonical_result,
            "result_sha256": _canonical_sha256(canonical_result),
        }
        document = {
            **unsigned,
            "integrity_sha256": _canonical_sha256(unsigned),
        }
        _atomic_write_json(checkpoint_path, document)

    def _validate_checkpoint_artifact(
        self,
        *,
        stage_name: str,
        checkpoint_path: Path,
        source_stage_name: str,
        label: str,
        record: Any,
        expected_path: Path,
    ) -> tuple[Path, str]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"relative_path", "sha256"}
            or not isinstance(record.get("relative_path"), str)
            or not record["relative_path"]
            or Path(record["relative_path"]).is_absolute()
            or not isinstance(record.get("sha256"), str)
            or _SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"{label} artifact contract is invalid",
                source_stage_name=source_stage_name,
            )
        artifact_path = (
            checkpoint_path.parent / record["relative_path"]
        ).resolve()
        if artifact_path != expected_path.resolve():
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"{label} artifact provenance points to the wrong path",
                source_stage_name=source_stage_name,
            )
        try:
            actual_sha256 = _file_sha256(artifact_path)
        except OSError as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"{label} artifact cannot be read: {error}",
                source_stage_name=source_stage_name,
            ) from error
        if actual_sha256 != record["sha256"]:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"{label} artifact SHA-256 mismatch",
                source_stage_name=source_stage_name,
            )
        return artifact_path, actual_sha256

    def _reuse_validated_checkpoint(
        self,
        *,
        stage_name: str,
        stage_kind: str,
        payload: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if self.checkpoint_dir is None or not self.reuse_checkpoints:
            return None
        checkpoint_path = self._checkpoint_path(stage_name)
        if not checkpoint_path.exists():
            return None
        try:
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"checkpoint JSON cannot be read: {error}",
            ) from error
        expected_fields = {
            "schema_version",
            "stage_name",
            "stage_kind",
            "input_payload_sha256",
            "checkpoint_identity_sha256",
            "provenance",
            "result",
            "result_sha256",
            "integrity_sha256",
        }
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_fields:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "checkpoint fields are invalid",
            )
        source_stage_name = None
        provenance = checkpoint.get("provenance")
        if isinstance(provenance, Mapping):
            candidate_source = provenance.get("source_stage_name")
            if isinstance(candidate_source, str) and candidate_source:
                source_stage_name = candidate_source
        if (
            checkpoint.get("schema_version")
            != VALIDATED_STAGE_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("stage_name") != stage_name
            or checkpoint.get("stage_kind") != stage_kind
            or checkpoint.get("checkpoint_identity_sha256")
            != self.checkpoint_identity_sha256
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "schema, stage, kind, or checkpoint identity mismatch",
                source_stage_name=source_stage_name,
            )
        if (
            not isinstance(checkpoint.get("integrity_sha256"), str)
            or _SHA256_RE.fullmatch(checkpoint["integrity_sha256"]) is None
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "integrity SHA-256 is invalid",
                source_stage_name=source_stage_name,
            )
        unsigned = dict(checkpoint)
        integrity_sha256 = unsigned.pop("integrity_sha256")
        if _canonical_sha256(unsigned) != integrity_sha256:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "whole-checkpoint integrity SHA-256 mismatch",
                source_stage_name=source_stage_name,
            )
        if checkpoint.get("input_payload_sha256") != _canonical_sha256(payload):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "original input payload SHA-256 mismatch",
                source_stage_name=source_stage_name,
            )
        if (
            not isinstance(source_stage_name, str)
            or (
                source_stage_name != stage_name
                and re.fullmatch(
                    re.escape(stage_name) + r"_retry[1-9][0-9]*",
                    source_stage_name,
                )
                is None
            )
            or not isinstance(provenance, Mapping)
            or set(provenance)
            != {"source_stage_name", "raw_output", "parse_audit"}
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "artifact provenance is invalid",
                source_stage_name=source_stage_name,
            )
        safe_source = self._safe_stage_name(source_stage_name)
        raw_path, raw_sha256 = self._validate_checkpoint_artifact(
            stage_name=stage_name,
            checkpoint_path=checkpoint_path,
            source_stage_name=source_stage_name,
            label="raw output",
            record=provenance["raw_output"],
            expected_path=self.raw_output_dir / f"{safe_source}.raw.txt",
        )
        parse_path, _ = self._validate_checkpoint_artifact(
            stage_name=stage_name,
            checkpoint_path=checkpoint_path,
            source_stage_name=source_stage_name,
            label="parse audit",
            record=provenance["parse_audit"],
            expected_path=self.raw_output_dir / f"{safe_source}.parse.json",
        )
        try:
            audit = json.loads(parse_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"parse audit cannot be decoded: {error}",
                source_stage_name=source_stage_name,
            ) from error
        if (
            not isinstance(audit, Mapping)
            or audit.get("raw_sha256") != raw_sha256
            or audit.get("strict_json_valid") is not True
            or audit.get("schema_validation_status") != "valid"
            or audit.get("schema_valid") is not True
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "parse audit no longer proves strict JSON and schema validation",
                source_stage_name=source_stage_name,
            )
        result = checkpoint.get("result")
        if (
            not isinstance(result, Mapping)
            or not isinstance(checkpoint.get("result_sha256"), str)
            or _SHA256_RE.fullmatch(checkpoint["result_sha256"]) is None
            or _canonical_sha256(result) != checkpoint["result_sha256"]
        ):
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "validated result contract or SHA-256 is invalid",
                source_stage_name=source_stage_name,
            )
        try:
            validated = dict(validator(result))
        except (QwenResponseError, StagedAnalysisError, TypeError, ValueError) as error:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                f"cached result fails the current validator: {error}",
                source_stage_name=source_stage_name,
            ) from error
        if _canonical_sha256(validated) != checkpoint["result_sha256"]:
            raise self._checkpoint_error(
                stage_name,
                checkpoint_path,
                "current validator would normalize or change the cached result",
                source_stage_name=source_stage_name,
            )
        return deepcopy(validated)

    def _invoke(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        if self.generation_runner is None:
            raw = self.runner(payload)
            metadata = None
        else:
            if max_new_tokens is None:
                result = self.generation_runner(payload)
            else:
                result = self.generation_runner(
                    payload, max_new_tokens=max_new_tokens
                )
            if not isinstance(result, LocalGenerationResult):
                raise StagedAnalysisError(
                    "local Qwen metadata runner returned an invalid result"
                )
            raw = result.text
            metadata = result.metadata()
        if not isinstance(raw, str):
            raise StagedAnalysisError("local Qwen runner returned non-text output")
        safe_name = self._safe_stage_name(name)
        (self.raw_output_dir / f"{safe_name}.raw.txt").write_text(raw, encoding="utf-8")
        return raw, metadata

    def _record_generation(
        self,
        name: str,
        *,
        attempt: int,
        metadata: Mapping[str, Any] | None,
        status: str,
        error_reason: str | None = None,
    ) -> None:
        if metadata is None:
            return
        event = {
            "stage_name": name,
            "attempt": attempt,
            "budget": metadata["max_new_tokens"],
            "generated_tokens": metadata["generated_tokens"],
            "hit_token_limit": metadata["hit_token_limit"],
            "eos_detected": metadata["eos_detected"],
            "truncated": metadata["truncated"],
            "status": status,
        }
        if error_reason is not None:
            event["error_reason"] = error_reason
        self.generation_events.append(event)
        sidecar = {**dict(metadata), **event}
        safe_name = self._safe_stage_name(name)
        (self.raw_output_dir / f"{safe_name}.generation.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.generation_event_callback is not None:
            self.generation_event_callback(dict(event))

    def _write_parse_audit(
        self, name: str, audit: Mapping[str, Any]
    ) -> None:
        safe_name = self._safe_stage_name(name)
        (self.raw_output_dir / f"{safe_name}.parse.json").write_text(
            json.dumps(dict(audit), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _parse_document(
        self, name: str, raw: str
    ) -> Mapping[str, Any]:
        try:
            document, audit = parse_plan_content_with_audit(raw)
        except QwenContentParseError as error:
            if error.parse_audit is not None:
                self._write_parse_audit(name, error.parse_audit)
            error.stage_name = name
            error.raw_output_path = (
                self.raw_output_dir
                / f"{self._safe_stage_name(name)}.raw.txt"
            )
            raise
        self._write_parse_audit(name, audit)
        return document

    def _record_schema_validation(
        self,
        name: str,
        *,
        status: str,
        error: Exception | None = None,
    ) -> None:
        """Append stage-schema validation to an existing transport audit."""

        safe_name = self._safe_stage_name(name)
        path = self.raw_output_dir / f"{safe_name}.parse.json"
        if not path.is_file():
            return
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(audit, dict):
            return
        audit["schema_validation_status"] = status
        audit["schema_valid"] = status == "valid"
        if error is not None:
            audit["schema_error"] = str(error)
        else:
            audit.pop("schema_error", None)
        self._write_parse_audit(name, audit)

    @staticmethod
    def _error_reason(error: Exception) -> str:
        if isinstance(error, QwenContentParseError):
            return error.reason
        if isinstance(error, QwenResponseError):
            return "invalid_qwen_response"
        if isinstance(error, StagedAnalysisError):
            return "palette_schema_invalid"
        return "generation_failed"

    def _run(self, name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raw, metadata = self._invoke(
            name, payload, max_new_tokens=self.max_new_tokens
        )
        try:
            document = self._parse_document(name, raw)
        except QwenResponseError as error:
            self._record_generation(
                name,
                attempt=1,
                metadata=metadata,
                status="invalid",
                error_reason=self._error_reason(error),
            )
            raise
        self._record_generation(
            name, attempt=1, metadata=metadata, status="complete"
        )
        return document

    def extract_palette(
        self,
        reference_view: Mapping[str, Any],
        *,
        run_label: str | None = None,
    ) -> dict[str, Any]:
        reference = _view(reference_view, "reference_view", geometry=False)
        stage_name = "01_palette"
        if run_label is not None:
            stage_name += "_" + _SAFE_NAME_RE.sub(
                "_", _nonempty(run_label, "run_label")
            )
        payload = build_palette_payload(self.model, reference)
        reused = self._reuse_validated_checkpoint(
            stage_name=stage_name,
            stage_kind="palette",
            payload=payload,
            validator=lambda candidate: validate_palette(
                candidate,
                allowed_reference_view_ids={reference["id"]},
            ),
        )
        if reused is not None:
            return reused
        budget = self.max_new_tokens
        attempt = 1
        attempt_payload = payload
        schema_retry_used = False
        schema_retry_active = False
        content_retry_used = False
        while True:
            attempt_name = (
                stage_name if attempt == 1 else f"{stage_name}_retry{attempt - 1}"
            )
            raw, metadata = self._invoke(
                attempt_name,
                attempt_payload,
                max_new_tokens=budget,
            )
            schema_error: StagedAnalysisError | None = None
            try:
                document = self._parse_document(attempt_name, raw)
                try:
                    result = validate_palette(
                        document,
                        allowed_reference_view_ids={reference["id"]},
                    )
                except StagedAnalysisError as error:
                    self._record_schema_validation(
                        attempt_name, status="invalid", error=error
                    )
                    schema_error = error
                    raise
            except (QwenResponseError, StagedAnalysisError) as error:
                truncated = bool(metadata and metadata.get("truncated") is True)
                current_budget = (
                    metadata.get("max_new_tokens") if metadata is not None else budget
                )
                can_retry = (
                    truncated
                    and not schema_retry_active
                    and isinstance(current_budget, int)
                    and self.max_new_tokens_ceiling is not None
                    and current_budget < self.max_new_tokens_ceiling
                )
                can_schema_retry = (
                    schema_error is not None
                    and not truncated
                    and not schema_retry_used
                )
                can_content_retry = (
                    isinstance(error, QwenContentParseError)
                    and not truncated
                    and not content_retry_used
                )
                self._record_generation(
                    attempt_name,
                    attempt=attempt,
                    metadata=metadata,
                    status=(
                        "content_retry"
                        if can_content_retry
                        else "schema_retry"
                        if can_schema_retry
                        else "truncated_retry"
                        if can_retry
                        else "truncated_exhausted"
                        if truncated
                        else "invalid"
                    ),
                    error_reason=self._error_reason(error),
                )
                if can_content_retry:
                    attempt_payload = _build_strict_json_retry_payload(
                        payload,
                        rejected_content=raw,
                        parse_error=error,
                        contract_context=(
                            "Preserve source_view_id exactly as "
                            f"{reference['id']!r} and obey the original palette "
                            "schema and visible-evidence constraints."
                        ),
                    )
                    content_retry_used = True
                    attempt += 1
                    continue
                if can_schema_retry:
                    attempt_payload = _build_palette_schema_retry_payload(
                        payload,
                        source_view_id=reference["id"],
                        rejected_content=raw,
                        schema_error=schema_error,
                    )
                    schema_retry_used = True
                    schema_retry_active = True
                    attempt += 1
                    continue
                if can_retry:
                    budget = min(
                        current_budget * 2, self.max_new_tokens_ceiling
                    )
                    attempt += 1
                    continue
                if truncated:
                    raise StagedAnalysisError(
                        "palette generation remained truncated after "
                        f"{attempt} attempt(s); max_new_tokens ceiling="
                        f"{current_budget}"
                    ) from error
                raise
            self._record_schema_validation(attempt_name, status="valid")
            self._record_generation(
                attempt_name,
                attempt=attempt,
                metadata=metadata,
                status="valid",
            )
            self._write_validated_checkpoint(
                stage_name=stage_name,
                stage_kind="palette",
                source_stage_name=attempt_name,
                payload=payload,
                result=result,
            )
            return result

    def map_part_batch(
        self,
        *,
        reference_view: Mapping[str, Any],
        cad_view: Mapping[str, Any],
        part_id_view: Mapping[str, Any],
        batch_sheet_view: Mapping[str, Any],
        palette: Mapping[str, Any],
        target_parts: Sequence[Mapping[str, Any]],
        batch_id: str,
        support_reference_view: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = build_part_palette_payload(
            self.model,
            reference_view=reference_view,
            cad_view=cad_view,
            part_id_view=part_id_view,
            batch_sheet_view=batch_sheet_view,
            palette=palette,
            target_parts=target_parts,
            batch_id=batch_id,
            support_reference_view=support_reference_view,
        )
        target_part_ids = {part["part_id"] for part in target_parts}
        max_pixels = {
            summary["part_id"]: summary.get(
                "evidence_visible_pixels", summary["max_visible_pixels"]
            )
            for summary in (_part_summary(part) for part in target_parts)
        }

        def normalize(
            candidate: Mapping[str, Any], *, quarantine_invalid_rows: bool = False
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            result = normalize_part_palette_batch(
                candidate,
                target_part_ids=target_part_ids,
                palette=palette,
                expected_batch_id=batch_id,
                visible_pixels_by_part=max_pixels,
                quarantine_invalid_rows=quarantine_invalid_rows,
                audit_events=events,
            )
            return result, events

        def record_normalizations(
            events: Sequence[Mapping[str, Any]],
            *,
            action: str,
            initial_error: str | None = None,
            retry_error: str | None = None,
        ) -> None:
            if not events and action == "normalized_model_rows":
                return
            normalized_parts = sorted(
                {
                    event["part_id"]
                    for event in events
                    if event.get("action") == "normalized"
                    and isinstance(event.get("part_id"), str)
                }
            )
            quarantined_parts = sorted(
                {
                    event["part_id"]
                    for event in events
                    if event.get("action") == "quarantined"
                    and isinstance(event.get("part_id"), str)
                }
            )
            record: dict[str, Any] = {
                "batch_id": batch_id,
                "action": action,
                "normalized_parts": normalized_parts,
                "quarantined_parts": quarantined_parts,
                "normalizations": [dict(event) for event in events],
            }
            if initial_error is not None:
                record["initial_error"] = initial_error
            if retry_error is not None:
                record["retry_error"] = retry_error
            self.repair_events.append(record)

        stage_name = f"02_map_{batch_id}"

        def validate_cached_mapping(
            candidate: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            validated, _events = normalize(candidate)
            return validated

        reused = self._reuse_validated_checkpoint(
            stage_name=stage_name,
            stage_kind="mapping",
            payload=payload,
            validator=validate_cached_mapping,
        )
        if reused is not None:
            return reused

        document: Mapping[str, Any] | None = None
        try:
            document = self._run(stage_name, payload)
            result, normalization_events = normalize(document)
            record_normalizations(normalization_events, action="normalized_model_rows")
            self._record_schema_validation(stage_name, status="valid")
            self._write_validated_checkpoint(
                stage_name=stage_name,
                stage_kind="mapping",
                source_stage_name=stage_name,
                payload=payload,
                result=result,
            )
            return result
        except (QwenResponseError, StagedAnalysisError) as first_error:
            if isinstance(first_error, StagedAnalysisError):
                self._record_schema_validation(
                    stage_name, status="invalid", error=first_error
                )
            # A malformed transport document or genuinely invalid semantic
            # citation receives one bounded causal retry. Mechanical
            # status/evidence/threshold disagreements have already been
            # normalized above and never reach this path.
            repair_payload = deepcopy(payload)
            valid_box_indices = {
                group["group_id"]: list(range(len(group["boxes"])))
                for group in validate_palette(palette)["groups"]
            }
            raw_path = (
                self.raw_output_dir
                / f"{self._safe_stage_name(stage_name)}.raw.txt"
            )
            if document is not None:
                rejected_output = json.dumps(document, ensure_ascii=False)
            elif raw_path.is_file():
                rejected_output = raw_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            else:
                rejected_output = "<unavailable>"
            if isinstance(first_error, QwenContentParseError):
                repair_payload = _build_strict_json_retry_payload(
                    payload,
                    rejected_content=rejected_output,
                    parse_error=first_error,
                    contract_context=(
                        "Return the original compact exact-cover mapping "
                        f"contract for batch_id {batch_id!r}. Valid box indices "
                        "by group are "
                        + json.dumps(valid_box_indices, ensure_ascii=False)
                        + "."
                    ),
                )
            else:
                repair_payload["messages"][1]["content"].append(
                    {
                        "type": "text",
                        "text": "\n".join(
                            (
                                "VALIDATOR REPAIR: your previous JSON was rejected.",
                                "Validation error: " + str(first_error),
                                "Previous rejected JSON: " + rejected_output,
                                "Regenerate the complete compact exact-cover JSON object, not an explanation.",
                                "Do not invent or clamp evidence indices. Valid box indices by group: "
                                + json.dumps(valid_box_indices, ensure_ascii=False),
                                "If a part cannot cite a valid box for its chosen group, use null group_id and evidence_box_index, confidence below 0.60, and an allowed unknown reason_code.",
                            )
                        ),
                    }
                )
            repair_name = f"{stage_name}_retry1"
            repaired = self._run(repair_name, repair_payload)
            try:
                result, normalization_events = normalize(repaired)
            except StagedAnalysisError as retry_error:
                self._record_schema_validation(
                    repair_name, status="invalid", error=retry_error
                )
                # Preserve every independently valid row and replace only the
                # remaining malformed/missing/duplicate rows with explicit
                # unknowns.  This fallback never fabricates a palette citation.
                try:
                    result, normalization_events = normalize(
                        repaired, quarantine_invalid_rows=True
                    )
                except StagedAnalysisError as quarantine_error:
                    quarantine_error.stage_name = repair_name
                    quarantine_error.raw_output_path = (
                        self.raw_output_dir
                        / f"{repair_name}.raw.txt"
                    )
                    raise
                self._record_schema_validation(repair_name, status="valid")
                record_normalizations(
                    normalization_events,
                    action="quarantined_invalid_rows",
                    initial_error=str(first_error),
                    retry_error=str(retry_error),
                )
                self._write_validated_checkpoint(
                    stage_name=stage_name,
                    stage_kind="mapping",
                    source_stage_name=repair_name,
                    payload=payload,
                    result=result,
                )
                return result
            self._record_schema_validation(repair_name, status="valid")
            record_normalizations(
                normalization_events,
                action="accepted_validated_retry",
                initial_error=str(first_error),
            )
            self._write_validated_checkpoint(
                stage_name=stage_name,
                stage_kind="mapping",
                source_stage_name=repair_name,
                payload=payload,
                result=result,
            )
            return result

    def choose_group_material(
        self,
        *,
        reference_crop_view: Mapping[str, Any],
        group: Mapping[str, Any],
        candidate_materials: Sequence[Mapping[str, Any]],
        run_label: str,
    ) -> dict[str, Any]:
        payload = build_group_material_payload(
            self.model,
            reference_crop_view=reference_crop_view,
            group=group,
            candidate_materials=candidate_materials,
        )
        stage_name = f"03_material_{group['group_id']}_{run_label}"
        allowed_material_ids = [
            candidate["material_id"] for candidate in candidate_materials
        ]

        def validate(
            document: Mapping[str, Any],
            *,
            name: str,
        ) -> dict[str, Any]:
            try:
                result = validate_group_material_choice(
                    document,
                    group_id=group["group_id"],
                    allowed_material_ids=allowed_material_ids,
                )
            except StagedAnalysisError as error:
                self._record_schema_validation(
                    name, status="invalid", error=error
                )
                error.stage_name = name
                error.raw_output_path = (
                    self.raw_output_dir
                    / f"{self._safe_stage_name(name)}.raw.txt"
                )
                raise
            self._record_schema_validation(name, status="valid")
            return result

        reused = self._reuse_validated_checkpoint(
            stage_name=stage_name,
            stage_kind="material",
            payload=payload,
            validator=lambda candidate: validate_group_material_choice(
                candidate,
                group_id=group["group_id"],
                allowed_material_ids=allowed_material_ids,
            ),
        )
        if reused is not None:
            return reused

        first_document: Mapping[str, Any] | None = None
        try:
            first_document = self._run(stage_name, payload)
            result = validate(first_document, name=stage_name)
            self._write_validated_checkpoint(
                stage_name=stage_name,
                stage_kind="material",
                source_stage_name=stage_name,
                payload=payload,
                result=result,
            )
            return result
        except (QwenResponseError, StagedAnalysisError) as first_error:
            # The retry is causal rather than an identical rerun: it includes
            # the rejected output, exact validator error, expected group and
            # the complete candidate allowlist.  It still cannot introduce a
            # material outside the deterministic candidate set.
            repair_payload = deepcopy(payload)
            raw_path = (
                self.raw_output_dir
                / f"{self._safe_stage_name(stage_name)}.raw.txt"
            )
            if first_document is not None:
                rejected_output = json.dumps(
                    first_document, ensure_ascii=False
                )
            elif raw_path.is_file():
                rejected_output = raw_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            else:
                rejected_output = "<unavailable>"
            repair_payload["messages"][1]["content"].append(
                {
                    "type": "text",
                    "text": "\n".join(
                        (
                            "VALIDATOR REPAIR: the previous material-choice "
                            "response was rejected.",
                            "Validation error: " + str(first_error),
                            "Previous rejected response: "
                            + json.dumps(rejected_output, ensure_ascii=False),
                            "Return one complete strict JSON object only, with "
                            "no Markdown or prose.",
                            "Required group_id: "
                            + json.dumps(str(group["group_id"])),
                            "Allowed material_id values: "
                            + json.dumps(
                                allowed_material_ids, ensure_ascii=False
                            ),
                            "Do not invent, shorten, normalize, or modify a "
                            "material_id.",
                        )
                    ),
                }
            )
            repair_name = f"{stage_name}_retry1"
            try:
                repaired_document = self._run(repair_name, repair_payload)
                result = validate(repaired_document, name=repair_name)
            except (QwenResponseError, StagedAnalysisError) as retry_error:
                retry_error.stage_name = repair_name
                retry_error.raw_output_path = (
                    self.raw_output_dir
                    / f"{self._safe_stage_name(repair_name)}.raw.txt"
                )
                raise
            self.repair_events.append(
                {
                    "group_id": str(group["group_id"]),
                    "run_label": run_label,
                    "action": "accepted_validated_material_choice_retry",
                    "initial_error": str(first_error),
                    "stage_name": stage_name,
                    "retry_stage_name": repair_name,
                }
            )
            self._write_validated_checkpoint(
                stage_name=stage_name,
                stage_kind="material",
                source_stage_name=repair_name,
                payload=payload,
                result=result,
            )
            return result


__all__ = [
    "LocalStagedQwenClient",
    "MATERIAL_CHOICE_SCHEMA_VERSION",
    "build_group_material_payload",
    "build_palette_payload",
    "build_part_palette_payload",
    "validate_group_material_choice",
]
