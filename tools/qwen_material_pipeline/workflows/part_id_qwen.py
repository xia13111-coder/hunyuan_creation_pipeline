#!/usr/bin/env python3
"""Qwen visual reranking for independent CAD Part-ID material candidates."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from qwen_material_pipeline.evidence.part_id_projection import SCHEMA_VERSION
from qwen_material_pipeline.materials.perceptual_color import (
    perceptual_similarity,
    srgb_delta_e,
)
from qwen_material_pipeline.materials.tuning import (
    tuning_profile_for_material,
)
from qwen_material_pipeline.qwen.client import (
    load_image_url,
    parse_plan_content_with_audit,
)
from qwen_material_pipeline.qwen.local_vl import TransformersQwen3VLRunner


OUTPUT_SCHEMA_VERSION = "qwen-part-id-material-rerank/v1"
BATCH_SCHEMA_VERSION = "qwen-part-id-material-rerank-batch/v2"
MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION = (
    "qwen-part-id-material-rerank-batch/v3"
)
MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION = (
    "qwen-part-id-material-family-prediction-batch/v2"
)
SELECTIVE_REGRESSION_SCHEMA_VERSION = "qwen-part-id-selective-visual-regression/v1"
DEFAULT_MAXIMUM_LOCAL_SCORE_REGRESSION = 0.02
COLOR_CRITICAL_MINIMUM_TRUSTED_PIXELS = 512
COLOR_CRITICAL_MINIMUM_SATURATION = 0.20
MINIMUM_PART_ID_APPEARANCE_PIXELS = 6
COLOR_CRITICAL_MAXIMUM_DELTA_E = 25.0
COLOR_CRITICAL_MAXIMUM_HUE_DISTANCE_DEGREES = 30.0
COLOR_CRITICAL_MINIMUM_CANDIDATE_SATURATION = 0.10
MINIMUM_MATERIAL_FAMILY_CONFIDENCE = 0.60
MINIMUM_MATERIAL_SPECIES_CONFIDENCE = 0.75
MINIMUM_EXACT_TREATMENT_CONFIDENCE = 0.85
MINIMUM_COMPONENT_EXACT_PRESET_CONFIDENCE = 0.80
EXACT_LIBRARY_PRESET_MAXIMUM_DELTA_E = 25.0
MINIMUM_COMPONENT_REFINEMENT_PIXELS = 256
MINIMUM_COMPONENT_REFINEMENT_INLIER_FRACTION = 0.75
MAXIMUM_COMPONENT_REFINEMENT_DELTA_E = 18.0
MINIMUM_COMPONENT_REFINEMENT_SPATIAL_SUPPORT = 0.60
MAXIMUM_DIRECT_EXACT_PBR_ERROR = 0.12
MINIMUM_DIRECT_EXACT_PBR_MARGIN = 0.08
MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH = 4
MATERIAL_FINISH_OPTIONS = frozenset(
    {
        "unknown",
        "matte",
        "satin",
        "glossy",
        "smooth",
        "rough",
        "brushed",
        "polished",
        "cast",
        "anodized",
        "rusty",
        "frosted",
    }
)
MATERIAL_SUBSTRATE_OPTIONS = frozenset(
    {
        "unknown",
        "metal",
        "polymer",
        "elastomer",
        "glass",
        "wood",
        "textile",
        "leather",
        "ceramic",
        "mineral",
        "stone",
        "paper",
        "liquid",
    }
)
MATERIAL_TREATMENT_OPTIONS = frozenset(
    {
        "unknown",
        "bare",
        "paint",
        "anodized",
        "plated",
        "oxidized",
        "conversion_coating",
        "emissive",
    }
)
MATERIAL_OPTICAL_OPTIONS = frozenset(
    {"unknown", "opaque", "transparent", "translucent", "emissive"}
)


# ``family=metal`` is not a material identity: it still permits copper,
# chrome, aluminium and steel to replace one another.  Keep a deliberately
# small, physical species ontology that can be inferred from NVIDIA's stable
# authored identifier or supplied explicitly by a future catalog schema.  The
# fallback species (metal/plastic/wood/...) remains useful when the library has
# no more specific identity.
_MATERIAL_SPECIES_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aluminum", ("aluminum", "aluminium")),
    ("copper", ("copper",)),
    ("brass", ("brass",)),
    ("bronze", ("bronze",)),
    ("steel", ("steel", "stainless")),
    ("iron", ("iron",)),
    ("chromium", ("chrome", "chromium")),
    ("gold", ("gold",)),
    ("silver", ("silver",)),
    ("rubber", ("rubber", "elastomer", "silicone", "neoprene")),
    ("acrylic", ("acrylic",)),
    ("abs", ("plastic_abs", "abs_plastic")),
    ("vinyl", ("vinyl",)),
    ("glass", ("glass", "mirror")),
    ("paint", ("paint",)),
    ("ceramic", ("ceramic", "porcelain")),
    ("concrete", ("concrete",)),
    ("stone", ("stone", "granite", "marble", "slate")),
    ("leather", ("leather",)),
    ("textile", ("cloth", "linen", "fabric", "carpet")),
    ("paper", ("paper", "cardboard")),
    ("wood", ("wood", "veneer", "walnut", "oak", "birch", "ash", "bamboo", "cherry", "mahogany", "timber", "plywood", "cork")),
    ("plastic", ("plastic", "polymer", "polycarbonate", "polypropylene", "polyethylene")),
)

_GENERIC_MATERIAL_SPECIES = frozenset(
    {
        "unknown",
        "metal",
        "plastic",
        "rubber",
        "glass",
        "paint",
        "wood",
        "textile",
        "leather",
        "ceramic",
        "concrete",
        "stone",
        "paper",
        "liquid",
    }
)


class PartIdQwenError(ValueError):
    """Raised when Qwen cannot produce a bounded Part-ID decision."""


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartIdQwenError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PartIdQwenError(f"{label} must be a JSON object")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_grayscale_identity_crop(source: Path, output: Path) -> Path:
    try:
        resolved = source.expanduser().resolve(strict=True)
        with Image.open(resolved) as opened:
            grayscale = ImageOps.grayscale(opened).convert("RGB")
    except (OSError, ValueError) as exc:
        raise PartIdQwenError(
            f"unable to prepare grayscale material-identity evidence: {source}: {exc}"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    grayscale.save(output)
    return output


def _color_free_identity_descriptor(value: Any) -> Any:
    """Remove chromatic fields before physical identity/species prediction."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            lowered = key.casefold()
            if any(
                token in lowered
                for token in ("color", "colour", "rgb", "albedo", "chromatic")
            ):
                continue
            output[key] = _color_free_identity_descriptor(raw_value)
        return output
    if isinstance(value, list):
        return [_color_free_identity_descriptor(item) for item in value]
    return value


def _catalog_by_id(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    materials = catalog.get("materials")
    if not isinstance(materials, list) or not materials:
        raise PartIdQwenError("material catalog has no materials")
    result: dict[str, dict[str, Any]] = {}
    for raw in materials:
        material_id = raw.get("material_id") if isinstance(raw, Mapping) else None
        if not isinstance(material_id, str) or not material_id or material_id in result:
            raise PartIdQwenError("material catalog contains invalid duplicate IDs")
        result[material_id] = dict(raw)
    return result


def _catalog_material_species(
    material_id: str,
    record: Mapping[str, Any],
) -> str:
    """Return a stable physical species for one authored catalog material.

    Catalog schema v1 has only a broad family and surface semantics.  Prefer a
    future explicit ``material_species`` value, otherwise infer the species
    from NVIDIA's authored sub-identifier/path.  Colour tokens are never used
    as species, so Aluminum_Anodized_Black and Aluminum_Anodized_Blue both map
    to ``aluminum``.
    """

    semantics = record.get("surface_semantics")
    explicit = (
        semantics.get("material_species")
        if isinstance(semantics, Mapping)
        else None
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().casefold()
    identifier = record.get("sub_identifier")
    if not isinstance(identifier, str) or not identifier:
        identifier = material_id.rsplit("#", 1)[-1]
    normalized = identifier.casefold().replace("-", "_").replace(" ", "_")
    padded = f"_{normalized}_"
    for species, aliases in _MATERIAL_SPECIES_PATTERNS:
        if any(
            f"_{alias}_" in padded
            or padded.startswith(f"_{alias}_")
            or normalized.startswith(f"{alias}_")
            or normalized.endswith(f"_{alias}")
            for alias in aliases
        ):
            return species
    family = record.get("family")
    if isinstance(family, str) and family.strip():
        return family.strip().casefold()
    return "unknown"


def _catalog_family_options(
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    families = sorted(
        {
            str(record.get("family", "")).strip().casefold()
            for record in catalog_by_id.values()
            if isinstance(record.get("family"), str)
            and str(record.get("family", "")).strip()
        }
    )
    if len(families) < 2:
        raise PartIdQwenError("material catalog has too few material families")
    return families


def _catalog_identity_options(
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    substrates: set[str] = set()
    treatments: set[str] = set()
    optical_behaviors: set[str] = set()
    for material_id, record in catalog_by_id.items():
        semantics = record.get("surface_semantics")
        if not isinstance(semantics, Mapping):
            raise PartIdQwenError(
                f"material catalog record {material_id} has no surface semantics"
            )
        raw_substrates = semantics.get("compatible_substrates")
        treatment = semantics.get("surface_treatment")
        optical = semantics.get("optical_behavior")
        if (
            not isinstance(raw_substrates, list)
            or not raw_substrates
            or not all(isinstance(value, str) for value in raw_substrates)
            or not isinstance(treatment, str)
            or not isinstance(optical, str)
        ):
            raise PartIdQwenError(
                f"material catalog record {material_id} has invalid surface semantics"
            )
        substrates.update(value.casefold() for value in raw_substrates)
        treatments.add(treatment.casefold())
        optical_behaviors.add(optical.casefold())
    species = {
        _catalog_material_species(material_id, record)
        for material_id, record in catalog_by_id.items()
    }
    if not substrates or not treatments or not optical_behaviors or not species:
        raise PartIdQwenError("material catalog has no usable identity semantics")
    return {
        "physical_substrates": sorted(substrates | {"unknown"}),
        "material_species": sorted(species | {"unknown"}),
        "surface_treatments": sorted(treatments | {"unknown"}),
        "optical_behaviors": sorted(optical_behaviors | {"unknown"}),
    }


def _catalog_family_for_identity(
    *,
    physical_substrate: str,
    surface_treatment: str,
    optical_behavior: str,
) -> str:
    if surface_treatment == "paint":
        return "paint"
    if surface_treatment == "emissive" or optical_behavior == "emissive":
        return "emissives"
    return {
        "metal": "metal",
        "polymer": "plastic",
        "elastomer": "rubber",
        "glass": "glass",
        "wood": "wood",
        "textile": "textiles",
        "leather": "leather",
        "ceramic": "ceramic",
        "mineral": "stone",
        "stone": "stone",
        "paper": "paper",
        "liquid": "liquid",
    }.get(physical_substrate, "unknown")


def _validate_material_prediction_batch(
    document: Mapping[str, Any],
    *,
    expected: Sequence[Mapping[str, Any]],
    identity_options: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    if set(document) != {"schema_version", "predictions"}:
        raise PartIdQwenError(
            "Qwen material-identity prediction returned unexpected fields"
        )
    if (
        document.get("schema_version")
        != MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION
    ):
        raise PartIdQwenError("Qwen material-identity prediction uses the wrong schema")
    rows = document.get("predictions")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise PartIdQwenError("Qwen material-identity prediction count is invalid")
    expected_ids = {str(row["part_id"]) for row in expected}
    allowed_substrates = set(identity_options["physical_substrates"])
    allowed_species = set(identity_options["material_species"])
    allowed_treatments = set(identity_options["surface_treatments"])
    allowed_optical = set(identity_options["optical_behaviors"])
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "part_id",
            "physical_substrate",
            "material_species",
            "surface_treatment",
            "optical_behavior",
            "surface_finish",
            "substrate_confidence",
            "species_confidence",
            "treatment_confidence",
        }:
            raise PartIdQwenError(
                f"Qwen material-identity prediction {index} has invalid fields"
            )
        part_id = raw.get("part_id")
        substrate = raw.get("physical_substrate")
        species = raw.get("material_species")
        treatment = raw.get("surface_treatment")
        optical = raw.get("optical_behavior")
        finish = raw.get("surface_finish")
        substrate_confidence = raw.get("substrate_confidence")
        species_confidence = raw.get("species_confidence")
        treatment_confidence = raw.get("treatment_confidence")
        if (
            not isinstance(part_id, str)
            or part_id not in expected_ids
            or part_id in output
        ):
            raise PartIdQwenError(
                f"Qwen material-identity prediction {index} cites an invalid part"
            )
        if not isinstance(substrate, str) or substrate not in allowed_substrates:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported physical substrate for {part_id}"
            )
        if not isinstance(species, str) or species not in allowed_species:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported material species for {part_id}"
            )
        if not isinstance(treatment, str) or treatment not in allowed_treatments:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported surface treatment for {part_id}"
            )
        if not isinstance(optical, str) or optical not in allowed_optical:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported optical behavior for {part_id}"
            )
        if not isinstance(finish, str) or finish not in MATERIAL_FINISH_OPTIONS:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported surface finish for {part_id}"
            )
        for label, value in (
            ("substrate", substrate_confidence),
            ("species", species_confidence),
            ("treatment", treatment_confidence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise PartIdQwenError(
                    f"Qwen predicted an invalid {label} confidence for {part_id}"
                )
        substrate_is_known = (
            substrate != "unknown"
            and optical != "unknown"
            and float(substrate_confidence) >= MINIMUM_MATERIAL_FAMILY_CONFIDENCE
        )
        exact_treatment_is_known = (
            treatment != "unknown"
            and float(treatment_confidence)
            >= MINIMUM_EXACT_TREATMENT_CONFIDENCE
        )
        exact_species_is_known = (
            species != "unknown"
            and float(species_confidence)
            >= MINIMUM_MATERIAL_SPECIES_CONFIDENCE
        )
        identity_resolution = (
            "exact_material"
            if (
                substrate_is_known
                and exact_species_is_known
                and exact_treatment_is_known
            )
            else "corresponding_material"
            if substrate_is_known
            else "insufficient_evidence"
        )
        confidence = (
            min(
                float(substrate_confidence),
                float(species_confidence),
                float(treatment_confidence),
            )
            if identity_resolution == "exact_material"
            else float(substrate_confidence)
        )
        output[part_id] = {
            "part_id": part_id,
            "catalog_family": _catalog_family_for_identity(
                physical_substrate=substrate,
                surface_treatment=treatment,
                optical_behavior=optical,
            ),
            "physical_substrate": substrate,
            "material_species": species,
            "surface_treatment": treatment,
            "optical_behavior": optical,
            "surface_finish": finish,
            "substrate_confidence": float(substrate_confidence),
            "species_confidence": float(species_confidence),
            "treatment_confidence": float(treatment_confidence),
            "confidence": confidence,
            "identity_resolution": identity_resolution,
            "status": (
                "APPLYABLE"
                if substrate_is_known
                else "INSUFFICIENT_EVIDENCE"
            ),
        }
    if set(output) != expected_ids:
        raise PartIdQwenError(
            "Qwen material-identity prediction does not exactly cover its input"
        )
    return [output[str(row["part_id"])] for row in expected]


def _material_prediction_payload(
    *,
    model: str,
    batch: Sequence[Mapping[str, Any]],
    identity_options: Mapping[str, Sequence[str]],
    retry_detail: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in batch:
        views = item.get("views")
        if not isinstance(views, list) or not views:
            raise PartIdQwenError(
                f"material prediction target {item['part_id']} has no views"
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"MATERIAL IDENTITY TARGET {item['part_id']}. All following "
                    "images are grayscale, neutral-background crops of the same "
                    "physical surface from trusted views or linked CAD parts. "
                    "Colour has deliberately been removed. Descriptor: "
                    + json.dumps(
                        _color_free_identity_descriptor(
                            item.get("descriptor", {})
                        ),
                        ensure_ascii=False,
                    )
                ),
            }
        )
        for view in views:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"VIEW {view['view_id']}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": load_image_url(view["crop"])},
                    },
                ]
            )
    output_shape = {
        "schema_version": MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION,
        "predictions": [
            {
                "part_id": str(item["part_id"]),
                "physical_substrate": "one allowed substrate or unknown",
                "material_species": "one allowed species or unknown",
                "surface_treatment": "one allowed treatment or unknown",
                "optical_behavior": "one allowed optical behavior or unknown",
                "surface_finish": "one allowed finish",
                "substrate_confidence": 0.75,
                "species_confidence": 0.55,
                "treatment_confidence": 0.55,
            }
            for item in batch
        ],
    }
    prompt = "\n".join(
        [
            "Predict physical material identity before seeing any material candidate.",
            "Colour is unavailable and must not influence the decision. Identify the substrate, material species, surface treatment, optical behavior, and finish from geometry, texture, reflectance and manufacturing cues.",
            "Examples: a powder-coated enclosure is substrate=metal, species=paint, treatment=paint, optical=opaque; exposed stainless steel is metal/steel/bare/opaque; anodized aluminium is metal/aluminum/anodized/opaque; hard ABS is polymer/abs/bare/opaque; rubber is elastomer/rubber/bare/opaque; clear glazing is glass/glass/bare/transparent.",
            "Score substrate_confidence, species_confidence and treatment_confidence separately. A visible rigid polymer or metal substrate can be high-confidence even when copper versus steel, or paint versus anodized versus bare, is uncertain. Use material_species=unknown or treatment=unknown with a low confidence instead of guessing. Exact species and treatment require exceptional evidence; the program will otherwise use bounded candidate evidence to resolve a corresponding material. Never guess from background, CAD names, or colour.",
            "Allowed physical_substrate values: "
            + json.dumps(identity_options["physical_substrates"], ensure_ascii=False),
            "Allowed material_species values: "
            + json.dumps(identity_options["material_species"], ensure_ascii=False),
            "Allowed surface_treatment values: "
            + json.dumps(identity_options["surface_treatments"], ensure_ascii=False),
            "Allowed optical_behavior values: "
            + json.dumps(identity_options["optical_behaviors"], ensure_ascii=False),
            "Allowed surface_finish values: "
            + json.dumps(sorted(MATERIAL_FINISH_OPTIONS), ensure_ascii=False),
            "Return exactly one strict JSON object with no Markdown or prose and this exact shape: "
            + json.dumps(output_shape, ensure_ascii=False),
            *(
                [
                    "The previous response was rejected. Correct only this contract error: "
                    + retry_detail
                ]
                if retry_detail
                else []
            ),
        ]
    )
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative physical material classifier. "
                    "Classify identity before colour and obey the exact JSON contract."
                ),
            },
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
        "enable_thinking": False,
    }


def _predict_material_families(
    *,
    items: Sequence[Mapping[str, Any]],
    runner: Any,
    model: str,
    identity_options: Mapping[str, Sequence[str]],
    batch_size: int,
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    batches: list[list[Mapping[str, Any]]] = []
    pending: list[Mapping[str, Any]] = []
    pending_image_count = 0
    for item in items:
        views = item.get("views")
        if not isinstance(views, list) or not views:
            raise PartIdQwenError(
                f"material prediction target {item.get('part_id')} has no views"
            )
        image_count = len(views)
        if image_count > MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH:
            raise PartIdQwenError(
                f"material prediction target {item.get('part_id')} exceeds the "
                "bounded multi-view image budget"
            )
        if pending and (
            len(pending) >= batch_size
            or pending_image_count + image_count
            > MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH
        ):
            batches.append(pending)
            pending = []
            pending_image_count = 0
        pending.append(item)
        pending_image_count += image_count
    if pending:
        batches.append(pending)
    for batch_index, batch in enumerate(batches, start=1):
        final_error: Exception | None = None
        for attempt in range(1, 3):
            generated = runner.generate_with_metadata(
                _material_prediction_payload(
                    model=model,
                    batch=batch,
                    identity_options=identity_options,
                    retry_detail=str(final_error) if final_error is not None else None,
                )
            )
            raw_path = (
                raw_dir / f"material_prediction_{batch_index:03d}_attempt_{attempt}.txt"
            )
            raw_path.write_text(generated.text, encoding="utf-8")
            try:
                document, parse_audit = parse_plan_content_with_audit(generated.text)
                validated = _validate_material_prediction_batch(
                    document,
                    expected=batch,
                    identity_options=identity_options,
                )
            except Exception as exc:
                final_error = exc
                _write(
                    raw_dir
                    / f"material_prediction_{batch_index:03d}_attempt_{attempt}.parse.json",
                    {
                        "status": "invalid",
                        "error": str(exc),
                        "generation": generated.metadata(),
                    },
                )
                continue
            predictions.extend(validated)
            audit_path = _write(
                raw_dir
                / f"material_prediction_{batch_index:03d}_attempt_{attempt}.parse.json",
                {
                    **parse_audit,
                    "status": "valid",
                    "generation": generated.metadata(),
                },
            )
            audits.append(
                {
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "part_ids": [str(item["part_id"]) for item in batch],
                    "parse_audit": str(audit_path),
                }
            )
            print(
                f"[PART-ID MATERIAL PREDICTION] {batch_index}/{len(batches)} "
                f"parts={','.join(str(item['part_id']) for item in batch)}",
                flush=True,
            )
            break
        else:
            raise PartIdQwenError(
                "Qwen material-identity prediction batch "
                f"{batch_index} failed twice: {final_error}"
            )
    return predictions, audits


_COLOR_IDENTITY_TOKENS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "charcoal",
        "cream",
        "cyan",
        "forest",
        "gold",
        "gray",
        "green",
        "grey",
        "olive",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "tan",
        "white",
        "yellow",
    }
)


def _catalog_identity_semantics(
    material_id: str,
    record: Mapping[str, Any],
) -> tuple[set[str], str, str, str, str]:
    semantics = record.get("surface_semantics")
    if not isinstance(semantics, Mapping):
        raise PartIdQwenError(
            f"material catalog record {material_id} has no surface semantics"
        )
    substrates = semantics.get("compatible_substrates")
    treatment = semantics.get("surface_treatment")
    optical = semantics.get("optical_behavior")
    semantic_finish = semantics.get("finish", "unknown")
    confidence = semantics.get("confidence")
    if (
        not isinstance(substrates, list)
        or not substrates
        or not all(isinstance(value, str) and value for value in substrates)
        or not isinstance(treatment, str)
        or not treatment
        or not isinstance(optical, str)
        or not optical
        or not isinstance(semantic_finish, str)
        or not isinstance(confidence, str)
    ):
        raise PartIdQwenError(
            f"material catalog record {material_id} has invalid surface semantics"
        )
    return (
        {value.casefold() for value in substrates},
        treatment.casefold(),
        optical.casefold(),
        semantic_finish.casefold(),
        confidence.casefold(),
    )


def _colorless_material_identity_key(
    material_id: str,
    *,
    treatment: str,
    optical: str,
    finish: str,
) -> tuple[str, str, str, str]:
    name = material_id.rsplit("#", 1)[-1].casefold()
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    identity_tokens = [
        token
        for token in tokens
        if token not in _COLOR_IDENTITY_TOKENS and token != "finish"
    ]
    return ("_".join(identity_tokens), treatment, optical, finish)


def _specific_library_preset(material_id: str) -> bool:
    name = material_id.rsplit("#", 1)[-1].casefold()
    tokens = {
        token for token in name.replace("-", "_").split("_") if token
    }
    return bool(tokens & _COLOR_IDENTITY_TOKENS)


def _annotate_library_preset_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    catalog_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep exact catalog presets while identifying their generic fallback.

    Stage 1 previously removed authored colour variants before the actual MDL
    comparison.  That made an exact preset such as Aluminum_Anodized_Black
    impossible to select even when it existed in the library.  Every preset is
    now retained.  The generic identity is metadata used only when the model
    reports CORRESPONDING_MATERIAL.
    """

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        material_id = str(row["material_id"])
        (
            _substrates,
            treatment,
            optical,
            semantic_finish,
            _confidence,
        ) = _catalog_identity_semantics(material_id, catalog_by_id[material_id])
        key = _colorless_material_identity_key(
            material_id,
            treatment=treatment,
            optical=optical,
            finish=semantic_finish,
        )
        groups.setdefault(key, []).append(row)
    annotated: list[dict[str, Any]] = []
    for key, variants in groups.items():
        generic_candidates = [
            row
            for row in variants
            if not _specific_library_preset(str(row["material_id"]))
        ]
        generic = (
            min(
                generic_candidates,
                key=lambda row: (
                    len(str(row["material_id"]).rsplit("#", 1)[-1]),
                    int(row.get("rank", 1_000_000)),
                    str(row["material_id"]),
                ),
            )
            if generic_candidates
            else None
        )
        generic_material_id = (
            str(generic["material_id"]) if generic is not None else None
        )
        for row in variants:
            material_id = str(row["material_id"])
            row["colorless_identity_key"] = list(key)
            row["generic_identity_material_id"] = generic_material_id
            row["specific_library_preset"] = (
                generic_material_id is None
                or material_id != generic_material_id
            )
            species = str(
                row.get(
                    "material_species",
                    _catalog_material_species(
                        material_id,
                        catalog_by_id[material_id],
                    ),
                )
            )
            row["material_species"] = species
            row["exact_authored_preset_candidate"] = bool(
                row["specific_library_preset"]
                or species not in _GENERIC_MATERIAL_SPECIES
            )
            annotated.append(row)
    return annotated


def _physical_pbr_evidence(
    descriptor: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Aggregate colour-free per-surface MVInverse evidence.

    Appearance-component predictions carry the member descriptors one level
    below the component descriptor.  Keeping this reader deliberately narrow
    prevents albedo/RGB fields from leaking into the identity stage.
    """

    if not isinstance(descriptor, Mapping):
        return {}
    descriptors: list[Mapping[str, Any]] = [descriptor]
    members = descriptor.get("member_descriptors")
    if isinstance(members, Mapping):
        descriptors.extend(
            value for value in members.values() if isinstance(value, Mapping)
        )
    result: dict[str, float] = {}
    for source_key, output_key in (
        ("roughness_hint", "roughness"),
        ("metallicity_hint", "metallic"),
    ):
        values = sorted(
            float(value)
            for row in descriptors
            for value in [row.get(source_key)]
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )
        if values:
            middle = len(values) // 2
            result[output_key] = (
                values[middle]
                if len(values) % 2
                else (values[middle - 1] + values[middle]) / 2.0
            )
    return result


def _library_pbr_fingerprint(
    profile: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return only authored scalar channels not overridden by a texture."""

    authored = profile.get("authored_mdl") if isinstance(profile, Mapping) else None
    if not isinstance(authored, Mapping):
        return {}
    orm_texture = (
        authored.get("ORM_texture")
        if authored.get("enable_ORM_texture") is True
        else None
    )
    result: dict[str, float] = {}
    for output_key, constant_key, texture_key in (
        (
            "roughness",
            "reflection_roughness_constant",
            "reflectionroughness_texture",
        ),
        ("metallic", "metallic_constant", "metallic_texture"),
    ):
        value = authored.get(constant_key)
        if (
            isinstance(orm_texture, str)
            and orm_texture
            or isinstance(authored.get(texture_key), str)
            and authored.get(texture_key)
        ):
            continue
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        ):
            result[output_key] = float(value)
    return result


def _rank_identity_candidates_with_pbr(
    ranking: Sequence[Mapping[str, Any]],
    *,
    descriptor: Mapping[str, Any] | None,
    profiles_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank the full compatible catalog using colour-free physical evidence."""

    observed = _physical_pbr_evidence(descriptor)
    ranked: list[dict[str, Any]] = []
    for raw in ranking:
        row = dict(raw)
        material_id = row.get("material_id")
        fingerprint = _library_pbr_fingerprint(
            profiles_by_id.get(material_id)
            if isinstance(material_id, str)
            else None
        )
        terms = {
            key: abs(observed[key] - fingerprint[key])
            for key in sorted(set(observed) & set(fingerprint))
        }
        mean_error = sum(terms.values()) / len(terms) if terms else None
        row["physical_pbr_evidence"] = dict(observed)
        row["library_pbr_fingerprint"] = fingerprint
        row["physical_pbr_term_errors"] = terms
        row["physical_pbr_mean_error"] = mean_error
        row["physical_pbr_similarity"] = (
            1.0 - mean_error if mean_error is not None else None
        )
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            0 if row.get("predicted_finish_match") is True else 1,
            (
                float(row["physical_pbr_mean_error"])
                if isinstance(row.get("physical_pbr_mean_error"), (int, float))
                else float("inf")
            ),
            str(row["material_id"]),
        )
    )
    return ranked


def _direct_exact_library_match(
    ranking: Sequence[Mapping[str, Any]],
    *,
    prediction: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Authorize a direct MDL only for a unique, high-confidence exact match.

    A unique physical contract may be accepted without numeric PBR evidence.
    When several exact MDLs share the contract, MVInverse must isolate one by
    both absolute error and runner-up margin.  Ambiguity always falls back to
    bounded Qwen comparison rather than guessing.
    """

    if (
        prediction.get("status") != "APPLYABLE"
        or prediction.get("identity_resolution") != "exact_material"
        or float(prediction.get("confidence", 0.0))
        < MINIMUM_EXACT_TREATMENT_CONFIDENCE
    ):
        return None
    exact = [
        dict(row)
        for row in ranking
        if row.get("identity_match_tier") == "exact_material_contract"
    ]
    finish = prediction.get("surface_finish")
    if isinstance(finish, str) and finish != "unknown":
        finish_matches = [
            row for row in exact if row.get("predicted_finish_match") is True
        ]
        if finish_matches:
            exact = finish_matches
    if len(exact) == 1:
        winner = exact[0]
        return {
            "material_id": winner["material_id"],
            "authority": "unique_full_catalog_physical_contract",
            "physical_pbr_mean_error": winner.get("physical_pbr_mean_error"),
            "physical_pbr_runner_up_margin": None,
            "physical_pbr_evidence": winner.get("physical_pbr_evidence", {}),
            "library_pbr_fingerprint": winner.get("library_pbr_fingerprint", {}),
        }
    comparable = [
        row
        for row in exact
        if isinstance(row.get("physical_pbr_mean_error"), (int, float))
    ]
    comparable.sort(
        key=lambda row: (
            float(row["physical_pbr_mean_error"]),
            str(row["material_id"]),
        )
    )
    if len(comparable) < 2:
        return None
    winner, runner_up = comparable[:2]
    error = float(winner["physical_pbr_mean_error"])
    margin = float(runner_up["physical_pbr_mean_error"]) - error
    if (
        error > MAXIMUM_DIRECT_EXACT_PBR_ERROR
        or margin < MINIMUM_DIRECT_EXACT_PBR_MARGIN
    ):
        return None
    return {
        "material_id": winner["material_id"],
        "authority": "unique_full_catalog_mvinverse_pbr_fingerprint",
        "physical_pbr_mean_error": error,
        "physical_pbr_runner_up_margin": margin,
        "physical_pbr_evidence": winner.get("physical_pbr_evidence", {}),
        "library_pbr_fingerprint": winner.get("library_pbr_fingerprint", {}),
    }


def _identity_filtered_ranking(
    *,
    ranking: Sequence[Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    confidence = float(prediction.get("confidence", 0.0))
    prediction_applyable = (
        prediction.get("status") == "APPLYABLE"
        and confidence >= MINIMUM_MATERIAL_FAMILY_CONFIDENCE
    )
    substrate = str(prediction.get("physical_substrate", "unknown"))
    predicted_species = str(prediction.get("material_species", "unknown"))
    species_confidence = float(prediction.get("species_confidence", 0.0))
    species_authoritative = (
        predicted_species != "unknown"
        and species_confidence >= MINIMUM_MATERIAL_SPECIES_CONFIDENCE
    )
    treatment = str(prediction.get("surface_treatment", "unknown"))
    optical = str(prediction.get("optical_behavior", "unknown"))
    finish = str(prediction.get("surface_finish", "unknown"))
    exact_treatment_authoritative = (
        prediction.get("identity_resolution") == "exact_material"
    )
    ranking_by_id = {
        str(row.get("material_id")): dict(row)
        for row in ranking
        if isinstance(row.get("material_id"), str)
    }
    missing = sorted(set(catalog_by_id) - ranking_by_id.keys())
    if missing:
        raise PartIdQwenError(
            "material-identity-first retrieval must cover the complete catalog; "
            f"the ranking is missing {len(missing)} material IDs"
        )
    if not prediction_applyable:
        unconstrained: list[dict[str, Any]] = []
        for material_id, record in catalog_by_id.items():
            row = dict(ranking_by_id[material_id])
            row["predicted_family"] = "unknown"
            row["predicted_substrate"] = "unknown"
            row["predicted_material_species"] = "unknown"
            row["predicted_material_species_authoritative"] = False
            row["material_species"] = _catalog_material_species(
                material_id,
                record,
            )
            row["predicted_treatment"] = "unknown"
            row["predicted_optical_behavior"] = "unknown"
            row["predicted_finish"] = "unknown"
            row["predicted_finish_match"] = False
            row["identity_match_tier"] = "insufficient_identity_evidence"
            row["catalog_surface_semantics"] = dict(
                record["surface_semantics"]
            )
            row["physical_identity_applyable"] = False
            unconstrained.append(row)
        unconstrained = _annotate_library_preset_variants(
            unconstrained,
            catalog_by_id=catalog_by_id,
        )
        unconstrained.sort(
            key=lambda row: (
                int(row.get("rank", 1_000_000)),
                str(row["material_id"]),
            )
        )
        return unconstrained
    exact: list[dict[str, Any]] = []
    corresponding: list[dict[str, Any]] = []
    for material_id, record in catalog_by_id.items():
        candidate_species = _catalog_material_species(material_id, record)
        substrates, candidate_treatment, candidate_optical, semantic_finish, semantic_confidence = (
            _catalog_identity_semantics(material_id, record)
        )
        if semantic_confidence == "low" or substrate not in substrates:
            continue
        if candidate_optical != optical:
            continue
        if species_authoritative and candidate_species != predicted_species:
            continue
        row = dict(ranking_by_id[material_id])
        record_finishes = {
            str(value).casefold()
            for value in record.get("finishes", [])
            if isinstance(value, str)
        }
        if semantic_finish != "unknown":
            record_finishes.add(semantic_finish)
        row["predicted_family"] = prediction.get("catalog_family")
        row["predicted_substrate"] = substrate
        row["predicted_material_species"] = predicted_species
        row["predicted_material_species_authoritative"] = species_authoritative
        row["material_species"] = candidate_species
        row["predicted_treatment"] = treatment
        row["predicted_optical_behavior"] = optical
        row["predicted_finish"] = finish
        row["predicted_finish_match"] = bool(
            finish != "unknown" and finish in record_finishes
        )
        row["physical_identity_applyable"] = True
        row["identity_match_tier"] = (
            "exact_material_contract"
            if exact_treatment_authoritative and candidate_treatment == treatment
            else "corresponding_material_fallback"
        )
        row["catalog_surface_semantics"] = dict(record["surface_semantics"])
        (
            exact
            if exact_treatment_authoritative and candidate_treatment == treatment
            else corresponding
        ).append(row)
    filtered = exact if exact else corresponding
    if not filtered:
        raise PartIdQwenError(
            "material_library_gap: no material has the predicted substrate, "
            "species and optical behavior for "
            f"{prediction.get('part_id')}"
        )
    # Preserve exact authored presets.  A specific preset may be selected only
    # as an exact full-library match; its generic sibling remains available as
    # the colour-independent corresponding-material fallback.
    filtered = _annotate_library_preset_variants(
        filtered,
        catalog_by_id=catalog_by_id,
    )
    filtered.sort(
        key=lambda row: (
            0 if row["predicted_finish_match"] else 1,
            str(row["catalog_surface_semantics"]["surface_treatment"]),
            int(row.get("rank", 1_000_000)),
            str(row["material_id"]),
        )
    )
    return filtered


def _family_filtered_ranking(
    *,
    ranking: Sequence[Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Backward-compatible private alias for the identity-first filter."""

    return _identity_filtered_ranking(
        ranking=ranking,
        catalog_by_id=catalog_by_id,
        prediction=prediction,
    )


def _identity_shortlist(
    ranking: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    required_material_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if candidate_count < 1:
        raise PartIdQwenError("candidate_count must be positive")
    # Physical prediction has already removed cross-substrate and optical
    # mismatches.  Within that safe set, retrieval rank is useful only to put a
    # possible exact authored preset on the bounded comparison sheet.  Generic
    # fallbacks remain represented across treatments, so colour cannot force a
    # wrong physical family when no exact preset exists.
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for raw in ranking:
        semantics = raw.get("catalog_surface_semantics")
        treatment = (
            semantics.get("surface_treatment")
            if isinstance(semantics, Mapping)
            else None
        )
        if not isinstance(treatment, str):
            raise PartIdQwenError("identity candidate has no surface treatment")
        buckets.setdefault(treatment, []).append(raw)
    exact_preset_order = sorted(
        ranking,
        key=lambda row: (
            0 if row.get("predicted_finish_match") is True else 1,
            int(row.get("rank", 1_000_000)),
            (
                float(row["physical_pbr_mean_error"])
                if isinstance(row.get("physical_pbr_mean_error"), (int, float))
                else float("inf")
            ),
            str(row["material_id"]),
        ),
    )
    generic_by_treatment = []
    for _treatment, rows in sorted(buckets.items()):
        generic_rows = [
            row
            for row in rows
            if (
                "generic_identity_material_id" not in row
                or (
                    isinstance(row.get("generic_identity_material_id"), str)
                    and str(row["material_id"])
                    == row["generic_identity_material_id"]
                )
            )
        ]
        if not generic_rows:
            continue
        generic_by_treatment.append(
            min(
                generic_rows,
                key=lambda row: (
                    0 if row.get("predicted_finish_match") is True else 1,
                    (
                        float(row["physical_pbr_mean_error"])
                        if isinstance(
                            row.get("physical_pbr_mean_error"), (int, float)
                        )
                        else float("inf")
                    ),
                    int(row.get("rank", 1_000_000)),
                    str(row["material_id"]),
                ),
            )
        )
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        material_id = str(row["material_id"])
        if material_id not in selected_ids and len(selected) < candidate_count:
            selected.append(row)
            selected_ids.add(material_id)

    ranking_by_id = {str(row["material_id"]): row for row in ranking}
    for material_id in required_material_ids:
        row = ranking_by_id.get(material_id)
        if row is None:
            raise PartIdQwenError(
                f"required exact material {material_id} is absent from the "
                "physical catalog set"
            )
        add(row)
    exact_budget = min(
        len(exact_preset_order),
        max(1, candidate_count // 2) if candidate_count > 1 else 0,
    )
    for row in exact_preset_order[:exact_budget]:
        add(row)
    for row in generic_by_treatment:
        add(row)
    for row in exact_preset_order:
        add(row)
    shortlist: list[dict[str, Any]] = []
    for index, raw in enumerate(selected, start=1):
        row = dict(raw)
        row.update(
            {
                "original_retrieval_rank": raw.get("rank"),
                "compatibility_rank": index,
                "visual_compatibility_score": None,
                "appearance_median_rgb": None,
                "color_similarity": None,
                "hue_similarity": None,
                "color_delta_e": None,
                "color_tunable": False,
                "color_gate_passed": None,
                "texture_similarity": None,
                "texture_gradient_energy": None,
                "transmission_risk": False,
                "texture_mismatch_risk": False,
                "intrinsic_pattern_risk": False,
                "selection_allowed": True,
                "selection_allowed_by_default_constraints": True,
                "library_gap_fallback": (
                    raw.get("identity_match_tier")
                    == "corresponding_material_fallback"
                ),
                "library_gap_fallback_tier": (
                    raw.get("identity_match_tier")
                    if raw.get("identity_match_tier")
                    == "corresponding_material_fallback"
                    else None
                ),
                "relaxed_constraints": [],
                "conditional_h1_evaluation": False,
                "color_evidence_used": False,
                "color_evidence_scope": (
                    "exact_library_preset_confirmation_only"
                ),
                "specific_library_preset": raw.get(
                    "specific_library_preset", False
                ),
                "material_species": raw.get("material_species", "unknown"),
                "exact_authored_preset_candidate": raw.get(
                    "exact_authored_preset_candidate", False
                ),
                "physical_identity_applyable": raw.get(
                    "physical_identity_applyable", True
                ),
                "generic_identity_material_id": raw.get(
                    "generic_identity_material_id", raw.get("material_id")
                ),
                "physical_pbr_evidence": raw.get("physical_pbr_evidence", {}),
                "library_pbr_fingerprint": raw.get(
                    "library_pbr_fingerprint", {}
                ),
                "physical_pbr_term_errors": raw.get(
                    "physical_pbr_term_errors", {}
                ),
                "physical_pbr_mean_error": raw.get("physical_pbr_mean_error"),
                "physical_pbr_similarity": raw.get("physical_pbr_similarity"),
            }
        )
        shortlist.append(row)
    return shortlist


def _candidate_summary(
    row: Mapping[str, Any],
    catalog_record: Mapping[str, Any],
    *,
    withhold_color: bool = False,
) -> dict[str, Any]:
    return {
        "material_id": row["material_id"],
        "display_name": catalog_record.get("display_name"),
        "description": catalog_record.get("description"),
        "family": catalog_record.get("family"),
        "category_path": catalog_record.get("category_path"),
        "keywords": catalog_record.get("keywords", []),
        "colors": [] if withhold_color else catalog_record.get("colors", []),
        "finishes": catalog_record.get("finishes", []),
        "surface_semantics": catalog_record.get("surface_semantics"),
        "identity_match_tier": row.get("identity_match_tier"),
        "specific_library_preset": row.get("specific_library_preset", False),
        "material_species": row.get("material_species", "unknown"),
        "exact_authored_preset_candidate": row.get(
            "exact_authored_preset_candidate", False
        ),
        "physical_identity_applyable": row.get(
            "physical_identity_applyable", True
        ),
        "generic_identity_material_id": row.get(
            "generic_identity_material_id", row.get("material_id")
        ),
        "color_evidence_withheld": withhold_color,
        "visual_retrieval_scores_withheld": withhold_color,
        "retrieval_rank": None if withhold_color else row.get("rank"),
        "siglip2_score": None if withhold_color else row.get("siglip2_score"),
        "dino_score": None if withhold_color else row.get("dino_score"),
        "color_score": None if withhold_color else row.get("color_score"),
        "mvinverse_score": None if withhold_color else row.get("mvinverse_score"),
        "physical_pbr_similarity": row.get("physical_pbr_similarity"),
        "physical_pbr_evidence": row.get("physical_pbr_evidence", {}),
        "library_pbr_fingerprint": row.get("library_pbr_fingerprint", {}),
    }


def _mean_profile_value(
    profile: Mapping[str, Any],
    key: str,
) -> float | None:
    appearance = profile.get("appearance")
    if not isinstance(appearance, Mapping):
        return None
    values = [
        float(record[key])
        for record in appearance.values()
        if isinstance(record, Mapping)
        and isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
        and math.isfinite(float(record[key]))
    ]
    return sum(values) / len(values) if values else None


def _median_profile_rgb(profile: Mapping[str, Any]) -> list[float] | None:
    appearance = profile.get("appearance")
    if not isinstance(appearance, Mapping):
        return None
    samples = [
        [float(value) for value in record["median_rgb"]]
        for record in appearance.values()
        if isinstance(record, Mapping)
        and isinstance(record.get("median_rgb"), list)
        and len(record["median_rgb"]) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in record["median_rgb"]
        )
    ]
    if not samples:
        return None
    return [
        float(value)
        for value in np.median(np.asarray(samples, dtype=np.float32), axis=0)
    ]


def _transmission_risk(catalog_record: Mapping[str, Any]) -> bool:
    """Identify fixed MDLs that visibly transmit the scene behind a surface."""

    family = str(catalog_record.get("family", "")).casefold()
    name = " ".join(
        str(catalog_record.get(key, ""))
        for key in ("material_id", "display_name", "category_path")
    ).casefold()
    if "mirror" in name:
        return False
    return (
        family == "glass"
        or "glasswithvolume" in name.replace("_", "")
        or "plastic_clear" in name
        or "plastic acrylic" in name
        or "plastic_acrylic" in name
    )


def _target_appearance(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    image_value = observation.get("image")
    mask_value = observation.get("mask")
    if not isinstance(image_value, str) or not isinstance(mask_value, str):
        return None
    try:
        with Image.open(Path(image_value).expanduser().resolve(strict=True)) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(Path(mask_value).expanduser().resolve(strict=True)) as opened:
            valid = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
    except (OSError, ValueError):
        return None
    if (
        rgb.shape[:2] != valid.shape
        or int(valid.sum()) < MINIMUM_PART_ID_APPEARANCE_PIXELS
    ):
        return None
    pixels = rgb[valid]
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    gradient_x = np.abs(np.diff(gray, axis=1))
    gradient_y = np.abs(np.diff(gray, axis=0))
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    gradients = np.concatenate((gradient_x[valid_x], gradient_y[valid_y]))
    texture_energy = float(gradients.mean()) if len(gradients) else 0.0
    chromatic_coverage = observation.get("chromatic_coverage")
    chromatic_component_authoritative = bool(
        isinstance(chromatic_coverage, Mapping)
        and chromatic_coverage.get("applied") is True
    )
    return {
        "trusted_pixels": int(valid.sum()),
        "median_rgb": [round(float(value), 8) for value in np.median(pixels, axis=0)],
        "rgb_std": [round(float(value), 8) for value in pixels.std(axis=0)],
        "texture_gradient_energy": round(texture_energy, 8),
        "likely_opaque": (
            texture_energy <= 0.04 and float(np.mean(pixels.std(axis=0))) <= 0.22
        ),
        "likely_smooth": (
            texture_energy <= 0.020 and float(np.mean(pixels.std(axis=0))) <= 0.16
        ),
        "likely_unpatterned": (
            texture_energy <= 0.020 and float(np.mean(pixels.std(axis=0))) <= 0.16
        ),
        "chromatic_component_authoritative": (chromatic_component_authoritative),
        "tiny_chromatic_rescue": bool(
            chromatic_component_authoritative
            and chromatic_coverage.get("tiny_part_rescue") is True
        ),
    }


def _is_color_critical_target(
    target: Mapping[str, Any] | None,
    *,
    saturation: float,
) -> bool:
    if target is None or saturation < COLOR_CRITICAL_MINIMUM_SATURATION:
        return False
    trusted_pixels = target.get("trusted_pixels")
    return bool(
        (
            isinstance(trusted_pixels, int)
            and not isinstance(trusted_pixels, bool)
            and trusted_pixels >= COLOR_CRITICAL_MINIMUM_TRUSTED_PIXELS
        )
        or target.get("chromatic_component_authoritative") is True
    )


def _observation_bank_context(
    retrieval: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]] | None:
    backends = retrieval.get("backends")
    siglip = backends.get("siglip2") if isinstance(backends, Mapping) else None
    bank = siglip.get("observation_bank") if isinstance(siglip, Mapping) else None
    value = bank.get("path") if isinstance(bank, Mapping) else None
    if not isinstance(value, str):
        return None
    try:
        root = Path(value).expanduser().resolve(strict=True)
        document = _read(root / "appearance_profiles.json", "appearance profiles")
    except (OSError, PartIdQwenError):
        return None
    rows = document.get("materials")
    if not isinstance(rows, list):
        return None
    by_id = {
        str(row["material_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("material_id"), str)
    }
    return root, by_id


def _candidate_render(
    *,
    material_id: str,
    profile: Mapping[str, Any] | None,
    bank_root: Path | None,
    catalog_record: Mapping[str, Any],
    material_root: Path | None,
) -> Path | None:
    if profile is not None and bank_root is not None:
        observations = profile.get("observations")
        if isinstance(observations, list):
            ordered = sorted(
                (
                    row
                    for row in observations
                    if isinstance(row, Mapping) and isinstance(row.get("image"), str)
                ),
                key=lambda row: (
                    0 if row.get("profile") == "neutral_iso" else 1,
                    str(row.get("profile", "")),
                ),
            )
            for row in ordered:
                relative = Path(str(row["image"]))
                try:
                    path = (bank_root / relative).resolve(strict=True)
                    path.relative_to(bank_root)
                except (OSError, ValueError):
                    continue
                if path.is_file():
                    return path
    thumbnail = catalog_record.get("thumbnail_path")
    if isinstance(thumbnail, str) and material_root is not None:
        try:
            path = (material_root / thumbnail).resolve(strict=True)
            path.relative_to(material_root)
        except (OSError, ValueError):
            return None
        if path.is_file():
            return path
    return None


def _compatibility_shortlist(
    *,
    ranking: Sequence[Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    profiles_by_id: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Any] | None,
    candidate_count: int,
    allow_color_tuning: bool = False,
) -> list[dict[str, Any]]:
    """Rerank retrieved MDLs by immutable, directly observable appearance."""

    target_rgb = (
        np.asarray(target["median_rgb"], dtype=np.float32)
        if target is not None
        else None
    )
    target_texture = (
        float(target["texture_gradient_energy"]) if target is not None else None
    )
    rows: list[dict[str, Any]] = []
    total = max(1, len(ranking))
    for original_rank, raw in enumerate(ranking, start=1):
        material_id = raw.get("material_id")
        if not isinstance(material_id, str) or material_id not in catalog_by_id:
            continue
        profile = profiles_by_id.get(material_id, {})
        candidate_rgb = _median_profile_rgb(profile)
        candidate_texture = _mean_profile_value(profile, "texture_gradient_energy")
        color_similarity: float | None = None
        hue_similarity: float | None = None
        color_delta_e: float | None = None
        color_gate_passed = True
        color_tunable = bool(
            allow_color_tuning and tuning_profile_for_material(material_id) is not None
        )
        if target_rgb is not None and candidate_rgb is not None:
            default_color_similarity = perceptual_similarity(
                target_rgb.tolist(),
                candidate_rgb,
            )
            color_delta_e = srgb_delta_e(target_rgb.tolist(), candidate_rgb)
            target_hue, target_saturation, _target_value = colorsys.rgb_to_hsv(
                *target_rgb.tolist()
            )
            candidate_hue, candidate_saturation, _candidate_value = colorsys.rgb_to_hsv(
                *candidate_rgb
            )
            hue_distance = min(
                abs(target_hue - candidate_hue),
                1.0 - abs(target_hue - candidate_hue),
            )
            chromatic_target = _is_color_critical_target(
                target,
                saturation=target_saturation,
            )
            if chromatic_target and candidate_saturation < (
                max(
                    COLOR_CRITICAL_MINIMUM_CANDIDATE_SATURATION,
                    0.35 * target_saturation,
                )
            ):
                hue_similarity = 0.0
            elif target_saturation >= 0.15 and candidate_saturation >= 0.10:
                hue_similarity = max(0.0, 1.0 - 2.0 * hue_distance)
            else:
                hue_similarity = 1.0 - abs(target_saturation - candidate_saturation)
            saturation_similarity = max(
                0.0, 1.0 - abs(target_saturation - candidate_saturation)
            )
            default_composite = (
                0.65 * default_color_similarity
                + 0.25 * hue_similarity
                + 0.10 * saturation_similarity
            )
            # A reviewed Base MDL colour input can reproduce the target colour
            # while retaining the preset's texture, normal and coating.  Its
            # default grey swatch is therefore not a colour mismatch.
            color_similarity = 1.0 if color_tunable else default_composite
            color_gate_passed = bool(
                not chromatic_target
                or color_tunable
                or (
                    candidate_saturation
                    >= max(
                        COLOR_CRITICAL_MINIMUM_CANDIDATE_SATURATION,
                        0.35 * target_saturation,
                    )
                    and hue_distance * 360.0
                    <= COLOR_CRITICAL_MAXIMUM_HUE_DISTANCE_DEGREES
                    and color_delta_e <= COLOR_CRITICAL_MAXIMUM_DELTA_E
                )
            )
        texture_similarity: float | None = None
        texture_mismatch = False
        if target_texture is not None and candidate_texture is not None:
            texture_similarity = math.exp(
                -abs(math.log((candidate_texture + 0.002) / (target_texture + 0.002)))
            )
            texture_mismatch = bool(
                target
                and target.get("likely_smooth")
                and candidate_texture > max(0.010, 1.20 * target_texture)
            )
        transmission = _transmission_risk(catalog_by_id[material_id])
        likely_opaque = bool(target and target.get("likely_opaque"))
        material_name = material_id.casefold()
        intrinsic_pattern_risk = bool(
            target
            and target.get("likely_unpatterned")
            and ("/grass_" in material_name or "#grass_" in material_name)
        )
        texture_mismatch = texture_mismatch or intrinsic_pattern_risk
        retrieval_prior = 1.0 - (original_rank - 1) / total
        score = 0.25 * retrieval_prior
        score += 0.50 * (color_similarity if color_similarity is not None else 0.5)
        score += 0.25 * (texture_similarity if texture_similarity is not None else 0.5)
        if transmission and likely_opaque:
            score -= 0.40
        if texture_mismatch:
            score -= 0.20
        if not color_gate_passed:
            score -= 0.50
        rows.append(
            {
                **dict(raw),
                "original_retrieval_rank": int(raw.get("rank", original_rank)),
                "visual_compatibility_score": round(score, 8),
                "color_similarity": (
                    round(color_similarity, 8) if color_similarity is not None else None
                ),
                "hue_similarity": (
                    round(hue_similarity, 8) if hue_similarity is not None else None
                ),
                "color_delta_e": (
                    round(color_delta_e, 8) if color_delta_e is not None else None
                ),
                "color_tunable": color_tunable,
                "color_gate_passed": color_gate_passed,
                "texture_similarity": (
                    round(texture_similarity, 8)
                    if texture_similarity is not None
                    else None
                ),
                "appearance_median_rgb": (
                    [round(value, 8) for value in candidate_rgb]
                    if candidate_rgb is not None
                    else None
                ),
                "texture_gradient_energy": (
                    round(candidate_texture, 8)
                    if candidate_texture is not None
                    else None
                ),
                "transmission_risk": transmission,
                "texture_mismatch_risk": texture_mismatch,
                "intrinsic_pattern_risk": intrinsic_pattern_risk,
                "selection_allowed": not (
                    (transmission and likely_opaque)
                    or texture_mismatch
                    or not color_gate_passed
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["visual_compatibility_score"]),
            int(row["original_retrieval_rank"]),
            str(row["material_id"]),
        )
    )
    # A single scalar ranking can hide an important trade-off (for example,
    # the only color-compatible fixed MDL may carry an incompatible texture).
    # Keep explicit color, texture and retrieval anchors so the VLM sees the
    # real library limitation instead of being forced into a homogeneous list.
    shortlist: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(values: Sequence[dict[str, Any]], limit: int) -> None:
        if len(shortlist) >= limit:
            return
        for row in values:
            material_id = str(row["material_id"])
            if material_id in selected_ids:
                continue
            shortlist.append(row)
            selected_ids.add(material_id)
            if len(shortlist) >= limit:
                return

    composite_limit = max(2, candidate_count - 4)
    add(rows, composite_limit)
    opaque_rows = [
        row
        for row in rows
        if not (
            bool(target and target.get("likely_opaque")) and row["transmission_risk"]
        )
    ]
    add(
        sorted(
            (row for row in opaque_rows if row["color_similarity"] is not None),
            key=lambda row: (
                -float(row["color_similarity"]),
                int(row["original_retrieval_rank"]),
                str(row["material_id"]),
            ),
        ),
        min(candidate_count, composite_limit + 2),
    )
    add(
        sorted(
            (row for row in opaque_rows if row["texture_similarity"] is not None),
            key=lambda row: (
                -float(row["texture_similarity"]),
                int(row["original_retrieval_rank"]),
                str(row["material_id"]),
            ),
        ),
        min(candidate_count, composite_limit + 3),
    )
    add(
        sorted(
            opaque_rows,
            key=lambda row: (
                int(row["original_retrieval_rank"]),
                str(row["material_id"]),
            ),
        ),
        candidate_count,
    )
    add(rows, candidate_count)
    for index, row in enumerate(shortlist, start=1):
        row["compatibility_rank"] = index
    return shortlist


def _promote_library_gap_candidates(
    shortlist: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    target: Mapping[str, Any] | None = None,
    allow_color_tuning: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Keep an exact-cover decision possible when every hard gate rejects.

    The normal visual compatibility gate remains authoritative whenever it
    admits at least one candidate.  If it admits none, fail-open only inside
    the already retrieved NVIDIA Base shortlist.

    An immutable library needs a distinct policy from a parameter-tunable one:
    a highly chromatic reference can have no fixed *opaque* Base MDL even
    though a color-faithful fixed MDL exists (for example ``Green_Glass``).
    With parameter mutation disabled, replacing that candidate with neutral
    steel destroys the requested visual result.  In that precise library-gap
    case, promote the best fixed colour-compatible candidate and record every
    relaxed physical constraint.  This never writes an MDL parameter and is
    bounded to the retrieved Base-only shortlist.
    """

    rows = [dict(row) for row in shortlist]
    for row in rows:
        row["selection_allowed_by_default_constraints"] = bool(
            row.get("selection_allowed") is True
        )
        row["library_gap_fallback"] = False
        row["library_gap_fallback_tier"] = None
        row["relaxed_constraints"] = []
        row["conditional_h1_evaluation"] = False
    if any(row["selection_allowed_by_default_constraints"] for row in rows):
        return rows, None

    target_is_chromatic = False
    if target is not None:
        median_rgb = target.get("median_rgb")
        if (
            isinstance(median_rgb, list)
            and len(median_rgb) == 3
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in median_rgb
            )
        ):
            _hue, saturation, _value = colorsys.rgb_to_hsv(
                *(float(value) for value in median_rgb)
            )
            target_is_chromatic = _is_color_critical_target(
                target,
                saturation=saturation,
            )

    if target_is_chromatic and not allow_color_tuning:
        # A color-gated candidate has the correct hue/chroma and a bounded
        # perceptual Delta-E.  Its transmission or texture risk is less
        # damaging than replacing an observed saturated coating with an
        # unrelated neutral material when the MDL must remain immutable.
        color_faithful = [
            row for row in rows if row.get("color_gate_passed") is True
        ]
        if color_faithful:
            selected = min(
                color_faithful,
                key=lambda row: (
                    -float(row.get("visual_compatibility_score", float("-inf"))),
                    float(row.get("color_delta_e", float("inf"))),
                    int(row.get("original_retrieval_rank", 1_000_000)),
                    str(row.get("material_id")),
                ),
            )
            relaxed_constraints: list[str] = []
            if bool(selected.get("transmission_risk")):
                relaxed_constraints.append("opacity_gate")
            if bool(selected.get("texture_mismatch_risk")):
                relaxed_constraints.append("texture_gate")
            # The candidate passed the fixed-colour gate.  It must be the only
            # VLM option so a semantic material-name preference cannot undo
            # the evidence-bounded visual decision.
            selected["selection_allowed"] = True
            selected["library_gap_fallback"] = True
            selected["library_gap_fallback_tier"] = (
                "immutable_chromatic_visual_priority"
            )
            selected["relaxed_constraints"] = relaxed_constraints
            selected["conditional_h1_evaluation"] = False
            selected["compatibility_rank"] = 1
            return rows, {
                "status": "LIBRARY_GAP_BOUNDED_FALLBACK",
                "policy": "retrieved_nvidia_base_immutable_chromatic_visual_priority/v1",
                "promoted_material_ids": [str(selected["material_id"])],
                "applied_tiers": ["immutable_chromatic_visual_priority"],
                "parameter_candidate_count": 0,
                "default_constraints_remain_unmet": True,
                "parameter_write_authorized": False,
                "final_authority": "immutable_base_visual_compatibility",
            }

    def color_parameter_capable(row: Mapping[str, Any]) -> bool:
        material_id = row.get("material_id")
        return bool(
            isinstance(material_id, str)
            and tuning_profile_for_material(material_id) is not None
        )

    tiers: tuple[
        tuple[str, tuple[str, ...], Any],
        ...,
    ] = (
        (
            "opaque_texture_compatible_color_parameter_candidate",
            ("default_color_gate",),
            lambda row: (
                not bool(row.get("transmission_risk"))
                and not bool(row.get("texture_mismatch_risk"))
                and color_parameter_capable(row)
            ),
        ),
        (
            "opaque_texture_compatible_visual_nearest",
            ("default_color_gate",),
            lambda row: (
                not bool(row.get("transmission_risk"))
                and not bool(row.get("texture_mismatch_risk"))
            ),
        ),
        (
            "opaque_visual_nearest",
            ("default_color_gate", "texture_gate"),
            lambda row: not bool(row.get("transmission_risk")),
        ),
        (
            "bounded_retrieval_visual_nearest",
            ("default_color_gate", "texture_gate", "transmission_gate"),
            lambda _row: True,
        ),
    )
    minimum_promoted = min(2, len(rows))
    maximum_promoted = min(candidate_count, len(rows))
    promoted_ids: set[str] = set()
    promoted: list[dict[str, Any]] = []
    applied_tiers: list[str] = []
    for tier_name, relaxed_constraints, predicate in tiers:
        additions = [
            row
            for row in rows
            if str(row.get("material_id")) not in promoted_ids and predicate(row)
        ]
        if not additions:
            continue
        applied_tiers.append(tier_name)
        for row in additions:
            material_id = str(row.get("material_id"))
            row["selection_allowed"] = True
            row["library_gap_fallback"] = True
            row["library_gap_fallback_tier"] = tier_name
            row["relaxed_constraints"] = list(relaxed_constraints)
            row["conditional_h1_evaluation"] = color_parameter_capable(row)
            promoted.append(row)
            promoted_ids.add(material_id)
            if len(promoted) >= maximum_promoted:
                break
        # A complete first-tier set is preferable to mixing in weaker
        # opacity/texture relaxations merely to fill the original shortlist.
        if len(promoted) >= minimum_promoted:
            break

    if not promoted:
        raise PartIdQwenError(
            "material library gap fallback found no bounded retrieval candidate"
        )
    for index, row in enumerate(promoted, start=1):
        row["compatibility_rank"] = index
    return rows, {
        "status": "LIBRARY_GAP_BOUNDED_FALLBACK",
        "policy": "retrieved_nvidia_base_hierarchical_relaxation/v1",
        "promoted_material_ids": [str(row["material_id"]) for row in promoted],
        "applied_tiers": applied_tiers,
        "parameter_candidate_count": sum(
            bool(row["conditional_h1_evaluation"]) for row in promoted
        ),
        "default_constraints_remain_unmet": True,
        "parameter_write_authorized": False,
        "final_authority": "part_id_h0_h1_actual_cad_render_tournament",
    }


def _candidate_local_score(
    candidate: Mapping[str, Any],
    *,
    target: Mapping[str, Any] | None,
) -> tuple[float, str]:
    """Score one fresh candidate inside one Part-ID reference mask.

    The score never reads a previous material plan.  Most parts use the
    immutable-render compatibility score directly.  Large chromatic surfaces
    receive a color-first score so their visual identity cannot be dominated
    by a texture embedding from a differently colored material.
    """

    compatibility = float(candidate.get("visual_compatibility_score", 0.0))
    if target is None:
        return compatibility, "fresh_composite_candidate_score"
    median_rgb = target.get("median_rgb")
    trusted_pixels = target.get("trusted_pixels")
    if (
        not isinstance(median_rgb, list)
        or len(median_rgb) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in median_rgb
        )
        or isinstance(trusted_pixels, bool)
        or not isinstance(trusted_pixels, int)
    ):
        return compatibility, "fresh_composite_candidate_score"
    _hue, saturation, _value = colorsys.rgb_to_hsv(
        *(float(value) for value in median_rgb)
    )
    color_similarity = candidate.get("color_similarity")
    texture_similarity = candidate.get("texture_similarity")
    if (
        not _is_color_critical_target(target, saturation=saturation)
        or isinstance(color_similarity, bool)
        or not isinstance(color_similarity, (int, float))
        or isinstance(texture_similarity, bool)
        or not isinstance(texture_similarity, (int, float))
    ):
        return compatibility, "fresh_composite_candidate_score"
    score = (
        0.55 * float(color_similarity)
        + 0.20 * float(texture_similarity)
        + 0.25 * compatibility
    )
    return score, "fresh_large_chromatic_part_color_first_score"


def _apply_part_id_selective_regression(
    *,
    jobs: Sequence[Mapping[str, Any]],
    qwen_selections: Sequence[Mapping[str, Any]],
    maximum_local_score_regression: float = (DEFAULT_MAXIMUM_LOCAL_SCORE_REGRESSION),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep each Qwen choice only when it does not regress its own Part-ID.

    The comparison baseline is the best candidate from the *current* fresh
    shortlist.  It is deliberately unrelated to v18, a prior run, or the
    source USD's existing material bindings.
    """

    if (
        isinstance(maximum_local_score_regression, bool)
        or not isinstance(maximum_local_score_regression, (int, float))
        or not math.isfinite(float(maximum_local_score_regression))
        or float(maximum_local_score_regression) < 0.0
    ):
        raise PartIdQwenError(
            "maximum_local_score_regression must be a finite non-negative number"
        )
    jobs_by_id = {
        str(job["part_id"]): job
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("part_id"), str)
    }
    qwen_by_id = {
        str(row["part_id"]): row
        for row in qwen_selections
        if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
    }
    if set(jobs_by_id) != set(qwen_by_id):
        raise PartIdQwenError(
            "selective regression inputs do not cover the same Part IDs"
        )

    final: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for part_id in sorted(jobs_by_id):
        job = jobs_by_id[part_id]
        target = (
            job.get("target_appearance")
            if isinstance(job.get("target_appearance"), Mapping)
            else None
        )
        candidates = [
            candidate
            for candidate in job.get("candidates", [])
            if isinstance(candidate, Mapping)
            and candidate.get("selection_allowed", True) is True
            and isinstance(candidate.get("material_id"), str)
        ]
        if not candidates:
            raise PartIdQwenError(
                f"Part-ID {part_id} has no allowed selective-regression candidates"
            )
        scored: list[tuple[float, int, str, str, Mapping[str, Any]]] = []
        for candidate in candidates:
            score, score_mode = _candidate_local_score(
                candidate,
                target=target,
            )
            compatibility_rank = candidate.get("compatibility_rank")
            rank = (
                int(compatibility_rank)
                if isinstance(compatibility_rank, int)
                and not isinstance(compatibility_rank, bool)
                else 1_000_000
            )
            scored.append(
                (
                    float(score),
                    rank,
                    str(candidate["material_id"]),
                    score_mode,
                    candidate,
                )
            )
        best = min(
            scored,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        qwen = qwen_by_id[part_id]
        qwen_material_id = str(qwen.get("material_id"))
        qwen_scored = next(
            (item for item in scored if item[2] == qwen_material_id),
            None,
        )
        if qwen_scored is None:
            raise PartIdQwenError(
                f"Qwen choice for {part_id} is not an allowed current candidate"
            )
        regression = max(0.0, best[0] - qwen_scored[0])
        strict_color_nonregression = False
        if target is not None:
            median_rgb = target.get("median_rgb")
            if (
                isinstance(median_rgb, list)
                and len(median_rgb) == 3
                and all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in median_rgb
                )
            ):
                _hue, saturation, _value = colorsys.rgb_to_hsv(
                    *(float(value) for value in median_rgb)
                )
                strict_color_nonregression = _is_color_critical_target(
                    target,
                    saturation=saturation,
                )
        effective_maximum_regression = (
            0.0 if strict_color_nonregression else float(maximum_local_score_regression)
        )
        gate_changed = regression > effective_maximum_regression
        selected = best if gate_changed else qwen_scored
        final.append(
            {
                "part_id": part_id,
                "material_id": selected[2],
                "confidence": float(qwen.get("confidence", 0.0)),
            }
        )
        audits.append(
            {
                "part_id": part_id,
                "qwen_material_id": qwen_material_id,
                "fresh_local_baseline_material_id": best[2],
                "selected_material_id": selected[2],
                "score_mode": selected[3],
                "qwen_local_score": round(qwen_scored[0], 8),
                "fresh_local_baseline_score": round(best[0], 8),
                "observed_local_score_regression": round(regression, 8),
                "maximum_local_score_regression": round(
                    effective_maximum_regression, 8
                ),
                "strict_color_nonregression": strict_color_nonregression,
                "selection_source": (
                    "fresh_local_baseline"
                    if gate_changed
                    else "qwen_within_local_nonregression_limit"
                ),
                "gate_changed_qwen_choice": gate_changed,
                "previous_material_plan_consulted": False,
            }
        )
    audit = {
        "schema_version": SELECTIVE_REGRESSION_SCHEMA_VERSION,
        "baseline_authority": "fresh_current_run_candidate_shortlist_only",
        "previous_material_plan_consulted": False,
        "maximum_local_score_regression": float(maximum_local_score_regression),
        "large_chromatic_part_policy": {
            "minimum_trusted_pixels": COLOR_CRITICAL_MINIMUM_TRUSTED_PIXELS,
            "minimum_saturation": COLOR_CRITICAL_MINIMUM_SATURATION,
            "single_view_chromatic_component_also_qualifies": True,
            "minimum_chromatic_component_pixels": (MINIMUM_PART_ID_APPEARANCE_PIXELS),
            "weights": {
                "color_similarity": 0.55,
                "texture_similarity": 0.20,
                "visual_compatibility": 0.25,
            },
        },
        "parts": audits,
        "summary": {
            "part_count": len(audits),
            "qwen_choice_retained_count": sum(
                not row["gate_changed_qwen_choice"] for row in audits
            ),
            "fresh_local_baseline_selected_count": sum(
                row["gate_changed_qwen_choice"] for row in audits
            ),
            "exact_cover": len(audits) == len(jobs_by_id),
        },
    }
    return final, audit


def _comparison_sheet(
    *,
    part_id: str,
    crop: Path,
    candidates: Sequence[Mapping[str, Any]],
    render_paths: Mapping[str, Path | None],
    output: Path,
    grayscale: bool = False,
) -> Path:
    tile_size = 320
    columns = 3
    tiles = 1 + len(candidates)
    rows = int(math.ceil(tiles / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_size, rows * tile_size),
        (28, 30, 34),
    )
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = (
        ImageFont.truetype(str(font_path), 22)
        if font_path.is_file()
        else ImageFont.load_default()
    )

    def paste_tile(path: Path | None, index: int, label: str) -> None:
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        area = (x + 8, y + 42, x + tile_size - 8, y + tile_size - 8)
        if path is not None:
            try:
                with Image.open(path) as opened:
                    source_image = opened.convert("RGB")
                    if grayscale:
                        source_image = ImageOps.grayscale(source_image).convert("RGB")
                    image = ImageOps.contain(
                        source_image,
                        (area[2] - area[0], area[3] - area[1]),
                    )
                background = Image.new(
                    "RGB",
                    (area[2] - area[0], area[3] - area[1]),
                    (128, 128, 128),
                )
                background.paste(
                    image,
                    (
                        (background.width - image.width) // 2,
                        (background.height - image.height) // 2,
                    ),
                )
                canvas.paste(background, (area[0], area[1]))
            except OSError:
                pass
        draw.rectangle(
            (x + 4, y + 4, x + tile_size - 4, y + tile_size - 4),
            outline=(210, 215, 224),
            width=2,
        )
        draw.text((x + 12, y + 10), label, fill=(255, 255, 255), font=font)

    paste_tile(crop, 0, f"TARGET {part_id}")
    for index, candidate in enumerate(candidates, start=1):
        paste_tile(
            render_paths.get(str(candidate["material_id"])),
            index,
            f"CANDIDATE {index}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _validate_batch(
    document: Mapping[str, Any],
    *,
    expected: Sequence[dict[str, Any]],
    require_material_identity_match: bool = False,
) -> list[dict[str, Any]]:
    if set(document) != {"schema_version", "selections"}:
        raise PartIdQwenError("Qwen Part-ID batch returned unexpected fields")
    expected_schema = (
        MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION
        if require_material_identity_match
        else BATCH_SCHEMA_VERSION
    )
    if document.get("schema_version") != expected_schema:
        raise PartIdQwenError("Qwen Part-ID batch uses the wrong schema")
    rows = document.get("selections")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise PartIdQwenError("Qwen Part-ID batch selection count is invalid")
    expected_by_id = {str(item["part_id"]): item for item in expected}
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        expected_fields = {"part_id", "candidate_index", "confidence"}
        if require_material_identity_match:
            expected_fields.add("match_type")
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise PartIdQwenError(f"Qwen Part-ID selection {index} has invalid fields")
        part_id = raw.get("part_id")
        candidate_index = raw.get("candidate_index")
        confidence = raw.get("confidence")
        match_type = raw.get("match_type")
        if (
            not isinstance(part_id, str)
            or part_id not in expected_by_id
            or part_id in output
        ):
            raise PartIdQwenError(
                f"Qwen Part-ID selection {index} cites an invalid part"
            )
        candidates = expected_by_id[part_id]["candidates"]
        requested_candidate_index = candidate_index
        # Some otherwise valid local Qwen JSON generations quote scalar
        # indices.  Normalize only the canonical decimal spelling of an
        # integer. Arbitrary strings, signs and decimals remain fail-closed.
        if (
            isinstance(candidate_index, str)
            and candidate_index.isascii()
            and candidate_index.isdecimal()
            and candidate_index == str(int(candidate_index))
        ):
            candidate_index = int(candidate_index)
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise PartIdQwenError(
                f"Qwen selected an invalid candidate_index for {part_id}"
            )
        index_resolution = "exact"
        if candidate_index < 1 or candidate_index > len(candidates):
            # An out-of-range integer cannot identify a supplied MDL. Use the
            # highest-ranked allowed candidate deterministically instead of
            # letting one malformed scalar abort every other Part-ID. The
            # subsequent per-Part-ID non-regression gate still compares this
            # fallback against every current allowed candidate.
            candidate_index = 1
            index_resolution = "bounded_top_candidate_fallback"
        candidate = candidates[candidate_index - 1]
        if candidate.get("candidate_index") != candidate_index:
            raise PartIdQwenError(
                f"Part-ID {part_id} candidate order is internally inconsistent"
            )
        if candidate.get("selection_allowed") is not True:
            raise PartIdQwenError(
                f"Qwen selected a blocked candidate_index for {part_id}"
            )
        material_id = candidate.get("material_id")
        if not isinstance(material_id, str):
            raise PartIdQwenError(
                f"Part-ID {part_id} candidate has no exact material_id"
            )
        requested_material_id = material_id
        requested_exact_preset_color_delta_e = candidate.get(
            "exact_preset_color_delta_e"
        )
        requested_exact_preset_color_gate_passed = candidate.get(
            "exact_preset_color_gate_passed"
        )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise PartIdQwenError(f"Qwen returned invalid confidence for {part_id}")
        if require_material_identity_match and match_type not in {
            "EXACT_LIBRARY_MATCH",
            "CORRESPONDING_MATERIAL",
        }:
            raise PartIdQwenError(
                f"Qwen returned an invalid material match type for {part_id}"
            )
        if (
            require_material_identity_match
            and match_type == "EXACT_LIBRARY_MATCH"
            and candidate.get("exact_preset_color_gate_passed") is False
        ):
            match_type = "CORRESPONDING_MATERIAL"
            index_resolution = "exact_preset_color_gate_rejected"
        if (
            require_material_identity_match
            and match_type == "CORRESPONDING_MATERIAL"
            and candidate.get("physical_identity_applyable") is True
            and candidate.get("exact_authored_preset_candidate") is True
            and candidate.get("exact_preset_color_gate_passed") is True
            and float(confidence) >= MINIMUM_MATERIAL_SPECIES_CONFIDENCE
        ):
            # Qwen ranks the actual MDL renders but does not own the final
            # exact/corresponding state transition.  A physically compatible,
            # identity-specific authored material that Qwen selected and the
            # measured preset appearance independently confirms is an exact
            # library match.  Without this deterministic promotion, a correct
            # Copper choice can be discarded merely because the model emitted
            # the conservative label CORRESPONDING_MATERIAL, after which the
            # grayscale fallback is free to replace it with Chrome.
            match_type = "EXACT_LIBRARY_MATCH"
            index_resolution = "deterministic_exact_authored_preset_promotion"
        if (
            require_material_identity_match
            and match_type == "CORRESPONDING_MATERIAL"
            and candidate.get("specific_library_preset") is True
        ):
            generic_material_id = candidate.get("generic_identity_material_id")
            generic_candidate = next(
                (
                    row
                    for row in candidates
                    if row.get("material_id") == generic_material_id
                    and row.get("specific_library_preset") is not True
                    and row.get("selection_allowed") is True
                ),
                None,
            )
            if generic_candidate is None:
                raise PartIdQwenError(
                    f"Part-ID {part_id} selected a specific authored preset as a "
                    "corresponding-material fallback, but its generic identity is "
                    "not in the bounded shortlist"
                )
            candidate = generic_candidate
            material_id = str(candidate["material_id"])
            candidate_index = int(candidate["candidate_index"])
            index_resolution = (
                "exact_preset_color_gate_rejected_to_generic"
                if index_resolution == "exact_preset_color_gate_rejected"
                else "specific_preset_to_generic_corresponding_fallback"
            )
        output[part_id] = {
            "part_id": part_id,
            "material_id": material_id,
            "candidate_index": candidate_index,
            "requested_candidate_index": requested_candidate_index,
            "requested_material_id": requested_material_id,
            "index_resolution": index_resolution,
            "confidence": float(confidence),
            "exact_preset_color_delta_e": (
                requested_exact_preset_color_delta_e
            ),
            "exact_preset_color_gate_passed": (
                requested_exact_preset_color_gate_passed
            ),
            "material_species": candidate.get("material_species", "unknown"),
        }
        if require_material_identity_match:
            output[part_id]["match_type"] = match_type
    if set(output) != set(expected_by_id):
        raise PartIdQwenError("Qwen Part-ID batch does not exactly cover its input")
    return [output[str(item["part_id"])] for item in expected]


def _payload(
    *,
    model: str,
    batch: Sequence[dict[str, Any]],
    allow_color_tuning: bool,
    require_material_family_prediction: bool = False,
    corresponding_material_only: bool = False,
    entity_label: str = "exact CAD Part-ID",
    retry_detail: str | None = None,
) -> dict[str, Any]:
    if require_material_family_prediction and corresponding_material_only:
        raise PartIdQwenError(
            "exact-preset and corresponding-material selection modes are exclusive"
        )
    content: list[dict[str, Any]] = []
    for item in batch:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"VISUAL COMPARISON SHEET FOR {entity_label.upper()} "
                        f"{item['part_id']}. Tile TARGET is a reference-photo "
                        "material-core crop; non-target pixels are neutralized. "
                        "numbered tiles are the default NVIDIA Base MDL renders "
                        "listed by candidate_index."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": load_image_url(
                            item.get("comparison_sheet", item["crop"])
                        )
                    },
                },
            ]
        )
    output_shape = {
        "schema_version": (
            MATERIAL_IDENTITY_SELECTION_BATCH_SCHEMA_VERSION
            if require_material_family_prediction
            else BATCH_SCHEMA_VERSION
        ),
        "selections": [
            {
                "part_id": item["part_id"],
                "candidate_index": (
                    "one supplied integer candidate_index for this part"
                ),
                "confidence": 0.75,
                **(
                    {
                        "match_type": (
                            "EXACT_LIBRARY_MATCH or CORRESPONDING_MATERIAL"
                        )
                    }
                    if require_material_family_prediction
                    else {}
                ),
            }
            for item in batch
        ],
    }
    prompt = "\n".join(
        [
            f"Choose one NVIDIA Base MDL independently for each {entity_label}.",
            "Each TARGET is the bounded, core-masked material evidence from a globally registered CAD Part-ID. Neutral pixels are not material evidence. Use the supplied descriptor and the visible core; do not infer a material from an adjacent part or background.",
            "Do not create new material groups or copy a decision to an unrelated target.",
            (
                "Material identity is bounded by the independent first pass. Every "
                "numbered candidate satisfies the predicted substrate and optical "
                "behavior; when treatment evidence is authoritative, every candidate "
                "also satisfies that treatment. First decide whether one authored MDL "
                "preset is an exact full-appearance match, including its fixed texture "
                "and authored swatch. Use colour only for that exact-preset confirmation. "
                "If no exact preset exists, choose the supplied generic corresponding "
                "material using manufacturing type, finish, texture, highlight shape "
                "and roughness; colour must not choose among fallbacks."
                if require_material_family_prediction
                else (
                    "This is the second, corresponding-material-only pass. No exact "
                    "authored preset was confirmed in the independent first pass. "
                    "Every displayed target and candidate is grayscale. Choose the "
                    "closest physical material using substrate, manufacturing "
                    "treatment, finish, texture scale, normal structure, opacity, "
                    "roughness and highlight shape. Color is forbidden evidence."
                    if corresponding_material_only
                    else "Optimize visible similarity only. Exact engineering/semantic category may differ, but visual behavior may not: opacity versus transmission, surface continuity, texture scale, color, brightness, highlight response and roughness are mandatory constraints."
                )
            ),
            "Compare TARGET directly with the numbered actual MDL renders. The tiles, not colour words embedded in an ID, are the authority for an exact preset. Never choose a transmissive candidate for an opaque target or introduce a different substrate.",
            (
                "For color_tunable=true candidates, a Part-ID-specific H1 may "
                "replace only the reviewed MDL color input. Judge those default "
                "render tiles by surface texture, normal structure, coating, "
                "opacity and highlight response, not by their default swatch "
                "color. H1 is not guaranteed: it is accepted only if the "
                "evidence gate passes and its actual CAD render beats untouched "
                "H0. Texture, normal and coating will not be modified. For "
                "color_tunable=false candidates, default color remains mandatory."
                if allow_color_tuning
                else (
                    "Select only one supplied candidate_index for each part. "
                    "The program maps that index to the exact MDL ID."
                )
            ),
            "SigLIP2, DINO, MVInverse and compatibility scores are evidence, not an instruction to choose rank 1. The actual comparison image has final authority.",
            "Every supplied candidate is selectable. Return its integer candidate_index exactly; never write, reconstruct or guess a material_id path.",
            *(
                [
                    "Set match_type=EXACT_LIBRARY_MATCH only when the independent "
                    "physical prediction and the colour target jointly support the same "
                    "authored library preset: substrate, treatment, finish, texture and "
                    "fixed swatch must agree. Otherwise set CORRESPONDING_MATERIAL and "
                    "select the candidate whose specific_library_preset=false. A "
                    "corresponding material is a valid conservative fallback, not a "
                    "failure."
                ]
                if require_material_family_prediction
                else []
            ),
            "Valid candidate_index ranges: "
            + ", ".join(
                f"{item['part_id']}=1..{len(item['candidates'])}" for item in batch
            )
            + ".",
            "Return exactly one strict JSON object with no Markdown or prose and this exact shape: "
            + json.dumps(output_shape, ensure_ascii=False),
            "Independent Part-ID candidates: "
            + json.dumps(list(batch), ensure_ascii=False),
            *(
                [
                    "The previous response was rejected. Correct only this "
                    f"contract error and return strict JSON: {retry_detail}"
                ]
                if retry_detail
                else []
            ),
        ]
    )
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    (
                        "You are a bounded physical material classifier operating on "
                        "grayscale evidence. Select only the closest corresponding "
                        "physical material. Never use or infer color. Obey the exact "
                        "JSON contract."
                        if corresponding_material_only
                        else "You are a bounded physical material classifier. Physical identity "
                        "is decided before colour. Colour may only confirm an exact authored "
                        "preset and may never authorize a cross-family fallback. Obey the "
                        "exact JSON contract."
                    )
                ),
            },
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
        "enable_thinking": False,
    }


def _component_refinement_rgb(part: Mapping[str, Any]) -> list[float] | None:
    descriptor = part.get("descriptor")
    if not isinstance(descriptor, Mapping):
        return None
    robust = descriptor.get("robust_color_evidence")
    value = (
        robust.get("robust_reference_srgb")
        if isinstance(robust, Mapping)
        else None
    )
    if value is None:
        value = descriptor.get("median_rgb")
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not math.isfinite(float(channel))
            or not 0.0 <= float(channel) <= 1.0
            for channel in value
        )
    ):
        return None
    return [float(channel) for channel in value]


def _component_refinement_bbox(observation: Mapping[str, Any]) -> list[float] | None:
    value = observation.get("target_box_xyxy")
    if value is None:
        value = observation.get("projected_box_xyxy")
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
            for coordinate in value
        )
    ):
        return None
    left, top, right, bottom = (float(coordinate) for coordinate in value)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _component_refinement_bbox_gap(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    left_a, top_a, right_a, bottom_a = first
    left_b, top_b, right_b, bottom_b = second
    dx = max(0.0, left_a - right_b, left_b - right_a)
    dy = max(0.0, top_a - bottom_b, top_b - bottom_a)
    return float(math.hypot(dx, dy))


def _component_refinement_proximity_limit(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    sizes = (
        first[2] - first[0],
        first[3] - first[1],
        second[2] - second[0],
        second[3] - second[1],
    )
    return float(max(10.0, min(36.0, 0.06 * max(sizes))))


def _component_refinement_assembly_branch(
    prim_path: str,
    *,
    default_prim: str | None,
) -> str | None:
    parts = [part for part in prim_path.split("/") if part]
    if not parts:
        return None
    root = (
        default_prim.strip("/").split("/")[-1]
        if isinstance(default_prim, str) and default_prim.strip("/")
        else parts[0]
    )
    while parts and parts[0] == root:
        parts.pop(0)
    return parts[0] if parts else None


def _component_refinement_observation_weight(
    observation: Mapping[str, Any],
) -> float:
    weight = observation.get("camera_alignment_evidence_weight")
    pixels = observation.get("trusted_foreground_pixels")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or isinstance(pixels, bool)
        or not isinstance(pixels, int)
        or pixels < 1
    ):
        return 0.0
    return float(
        max(0.0, min(1.0, float(weight)))
        * min(1.0, math.log2(max(2, pixels)) / 12.0)
    )


def _refine_component_memberships_with_final_evidence(
    *,
    appearance_components: Mapping[str, Any] | None,
    component_members: Mapping[str, Sequence[str]],
    part_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Expand sealed component seeds using final Part-ID evidence.

    The early appearance-component pass is deliberately conservative and runs
    before local Part-ID evidence is available.  A split housing panel can
    therefore be marked independent even when later evidence proves that it is
    the same continuous coating.  This late pass never invents a component: it
    only expands an existing photo-supported seed when final colour, physical
    surface type, CAD assembly branch, and multi-view spatial evidence all
    agree.  Ambiguous candidates remain independent.
    """

    result = {
        component_id: sorted(set(members))
        for component_id, members in component_members.items()
    }
    disabled = {
        "schema_version": "qwen-final-evidence-component-refinement/v1",
        "status": "NOT_AVAILABLE",
        "components": [],
        "ambiguous_part_ids": [],
        "summary": {
            "source_component_count": len(result),
            "source_member_count": sum(len(members) for members in result.values()),
            "added_member_count": 0,
            "refined_member_count": sum(len(members) for members in result.values()),
        },
    }
    if appearance_components is None or not result:
        return result, disabled
    inputs = appearance_components.get("inputs")
    registry_value = (
        inputs.get("rendered_registry") if isinstance(inputs, Mapping) else None
    )
    registry_sha256 = (
        inputs.get("rendered_registry_sha256")
        if isinstance(inputs, Mapping)
        else None
    )
    if not isinstance(registry_value, str) or not isinstance(registry_sha256, str):
        return result, disabled
    try:
        registry_path = Path(registry_value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PartIdQwenError(
            f"appearance-component rendered registry is unavailable: {exc}"
        ) from exc
    if not registry_path.is_file() or _sha256_file(registry_path) != registry_sha256:
        raise PartIdQwenError(
            "appearance-component rendered registry failed its file seal"
        )
    registry = _read(registry_path, "appearance-component rendered registry")
    raw_registry_parts = registry.get("parts")
    if not isinstance(raw_registry_parts, list):
        raise PartIdQwenError("appearance-component rendered registry has no parts")
    registry_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_registry_parts:
        part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
        prim_path = raw.get("prim_path") if isinstance(raw, Mapping) else None
        if (
            not isinstance(part_id, str)
            or not isinstance(prim_path, str)
            or part_id in registry_by_id
        ):
            raise PartIdQwenError(
                "appearance-component rendered registry has invalid Part IDs"
            )
        registry_by_id[part_id] = raw
    if not set(part_by_id) <= set(registry_by_id):
        raise PartIdQwenError(
            "final Part-ID evidence is not covered by the rendered registry"
        )
    raw_components = appearance_components.get("components")
    component_metadata = {
        str(raw["component_id"]): raw
        for raw in raw_components
        if isinstance(raw, Mapping) and isinstance(raw.get("component_id"), str)
    } if isinstance(raw_components, list) else {}
    default_prim = registry.get("default_prim")
    assigned = {
        part_id for members in result.values() for part_id in members
    }
    candidate_matches: dict[str, list[dict[str, Any]]] = {}
    for part_id in sorted(set(part_by_id) - assigned):
        part = part_by_id[part_id]
        descriptor = part.get("descriptor")
        robust = (
            descriptor.get("robust_color_evidence")
            if isinstance(descriptor, Mapping)
            else None
        )
        sample_count = robust.get("sample_count") if isinstance(robust, Mapping) else None
        inlier_fraction = (
            robust.get("inlier_fraction") if isinstance(robust, Mapping) else None
        )
        candidate_rgb = _component_refinement_rgb(part)
        candidate_surface = (
            descriptor.get("surface_class")
            if isinstance(descriptor, Mapping)
            else None
        )
        registry_row = registry_by_id[part_id]
        candidate_branch = _component_refinement_assembly_branch(
            str(registry_row["prim_path"]),
            default_prim=default_prim if isinstance(default_prim, str) else None,
        )
        if (
            candidate_rgb is None
            or not isinstance(candidate_surface, str)
            or not candidate_surface
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < MINIMUM_COMPONENT_REFINEMENT_PIXELS
            or isinstance(inlier_fraction, bool)
            or not isinstance(inlier_fraction, (int, float))
            or float(inlier_fraction)
            < MINIMUM_COMPONENT_REFINEMENT_INLIER_FRACTION
            or candidate_branch is None
        ):
            continue
        matches: list[dict[str, Any]] = []
        for component_id, members in sorted(result.items()):
            metadata = component_metadata.get(component_id)
            canonical_rgb = (
                metadata.get("canonical_reference_rgb")
                if isinstance(metadata, Mapping)
                else None
            )
            if (
                not isinstance(canonical_rgb, list)
                or len(canonical_rgb) != 3
                or any(
                    isinstance(channel, bool)
                    or not isinstance(channel, (int, float))
                    for channel in canonical_rgb
                )
            ):
                continue
            seed_surfaces = {
                str(seed_descriptor["surface_class"])
                for member in members
                for seed_descriptor in [part_by_id[member].get("descriptor")]
                if isinstance(seed_descriptor, Mapping)
                and isinstance(seed_descriptor.get("surface_class"), str)
            }
            if seed_surfaces != {candidate_surface}:
                continue
            member_branches = {
                branch
                for member in members
                for branch in [
                    _component_refinement_assembly_branch(
                        str(registry_by_id[member]["prim_path"]),
                        default_prim=(
                            default_prim if isinstance(default_prim, str) else None
                        ),
                    )
                ]
                if branch is not None
            }
            if candidate_branch not in member_branches:
                continue
            delta_e = srgb_delta_e(candidate_rgb, canonical_rgb)
            if delta_e > MAXIMUM_COMPONENT_REFINEMENT_DELTA_E:
                continue
            best_by_view: dict[str, tuple[float, str, float]] = {}
            for candidate_observation in part.get("observations", []):
                if not isinstance(candidate_observation, Mapping):
                    continue
                view_id = candidate_observation.get("view_id")
                candidate_bbox = _component_refinement_bbox(candidate_observation)
                if not isinstance(view_id, str) or candidate_bbox is None:
                    continue
                for member in members:
                    for member_observation in part_by_id[member].get(
                        "observations", []
                    ):
                        if (
                            not isinstance(member_observation, Mapping)
                            or member_observation.get("view_id") != view_id
                        ):
                            continue
                        member_bbox = _component_refinement_bbox(member_observation)
                        if member_bbox is None:
                            continue
                        gap = _component_refinement_bbox_gap(
                            candidate_bbox, member_bbox
                        )
                        if gap > _component_refinement_proximity_limit(
                            candidate_bbox, member_bbox
                        ):
                            continue
                        support = min(
                            _component_refinement_observation_weight(
                                candidate_observation
                            ),
                            _component_refinement_observation_weight(
                                member_observation
                            ),
                        )
                        current = best_by_view.get(view_id)
                        if current is None or (support, member, -gap) > (
                            current[0], current[1], -current[2]
                        ):
                            best_by_view[view_id] = (support, member, gap)
            spatial_support = sum(row[0] for row in best_by_view.values())
            if spatial_support < MINIMUM_COMPONENT_REFINEMENT_SPATIAL_SUPPORT:
                continue
            matches.append(
                {
                    "component_id": component_id,
                    "assembly_branch": candidate_branch,
                    "surface_class": candidate_surface,
                    "sample_count": sample_count,
                    "inlier_fraction": round(float(inlier_fraction), 8),
                    "color_delta_e": round(float(delta_e), 8),
                    "spatial_support": round(float(spatial_support), 8),
                    "supporting_views": {
                        view_id: {
                            "support": round(values[0], 8),
                            "adjacent_member_part_id": values[1],
                            "bbox_gap_px": round(values[2], 8),
                        }
                        for view_id, values in sorted(best_by_view.items())
                    },
                }
            )
        if matches:
            candidate_matches[part_id] = matches
    additions: list[dict[str, Any]] = []
    ambiguous: list[str] = []
    for part_id, matches in sorted(candidate_matches.items()):
        if len(matches) != 1:
            ambiguous.append(part_id)
            continue
        match = matches[0]
        component_id = str(match["component_id"])
        result[component_id] = sorted([*result[component_id], part_id])
        additions.append({"part_id": part_id, **match})
    component_audits = []
    additions_by_component = {
        component_id: [
            row for row in additions if row["component_id"] == component_id
        ]
        for component_id in result
    }
    for component_id, members in sorted(result.items()):
        source_members = sorted(set(component_members[component_id]))
        component_audits.append(
            {
                "component_id": component_id,
                "source_member_part_ids": source_members,
                "refined_member_part_ids": list(members),
                "added_members": additions_by_component[component_id],
            }
        )
    return (
        result,
        {
            "schema_version": "qwen-final-evidence-component-refinement/v1",
            "status": "COMPLETED",
            "rendered_registry": str(registry_path),
            "rendered_registry_sha256": registry_sha256,
            "policy": {
                "existing_photo_supported_seed_required": True,
                "final_part_id_evidence_required": True,
                "same_cad_assembly_branch_required": True,
                "same_physical_surface_class_required": True,
                "minimum_sample_count": MINIMUM_COMPONENT_REFINEMENT_PIXELS,
                "minimum_inlier_fraction": (
                    MINIMUM_COMPONENT_REFINEMENT_INLIER_FRACTION
                ),
                "maximum_color_delta_e": MAXIMUM_COMPONENT_REFINEMENT_DELTA_E,
                "minimum_multiview_spatial_support": (
                    MINIMUM_COMPONENT_REFINEMENT_SPATIAL_SUPPORT
                ),
                "ambiguous_candidates_remain_independent": True,
            },
            "components": component_audits,
            "ambiguous_part_ids": ambiguous,
            "summary": {
                "source_component_count": len(result),
                "source_member_count": sum(
                    len(members) for members in component_members.values()
                ),
                "added_member_count": len(additions),
                "refined_member_count": sum(
                    len(members) for members in result.values()
                ),
            },
        },
    )


def _validated_appearance_component_memberships(
    appearance_components: Mapping[str, Any] | None,
    *,
    observed_part_ids: set[str],
) -> dict[str, list[str]]:
    if appearance_components is None:
        return {}
    raw_components = appearance_components.get("components")
    if not isinstance(raw_components, list):
        raise PartIdQwenError("appearance-components document has no components")
    result: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for index, raw in enumerate(raw_components):
        component_id = raw.get("component_id") if isinstance(raw, Mapping) else None
        members = raw.get("member_part_ids") if isinstance(raw, Mapping) else None
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in result
            or not isinstance(members, list)
            or len(members) < 2
            or not all(isinstance(part_id, str) for part_id in members)
            or len(set(members)) != len(members)
        ):
            raise PartIdQwenError(
                f"appearance component {index} has an invalid identity scope"
            )
        member_set = set(members)
        if not member_set <= observed_part_ids:
            raise PartIdQwenError(
                f"appearance component {component_id} contains an unobserved Part-ID"
            )
        if assigned & member_set:
            raise PartIdQwenError("appearance components overlap in Part-ID identity")
        assigned.update(member_set)
        result[component_id] = sorted(member_set)
    return result


def _apply_component_identity_consensus(
    *,
    selections: Sequence[Mapping[str, Any]],
    component_members: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {
        str(row["part_id"]): dict(row)
        for row in selections
        if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
    }
    audits: list[dict[str, Any]] = []
    for component_id, members in sorted(component_members.items()):
        if not set(members) <= set(by_id):
            raise PartIdQwenError(
                f"appearance component {component_id} is missing a member selection"
            )
        votes: dict[str, tuple[int, float]] = {}
        before: dict[str, str] = {}
        for part_id in members:
            row = by_id[part_id]
            material_id = row.get("material_id")
            confidence = row.get("confidence")
            if not isinstance(material_id, str) or not isinstance(
                confidence, (int, float)
            ):
                raise PartIdQwenError(
                    f"appearance component {component_id} has an invalid selection"
                )
            before[part_id] = material_id
            count, total = votes.get(material_id, (0, 0.0))
            votes[material_id] = (count + 1, total + float(confidence))
        protected_exact_ids = {
            str(by_id[part_id]["material_id"])
            for part_id in members
            if by_id[part_id].get("match_type") == "EXACT_LIBRARY_MATCH"
            and float(by_id[part_id].get("confidence", 0.0))
            >= MINIMUM_COMPONENT_EXACT_PRESET_CONFIDENCE
        }
        protected_exact_support = {
            material_id: sorted(
                part_id
                for part_id in members
                if by_id[part_id].get("match_type") == "EXACT_LIBRARY_MATCH"
                and float(by_id[part_id].get("confidence", 0.0))
                >= MINIMUM_COMPONENT_EXACT_PRESET_CONFIDENCE
                and str(by_id[part_id]["material_id"]) == material_id
            )
            for material_id in protected_exact_ids
        }
        if len(protected_exact_ids) == 1:
            winner = next(iter(protected_exact_ids))
            consensus_match_type = "EXACT_LIBRARY_MATCH"
            consensus_mode = "PROTECTED_EXACT_PRESET_PROPAGATED"
            consensus_applied = True
        elif len(protected_exact_ids) > 1:
            winner = None
            consensus_match_type = None
            consensus_mode = "CONFLICTING_EXACT_PRESETS_PRESERVED"
            consensus_applied = False
        else:
            winner = min(
                votes,
                key=lambda material_id: (
                    -votes[material_id][0],
                    -votes[material_id][1],
                    material_id,
                ),
            )
            consensus_match_type = "CORRESPONDING_MATERIAL"
            consensus_mode = "GENERIC_CORRESPONDING_CONSENSUS"
            consensus_applied = True
        for part_id in members:
            row = by_id[part_id]
            row["pre_component_consensus_material_id"] = row["material_id"]
            row["pre_component_consensus_match_type"] = row.get("match_type")
            row["component_id"] = component_id
            row["component_identity_consensus_applied"] = consensus_applied
            if consensus_applied:
                was_protected_source = (
                    consensus_mode == "PROTECTED_EXACT_PRESET_PROPAGATED"
                    and part_id in protected_exact_support.get(str(winner), [])
                )
                row["material_id"] = winner
                row["match_type"] = consensus_match_type
                if consensus_mode == "PROTECTED_EXACT_PRESET_PROPAGATED":
                    row["component_exact_preset_source"] = was_protected_source
                    row["component_exact_preset_source_part_ids"] = list(
                        protected_exact_support[str(winner)]
                    )
                    if not was_protected_source:
                        row["index_resolution"] = (
                            "component_exact_preset_propagation"
                        )
                        row["selection_authority"] = (
                            "appearance_component_protected_exact_preset"
                        )
        audits.append(
            {
                "component_id": component_id,
                "member_part_ids": list(members),
                "member_material_ids_before_consensus": before,
                "selected_material_id": winner,
                "match_type": consensus_match_type,
                "vote_count": votes[winner][0] if winner is not None else 0,
                "member_count": len(members),
                "consensus_mode": consensus_mode,
                "protected_exact_material_ids": sorted(protected_exact_ids),
                "protected_exact_support": protected_exact_support,
                "exact_shared_material_enforced": consensus_applied,
            }
        )
    return (
        [by_id[str(row["part_id"])] for row in selections],
        {
            "schema_version": "qwen-component-material-identity-consensus/v2",
            "components": audits,
            "summary": {
                "component_count": len(audits),
                "constrained_part_count": sum(
                    len(row["member_part_ids"]) for row in audits
                ),
                "all_components_share_one_exact_mdl": all(
                    len(
                        {
                            by_id[part_id]["material_id"]
                            for part_id in row["member_part_ids"]
                        }
                    )
                    == 1
                    for row in audits
                ),
            },
        },
    )


def run_part_id_qwen_rerank(
    *,
    evidence: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    catalog: Mapping[str, Any],
    runner: Any,
    model: str,
    output_dir: Path,
    batch_size: int = 4,
    candidate_count: int = 4,
    allow_color_tuning: bool = False,
    require_material_family_prediction: bool = False,
    appearance_components: Mapping[str, Any] | None = None,
    entity_label: str = "exact CAD Part-ID",
) -> dict[str, Any]:
    """Rerank each Part-ID shortlist without any palette-group decision."""

    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise PartIdQwenError("unsupported Part-ID evidence schema")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > 8
    ):
        raise PartIdQwenError("batch_size must be from 1 to 8")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 2
        or candidate_count > 8
    ):
        raise PartIdQwenError("candidate_count must be from 2 to 8")
    if not isinstance(entity_label, str) or not entity_label.strip():
        raise PartIdQwenError("entity_label must be a non-empty string")
    if not isinstance(require_material_family_prediction, bool):
        raise PartIdQwenError("require_material_family_prediction must be boolean")
    if require_material_family_prediction and allow_color_tuning:
        raise PartIdQwenError(
            "material-identity-first selection cannot tune colour in the identity stage"
        )
    parts = evidence.get("parts")
    groups = retrieval.get("groups")
    if not isinstance(parts, list) or not isinstance(groups, list):
        raise PartIdQwenError("Part-ID evidence/retrieval inputs are invalid")
    part_by_id = {
        str(part["part_id"]): part
        for part in parts
        if isinstance(part, Mapping)
        and part.get("status") == "observed"
        and isinstance(part.get("part_id"), str)
    }
    retrieval_by_id = {
        str(group["group_id"]): group
        for group in groups
        if isinstance(group, Mapping) and isinstance(group.get("group_id"), str)
    }
    if set(part_by_id) != set(retrieval_by_id):
        raise PartIdQwenError(
            "Qwen rerank inputs do not exactly cover the same observed Part IDs"
        )
    catalog_by_id = _catalog_by_id(catalog)
    bank_context = _observation_bank_context(retrieval)
    bank_root = bank_context[0] if bank_context is not None else None
    profiles_by_id = bank_context[1] if bank_context is not None else {}
    material_root_value = (
        retrieval.get("catalog", {}).get("material_root")
        if isinstance(retrieval.get("catalog"), Mapping)
        else None
    )
    try:
        material_root = (
            Path(material_root_value).expanduser().resolve(strict=True)
            if isinstance(material_root_value, str)
            else None
        )
    except OSError:
        material_root = None
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    identity_evidence_dir = output_dir / "identity_evidence_grayscale"
    material_predictions: list[dict[str, Any]] = []
    material_prediction_batches: list[dict[str, Any]] = []
    component_members = _validated_appearance_component_memberships(
        appearance_components if require_material_family_prediction else None,
        observed_part_ids=set(part_by_id),
    )
    component_members, component_membership_refinement = (
        _refine_component_memberships_with_final_evidence(
            appearance_components=(
                appearance_components
                if require_material_family_prediction
                else None
            ),
            component_members=component_members,
            part_by_id=part_by_id,
        )
    )
    part_to_component = {
        part_id: component_id
        for component_id, members in component_members.items()
        for part_id in members
    }
    if require_material_family_prediction:
        prediction_items: list[dict[str, Any]] = []

        def prediction_views(part_id: str, *, maximum: int) -> list[dict[str, str]]:
            part = part_by_id[part_id]
            selected_observations = [
                observation
                for observation in part["observations"]
                if observation.get("selected_for_material_inference") is True
            ]
            if len(selected_observations) != 1:
                raise PartIdQwenError(
                    f"Part-ID {part_id} does not have exactly one selected observation"
                )
            observation = selected_observations[0]
            trusted_views = [
                {
                    "view_id": str(candidate["view_id"]),
                    "crop": candidate.get("isolated_crop", candidate["crop"]),
                }
                for candidate in part["observations"]
                if isinstance(candidate, Mapping)
                and isinstance(candidate.get("view_id"), str)
                and isinstance(
                    candidate.get("isolated_crop", candidate.get("crop")), str
                )
            ]
            if not trusted_views:
                raise PartIdQwenError(
                    f"Part-ID {part_id} has no trusted material-prediction crops"
                )
            selected_view_id = str(observation["view_id"])
            trusted_views.sort(
                key=lambda row: (
                    0 if row["view_id"] == selected_view_id else 1,
                    row["view_id"],
                )
            )
            trusted_views = trusted_views[:maximum]
            grayscale_views: list[dict[str, str]] = []
            for index, row in enumerate(trusted_views, start=1):
                output = identity_evidence_dir / (
                    f"{part_id}_{row['view_id']}_{index:02d}.png"
                )
                _write_grayscale_identity_crop(Path(row["crop"]), output)
                grayscale_views.append(
                    {"view_id": row["view_id"], "crop": str(output)}
                )
            return grayscale_views

        for component_id, members in sorted(component_members.items()):
            representative_members = list(members)[:MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH]
            trusted_views = [
                prediction_views(part_id, maximum=1)[0]
                for part_id in representative_members
            ]
            prediction_items.append(
                {
                    "part_id": component_id,
                    "views": trusted_views,
                    "descriptor": {
                        "prediction_scope": "appearance_component",
                        "member_part_ids": list(members),
                        "member_descriptors": {
                            part_id: part_by_id[part_id].get("descriptor", {})
                            for part_id in representative_members
                        },
                    },
                }
            )
        for part_id in sorted(set(part_by_id) - set(part_to_component)):
            prediction_items.append(
                {
                    "part_id": part_id,
                    "views": prediction_views(
                        part_id, maximum=MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH
                    ),
                    "descriptor": part_by_id[part_id].get("descriptor", {}),
                }
            )
        target_predictions, material_prediction_batches = _predict_material_families(
            items=prediction_items,
            runner=runner,
            model=model,
            identity_options=_catalog_identity_options(catalog_by_id),
            batch_size=batch_size,
            raw_dir=raw_dir,
        )
        target_prediction_by_id = {
            str(row["part_id"]): row for row in target_predictions
        }
        for component_id, members in sorted(component_members.items()):
            prediction = target_prediction_by_id[component_id]
            for part_id in members:
                material_predictions.append(
                    {
                        **prediction,
                        "part_id": part_id,
                        "prediction_scope": "appearance_component",
                        "component_id": component_id,
                        "component_member_part_ids": list(members),
                    }
                )
        for part_id in sorted(set(part_by_id) - set(part_to_component)):
            material_predictions.append(
                {
                    **target_prediction_by_id[part_id],
                    "prediction_scope": "independent_part_id",
                    "component_id": None,
                }
            )
        material_predictions.sort(key=lambda row: str(row["part_id"]))
    prediction_by_id = {str(row["part_id"]): row for row in material_predictions}
    jobs: list[dict[str, Any]] = []
    direct_selections: list[dict[str, Any]] = []
    direct_assignment_audits: list[dict[str, Any]] = []
    gate_audits: list[dict[str, Any]] = []
    for part_id in sorted(part_by_id):
        part = part_by_id[part_id]
        selected_observations = [
            observation
            for observation in part["observations"]
            if observation.get("selected_for_material_inference") is True
        ]
        if len(selected_observations) != 1:
            raise PartIdQwenError(
                f"Part-ID {part_id} does not have exactly one selected observation"
            )
        ranking = retrieval_by_id[part_id].get("fused_ranking")
        if not isinstance(ranking, list) or len(ranking) < 2:
            raise PartIdQwenError(f"Part-ID {part_id} has no candidate ranking")
        prediction = prediction_by_id.get(part_id)
        if require_material_family_prediction:
            if not isinstance(prediction, Mapping):
                raise PartIdQwenError(
                    f"Part-ID {part_id} has no material-identity prediction"
                )
            ranking = _identity_filtered_ranking(
                ranking=ranking,
                catalog_by_id=catalog_by_id,
                prediction=prediction,
            )
            component_id = prediction.get("component_id")
            if isinstance(component_id, str):
                members = component_members.get(component_id)
                if not members:
                    raise PartIdQwenError(
                        f"Part-ID {part_id} cites an unknown appearance component"
                    )
                identity_descriptor: Mapping[str, Any] | None = {
                    "member_descriptors": {
                        member_id: part_by_id[member_id].get("descriptor", {})
                        for member_id in members
                    }
                }
            else:
                raw_descriptor = part.get("descriptor")
                identity_descriptor = (
                    raw_descriptor
                    if isinstance(raw_descriptor, Mapping)
                    else None
                )
            ranking = _rank_identity_candidates_with_pbr(
                ranking,
                descriptor=identity_descriptor,
                profiles_by_id=profiles_by_id,
            )
        direct_match = (
            _direct_exact_library_match(ranking, prediction=prediction)
            if require_material_family_prediction
            and isinstance(prediction, Mapping)
            else None
        )
        target = _target_appearance(selected_observations[0])
        if require_material_family_prediction:
            shortlist = _identity_shortlist(
                ranking,
                candidate_count=candidate_count,
                required_material_ids=(
                    [str(direct_match["material_id"])]
                    if direct_match is not None
                    else []
                ),
            )
        else:
            shortlist = _compatibility_shortlist(
                ranking=ranking,
                catalog_by_id=catalog_by_id,
                profiles_by_id=profiles_by_id,
                target=target,
                candidate_count=candidate_count,
                allow_color_tuning=allow_color_tuning,
            )
        if len(shortlist) < (1 if require_material_family_prediction else 2):
            raise PartIdQwenError(
                f"Part-ID {part_id} has too few visually compatible candidates"
            )
        if require_material_family_prediction:
            library_gap_fallback = any(
                row.get("library_gap_fallback") is True for row in shortlist
            )
        else:
            shortlist, library_gap_fallback = _promote_library_gap_candidates(
                shortlist,
                candidate_count=candidate_count,
                target=target,
                allow_color_tuning=allow_color_tuning,
            )
        allowed_material_ids = [
            str(row["material_id"])
            for row in shortlist
            if row.get("selection_allowed") is True
        ]
        if not allowed_material_ids:
            raise PartIdQwenError(
                "material_library_gap: Part-ID "
                f"{part_id} has no NVIDIA Base MDL satisfying its color, "
                "opacity and texture constraints"
            )
        candidates = []
        for row in shortlist:
            # Blocked candidates remain in visual_compatibility_gate below for
            # audit, but are never shown to the VLM.  A forbidden choice
            # cannot improve an unattended decision and only creates an
            # avoidable structured-output failure.
            if row.get("selection_allowed") is not True:
                continue
            candidate_index = len(candidates) + 1
            material_id = row.get("material_id") if isinstance(row, Mapping) else None
            if not isinstance(material_id, str) or material_id not in catalog_by_id:
                raise PartIdQwenError(
                    f"Part-ID {part_id} cites an unknown material candidate"
                )
            candidate_rgb = _median_profile_rgb(profiles_by_id.get(material_id, {}))
            exact_preset_color_delta_e = (
                srgb_delta_e(target["median_rgb"], candidate_rgb)
                if target is not None and candidate_rgb is not None
                else None
            )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    **_candidate_summary(
                        row,
                        catalog_by_id[material_id],
                        withhold_color=False,
                    ),
                    "original_retrieval_rank": row.get("original_retrieval_rank"),
                    "compatibility_rank": row.get("compatibility_rank"),
                    "visual_compatibility_score": row.get("visual_compatibility_score"),
                    "appearance_median_rgb": row.get("appearance_median_rgb"),
                    "color_similarity": row.get("color_similarity"),
                    "hue_similarity": row.get("hue_similarity"),
                    "color_delta_e": row.get("color_delta_e"),
                    "color_tunable": row.get("color_tunable"),
                    "color_gate_passed": row.get("color_gate_passed"),
                    "texture_similarity": row.get("texture_similarity"),
                    "texture_gradient_energy": row.get("texture_gradient_energy"),
                    "transmission_risk": row.get("transmission_risk"),
                    "texture_mismatch_risk": row.get("texture_mismatch_risk"),
                    "intrinsic_pattern_risk": row.get("intrinsic_pattern_risk"),
                    "selection_allowed": row.get("selection_allowed"),
                    "selection_allowed_by_default_constraints": row.get(
                        "selection_allowed_by_default_constraints"
                    ),
                    "library_gap_fallback": row.get("library_gap_fallback"),
                    "library_gap_fallback_tier": row.get("library_gap_fallback_tier"),
                    "relaxed_constraints": row.get("relaxed_constraints"),
                    "conditional_h1_evaluation": row.get("conditional_h1_evaluation"),
                    "identity_match_tier": row.get("identity_match_tier"),
                    "color_evidence_used": row.get("color_evidence_used"),
                    "color_evidence_scope": row.get("color_evidence_scope"),
                    "specific_library_preset": row.get(
                        "specific_library_preset", False
                    ),
                    "material_species": row.get(
                        "material_species", "unknown"
                    ),
                    "exact_authored_preset_candidate": row.get(
                        "exact_authored_preset_candidate", False
                    ),
                    "generic_identity_material_id": row.get(
                        "generic_identity_material_id", material_id
                    ),
                    "physical_pbr_similarity": row.get(
                        "physical_pbr_similarity"
                    ),
                    "physical_pbr_mean_error": row.get(
                        "physical_pbr_mean_error"
                    ),
                    "authored_preset_median_rgb": (
                        [round(float(value), 8) for value in candidate_rgb]
                        if candidate_rgb is not None
                        else None
                    ),
                    "exact_preset_color_delta_e": (
                        round(float(exact_preset_color_delta_e), 8)
                        if exact_preset_color_delta_e is not None
                        else None
                    ),
                    "exact_preset_color_gate_passed": (
                        exact_preset_color_delta_e
                        <= EXACT_LIBRARY_PRESET_MAXIMUM_DELTA_E
                        if exact_preset_color_delta_e is not None
                        else None
                    ),
                }
            )
        job = {
            "part_id": part_id,
            # The annotated correspondence crop is intentionally not a VLM
            # material target. It may contain other parts inside a valid
            # tolerant box. Prefer the neutralized exact-core crop emitted by
            # Part-ID evidence, retaining the legacy crop only for old sealed
            # checkpoints.
            "crop": selected_observations[0].get(
                "isolated_crop", selected_observations[0]["crop"]
            ),
            "descriptor": part["descriptor"],
            "target_appearance": target,
            "candidates": candidates,
            "library_gap_fallback": library_gap_fallback,
            "material_prediction": prediction,
        }
        if direct_match is None:
            jobs.append(job)
        else:
            direct_material_id = str(direct_match["material_id"])
            candidate_index = next(
                (
                    int(candidate["candidate_index"])
                    for candidate in candidates
                    if candidate["material_id"] == direct_material_id
                ),
                None,
            )
            if candidate_index is None:
                raise PartIdQwenError(
                    f"direct exact match for {part_id} is absent from the "
                    "physical shortlist"
                )
            direct_selections.append(
                {
                    "part_id": part_id,
                    "material_id": direct_material_id,
                    "candidate_index": candidate_index,
                    "requested_candidate_index": None,
                    "index_resolution": "direct_exact_library_match",
                    "confidence": float(prediction["confidence"]),
                    "match_type": "EXACT_LIBRARY_MATCH",
                    "selection_authority": direct_match["authority"],
                }
            )
            direct_assignment_audits.append(
                {
                    "part_id": part_id,
                    **direct_match,
                }
            )
        gate_audits.append(
            {
                "part_id": part_id,
                "target_appearance": target,
                "library_gap_fallback": library_gap_fallback,
                "material_prediction": prediction,
                "direct_exact_library_assignment": direct_match,
                "authorized_catalog_family": (
                    prediction.get("catalog_family")
                    if isinstance(prediction, Mapping)
                    and prediction.get("status") == "APPLYABLE"
                    else None
                ),
                "authorized_physical_identity": (
                    {
                        "physical_substrate": prediction.get(
                            "physical_substrate"
                        ),
                        "material_species": prediction.get(
                            "material_species"
                        ),
                        "surface_treatment": prediction.get("surface_treatment"),
                        "optical_behavior": prediction.get("optical_behavior"),
                        "surface_finish": prediction.get("surface_finish"),
                    }
                    if isinstance(prediction, Mapping)
                    and prediction.get("status") == "APPLYABLE"
                    else None
                ),
                "authorized_material_ids": [
                    str(row["material_id"])
                    for row in shortlist
                    if row.get("selection_allowed") is True
                ],
                "input_candidate_count": len(ranking),
                "shortlist": [
                    {
                        "material_id": row["material_id"],
                        "original_retrieval_rank": row["original_retrieval_rank"],
                        "compatibility_rank": row["compatibility_rank"],
                        "visual_compatibility_score": row["visual_compatibility_score"],
                        "color_delta_e": row["color_delta_e"],
                        "color_tunable": row["color_tunable"],
                        "color_gate_passed": row["color_gate_passed"],
                        "transmission_risk": row["transmission_risk"],
                        "texture_mismatch_risk": row["texture_mismatch_risk"],
                        "selection_allowed": row["selection_allowed"],
                        "selection_allowed_by_default_constraints": row[
                            "selection_allowed_by_default_constraints"
                        ],
                        "library_gap_fallback": row["library_gap_fallback"],
                        "library_gap_fallback_tier": row["library_gap_fallback_tier"],
                        "relaxed_constraints": row["relaxed_constraints"],
                        "conditional_h1_evaluation": row["conditional_h1_evaluation"],
                        "identity_match_tier": row.get("identity_match_tier"),
                        "color_evidence_used": row.get("color_evidence_used"),
                        "color_evidence_scope": row.get(
                            "color_evidence_scope"
                        ),
                        "specific_library_preset": row.get(
                            "specific_library_preset", False
                        ),
                        "material_species": row.get(
                            "material_species", "unknown"
                        ),
                        "exact_authored_preset_candidate": row.get(
                            "exact_authored_preset_candidate", False
                        ),
                        "generic_identity_material_id": row.get(
                            "generic_identity_material_id", row["material_id"]
                        ),
                        "physical_pbr_similarity": row.get(
                            "physical_pbr_similarity"
                        ),
                        "physical_pbr_mean_error": row.get(
                            "physical_pbr_mean_error"
                        ),
                        "physical_pbr_evidence": row.get(
                            "physical_pbr_evidence", {}
                        ),
                        "library_pbr_fingerprint": row.get(
                            "library_pbr_fingerprint", {}
                        ),
                    }
                    for row in shortlist
                ],
            }
        )

    comparison_dir = output_dir / "comparison_sheets"
    if bank_context is not None or material_root is not None:
        for job in jobs:
            render_paths = {
                str(candidate["material_id"]): _candidate_render(
                    material_id=str(candidate["material_id"]),
                    profile=profiles_by_id.get(str(candidate["material_id"])),
                    bank_root=bank_root,
                    catalog_record=catalog_by_id[str(candidate["material_id"])],
                    material_root=material_root,
                )
                for candidate in job["candidates"]
            }
            sheet = _comparison_sheet(
                part_id=str(job["part_id"]),
                crop=Path(str(job["crop"])).expanduser().resolve(strict=True),
                candidates=job["candidates"],
                render_paths=render_paths,
                output=comparison_dir / f"{job['part_id']}.png",
                grayscale=False,
            )
            job["comparison_sheet"] = str(sheet)
    def run_selection_batches(
        stage_jobs: Sequence[dict[str, Any]],
        *,
        stage: str,
        require_match_type: bool,
        corresponding_only: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stage_selections: list[dict[str, Any]] = []
        stage_audits: list[dict[str, Any]] = []
        stage_batches = [
            stage_jobs[index : index + batch_size]
            for index in range(0, len(stage_jobs), batch_size)
        ]
        for batch_index, batch in enumerate(stage_batches, start=1):
            final_error: Exception | None = None
            for attempt in range(1, 3):
                payload = _payload(
                    model=model,
                    batch=batch,
                    allow_color_tuning=allow_color_tuning,
                    require_material_family_prediction=require_match_type,
                    corresponding_material_only=corresponding_only,
                    entity_label=entity_label,
                    retry_detail=(
                        str(final_error) if final_error is not None else None
                    ),
                )
                generated = runner.generate_with_metadata(payload)
                artifact_stem = (
                    f"{stage}_batch_{batch_index:03d}_attempt_{attempt}"
                )
                raw_path = raw_dir / f"{artifact_stem}.txt"
                raw_path.write_text(generated.text, encoding="utf-8")
                try:
                    document, parse_audit = parse_plan_content_with_audit(
                        generated.text
                    )
                    validated = _validate_batch(
                        document,
                        expected=batch,
                        require_material_identity_match=require_match_type,
                    )
                except Exception as exc:
                    final_error = exc
                    _write(
                        raw_dir / f"{artifact_stem}.parse.json",
                        {
                            "status": "invalid",
                            "error": str(exc),
                            "generation": generated.metadata(),
                        },
                    )
                    continue
                stage_selections.extend(dict(row) for row in validated)
                audit_path = _write(
                    raw_dir / f"{artifact_stem}.parse.json",
                    {
                        **parse_audit,
                        "status": "valid",
                        "generation": generated.metadata(),
                    },
                )
                stage_audits.append(
                    {
                        "stage": stage,
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "part_ids": [str(item["part_id"]) for item in batch],
                        "parse_audit": str(audit_path),
                    }
                )
                print(
                    f"[PART-ID QWEN {stage.upper()}] "
                    f"{batch_index}/{len(stage_batches)} "
                    f"parts={','.join(item['part_id'] for item in batch)}",
                    flush=True,
                )
                break
            else:
                raise PartIdQwenError(
                    f"Qwen Part-ID {stage} batch {batch_index} failed twice: "
                    f"{final_error}"
                )
        return stage_selections, stage_audits

    exact_preset_qwen_selections, exact_preset_batch_audits = (
        run_selection_batches(
            jobs,
            stage=(
                "exact_preset" if require_material_family_prediction else "selection"
            ),
            require_match_type=require_material_family_prediction,
            corresponding_only=False,
        )
    )
    corresponding_qwen_selections: list[dict[str, Any]] = []
    corresponding_batch_audits: list[dict[str, Any]] = []
    corresponding_jobs: list[dict[str, Any]] = []
    qwen_selections = [dict(row) for row in exact_preset_qwen_selections]
    if require_material_family_prediction:
        initial_by_id = {
            str(row["part_id"]): row for row in exact_preset_qwen_selections
        }
        job_by_id = {str(job["part_id"]): job for job in jobs}
        corresponding_comparison_dir = output_dir / "comparison_sheets_corresponding"
        for part_id, initial in sorted(initial_by_id.items()):
            if initial.get("match_type") != "CORRESPONDING_MATERIAL":
                initial["selection_authority"] = (
                    "deterministic_material_identity_and_preset_gate"
                    if initial.get("index_resolution")
                    == "deterministic_exact_authored_preset_promotion"
                    else "color_confirmed_exact_authored_preset"
                )
                continue
            source_job = job_by_id[part_id]
            generic_candidates: list[dict[str, Any]] = []
            for candidate in source_job["candidates"]:
                if candidate.get("specific_library_preset") is True:
                    continue
                sanitized = {
                    key: value
                    for key, value in candidate.items()
                    if key
                    not in {
                        "appearance_median_rgb",
                        "color_similarity",
                        "hue_similarity",
                        "color_delta_e",
                    }
                }
                sanitized["candidate_index"] = len(generic_candidates) + 1
                sanitized["color_evidence_used"] = False
                sanitized["color_evidence_scope"] = (
                    "forbidden_in_corresponding_material_pass"
                )
                generic_candidates.append(sanitized)
            if not generic_candidates:
                raise PartIdQwenError(
                    f"Part-ID {part_id} has no generic candidate for the "
                    "corresponding-material pass"
                )
            fallback_job = {
                **source_job,
                "candidates": generic_candidates,
                "selection_stage": "corresponding_material_grayscale",
            }
            if bank_context is not None or material_root is not None:
                render_paths = {
                    str(candidate["material_id"]): _candidate_render(
                        material_id=str(candidate["material_id"]),
                        profile=profiles_by_id.get(str(candidate["material_id"])),
                        bank_root=bank_root,
                        catalog_record=catalog_by_id[str(candidate["material_id"])],
                        material_root=material_root,
                    )
                    for candidate in generic_candidates
                }
                sheet = _comparison_sheet(
                    part_id=part_id,
                    crop=Path(str(source_job["crop"])).expanduser().resolve(
                        strict=True
                    ),
                    candidates=generic_candidates,
                    render_paths=render_paths,
                    output=corresponding_comparison_dir / f"{part_id}.png",
                    grayscale=True,
                )
                fallback_job["comparison_sheet"] = str(sheet)
            corresponding_jobs.append(fallback_job)
        corresponding_qwen_selections, corresponding_batch_audits = (
            run_selection_batches(
                corresponding_jobs,
                stage="corresponding_material",
                require_match_type=False,
                corresponding_only=True,
            )
        )
        for row in corresponding_qwen_selections:
            part_id = str(row["part_id"])
            initial = initial_by_id[part_id]
            row["match_type"] = "CORRESPONDING_MATERIAL"
            row["selection_authority"] = (
                "grayscale_corresponding_material_second_pass"
            )
            row["exact_preset_decision"] = dict(initial)
            initial_by_id[part_id] = row
        qwen_selections = [
            dict(initial_by_id[part_id]) for part_id in sorted(initial_by_id)
        ]
    selections: list[dict[str, Any]] = [
        *[dict(row) for row in direct_selections],
        *[dict(row) for row in qwen_selections],
    ]
    batch_audits = [
        *exact_preset_batch_audits,
        *corresponding_batch_audits,
    ]
    selections.sort(key=lambda row: str(row["part_id"]))
    if require_material_family_prediction:
        selections, component_identity_consensus = (
            _apply_component_identity_consensus(
                selections=selections,
                component_members=component_members,
            )
        )
        selective_regression = {
            "schema_version": SELECTIVE_REGRESSION_SCHEMA_VERSION,
            "status": "DISABLED_IDENTITY_STAGE_IGNORES_COLOR",
            "parts": [],
            "summary": {
                "part_count": len(selections),
                "qwen_choice_retained_count": len(qwen_selections),
                "direct_exact_library_assignment_count": len(
                    direct_selections
                ),
                "fresh_local_baseline_selected_count": 0,
                "exact_cover": True,
            },
        }
        for selection in selections:
            part_id = str(selection["part_id"])
            prediction = prediction_by_id[part_id]
            prediction_applyable = prediction.get("status") == "APPLYABLE"
            if prediction_applyable:
                authorized_material_ids = {
                    str(row["material_id"])
                    for row in _identity_filtered_ranking(
                        ranking=retrieval_by_id[part_id]["fused_ranking"],
                        catalog_by_id=catalog_by_id,
                        prediction=prediction,
                    )
                }
                if str(selection["material_id"]) not in authorized_material_ids:
                    raise PartIdQwenError(
                        f"material selection escaped predicted physical identity "
                        f"for {part_id}"
                    )
            selection_confidence = float(selection.get("confidence", 0.0))
            selection["selection_confidence"] = selection_confidence
            selection["confidence"] = (
                min(selection_confidence, float(prediction["confidence"]))
                if prediction_applyable
                else 0.0
            )
            selection["material_prediction"] = dict(prediction)
    else:
        selections, selective_regression = _apply_part_id_selective_regression(
            jobs=jobs,
            qwen_selections=qwen_selections,
        )
        component_identity_consensus = {
            "schema_version": "qwen-component-material-identity-consensus/v1",
            "components": [],
            "summary": {
                "component_count": 0,
                "constrained_part_count": 0,
                "all_components_share_one_exact_mdl": True,
            },
        }
    choices = {row["part_id"]: row["material_id"] for row in selections}
    if set(choices) != set(part_by_id):
        raise PartIdQwenError("Qwen Part-ID rerank did not exactly cover its jobs")
    unsigned = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "model": model,
        "model_identity": getattr(runner, "model_identity", None),
        "evidence_sha256": evidence["integrity"]["document_sha256"],
        "retrieval_sha256": _canonical_sha256(retrieval),
        "batch_size": batch_size,
        "candidate_count": candidate_count,
        "allow_color_tuning": allow_color_tuning,
        "material_prediction_mode": (
            "catalog_family_first" if require_material_family_prediction else "disabled"
        ),
        "selection_order": (
            [
                "physical_material_identity_prediction_without_color",
                "final_evidence_component_membership_refinement",
                "exact_substrate_species_treatment_optical_filter",
                "full_catalog_specific_preset_preservation",
                "exact_preset_confirmation_with_bounded_color_evidence",
                "independent_grayscale_corresponding_material_selection_when_unresolved",
                "component_consensus_without_overwriting_protected_exact_presets",
            ]
            if require_material_family_prediction
            else ["visual_candidate_selection"]
        ),
        "material_predictions": material_predictions,
        "material_prediction_batches": material_prediction_batches,
        "comparison_sheets_used": all(
            isinstance(job.get("comparison_sheet"), str) for job in jobs
        ),
        "corresponding_material_comparison_sheets_used": (
            all(
                isinstance(job.get("comparison_sheet"), str)
                for job in corresponding_jobs
            )
            if require_material_family_prediction
            else False
        ),
        "qwen_raw_selections": qwen_selections,
        "exact_preset_qwen_selections": exact_preset_qwen_selections,
        "corresponding_material_qwen_selections": corresponding_qwen_selections,
        "direct_exact_library_assignments": direct_assignment_audits,
        "part_id_selective_regression": selective_regression,
        "component_identity_consensus": component_identity_consensus,
        "component_membership_refinement": component_membership_refinement,
        "visual_compatibility_gate": {
            "policy": (
                "physical_identity_species_then_exact_authored_preset/v6"
                if require_material_family_prediction
                else "perceptual_color_texture_transmission_gate/v2"
            ),
            "observation_bank": str(bank_root) if bank_root is not None else None,
            "exact_preset_color_gate": {
                "metric": "CIE76_delta_e_on_sealed_target_and_library_profile_medians",
                "maximum_delta_e": EXACT_LIBRARY_PRESET_MAXIMUM_DELTA_E,
                "action_on_failure": (
                    "reject_exact_match_and_run_independent_grayscale_corresponding_pass"
                ),
            },
            "parts": gate_audits,
        },
        "selections": selections,
        "choices": choices,
        "batches": batch_audits,
        "summary": {
            "observed_part_count": len(part_by_id),
            "selection_count": len(selections),
            "exact_cover_of_observed_parts": len(selections) == len(part_by_id),
            "material_library_gap_fallback_count": (
                sum(
                    row.get("match_type") == "CORRESPONDING_MATERIAL"
                    and row.get("confidence", 0.0) > 0.0
                    for row in selections
                )
                if require_material_family_prediction
                else sum(
                    job.get("library_gap_fallback") is not None for job in jobs
                )
            ),
            "selective_regression_changed_count": (
                selective_regression["summary"]["fresh_local_baseline_selected_count"]
            ),
            "material_prediction_count": len(material_predictions),
            "material_prediction_applyable_count": sum(
                row.get("status") == "APPLYABLE" for row in material_predictions
            ),
            "material_prediction_insufficient_evidence_count": sum(
                row.get("status") == "INSUFFICIENT_EVIDENCE"
                for row in material_predictions
            ),
            "physical_cross_family_fallback_count": 0,
            "component_identity_count": len(component_members),
            "component_identity_constrained_part_count": sum(
                len(members) for members in component_members.values()
            ),
            "color_evidence_used_for_identity_count": sum(
                row.get("match_type") == "EXACT_LIBRARY_MATCH"
                and row.get("selection_authority")
                not in {
                    "unique_full_catalog_physical_contract",
                    "unique_full_catalog_mvinverse_pbr_fingerprint",
                }
                for row in selections
            ),
            "mvinverse_pbr_evidence_available_count": sum(
                bool(_physical_pbr_evidence(part.get("descriptor")))
                for part in part_by_id.values()
            ),
            "direct_exact_library_assignment_count": len(direct_selections),
            "qwen_corresponding_selection_count": sum(
                row.get("match_type") == "CORRESPONDING_MATERIAL"
                for row in qwen_selections
            ),
            "exact_library_match_count": sum(
                row.get("match_type") == "EXACT_LIBRARY_MATCH"
                and row.get("confidence", 0.0) > 0.0
                for row in selections
            ),
            "corresponding_material_count": sum(
                row.get("match_type") == "CORRESPONDING_MATERIAL"
                and row.get("confidence", 0.0) > 0.0
                for row in selections
            ),
        },
    }
    return {
        **unsigned,
        "integrity": {"document_sha256": _canonical_sha256(unsigned)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank independent CAD Part-ID material candidates with Qwen"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--appearance-components",
        type=Path,
        help=(
            "sealed same-surface appearance components whose members must share "
            "one physical identity and exact MDL"
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3.5-local-part-id-reranker")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument(
        "--allow-mdl-color-tuning",
        action="store_true",
        help=(
            "treat reviewed Base MDL colour inputs as target-reproducible while "
            "keeping texture, normal and coating on the selected preset"
        ),
    )
    parser.add_argument(
        "--require-material-family-prediction",
        action="store_true",
        help=(
            "predict physical substrate/treatment/optical identity before candidate "
            "selection, prefer an exact library material, and otherwise choose a "
            "corresponding material without using colour"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence_path = args.evidence.expanduser().resolve(strict=True)
    retrieval_path = args.retrieval.expanduser().resolve(strict=True)
    catalog_path = args.catalog.expanduser().resolve(strict=True)
    appearance_components_path = (
        args.appearance_components.expanduser().resolve(strict=True)
        if args.appearance_components is not None
        else None
    )
    model_path = args.model_path.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    runner = TransformersQwen3VLRunner(
        model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=512 * 512,
        max_total_pixels=4 * 512 * 512,
    )
    runner.preflight()
    try:
        result = run_part_id_qwen_rerank(
            evidence=_read(evidence_path, "Part-ID evidence"),
            retrieval=_read(retrieval_path, "Part-ID retrieval"),
            catalog=_read(catalog_path, "material catalog"),
            appearance_components=(
                _read(appearance_components_path, "appearance components")
                if appearance_components_path is not None
                else None
            ),
            runner=runner,
            model=args.model_name,
            output_dir=output_dir,
            batch_size=args.batch_size,
            candidate_count=args.candidate_count,
            allow_color_tuning=args.allow_mdl_color_tuning,
            require_material_family_prediction=(
                args.require_material_family_prediction
            ),
        )
    finally:
        runner.unload()
    _write(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                **result["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
