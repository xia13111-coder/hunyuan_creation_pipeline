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
MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION = (
    "qwen-part-id-material-family-prediction-batch/v1"
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


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


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


def _validate_material_prediction_batch(
    document: Mapping[str, Any],
    *,
    expected: Sequence[Mapping[str, Any]],
    allowed_families: Sequence[str],
) -> list[dict[str, Any]]:
    if set(document) != {"schema_version", "predictions"}:
        raise PartIdQwenError(
            "Qwen material-family prediction returned unexpected fields"
        )
    if (
        document.get("schema_version")
        != MATERIAL_FAMILY_PREDICTION_BATCH_SCHEMA_VERSION
    ):
        raise PartIdQwenError("Qwen material-family prediction uses the wrong schema")
    rows = document.get("predictions")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise PartIdQwenError("Qwen material-family prediction count is invalid")
    expected_ids = {str(row["part_id"]) for row in expected}
    allowed = set(allowed_families)
    allowed.add("unknown")
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "part_id",
            "catalog_family",
            "surface_finish",
            "confidence",
        }:
            raise PartIdQwenError(
                f"Qwen material-family prediction {index} has invalid fields"
            )
        part_id = raw.get("part_id")
        family = raw.get("catalog_family")
        finish = raw.get("surface_finish")
        confidence = raw.get("confidence")
        if (
            not isinstance(part_id, str)
            or part_id not in expected_ids
            or part_id in output
        ):
            raise PartIdQwenError(
                f"Qwen material-family prediction {index} cites an invalid part"
            )
        if not isinstance(family, str) or family not in allowed:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported catalog family for {part_id}"
            )
        if not isinstance(finish, str) or finish not in MATERIAL_FINISH_OPTIONS:
            raise PartIdQwenError(
                f"Qwen predicted an unsupported surface finish for {part_id}"
            )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise PartIdQwenError(
                f"Qwen predicted an invalid material confidence for {part_id}"
            )
        output[part_id] = {
            "part_id": part_id,
            "catalog_family": family,
            "surface_finish": finish,
            "confidence": float(confidence),
            "status": (
                "APPLYABLE"
                if family != "unknown"
                and float(confidence) >= MINIMUM_MATERIAL_FAMILY_CONFIDENCE
                else "INSUFFICIENT_EVIDENCE"
            ),
        }
    if set(output) != expected_ids:
        raise PartIdQwenError(
            "Qwen material-family prediction does not exactly cover its input"
        )
    return [output[str(row["part_id"])] for row in expected]


def _material_prediction_payload(
    *,
    model: str,
    batch: Sequence[Mapping[str, Any]],
    allowed_families: Sequence[str],
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
                    "images are neutral-background crops of this exact CAD Part-ID "
                    "from different trusted views. Descriptor: "
                    + json.dumps(item.get("descriptor", {}), ensure_ascii=False)
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
                "catalog_family": "one allowed catalog family or unknown",
                "surface_finish": "one allowed finish",
                "confidence": 0.75,
            }
            for item in batch
        ],
    }
    prompt = "\n".join(
        [
            "Predict physical material identity before seeing any material candidate.",
            "Ignore colour as evidence of material family. Green, black, silver or white can all be paint, metal, plastic or rubber.",
            "Choose the NVIDIA catalog family that describes the visible surface material. A painted or powder-coated metal enclosure is catalog_family=paint; exposed steel/aluminium is metal; an elastomer is rubber; a polymer housing is plastic; transparent glazing is glass.",
            "Use unknown with confidence below 0.60 when the crop cannot support a physical identity. Never guess from an adjacent part, background, CAD part name, or colour alone.",
            "Allowed catalog_family values: "
            + json.dumps([*allowed_families, "unknown"], ensure_ascii=False),
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
    allowed_families: Sequence[str],
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
                    allowed_families=allowed_families,
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
                    allowed_families=allowed_families,
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
                "Qwen material-family prediction batch "
                f"{batch_index} failed twice: {final_error}"
            )
    return predictions, audits


def _family_filtered_ranking(
    *,
    ranking: Sequence[Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    family = str(prediction.get("catalog_family", "unknown"))
    confidence = float(prediction.get("confidence", 0.0))
    if family == "unknown" or confidence < MINIMUM_MATERIAL_FAMILY_CONFIDENCE:
        return [dict(row) for row in ranking]
    expected_ids = {
        material_id
        for material_id, record in catalog_by_id.items()
        if str(record.get("family", "")).strip().casefold() == family
    }
    if not expected_ids:
        raise PartIdQwenError(
            f"predicted catalog family {family!r} has no NVIDIA MDL materials"
        )
    ranking_by_id = {
        str(row.get("material_id")): dict(row)
        for row in ranking
        if isinstance(row.get("material_id"), str)
    }
    missing = sorted(expected_ids - ranking_by_id.keys())
    if missing:
        raise PartIdQwenError(
            "material-family-first retrieval must cover the complete catalog; "
            f"family {family!r} is missing {len(missing)} material IDs"
        )
    finish = str(prediction.get("surface_finish", "unknown"))
    filtered = [ranking_by_id[material_id] for material_id in expected_ids]
    for row in filtered:
        record = catalog_by_id[str(row["material_id"])]
        record_finishes = {
            str(value).casefold()
            for value in record.get("finishes", [])
            if isinstance(value, str)
        }
        row["predicted_family"] = family
        row["predicted_finish"] = finish
        row["predicted_finish_match"] = bool(
            finish != "unknown" and finish in record_finishes
        )
    filtered.sort(
        key=lambda row: (
            0 if row["predicted_finish_match"] else 1,
            int(row.get("rank", 1_000_000)),
            str(row["material_id"]),
        )
    )
    return filtered


def _candidate_summary(
    row: Mapping[str, Any],
    catalog_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "material_id": row["material_id"],
        "display_name": catalog_record.get("display_name"),
        "description": catalog_record.get("description"),
        "family": catalog_record.get("family"),
        "category_path": catalog_record.get("category_path"),
        "keywords": catalog_record.get("keywords", []),
        "colors": catalog_record.get("colors", []),
        "finishes": catalog_record.get("finishes", []),
        "retrieval_rank": row.get("rank"),
        "siglip2_score": row.get("siglip2_score"),
        "dino_score": row.get("dino_score"),
        "color_score": row.get("color_score"),
        "mvinverse_score": row.get("mvinverse_score"),
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
                    image = ImageOps.contain(
                        opened.convert("RGB"),
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
) -> list[dict[str, Any]]:
    if set(document) != {"schema_version", "selections"}:
        raise PartIdQwenError("Qwen Part-ID batch returned unexpected fields")
    if document.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise PartIdQwenError("Qwen Part-ID batch uses the wrong schema")
    rows = document.get("selections")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise PartIdQwenError("Qwen Part-ID batch selection count is invalid")
    expected_by_id = {str(item["part_id"]): item for item in expected}
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "part_id",
            "candidate_index",
            "confidence",
        }:
            raise PartIdQwenError(f"Qwen Part-ID selection {index} has invalid fields")
        part_id = raw.get("part_id")
        candidate_index = raw.get("candidate_index")
        confidence = raw.get("confidence")
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
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise PartIdQwenError(f"Qwen returned invalid confidence for {part_id}")
        output[part_id] = {
            "part_id": part_id,
            "material_id": material_id,
            "candidate_index": candidate_index,
            "requested_candidate_index": requested_candidate_index,
            "index_resolution": index_resolution,
            "confidence": float(confidence),
        }
    if set(output) != set(expected_by_id):
        raise PartIdQwenError("Qwen Part-ID batch does not exactly cover its input")
    return [output[str(item["part_id"])] for item in expected]


def _payload(
    *,
    model: str,
    batch: Sequence[dict[str, Any]],
    allow_color_tuning: bool,
    require_material_family_prediction: bool = False,
    entity_label: str = "exact CAD Part-ID",
    retry_detail: str | None = None,
) -> dict[str, Any]:
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
        "schema_version": BATCH_SCHEMA_VERSION,
        "selections": [
            {
                "part_id": item["part_id"],
                "candidate_index": (
                    "one supplied integer candidate_index for this part"
                ),
                "confidence": 0.75,
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
                "Material identity is already fixed by the independent first pass. "
                "Every numbered candidate belongs to that exact predicted catalog "
                "family. Keep identity fixed and select the closest finish, texture, "
                "highlight response and roughness; colour is only a secondary tie-break."
                if require_material_family_prediction
                else "Optimize visible similarity only. Exact engineering/semantic category may differ, but visual behavior may not: opacity versus transmission, surface continuity, texture scale, color, brightness, highlight response and roughness are mandatory constraints."
            ),
            "Compare TARGET directly with the numbered actual MDL renders. Never choose a transmissive/transparent candidate for an opaque target, and never introduce grass, fabric, wood, stone or other repeating/high-frequency texture when TARGET is smooth.",
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
                    "You are a bounded visual material classifier. Decisions "
                    "are independent per CAD part_id. Obey the exact JSON contract."
                ),
            },
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
        "enable_thinking": False,
    }


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
            "material-family-first selection cannot tune colour in the identity stage"
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
    material_predictions: list[dict[str, Any]] = []
    material_prediction_batches: list[dict[str, Any]] = []
    if require_material_family_prediction:
        prediction_items: list[dict[str, Any]] = []
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
            trusted_views = trusted_views[:MAX_MATERIAL_PREDICTION_IMAGES_PER_BATCH]
            prediction_items.append(
                {
                    "part_id": part_id,
                    "views": trusted_views,
                    "selected_view_id": selected_view_id,
                    "descriptor": part.get("descriptor", {}),
                }
            )
        material_predictions, material_prediction_batches = _predict_material_families(
            items=prediction_items,
            runner=runner,
            model=model,
            allowed_families=_catalog_family_options(catalog_by_id),
            batch_size=batch_size,
            raw_dir=raw_dir,
        )
    prediction_by_id = {str(row["part_id"]): row for row in material_predictions}
    jobs: list[dict[str, Any]] = []
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
                    f"Part-ID {part_id} has no material-family prediction"
                )
            ranking = _family_filtered_ranking(
                ranking=ranking,
                catalog_by_id=catalog_by_id,
                prediction=prediction,
            )
        target = _target_appearance(selected_observations[0])
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
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    **_candidate_summary(row, catalog_by_id[material_id]),
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
                }
            )
        jobs.append(
            {
                "part_id": part_id,
                # The annotated correspondence crop is intentionally not a
                # VLM material target.  It may contain other parts inside a
                # valid tolerant box.  Prefer the neutralized exact-core crop
                # emitted by Part-ID evidence, retaining the legacy crop only
                # for old sealed checkpoints.
                "crop": selected_observations[0].get(
                    "isolated_crop", selected_observations[0]["crop"]
                ),
                "descriptor": part["descriptor"],
                "target_appearance": target,
                "candidates": candidates,
                "library_gap_fallback": library_gap_fallback,
                "material_prediction": prediction,
            }
        )
        gate_audits.append(
            {
                "part_id": part_id,
                "target_appearance": target,
                "library_gap_fallback": library_gap_fallback,
                "material_prediction": prediction,
                "authorized_catalog_family": (
                    prediction.get("catalog_family")
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
            )
            job["comparison_sheet"] = str(sheet)
    selections: list[dict[str, Any]] = []
    batch_audits: list[dict[str, Any]] = []
    batches = [
        jobs[index : index + batch_size] for index in range(0, len(jobs), batch_size)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        final_error: Exception | None = None
        for attempt in range(1, 3):
            payload = _payload(
                model=model,
                batch=batch,
                allow_color_tuning=allow_color_tuning,
                require_material_family_prediction=(require_material_family_prediction),
                entity_label=entity_label,
                retry_detail=str(final_error) if final_error is not None else None,
            )
            generated = runner.generate_with_metadata(payload)
            raw_path = raw_dir / f"batch_{batch_index:03d}_attempt_{attempt}.txt"
            raw_path.write_text(generated.text, encoding="utf-8")
            try:
                document, parse_audit = parse_plan_content_with_audit(generated.text)
                validated = _validate_batch(document, expected=batch)
            except Exception as exc:
                final_error = exc
                _write(
                    raw_dir / f"batch_{batch_index:03d}_attempt_{attempt}.parse.json",
                    {
                        "status": "invalid",
                        "error": str(exc),
                        "generation": generated.metadata(),
                    },
                )
                continue
            selections.extend(validated)
            audit_path = _write(
                raw_dir / f"batch_{batch_index:03d}_attempt_{attempt}.parse.json",
                {
                    **parse_audit,
                    "status": "valid",
                    "generation": generated.metadata(),
                },
            )
            batch_audits.append(
                {
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "part_ids": [str(item["part_id"]) for item in batch],
                    "parse_audit": str(audit_path),
                }
            )
            print(
                f"[PART-ID QWEN] {batch_index}/{len(batches)} "
                f"parts={','.join(item['part_id'] for item in batch)}",
                flush=True,
            )
            break
        else:
            raise PartIdQwenError(
                f"Qwen Part-ID batch {batch_index} failed twice: {final_error}"
            )
    qwen_selections = [dict(row) for row in selections]
    selections, selective_regression = _apply_part_id_selective_regression(
        jobs=jobs,
        qwen_selections=qwen_selections,
    )
    if require_material_family_prediction:
        for selection in selections:
            part_id = str(selection["part_id"])
            prediction = prediction_by_id[part_id]
            selected_record = catalog_by_id[str(selection["material_id"])]
            prediction_applyable = prediction.get("status") == "APPLYABLE"
            if prediction_applyable and (
                str(selected_record.get("family", "")).strip().casefold()
                != prediction["catalog_family"]
            ):
                raise PartIdQwenError(
                    f"material selection escaped predicted family for {part_id}"
                )
            selection_confidence = float(selection.get("confidence", 0.0))
            selection["selection_confidence"] = selection_confidence
            selection["confidence"] = (
                min(selection_confidence, float(prediction["confidence"]))
                if prediction_applyable
                else 0.0
            )
            selection["material_prediction"] = dict(prediction)
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
                "material_family_prediction",
                "exact_catalog_family_filter",
                "exact_mdl_visual_selection",
            ]
            if require_material_family_prediction
            else ["visual_candidate_selection"]
        ),
        "material_predictions": material_predictions,
        "material_prediction_batches": material_prediction_batches,
        "comparison_sheets_used": all(
            isinstance(job.get("comparison_sheet"), str) for job in jobs
        ),
        "qwen_raw_selections": qwen_selections,
        "part_id_selective_regression": selective_regression,
        "visual_compatibility_gate": {
            "policy": ("perceptual_color_texture_transmission_gate/v2"),
            "observation_bank": str(bank_root) if bank_root is not None else None,
            "parts": gate_audits,
        },
        "selections": selections,
        "choices": choices,
        "batches": batch_audits,
        "summary": {
            "observed_part_count": len(part_by_id),
            "selection_count": len(selections),
            "exact_cover_of_observed_parts": len(selections) == len(part_by_id),
            "material_library_gap_fallback_count": sum(
                job.get("library_gap_fallback") is not None for job in jobs
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
            "predict a physical catalog family before candidate selection and "
            "forbid cross-family MDL choices"
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
