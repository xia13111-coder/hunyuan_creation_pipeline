"""Human-confirmed SAM3 foreground annotations and interactive inference.

The material pipeline uses this module only for the whole-workpiece foreground
stage. A user may incrementally confirm several disconnected instances per
reference view; each instance keeps its strict positive/negative event order,
selected first candidate, low-resolution refinement logits, and mask. The
final foreground is their deterministic union. Material-group segmentation
later in the pipeline remains automatic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


LEGACY_ANNOTATION_SCHEMA_VERSION = "sam3-human-foreground-annotations/v1"
ANNOTATION_SCHEMA_VERSION = "sam3-human-foreground-annotations/v2"
COORDINATE_GRID_SIZE = 1000
REFERENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
CONFIRMED_MASK_STRICT_MINIMUM_IOU = 0.995
# A fresh CUDA process may move a small number of boundary pixels in either
# direction while still reproducing the same confirmed workpiece.  This route
# is deliberately symmetric and much tighter than the omission-tolerant gate
# below: the total union disagreement must remain below 1.5%.
CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU = 0.985
# A confirmed mask remains authoritative when a fresh SAM3 process reproduces
# almost no pixels outside it but omits a small part of the approved union.
# This handles reproducible CUDA/session-state drift without allowing a replay
# that moved onto another object.
CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION = 0.99
CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL = 0.90


class HumanForegroundError(ValueError):
    """Raised when an interactive foreground artifact is unsafe or stale."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_revision(repository: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = process.stdout.strip()
    return revision if len(revision) == 40 else None


def parse_reference_specs(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    """Resolve 2..4 unique ``[ID=]IMAGE`` reference specifications."""

    if isinstance(values, (str, bytes)) or not 2 <= len(values) <= 4:
        raise HumanForegroundError("interactive SAM3 requires 2..4 reference images")
    result: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, raw_value in enumerate(values, start=1):
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise HumanForegroundError(f"reference {index} must be a non-empty string")
        raw = raw_value.strip()
        if "=" in raw:
            view_id, raw_path = raw.split("=", 1)
            view_id = view_id.strip()
        else:
            view_id, raw_path = f"ref_{index:02d}", raw
        if REFERENCE_ID.fullmatch(view_id) is None or view_id in seen_ids:
            raise HumanForegroundError(
                f"invalid or duplicate reference ID: {view_id!r}"
            )
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HumanForegroundError(
                f"reference image does not exist: {raw_path}"
            ) from exc
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise HumanForegroundError(f"unsupported reference image: {path}")
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise HumanForegroundError("reference images must have unique content")
        seen_ids.add(view_id)
        seen_hashes.add(digest)
        result.append((view_id, path))
    return tuple(result)


def _decoded_rgb(path: Path) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    return image, digest


def _resolve_file(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HumanForegroundError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HumanForegroundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise HumanForegroundError(f"{label} is not a file: {resolved}")
    return resolved


def _point(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise HumanForegroundError(f"{label} must be [x,y] integer coordinates")
    x, y = value
    if not (0 <= x <= COORDINATE_GRID_SIZE and 0 <= y <= COORDINATE_GRID_SIZE):
        raise HumanForegroundError(
            f"{label} must be within the 0..{COORDINATE_GRID_SIZE} grid"
        )
    return [x, y]


def validate_click_sets(value: Any, label: str) -> list[dict[str, list[list[int]]]]:
    if not isinstance(value, list) or not value:
        raise HumanForegroundError(f"{label} must contain at least one click set")
    result: list[dict[str, list[list[int]]]] = []
    for set_index, raw_set in enumerate(value):
        set_label = f"{label}[{set_index}]"
        if not isinstance(raw_set, Mapping):
            raise HumanForegroundError(f"{set_label} must be an object")
        unexpected = sorted(set(raw_set) - {"positive_points", "negative_points"})
        if unexpected:
            raise HumanForegroundError(
                f"{set_label} has unexpected fields: {', '.join(unexpected)}"
            )
        positives = raw_set.get("positive_points")
        negatives = raw_set.get("negative_points", [])
        if not isinstance(positives, list) or not positives:
            raise HumanForegroundError(f"{set_label}.positive_points must be non-empty")
        if not isinstance(negatives, list):
            raise HumanForegroundError(f"{set_label}.negative_points must be an array")
        normalized = {
            "positive_points": [
                _point(point, f"{set_label}.positive_points[{point_index}]")
                for point_index, point in enumerate(positives)
            ],
            "negative_points": [
                _point(point, f"{set_label}.negative_points[{point_index}]")
                for point_index, point in enumerate(negatives)
            ],
        }
        labelled = [(tuple(point), 1) for point in normalized["positive_points"]] + [
            (tuple(point), 0) for point in normalized["negative_points"]
        ]
        if len(set(labelled)) != len(labelled):
            raise HumanForegroundError(
                f"{set_label} contains duplicate labelled points"
            )
        positive_locations = {tuple(point) for point in normalized["positive_points"]}
        negative_locations = {tuple(point) for point in normalized["negative_points"]}
        if positive_locations & negative_locations:
            raise HumanForegroundError(
                f"{set_label} labels the same coordinate as foreground and background"
            )
        result.append(normalized)
    return result


def validate_ordered_click_sets(value: Any, label: str) -> list[dict[str, Any]]:
    """Validate v2 per-instance ordered prompts and their derived point lists."""

    if not isinstance(value, list) or not value:
        raise HumanForegroundError(f"{label} must contain at least one click set")
    result: list[dict[str, Any]] = []
    expected_fields = {
        "events",
        "positive_points",
        "negative_points",
        "initial_candidate_index",
    }
    for set_index, raw_set in enumerate(value):
        set_label = f"{label}[{set_index}]"
        if not isinstance(raw_set, Mapping) or set(raw_set) != expected_fields:
            raise HumanForegroundError(f"{set_label} fields are invalid")
        candidate_index = raw_set.get("initial_candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index <= 2
        ):
            raise HumanForegroundError(
                f"{set_label}.initial_candidate_index must be 0, 1, or 2"
            )
        raw_events = raw_set.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise HumanForegroundError(f"{set_label}.events must be non-empty")
        events: list[dict[str, Any]] = []
        seen_points: set[tuple[int, int]] = set()
        for event_index, raw_event in enumerate(raw_events):
            event_label = f"{set_label}.events[{event_index}]"
            if not isinstance(raw_event, Mapping) or set(raw_event) != {
                "point",
                "label",
            }:
                raise HumanForegroundError(f"{event_label} fields are invalid")
            point = _point(raw_event.get("point"), f"{event_label}.point")
            point_label = raw_event.get("label")
            if (
                isinstance(point_label, bool)
                or not isinstance(point_label, int)
                or point_label not in {0, 1}
            ):
                raise HumanForegroundError(f"{event_label}.label must be 0 or 1")
            location = tuple(point)
            if location in seen_points:
                raise HumanForegroundError(
                    f"{set_label} contains duplicate or relabelled coordinates"
                )
            seen_points.add(location)
            events.append({"point": point, "label": point_label})
        if events[0]["label"] != 1 or not any(event["label"] == 1 for event in events):
            raise HumanForegroundError(
                f"{set_label} must begin with a foreground point"
            )
        positives = [event["point"] for event in events if event["label"] == 1]
        negatives = [event["point"] for event in events if event["label"] == 0]
        supplied_positive = raw_set.get("positive_points")
        supplied_negative = raw_set.get("negative_points")
        if not isinstance(supplied_positive, list) or not isinstance(
            supplied_negative, list
        ):
            raise HumanForegroundError(
                f"{set_label} derived point arrays must be arrays"
            )
        normalized_positive = [
            _point(point, f"{set_label}.positive_points[{index}]")
            for index, point in enumerate(supplied_positive)
        ]
        normalized_negative = [
            _point(point, f"{set_label}.negative_points[{index}]")
            for index, point in enumerate(supplied_negative)
        ]
        if normalized_positive != positives or normalized_negative != negatives:
            raise HumanForegroundError(
                f"{set_label} point arrays do not match ordered events"
            )
        result.append(
            {
                "events": events,
                "positive_points": positives,
                "negative_points": negatives,
                "initial_candidate_index": candidate_index,
            }
        )
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HumanForegroundError(f"{label} must be a positive integer")
    return value


def _unit_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanForegroundError(f"{label} must be numeric")
    result = float(value)
    lower_bound_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_bound_ok or result > 1.0:
        interval = "(0,1]" if positive else "[0,1]"
        raise HumanForegroundError(f"{label} must be within {interval}")
    return result


def annotation_replay_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the point-replay gates frozen by the UI."""

    policy = document.get("policy")
    schema_version = document.get("schema_version")
    base_fields = {
        "minimum_model_score",
        "human_point_model_score_authority",
        "minimum_prompt_agreement",
        "maximum_image_fraction",
        "minimum_mask_pixels",
        "disconnected_region_policy",
    }
    expected_fields = set(base_fields)
    if schema_version == ANNOTATION_SCHEMA_VERSION:
        expected_fields.update({"interaction_policy", "ordered_replay_policy"})
    elif schema_version != LEGACY_ANNOTATION_SCHEMA_VERSION:
        raise HumanForegroundError(
            f"unsupported SAM3 foreground annotation schema: {schema_version!r}"
        )
    if not isinstance(policy, Mapping) or set(policy) != expected_fields:
        raise HumanForegroundError(
            "SAM3 foreground annotation policy fields are invalid"
        )
    if policy.get("human_point_model_score_authority") != "advisory":
        raise HumanForegroundError(
            "human-confirmed point masks must treat model score as advisory"
        )
    if schema_version == LEGACY_ANNOTATION_SCHEMA_VERSION:
        if policy.get("disconnected_region_policy") != "separate_click_sets_then_union":
            raise HumanForegroundError("SAM3 disconnected-region policy is invalid")
    else:
        if policy.get("disconnected_region_policy") != (
            "incremental_instances_then_union"
        ):
            raise HumanForegroundError("SAM3 disconnected-region policy is invalid")
        if policy.get("interaction_policy") != (
            "smart_outside_add_inside_refine_with_explicit_overrides"
        ):
            raise HumanForegroundError("SAM3 interaction policy is invalid")
        if policy.get("ordered_replay_policy") != (
            "first_multimask_then_previous_logits_single_mask"
        ):
            raise HumanForegroundError("SAM3 ordered replay policy is invalid")
    return {
        "minimum_model_score": _unit_float(
            policy.get("minimum_model_score"), "policy.minimum_model_score"
        ),
        "minimum_prompt_agreement": _unit_float(
            policy.get("minimum_prompt_agreement"),
            "policy.minimum_prompt_agreement",
        ),
        "maximum_image_fraction": _unit_float(
            policy.get("maximum_image_fraction"),
            "policy.maximum_image_fraction",
            positive=True,
        ),
        "minimum_mask_pixels": _positive_int(
            policy.get("minimum_mask_pixels"), "policy.minimum_mask_pixels"
        ),
    }


def require_replay_policy(
    document: Mapping[str, Any],
    *,
    minimum_prompt_agreement: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> None:
    """Fail early when UI and formal point-replay gates are not identical."""

    policy = annotation_replay_policy(document)
    expected = {
        "minimum_prompt_agreement": float(minimum_prompt_agreement),
        "maximum_image_fraction": float(maximum_image_fraction),
        "minimum_mask_pixels": int(minimum_mask_pixels),
    }
    actual = {key: policy[key] for key in expected}
    if actual != expected:
        raise HumanForegroundError(
            "SAM3 foreground annotation replay policy differs from the formal "
            f"pipeline: annotation={actual}, pipeline={expected}; relaunch the "
            "point UI with matching gate arguments"
        )


def materialize_annotation_bundle(
    document: Mapping[str, Any],
    *,
    destination: Path,
    references: Sequence[tuple[str, Path]],
    repository: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    """Copy confirmed masks beside a sealed, independently loadable JSON."""

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = destination.parent / f"{destination.stem}_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False))
    normalized.pop("integrity", None)
    raw_views = normalized.get("source_views")
    if not isinstance(raw_views, list):
        raise HumanForegroundError("SAM3 annotation bundle has no source views")
    for raw_view in raw_views:
        if not isinstance(raw_view, dict) or not isinstance(raw_view.get("id"), str):
            raise HumanForegroundError("SAM3 annotation bundle has an invalid view")
        mask_record = raw_view.get("confirmed_mask")
        if not isinstance(mask_record, dict):
            raise HumanForegroundError("SAM3 annotation bundle has an invalid mask")
        source_mask = (
            Path(str(mask_record.get("path", ""))).expanduser().resolve(strict=True)
        )
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_view["id"])
        target_mask = masks_dir / f"{safe_id}.png"
        if target_mask.exists():
            if sha256_file(target_mask) != sha256_file(source_mask):
                raise HumanForegroundError(
                    f"stored SAM3 annotation mask changed: {target_mask}"
                )
        elif source_mask != target_mask:
            temporary_mask = target_mask.with_name(target_mask.name + ".tmp")
            shutil.copyfile(source_mask, temporary_mask)
            temporary_mask.replace(target_mask)
        mask_record["path"] = target_mask.relative_to(destination.parent).as_posix()
    sealed = {
        **normalized,
        "integrity": {"document_sha256": canonical_sha256(normalized)},
    }
    payload = (
        json.dumps(sealed, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise HumanForegroundError(
                "stored SAM3 foreground annotation bundle differs from this run"
            )
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    verified, _masks = load_annotations(
        destination,
        references=references,
        repository=repository,
        checkpoint=checkpoint,
    )
    return verified


def load_annotations(
    path: Path,
    *,
    references: Sequence[tuple[str, Path]] | None = None,
    repository: Path | None = None,
    checkpoint: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load and fully verify one human-confirmed foreground artifact."""

    annotation_path = path.expanduser().resolve(strict=True)
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanForegroundError(
            f"unable to read SAM3 foreground annotations {annotation_path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise HumanForegroundError("SAM3 foreground annotations must be an object")
    schema_version = document.get("schema_version")
    if schema_version not in {
        LEGACY_ANNOTATION_SCHEMA_VERSION,
        ANNOTATION_SCHEMA_VERSION,
    }:
        raise HumanForegroundError(
            "unsupported SAM3 foreground annotation schema: "
            f"{document.get('schema_version')!r}"
        )
    if document.get("prompt_authority") != ("human_confirmed_sam3_interactive_points"):
        raise HumanForegroundError("SAM3 foreground prompt authority is invalid")
    integrity = document.get("integrity")
    if not isinstance(integrity, dict):
        raise HumanForegroundError("SAM3 foreground annotations lack integrity")
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    if integrity.get("document_sha256") != canonical_sha256(unsigned):
        raise HumanForegroundError("SAM3 foreground annotation integrity mismatch")
    coordinate_space = document.get("coordinate_space")
    if coordinate_space != {
        "type": "exif_transposed_image_grid",
        "grid_size": COORDINATE_GRID_SIZE,
        "origin": "top_left",
        "axes": "x_right_y_down",
    }:
        raise HumanForegroundError("SAM3 foreground coordinate contract is invalid")
    confirmation = document.get("confirmation")
    if (
        not isinstance(confirmation, dict)
        or confirmation.get("all_views_confirmed") is not True
        or confirmation.get("human_mask_is_authoritative") is not True
    ):
        raise HumanForegroundError("all SAM3 reference views must be confirmed")
    annotation_replay_policy(document)
    backend = document.get("sam3")
    if not isinstance(backend, dict):
        raise HumanForegroundError("SAM3 foreground annotations lack backend identity")
    if backend.get("mode") != "instance_interactivity":
        raise HumanForegroundError("SAM3 foreground annotations use an invalid mode")
    if repository is not None:
        resolved_repository = repository.expanduser().resolve(strict=True)
        if backend.get("repository_revision") != git_revision(resolved_repository):
            raise HumanForegroundError(
                "SAM3 repository revision differs from the confirmed annotation"
            )
    if checkpoint is not None:
        resolved_checkpoint = checkpoint.expanduser().resolve(strict=True)
        if backend.get("checkpoint_sha256") != sha256_file(resolved_checkpoint):
            raise HumanForegroundError(
                "SAM3 checkpoint differs from the confirmed annotation"
            )
    raw_views = document.get("source_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise HumanForegroundError("SAM3 foreground source_views must be non-empty")
    expected = dict(references or ())
    normalized_views: list[dict[str, Any]] = []
    masks: dict[str, Path] = {}
    seen: set[str] = set()
    for index, raw_view in enumerate(raw_views):
        label = f"source_views[{index}]"
        if not isinstance(raw_view, Mapping):
            raise HumanForegroundError(f"{label} must be an object")
        view_id = raw_view.get("id")
        if (
            not isinstance(view_id, str)
            or REFERENCE_ID.fullmatch(view_id) is None
            or view_id in seen
        ):
            raise HumanForegroundError(f"{label}.id is invalid or duplicate")
        seen.add(view_id)
        image_path = _resolve_file(
            raw_view.get("image"), base=annotation_path.parent, label=f"{label}.image"
        )
        if expected:
            expected_path = expected.get(view_id)
            if expected_path is None:
                raise HumanForegroundError(
                    f"annotation contains an unknown reference view: {view_id}"
                )
            expected_digest = sha256_file(expected_path)
            if sha256_file(image_path) != expected_digest:
                raise HumanForegroundError(
                    f"annotation image differs from current reference: {view_id}"
                )
        file_digest = sha256_file(image_path)
        if raw_view.get("image_sha256") != file_digest:
            raise HumanForegroundError(f"annotation source image changed: {view_id}")
        image, decoded_digest = _decoded_rgb(image_path)
        width, height = image.size
        image.close()
        if raw_view.get("decoded_rgb_sha256") != decoded_digest:
            raise HumanForegroundError(f"annotation decoded pixels changed: {view_id}")
        if raw_view.get("width") != width or raw_view.get("height") != height:
            raise HumanForegroundError(
                f"annotation image dimensions changed: {view_id}"
            )
        if schema_version == ANNOTATION_SCHEMA_VERSION:
            click_sets = validate_ordered_click_sets(
                raw_view.get("click_sets"), f"{label}.click_sets"
            )
        else:
            click_sets = validate_click_sets(
                raw_view.get("click_sets"), f"{label}.click_sets"
            )
        mask_record = raw_view.get("confirmed_mask")
        if not isinstance(mask_record, Mapping):
            raise HumanForegroundError(f"{label}.confirmed_mask must be an object")
        mask_path = _resolve_file(
            mask_record.get("path"),
            base=annotation_path.parent,
            label=f"{label}.confirmed_mask.path",
        )
        if mask_record.get("sha256") != sha256_file(mask_path):
            raise HumanForegroundError(f"confirmed mask changed: {view_id}")
        import numpy as np
        from PIL import Image

        with Image.open(mask_path) as opened:
            mask_pixels_raw = np.asarray(opened.convert("L"), dtype=np.uint8)
        if not set(int(value) for value in np.unique(mask_pixels_raw)) <= {0, 255}:
            raise HumanForegroundError(
                f"confirmed mask is not a binary 0/255 image: {view_id}"
            )
        mask_array = mask_pixels_raw > 0
        if mask_array.shape != (height, width):
            raise HumanForegroundError(
                f"confirmed mask dimensions differ from source view: {view_id}"
            )
        mask_pixels = int(np.count_nonzero(mask_array))
        if mask_pixels < 1 or mask_record.get("mask_pixels") != mask_pixels:
            raise HumanForegroundError(f"confirmed mask pixel audit failed: {view_id}")
        expected_fraction = round(mask_pixels / max(1, width * height), 8)
        if mask_record.get("image_fraction") != expected_fraction:
            raise HumanForegroundError(
                f"confirmed mask area-fraction audit failed: {view_id}"
            )
        decoded_mask_sha256 = hashlib.sha256(
            np.asarray(mask_array, dtype=np.uint8).tobytes(order="C")
        ).hexdigest()
        if mask_record.get("decoded_mask_sha256") != decoded_mask_sha256:
            raise HumanForegroundError(f"confirmed mask content changed: {view_id}")
        masks[view_id] = mask_path
        normalized_views.append(
            {
                **dict(raw_view),
                "image": str(image_path),
                "click_sets": click_sets,
                "confirmed_mask": {**dict(mask_record), "path": str(mask_path)},
            }
        )
    if expected and seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise HumanForegroundError(
            "annotations do not exactly cover current references; missing: "
            + ", ".join(missing)
        )
    if confirmation.get("confirmed_view_ids") != sorted(seen):
        raise HumanForegroundError("confirmed_view_ids do not match source_views")
    return {**document, "source_views": normalized_views}, masks


def grid_to_pixel(point: Sequence[int], *, width: int, height: int) -> list[float]:
    x, y = point
    return [
        float(x) * max(0, width - 1) / COORDINATE_GRID_SIZE,
        float(y) * max(0, height - 1) / COORDINATE_GRID_SIZE,
    ]


def pixel_to_grid(point: Sequence[float], *, width: int, height: int) -> list[int]:
    x, y = point
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in (x, y)
    ):
        raise HumanForegroundError("click coordinates must be finite numbers")
    if not (0 <= x < width and 0 <= y < height):
        raise HumanForegroundError("click coordinates fall outside the source image")
    return [
        int(round(float(x) * COORDINATE_GRID_SIZE / max(1, width - 1))),
        int(round(float(y) * COORDINATE_GRID_SIZE / max(1, height - 1))),
    ]


def point_candidate_metrics(
    mask: Any,
    *,
    positive_points: Sequence[Sequence[int]],
    negative_points: Sequence[Sequence[int]],
    width: int,
    height: int,
) -> dict[str, Any]:
    import numpy as np

    candidate = np.asarray(mask, dtype=bool)
    if candidate.shape != (height, width):
        raise HumanForegroundError(
            f"SAM3 returned mask shape {candidate.shape}, expected {(height, width)}"
        )

    def sample(point: Sequence[int]) -> bool:
        px, py = grid_to_pixel(point, width=width, height=height)
        x = min(width - 1, max(0, int(round(px))))
        y = min(height - 1, max(0, int(round(py))))
        return bool(candidate[y, x])

    positive_hits = sum(sample(point) for point in positive_points)
    negative_exclusions = sum(not sample(point) for point in negative_points)
    positive_hit_rate = positive_hits / max(1, len(positive_points))
    negative_exclusion_rate = negative_exclusions / max(1, len(negative_points))
    prompt_count = len(positive_points) + len(negative_points)
    prompt_agreement = (positive_hits + negative_exclusions) / max(1, prompt_count)
    mask_pixels = int(np.count_nonzero(candidate))
    return {
        "mask_pixels": mask_pixels,
        "image_fraction": mask_pixels / max(1, width * height),
        "positive_point_count": len(positive_points),
        "negative_point_count": len(negative_points),
        "positive_hits": positive_hits,
        "negative_exclusions": negative_exclusions,
        "positive_hit_rate": positive_hit_rate,
        "negative_exclusion_rate": negative_exclusion_rate,
        "prompt_agreement": prompt_agreement,
        "all_positive_inside": positive_hits == len(positive_points),
        "all_negative_outside": negative_exclusions == len(negative_points),
    }


def select_point_candidate(
    masks: Any,
    scores: Any,
    *,
    positive_points: Sequence[Sequence[int]],
    negative_points: Sequence[Sequence[int]],
    width: int,
    height: int,
    minimum_model_score: float,
    minimum_prompt_agreement: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    allow_rejected_preview: bool = False,
) -> tuple[Any | None, dict[str, Any]]:
    """Apply point-aware gates and select one deterministic SAM3 candidate."""

    import numpy as np

    masks_array = np.asarray(masks)
    scores_array = np.asarray(scores).reshape(-1)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if masks_array.ndim != 3 or len(masks_array) != len(scores_array):
        raise HumanForegroundError("SAM3 point masks/scores have incompatible shapes")
    candidates: list[tuple[tuple[float, float, int], Any, dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for index, (mask, raw_score) in enumerate(zip(masks_array, scores_array)):
        score = float(raw_score)
        metrics = point_candidate_metrics(
            mask,
            positive_points=positive_points,
            negative_points=negative_points,
            width=width,
            height=height,
        )
        reasons: list[str] = []
        if not math.isfinite(score) or score < minimum_model_score:
            reasons.append("model_score_below_threshold")
        if metrics["mask_pixels"] < minimum_mask_pixels:
            reasons.append("mask_too_small")
        if metrics["image_fraction"] > maximum_image_fraction:
            reasons.append("mask_too_large")
        if not metrics["all_positive_inside"]:
            reasons.append("positive_point_outside_mask")
        if not metrics["all_negative_outside"]:
            reasons.append("negative_point_inside_mask")
        if metrics["prompt_agreement"] < minimum_prompt_agreement:
            reasons.append("insufficient_point_agreement")
        audit = {
            "candidate_index": index,
            "model_score": round(score, 8),
            **{
                key: round(float(value), 8) if isinstance(value, float) else value
                for key, value in metrics.items()
            },
            "accepted": not reasons,
            "reason_codes": reasons,
        }
        audits.append(audit)
        if not reasons:
            candidates.append(
                (
                    (score, float(metrics["prompt_agreement"]), -index),
                    np.asarray(mask, dtype=bool),
                    audit,
                )
            )
    if not candidates:
        if not allow_rejected_preview or not audits:
            return None, {"accepted": False, "candidates": audits}
        preview_index = max(
            range(len(audits)),
            key=lambda index: (
                float(audits[index]["prompt_agreement"]),
                float(audits[index]["model_score"]),
                -index,
            ),
        )
        return np.asarray(masks_array[preview_index], dtype=bool), {
            "accepted": False,
            "preview_only": True,
            "preview_candidate_index": preview_index,
            "candidates": audits,
        }
    _rank, selected, selected_audit = max(candidates, key=lambda item: item[0])
    return selected, {
        "accepted": True,
        "selected_candidate_index": selected_audit["candidate_index"],
        "candidates": audits,
    }


def ordered_event_points(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[list[int]], list[list[int]]]:
    positives = [list(event["point"]) for event in events if event["label"] == 1]
    negatives = [list(event["point"]) for event in events if event["label"] == 0]
    return positives, negatives


def _prediction_arrays(
    masks: Any, scores: Any, low_res_logits: Any
) -> tuple[Any, Any, Any]:
    import numpy as np

    def to_numpy(value: Any) -> Any:
        detached = value.detach() if hasattr(value, "detach") else value
        on_cpu = detached.to("cpu") if hasattr(detached, "to") else detached
        return on_cpu.numpy() if hasattr(on_cpu, "numpy") else np.asarray(on_cpu)

    masks_array = to_numpy(masks)
    scores_array = to_numpy(scores).reshape(-1)
    logits_array = to_numpy(low_res_logits)
    if masks_array.ndim == 4:
        if masks_array.shape[0] == 1 and masks_array.shape[1] == len(scores_array):
            masks_array = masks_array[0]
        elif masks_array.shape[1] == 1 and masks_array.shape[0] == len(scores_array):
            masks_array = masks_array[:, 0]
    if logits_array.ndim == 4:
        if logits_array.shape[0] == 1 and logits_array.shape[1] == len(scores_array):
            logits_array = logits_array[0]
        elif logits_array.shape[1] == 1 and logits_array.shape[0] == len(scores_array):
            logits_array = logits_array[:, 0]
    if (
        masks_array.ndim != 3
        or logits_array.ndim != 3
        or len(masks_array) != len(scores_array)
        or len(logits_array) != len(scores_array)
    ):
        raise HumanForegroundError(
            "SAM3 point masks, scores, and low-resolution logits are incompatible"
        )
    return masks_array, scores_array, logits_array


def predict_ordered_point_step(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    events: Sequence[Mapping[str, Any]],
    previous_logits: Any | None,
    initial_candidate_index: int | None,
    minimum_prompt_agreement: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run one deterministic ordered SAM3 interaction step."""

    import numpy as np

    if not events or events[0].get("label") != 1:
        raise HumanForegroundError(
            "ordered SAM3 interaction must begin with a foreground point"
        )
    if (previous_logits is None) is not (len(events) == 1):
        raise HumanForegroundError(
            "ordered SAM3 history must use no logits for the first event and "
            "the previous logits for every later event"
        )
    if previous_logits is not None:
        previous_array = np.asarray(previous_logits)
        if (
            previous_array.ndim != 3
            or previous_array.shape[0] != 1
            or not np.all(np.isfinite(previous_array))
        ):
            raise HumanForegroundError(
                "previous SAM3 logits must be finite with shape (1,H,W)"
            )
    width, height = image.size
    positive_points, negative_points = ordered_event_points(events)
    point_coords = np.asarray(
        [grid_to_pixel(event["point"], width=width, height=height) for event in events],
        dtype=np.float32,
    )
    point_labels = np.asarray([int(event["label"]) for event in events], dtype=np.int32)
    first_step = previous_logits is None
    masks, scores, low_res_logits = model.predict_inst(
        image_state,
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=previous_logits,
        multimask_output=first_step,
    )
    masks_array, scores_array, logits_array = _prediction_arrays(
        masks, scores, low_res_logits
    )
    expected_candidate_count = 3 if first_step else 1
    if len(scores_array) != expected_candidate_count:
        raise HumanForegroundError(
            "SAM3 returned an unexpected number of interactive candidates: "
            f"expected {expected_candidate_count}, got {len(scores_array)}"
        )
    _selected_by_policy, policy_audit = select_point_candidate(
        masks_array,
        scores_array,
        positive_points=positive_points,
        negative_points=negative_points,
        width=width,
        height=height,
        minimum_model_score=0.0,
        minimum_prompt_agreement=minimum_prompt_agreement,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
        allow_rejected_preview=True,
    )
    if first_step and initial_candidate_index is not None:
        selected_index = initial_candidate_index
    else:
        selected_index = policy_audit.get(
            "selected_candidate_index",
            policy_audit.get("preview_candidate_index"),
        )
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < len(masks_array)
    ):
        raise HumanForegroundError("SAM3 did not provide a selectable point mask")
    candidate_audit = policy_audit["candidates"][selected_index]
    selected_mask = np.asarray(masks_array[selected_index], dtype=bool)
    selected_logits = np.asarray(
        logits_array[selected_index : selected_index + 1], dtype=np.float32
    ).copy()
    audit = {
        "accepted": candidate_audit.get("accepted") is True,
        "selected_candidate_index": selected_index,
        "candidate_selection": (
            "persisted_initial_candidate"
            if first_step and initial_candidate_index is not None
            else (
                "automatic_initial_candidate"
                if first_step
                else "single_mask_refinement"
            )
        ),
        "multimask_output": first_step,
        "used_previous_logits": not first_step,
        "event_count": len(events),
        "candidates": policy_audit["candidates"],
    }
    return selected_mask, selected_logits, audit


def replay_ordered_click_set(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    click_set: Mapping[str, Any],
    minimum_prompt_agreement: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Replay one v2 annotation click set exactly, including prior logits."""

    normalized = validate_ordered_click_sets([click_set], "click_set")[0]
    events = normalized["events"]
    previous_logits = None
    selected_mask = None
    event_audits: list[dict[str, Any]] = []
    for event_index in range(len(events)):
        selected_mask, previous_logits, step_audit = predict_ordered_point_step(
            model=model,
            image_state=image_state,
            image=image,
            events=events[: event_index + 1],
            previous_logits=previous_logits,
            initial_candidate_index=(
                normalized["initial_candidate_index"] if event_index == 0 else None
            ),
            minimum_prompt_agreement=minimum_prompt_agreement,
            maximum_image_fraction=maximum_image_fraction,
            minimum_mask_pixels=minimum_mask_pixels,
        )
        event_audits.append(
            {
                "event_index": event_index,
                "event": deepcopy(events[event_index]),
                **step_audit,
            }
        )
    if selected_mask is None or previous_logits is None:
        raise HumanForegroundError("ordered SAM3 replay produced no mask")
    return (
        selected_mask,
        previous_logits,
        {
            "accepted": event_audits[-1]["accepted"] is True,
            "initial_candidate_index": normalized["initial_candidate_index"],
            "event_count": len(events),
            "event_audits": event_audits,
        },
    )


@dataclass
class _InstanceState:
    events: list[dict[str, Any]]
    initial_candidate_index: int
    mask: Any
    logits: Any
    audit: dict[str, Any]


@dataclass
class _ViewState:
    view_id: str
    path: Path
    image: Any
    decoded_rgb_sha256: str
    instances: list[_InstanceState] = field(default_factory=list)
    active_instance_index: int | None = None
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    confirmed: bool = False


class InteractiveForegroundSession:
    """Single-user, GPU-serialized interactive SAM3 annotation session."""

    def __init__(
        self,
        *,
        references: Sequence[tuple[str, Path]],
        repository: Path,
        checkpoint: Path,
        output: Path,
        device: str = "cuda",
        minimum_model_score: float = 0.45,
        minimum_prompt_agreement: float = 0.25,
        maximum_image_fraction: float = 0.90,
        minimum_mask_pixels: int = 32,
        overwrite: bool = False,
    ) -> None:
        if device not in {"cuda", "cpu"}:
            raise HumanForegroundError("device must be cuda or cpu")
        self.repository = repository.expanduser().resolve(strict=True)
        self.checkpoint = checkpoint.expanduser().resolve(strict=True)
        self.output = output.expanduser().resolve()
        self.device = device
        self.minimum_model_score = minimum_model_score
        self.minimum_prompt_agreement = minimum_prompt_agreement
        self.maximum_image_fraction = maximum_image_fraction
        self.minimum_mask_pixels = minimum_mask_pixels
        self.overwrite = overwrite
        if self.output.exists() and not overwrite:
            raise FileExistsError(
                f"annotation output already exists; use --overwrite: {self.output}"
            )
        masks_output = self.output.parent / f"{self.output.stem}_masks"
        if masks_output.exists() and not overwrite:
            raise FileExistsError(
                "annotation mask directory already exists; use --overwrite: "
                f"{masks_output}"
            )
        self.views: dict[str, _ViewState] = {}
        for view_id, path in references:
            image, decoded_digest = _decoded_rgb(path)
            self.views[view_id] = _ViewState(
                view_id=view_id,
                path=path.expanduser().resolve(strict=True),
                image=image,
                decoded_rgb_sha256=decoded_digest,
            )
        if not self.views:
            raise HumanForegroundError("interactive session has no reference views")
        self.view_order = list(self.views)
        self.active_view_id = self.view_order[0]
        self._active_image_state: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._load_model()
        self.activate(self.active_view_id)

    def _load_model(self) -> None:
        sys.path.insert(0, str(self.repository))
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise HumanForegroundError(
                f"unable to import SAM3 from {self.repository}: {exc}"
            ) from exc
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.minimum_model_score,
        )
        if self.model.inst_interactive_predictor is None:
            raise HumanForegroundError("SAM3 interactive predictor was not enabled")

    def activate(self, view_id: str) -> Any:
        with self._lock:
            if view_id not in self.views:
                raise HumanForegroundError(f"unknown reference view: {view_id}")
            if view_id != self.active_view_id or self._active_image_state is None:
                self.active_view_id = view_id
                self._active_image_state = self.processor.set_image(
                    self.views[view_id].image
                )
            return self.render_preview(view_id)

    @staticmethod
    def _pack_instance(instance: _InstanceState) -> dict[str, Any]:
        import numpy as np

        mask = np.asarray(instance.mask, dtype=bool)
        return {
            "events": deepcopy(instance.events),
            "initial_candidate_index": instance.initial_candidate_index,
            "mask_shape": tuple(mask.shape),
            "mask_bits": np.packbits(mask.reshape(-1), bitorder="little"),
            "logits": np.asarray(instance.logits, dtype=np.float32).copy(),
            "audit": deepcopy(instance.audit),
        }

    @staticmethod
    def _unpack_instance(snapshot: Mapping[str, Any]) -> _InstanceState:
        import numpy as np

        shape = tuple(snapshot["mask_shape"])
        size = math.prod(shape)
        mask = np.unpackbits(
            np.asarray(snapshot["mask_bits"], dtype=np.uint8),
            count=size,
            bitorder="little",
        ).reshape(shape)
        return _InstanceState(
            events=deepcopy(snapshot["events"]),
            initial_candidate_index=int(snapshot["initial_candidate_index"]),
            mask=np.asarray(mask, dtype=bool),
            logits=np.asarray(snapshot["logits"], dtype=np.float32).copy(),
            audit=deepcopy(snapshot["audit"]),
        )

    @staticmethod
    def _undo_context(view: _ViewState) -> dict[str, Any]:
        return {
            "active_instance_index": view.active_instance_index,
            "confirmed": view.confirmed,
        }

    @staticmethod
    def _restore_undo_context(view: _ViewState, action: Mapping[str, Any]) -> None:
        context = action["context"]
        view.active_instance_index = context["active_instance_index"]
        view.confirmed = bool(context["confirmed"])

    @staticmethod
    def _record_undo(view: _ViewState, action: dict[str, Any]) -> None:
        view.undo_stack.append(action)
        if len(view.undo_stack) > 20:
            del view.undo_stack[0]

    def _apply_undo(self, view: _ViewState, action: Mapping[str, Any]) -> None:
        kind = action.get("kind")
        index = action.get("index")
        if kind == "created" and isinstance(index, int):
            del view.instances[index]
        elif kind == "updated" and isinstance(index, int):
            view.instances[index] = self._unpack_instance(action["instance"])
        elif kind == "removed" and isinstance(index, int):
            view.instances.insert(index, action["instance"])
        elif kind == "cleared":
            view.instances = action["instances"]
        else:
            raise HumanForegroundError("SAM3 undo history is invalid")
        self._restore_undo_context(view, action)

    @staticmethod
    def _click_set(instance: _InstanceState) -> dict[str, Any]:
        positives, negatives = ordered_event_points(instance.events)
        return {
            "events": deepcopy(instance.events),
            "positive_points": positives,
            "negative_points": negatives,
            "initial_candidate_index": instance.initial_candidate_index,
        }

    def _ensure_active_image(self, view: _ViewState) -> None:
        if view.view_id != self.active_view_id or self._active_image_state is None:
            self.activate(view.view_id)

    def _new_instance(self, view: _ViewState, point: list[int]) -> _InstanceState:
        self._ensure_active_image(view)
        events = [{"point": list(point), "label": 1}]
        mask, logits, audit = predict_ordered_point_step(
            model=self.model,
            image_state=self._active_image_state,
            image=view.image,
            events=events,
            previous_logits=None,
            initial_candidate_index=None,
            minimum_prompt_agreement=self.minimum_prompt_agreement,
            maximum_image_fraction=self.maximum_image_fraction,
            minimum_mask_pixels=self.minimum_mask_pixels,
        )
        return _InstanceState(
            events=events,
            initial_candidate_index=int(audit["selected_candidate_index"]),
            mask=mask,
            logits=logits,
            audit=audit,
        )

    def _replay_instance(self, view: _ViewState, index: int) -> None:
        self._ensure_active_image(view)
        instance = view.instances[index]
        mask, logits, audit = replay_ordered_click_set(
            model=self.model,
            image_state=self._active_image_state,
            image=view.image,
            click_set=self._click_set(instance),
            minimum_prompt_agreement=self.minimum_prompt_agreement,
            maximum_image_fraction=self.maximum_image_fraction,
            minimum_mask_pixels=self.minimum_mask_pixels,
        )
        instance.mask = mask
        instance.logits = logits
        instance.audit = audit["event_audits"][-1]

    def _cold_replay_view(self, view: _ViewState) -> None:
        """Canonicalize one view from its persisted events on a fresh image state."""

        snapshots = [self._pack_instance(instance) for instance in view.instances]
        previous_active_view_id = self.active_view_id
        previous_image_state = self._active_image_state
        try:
            # ``activate`` intentionally reuses the current image state.  Saving
            # must instead match the standalone runner, which always starts from
            # a newly encoded image and replays every instance in order.
            self.active_view_id = view.view_id
            self._active_image_state = self.processor.set_image(view.image)
            for index in range(len(view.instances)):
                self._replay_instance(view, index)
            rejected = [
                index + 1
                for index, instance in enumerate(view.instances)
                if instance.audit.get("accepted") is not True
            ]
            if rejected:
                raise HumanForegroundError(
                    f"view {view.view_id} cold replay rejected instances: "
                    + ", ".join(str(index) for index in rejected)
                )
        except Exception:
            view.instances = [self._unpack_instance(snapshot) for snapshot in snapshots]
            self.active_view_id = previous_active_view_id
            self._active_image_state = previous_image_state
            raise

    def _refine_instance(
        self,
        view: _ViewState,
        index: int,
        *,
        point: list[int],
        point_label: int,
    ) -> None:
        self._ensure_active_image(view)
        instance = view.instances[index]
        if any(event["point"] == point for event in instance.events):
            raise HumanForegroundError("该坐标已经点过；请撤销后重新选择，或点击相邻位置")
        events = [
            *deepcopy(instance.events),
            {"point": list(point), "label": point_label},
        ]
        mask, logits, audit = predict_ordered_point_step(
            model=self.model,
            image_state=self._active_image_state,
            image=view.image,
            events=events,
            previous_logits=instance.logits,
            initial_candidate_index=None,
            minimum_prompt_agreement=self.minimum_prompt_agreement,
            maximum_image_fraction=self.maximum_image_fraction,
            minimum_mask_pixels=self.minimum_mask_pixels,
        )
        instance.events = events
        instance.mask = mask
        instance.logits = logits
        instance.audit = audit

    @staticmethod
    def _mask_contains(
        view: _ViewState, instance: _InstanceState, point: list[int]
    ) -> bool:
        import numpy as np

        px, py = grid_to_pixel(point, width=view.image.width, height=view.image.height)
        x = min(view.image.width - 1, max(0, int(round(px))))
        y = min(view.image.height - 1, max(0, int(round(py))))
        return bool(np.asarray(instance.mask, dtype=bool)[y, x])

    def _instance_at_point(self, view: _ViewState, point: list[int]) -> int | None:
        active = view.active_instance_index
        if (
            active is not None
            and 0 <= active < len(view.instances)
            and self._mask_contains(view, view.instances[active], point)
        ):
            return active
        matches = [
            index
            for index, instance in enumerate(view.instances)
            if (
                instance.audit.get("accepted") is True
                or index == view.active_instance_index
            )
            and self._mask_contains(view, instance, point)
        ]
        if not matches:
            return None
        return min(matches, key=lambda index: int(view.instances[index].mask.sum()))

    def add_point(
        self,
        x: float,
        y: float,
        *,
        foreground: bool | None = None,
        mode: str | None = None,
    ) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            point = pixel_to_grid(
                [x, y], width=view.image.width, height=view.image.height
            )
            interaction_mode = mode or (
                "smart_foreground" if foreground is not False else "background"
            )
            if interaction_mode not in {
                "smart_foreground",
                "refine_active",
                "new_instance",
                "background",
            }:
                raise HumanForegroundError(
                    f"unknown interaction mode: {interaction_mode}"
                )
            context = self._undo_context(view)
            action: dict[str, Any] | None = None
            try:
                target = self._instance_at_point(view, point)
                creating = interaction_mode == "new_instance" or (
                    interaction_mode == "smart_foreground" and target is None
                )
                active = view.active_instance_index
                if (
                    creating
                    and active is not None
                    and 0 <= active < len(view.instances)
                    and view.instances[active].audit.get("accepted") is not True
                ):
                    raise HumanForegroundError("当前实例尚未通过门限；请继续修正、切换候选或删除后再新增零件")
                if creating:
                    action = {
                        "kind": "created",
                        "index": len(view.instances),
                        "context": context,
                    }
                    view.instances.append(self._new_instance(view, point))
                    view.active_instance_index = len(view.instances) - 1
                else:
                    if interaction_mode == "refine_active":
                        target = view.active_instance_index
                    elif interaction_mode == "background" and target is None:
                        target = view.active_instance_index
                    if target is None or not 0 <= target < len(view.instances):
                        raise HumanForegroundError("当前没有可修正实例；请先用智能前景或新增零件点击")
                    action = {
                        "kind": "updated",
                        "index": target,
                        "instance": self._pack_instance(view.instances[target]),
                        "context": context,
                    }
                    self._refine_instance(
                        view,
                        target,
                        point=point,
                        point_label=0 if interaction_mode == "background" else 1,
                    )
                    view.active_instance_index = target
                view.confirmed = False
            except Exception:
                if action is not None:
                    if action["kind"] == "created" and action["index"] < len(
                        view.instances
                    ):
                        del view.instances[action["index"]]
                    elif action["kind"] == "updated":
                        view.instances[action["index"]] = self._unpack_instance(
                            action["instance"]
                        )
                    self._restore_undo_context(view, action)
                raise
            if action is None:
                raise HumanForegroundError("SAM3 click produced no editable action")
            self._record_undo(view, action)
            return self.render_preview(view.view_id), self.status(view.view_id)

    def undo_point(self) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            if view.undo_stack:
                self._apply_undo(view, view.undo_stack.pop())
            return self.render_preview(view.view_id), self.status(view.view_id)

    def clear_draft(self) -> tuple[Any, str]:
        """Backward-compatible alias: remove the active incremental instance."""

        return self.remove_last_region()

    def clear_view(self) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            if view.instances:
                action = {
                    "kind": "cleared",
                    "instances": view.instances,
                    "context": self._undo_context(view),
                }
                view.instances = []
                view.active_instance_index = None
                view.confirmed = False
                self._record_undo(view, action)
            return self.render_preview(view.view_id), self.status(view.view_id)

    def confirm_region(self) -> tuple[Any, str]:
        """Finish editing the active instance; the union was already updated."""

        with self._lock:
            view = self.views[self.active_view_id]
            active = view.active_instance_index
            if (
                active is not None
                and view.instances[active].audit.get("accepted") is not True
            ):
                raise HumanForegroundError("当前实例尚未通过门限，请继续修正、切换候选或删除它")
            view.active_instance_index = None
            return self.render_preview(view.view_id), self.status(view.view_id)

    def cycle_active_candidate(self) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            index = view.active_instance_index
            if index is None or not 0 <= index < len(view.instances):
                raise HumanForegroundError("当前没有可切换候选的实例")
            action = {
                "kind": "updated",
                "index": index,
                "instance": self._pack_instance(view.instances[index]),
                "context": self._undo_context(view),
            }
            try:
                instance = view.instances[index]
                instance.initial_candidate_index = (
                    instance.initial_candidate_index + 1
                ) % 3
                self._replay_instance(view, index)
                view.confirmed = False
            except Exception:
                view.instances[index] = self._unpack_instance(action["instance"])
                self._restore_undo_context(view, action)
                raise
            self._record_undo(view, action)
            return self.render_preview(view.view_id), self.status(view.view_id)

    def remove_last_region(self) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            if view.instances:
                index = (
                    view.active_instance_index
                    if view.active_instance_index is not None
                    else len(view.instances) - 1
                )
                context = self._undo_context(view)
                removed = view.instances.pop(index)
                action = {
                    "kind": "removed",
                    "index": index,
                    "instance": removed,
                    "context": context,
                }
                view.active_instance_index = None
                view.confirmed = False
                self._record_undo(view, action)
            return self.render_preview(view.view_id), self.status(view.view_id)

    def confirm_view(self) -> tuple[Any, str]:
        with self._lock:
            view = self.views[self.active_view_id]
            if not view.instances:
                raise HumanForegroundError("当前视角至少要分割一个前景实例")
            rejected = [
                index + 1
                for index, instance in enumerate(view.instances)
                if instance.audit.get("accepted") is not True
            ]
            if rejected:
                raise HumanForegroundError(
                    "以下实例尚未通过门限: " + ", ".join(str(index) for index in rejected)
                )
            union = self.union_mask(view.view_id, include_draft=False)
            mask_pixels = int(union.sum())
            image_fraction = mask_pixels / max(1, union.size)
            if mask_pixels < self.minimum_mask_pixels:
                raise HumanForegroundError("当前视角的前景并集像素数过少")
            if image_fraction > self.maximum_image_fraction:
                raise HumanForegroundError("当前视角的前景并集覆盖范围过大；请删除错误实例或添加背景点")
            view.active_instance_index = None
            view.confirmed = True
            return self.render_preview(view.view_id), self.status(view.view_id)

    def union_mask(self, view_id: str, *, include_draft: bool = True) -> Any:
        import numpy as np

        view = self.views[view_id]
        union = np.zeros((view.image.height, view.image.width), dtype=bool)
        for index, instance in enumerate(view.instances):
            if instance.audit.get("accepted") is True or (
                include_draft and index == view.active_instance_index
            ):
                union |= np.asarray(instance.mask, dtype=bool)
        return union

    @staticmethod
    def _mask_boundary(mask: Any) -> Any:
        import numpy as np

        value = np.asarray(mask, dtype=bool)
        padded = np.pad(value, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        return value & ~eroded

    def render_preview(self, view_id: str | None = None) -> Any:
        import numpy as np
        from PIL import Image, ImageDraw

        view = self.views[view_id or self.active_view_id]
        pixels = np.asarray(view.image, dtype=np.uint8).copy()
        union = self.union_mask(view.view_id, include_draft=True)
        if np.any(union):
            colour = np.asarray([0, 190, 255], dtype=np.float32)
            pixels[union] = (
                pixels[union].astype(np.float32) * 0.72 + colour * 0.28
            ).astype(np.uint8)
            pixels[self._mask_boundary(union)] = np.asarray(
                [255, 0, 255], dtype=np.uint8
            )
        for instance in view.instances:
            if instance.audit.get("accepted") is not True:
                rejected_boundary = self._mask_boundary(instance.mask)
                pixels[rejected_boundary] = np.asarray([255, 35, 35], dtype=np.uint8)
        active = view.active_instance_index
        if active is not None and 0 <= active < len(view.instances):
            active_boundary = self._mask_boundary(view.instances[active].mask)
            active_colour = (
                [255, 235, 0]
                if view.instances[active].audit.get("accepted") is True
                else [255, 35, 35]
            )
            pixels[active_boundary] = np.asarray(active_colour, dtype=np.uint8)
        preview = Image.fromarray(pixels)
        draw = ImageDraw.Draw(preview)
        base_radius = max(4, min(view.image.size) // 100)
        for instance_index, instance in enumerate(view.instances):
            radius = base_radius + (2 if instance_index == active else 0)
            for event in instance.events:
                x, y = grid_to_pixel(
                    event["point"], width=view.image.width, height=view.image.height
                )
                if event["label"] == 1:
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(0, 255, 80),
                        outline=(20, 20, 20),
                        width=2,
                    )
                else:
                    draw.line(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(255, 30, 30),
                        width=3,
                    )
                    draw.line(
                        (x - radius, y + radius, x + radius, y - radius),
                        fill=(255, 30, 30),
                        width=3,
                    )
            if instance.events:
                label_x, label_y = grid_to_pixel(
                    instance.events[0]["point"],
                    width=view.image.width,
                    height=view.image.height,
                )
                draw.text(
                    (label_x + radius + 2, label_y - radius - 2),
                    f"#{instance_index + 1}",
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
        return preview

    def render_mask(self, view_id: str | None = None) -> Any:
        import numpy as np
        from PIL import Image

        view = self.views[view_id or self.active_view_id]
        mask = self.union_mask(view.view_id, include_draft=True)
        return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)

    def render_cutout(self, view_id: str | None = None) -> Any:
        import numpy as np
        from PIL import Image

        view = self.views[view_id or self.active_view_id]
        rgb = np.asarray(view.image, dtype=np.uint8)
        alpha = (
            np.asarray(
                self.union_mask(view.view_id, include_draft=True), dtype=np.uint8
            )
            * 255
        )
        return Image.fromarray(np.dstack([rgb, alpha]))

    def status(self, view_id: str | None = None) -> str:
        view = self.views[view_id or self.active_view_id]
        accepted_count = sum(
            instance.audit.get("accepted") is True for instance in view.instances
        )
        active = view.active_instance_index
        active_text = "无"
        if active is not None and 0 <= active < len(view.instances):
            instance = view.instances[active]
            positives, negatives = ordered_event_points(instance.events)
            active_text = (
                f"#{active + 1}，候选 {instance.initial_candidate_index + 1}/3，"
                f"正点 {len(positives)} / 负点 {len(negatives)}，"
                f"{'通过' if instance.audit.get('accepted') is True else '仅预览'}"
            )
        coverage = 100.0 * float(
            self.union_mask(view.view_id, include_draft=True).mean()
        )
        return (
            f"视角 {view.view_id}｜实例 {len(view.instances)}，通过 {accepted_count}｜"
            f"当前 {active_text}｜并集覆盖 {coverage:.2f}%｜"
            f"视角 {'已确认' if view.confirmed else '待确认'}"
        )

    def completion_status(self) -> str:
        confirmed = sum(view.confirmed for view in self.views.values())
        return f"已确认视角 {confirmed}/{len(self.views)}"

    def save(self) -> Path:
        import numpy as np
        from PIL import Image

        with self._lock:
            missing = [
                view_id for view_id, view in self.views.items() if not view.confirmed
            ]
            if missing:
                raise HumanForegroundError("以下视角尚未确认: " + ", ".join(missing))
            # Persist the cold-replayed masks, not a union that depends on the
            # order in which the user switched between views during a long UI
            # session.  The standalone material stage uses this same replay.
            for view_id in self.view_order:
                self._cold_replay_view(self.views[view_id])
            for view_id in self.view_order:
                union = self.union_mask(view_id, include_draft=False)
                mask_pixels = int(union.sum())
                image_fraction = mask_pixels / max(1, union.size)
                if mask_pixels < self.minimum_mask_pixels:
                    raise HumanForegroundError(
                        f"view {view_id} foreground union is too small to save"
                    )
                if image_fraction > self.maximum_image_fraction:
                    raise HumanForegroundError(
                        f"view {view_id} foreground union is too large to save"
                    )
            self.output.parent.mkdir(parents=True, exist_ok=True)
            masks_dir = self.output.parent / f"{self.output.stem}_masks"
            if masks_dir.exists():
                if not self.overwrite:
                    raise FileExistsError(
                        f"annotation mask directory already exists: {masks_dir}"
                    )
                if masks_dir.is_symlink():
                    raise HumanForegroundError(
                        "refusing to overwrite a symlinked annotation mask directory: "
                        f"{masks_dir}"
                    )
                if masks_dir.is_dir():
                    shutil.rmtree(masks_dir)
                else:
                    masks_dir.unlink()
            masks_dir.mkdir(parents=True, exist_ok=False)
            source_views: list[dict[str, Any]] = []
            for view_id in self.view_order:
                view = self.views[view_id]
                union = self.union_mask(view_id, include_draft=False)
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", view_id)
                mask_path = masks_dir / f"{safe_id}.png"
                Image.fromarray(np.asarray(union, dtype=np.uint8) * 255).save(mask_path)
                mask_pixels = int(np.count_nonzero(union))
                source_views.append(
                    {
                        "id": view_id,
                        "image": str(view.path),
                        "image_sha256": sha256_file(view.path),
                        "decoded_rgb_sha256": view.decoded_rgb_sha256,
                        "width": view.image.width,
                        "height": view.image.height,
                        "click_sets": [
                            self._click_set(instance) for instance in view.instances
                        ],
                        "confirmed_mask": {
                            "path": mask_path.relative_to(
                                self.output.parent
                            ).as_posix(),
                            "sha256": sha256_file(mask_path),
                            "decoded_mask_sha256": hashlib.sha256(
                                np.asarray(union, dtype=np.uint8).tobytes(order="C")
                            ).hexdigest(),
                            "mask_pixels": mask_pixels,
                            "image_fraction": round(
                                mask_pixels / max(1, union.size), 8
                            ),
                        },
                    }
                )
            unsigned: dict[str, Any] = {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "prompt_authority": "human_confirmed_sam3_interactive_points",
                "coordinate_space": {
                    "type": "exif_transposed_image_grid",
                    "grid_size": COORDINATE_GRID_SIZE,
                    "origin": "top_left",
                    "axes": "x_right_y_down",
                },
                "sam3": {
                    "repository": str(self.repository),
                    "repository_revision": git_revision(self.repository),
                    "checkpoint": str(self.checkpoint),
                    "checkpoint_sha256": sha256_file(self.checkpoint),
                    "device": self.device,
                    "mode": "instance_interactivity",
                },
                "policy": {
                    "minimum_model_score": self.minimum_model_score,
                    "human_point_model_score_authority": "advisory",
                    "minimum_prompt_agreement": self.minimum_prompt_agreement,
                    "maximum_image_fraction": self.maximum_image_fraction,
                    "minimum_mask_pixels": self.minimum_mask_pixels,
                    "disconnected_region_policy": ("incremental_instances_then_union"),
                    "interaction_policy": (
                        "smart_outside_add_inside_refine_with_explicit_overrides"
                    ),
                    "ordered_replay_policy": (
                        "first_multimask_then_previous_logits_single_mask"
                    ),
                },
                "source_views": source_views,
                "confirmation": {
                    "all_views_confirmed": True,
                    "confirmed_view_ids": sorted(self.views),
                    "human_mask_is_authoritative": True,
                },
            }
            document = {
                **unsigned,
                "integrity": {"document_sha256": canonical_sha256(unsigned)},
            }
            temporary = self.output.with_name(self.output.name + ".tmp")
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.output)
            load_annotations(
                self.output,
                references=[
                    (view_id, self.views[view_id].path) for view_id in self.view_order
                ],
                repository=self.repository,
                checkpoint=self.checkpoint,
            )
            return self.output


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "COORDINATE_GRID_SIZE",
    "LEGACY_ANNOTATION_SCHEMA_VERSION",
    "HumanForegroundError",
    "InteractiveForegroundSession",
    "canonical_sha256",
    "annotation_replay_policy",
    "git_revision",
    "grid_to_pixel",
    "load_annotations",
    "materialize_annotation_bundle",
    "ordered_event_points",
    "parse_reference_specs",
    "pixel_to_grid",
    "point_candidate_metrics",
    "predict_ordered_point_step",
    "replay_ordered_click_set",
    "require_replay_policy",
    "select_point_candidate",
    "sha256_file",
    "validate_click_sets",
    "validate_ordered_click_sets",
]
