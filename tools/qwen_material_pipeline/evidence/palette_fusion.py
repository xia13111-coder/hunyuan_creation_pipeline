"""Deterministic, lossless fusion of independently verified view palettes.

This module deliberately does not select a winning reference view.  Every
validated local palette group becomes a source observation of one canonical
appearance group.  Source view IDs, local group IDs, and original normalized
boxes remain authoritative in the output; canonical groups never invent a
contact-sheet coordinate system.

The default association is intentionally narrow.  Physical family and finish
normally match exactly, while blue/cyan and orange/brown use their
illumination-stable colour family.  Three bounded exceptions apply to a
chromatic colour that occurs at most once per view:

* unresolved pixel-only observations may join one unambiguous interpretation;
* identical normalized pixel-only observations from at least two independent
  views may join one unresolved appearance group;
* a single substrate-family outlier may join a painted appearance whose
  physical interpretation is independently dominant in at least two views.

The second rule reflects that an opaque coating is visible while its substrate
is often not.  It never crosses a finish disagreement, a per-view duplicate,
or a tied family vote.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.staged_analysis import validate_palette
from .color_semantics import fusion_color_label
from .palette_augmentation import (
    ACCENT_AUGMENTATION_SCHEMA_VERSION,
)


FUSION_SCHEMA_VERSION = "qwen-multiview-palette-fusion/v1"
CANONICAL_PALETTE_SCHEMA_VERSION = "qwen-canonical-material-palette/v1"

_UNCERTAIN_SEMANTICS = frozenset({"other", "unknown"})
_UNIQUE_CHROMATIC_COLORS = frozenset(
    {"red", "yellow", "green", "blue", "orange", "pink"}
)
UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION = (
    "identical_unresolved_pixel_chromatic_multiview"
)
UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION = (
    "identical_unresolved_pixel_light_neutral_multiview"
)
UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION = (
    "identical_unresolved_pixel_masked_dark_multiview"
)
ACCENT_FUSION_EVIDENCE_SCHEMA_VERSION = (
    "qwen-palette-accent-fusion-evidence/v1"
)
UNRESOLVED_PIXEL_ASSOCIATION_EVIDENCE_SCHEMA_VERSION = (
    "qwen-unresolved-pixel-chromatic-association/v1"
)


class PaletteFusionError(ValueError):
    """Raised when multiple palettes cannot be fused without ambiguity."""


def _appearance_key(
    group: Mapping[str, Any],
    *,
    view_id: str,
    local_group_id: str,
    chromatic_association: str | None = None,
) -> tuple[str, str, str, str, str]:
    family = str(group["family_hint"])
    color = fusion_color_label(str(group["base_color"]))
    finish = str(group["finish_hint"])
    if chromatic_association == UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION and (
        color in _UNIQUE_CHROMATIC_COLORS
    ):
        return ("unresolved_pixel_chromatic", color, "", "", "")
    if (
        chromatic_association == UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
        and color == "white"
    ):
        return ("unresolved_pixel_light_neutral", color, "", "", "")
    if (
        chromatic_association == UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
        and color == "black"
    ):
        return ("unresolved_pixel_masked_dark", color, "", "", "")
    if (
        chromatic_association == "unique_chromatic"
        and color in _UNIQUE_CHROMATIC_COLORS
    ):
        return ("unique_chromatic", color, "", "", "")
    if (
        family in _UNCERTAIN_SEMANTICS
        or color in _UNCERTAIN_SEMANTICS
        or finish in _UNCERTAIN_SEMANTICS
    ):
        # Two "unknown" observations are not evidence that the appearances
        # match.  Qualify the key so each remains an auditable singleton.
        return ("singleton", view_id, local_group_id, family, finish)
    return ("appearance", family, color, finish, "")


def _normalized_description(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _unresolved_pixel_description(group: Mapping[str, Any]) -> str | None:
    """Return a strict normalized pixel-only label, otherwise ``None``."""

    family = str(group["family_hint"]).casefold()
    finish = str(group["finish_hint"]).casefold()
    color = fusion_color_label(str(group["base_color"]))
    description = _normalized_description(group["visual_description"])
    if color == "white":
        expected = (
            "connected light neutral surface region detected from pixels; "
            "physical material unresolved"
        )
        if (
            family in _UNCERTAIN_SEMANTICS
            and finish in _UNCERTAIN_SEMANTICS
            and description == expected
        ):
            return description
        return None
    if color == "black":
        expected = (
            "connected dark surface region detected inside the trusted "
            "foreground mask; physical material unresolved"
        )
        if (
            family in _UNCERTAIN_SEMANTICS
            and finish in _UNCERTAIN_SEMANTICS
            and description == expected
        ):
            return description
        return None
    if (
        family not in _UNCERTAIN_SEMANTICS
        or finish not in _UNCERTAIN_SEMANTICS
        or color not in _UNIQUE_CHROMATIC_COLORS
        or not description.startswith("connected ")
        or " chromatic region detected from pixels" not in description
        or not description.endswith("physical material unresolved")
    ):
        return None
    return description


def _source_observation(
    view_id: str,
    group: Mapping[str, Any],
    *,
    augmentation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "view_id": view_id,
        "local_group_id": str(group["group_id"]),
        "family_hint": str(group["family_hint"]),
        "base_color": str(group["base_color"]),
        "finish_hint": str(group["finish_hint"]),
        "visual_description": str(group["visual_description"]),
        "confidence": float(group["confidence"]),
        "boxes": [list(box) for box in group["boxes"]],
    }
    if augmentation_evidence is not None:
        result["accent_augmentation_evidence"] = dict(augmentation_evidence)
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_group_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group_id": source.get("local_group_id"),
        "family_hint": source.get("family_hint"),
        "base_color": source.get("base_color"),
        "finish_hint": source.get("finish_hint"),
        "visual_description": source.get("visual_description"),
        "confidence": source.get("confidence"),
        "boxes": source.get("boxes"),
    }


def _augmentation_evidence(
    *,
    view_id: str,
    palette: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Return strict augmentation proof keyed by surviving local group ID."""

    if audit is None:
        return {}
    if audit.get("schema_version") != ACCENT_AUGMENTATION_SCHEMA_VERSION:
        raise PaletteFusionError(
            f"accent augmentation for {view_id} has an unsupported schema_version"
        )
    if audit.get("source_view_id") != view_id:
        raise PaletteFusionError(
            f"accent augmentation source view does not match palette {view_id}"
        )
    image_sha256 = audit.get("image_sha256")
    if not _is_sha256(image_sha256):
        raise PaletteFusionError(
            f"accent augmentation for {view_id} lacks image_sha256"
        )
    raw_added_group_ids = audit.get("added_group_ids")
    if (
        isinstance(raw_added_group_ids, (str, bytes))
        or not isinstance(raw_added_group_ids, Sequence)
        or any(
            not isinstance(group_id, str) or not group_id
            for group_id in raw_added_group_ids
        )
        or len(set(raw_added_group_ids)) != len(raw_added_group_ids)
    ):
        raise PaletteFusionError(
            f"accent augmentation for {view_id} has invalid added_group_ids"
        )
    groups_by_id = {
        str(group["group_id"]): group for group in palette["groups"]
    }
    raw_components = audit.get("components")
    if (
        isinstance(raw_components, (str, bytes))
        or not isinstance(raw_components, Sequence)
        or any(not isinstance(component, Mapping) for component in raw_components)
    ):
        raise PaletteFusionError(
            f"accent augmentation for {view_id} has invalid components"
        )
    added_components = [
        component
        for component in raw_components
        if component.get("decision") == "added"
    ]
    if len(added_components) != len(raw_added_group_ids):
        raise PaletteFusionError(
            f"accent augmentation for {view_id} has inconsistent added components"
        )
    audit_sha256 = _canonical_sha256(audit)
    result: dict[str, dict[str, Any]] = {}
    for group_id, component in zip(raw_added_group_ids, added_components):
        group = groups_by_id.get(group_id)
        if group is None:
            continue
        color = fusion_color_label(str(component.get("base_color", "")))
        raw_accepted = component.get("accepted_components")
        if (
            color not in (_UNIQUE_CHROMATIC_COLORS | {"white", "black"})
            or isinstance(raw_accepted, (str, bytes))
            or not isinstance(raw_accepted, Sequence)
            or not raw_accepted
            or any(not isinstance(item, Mapping) for item in raw_accepted)
        ):
            raise PaletteFusionError(
                f"accent augmentation group {view_id}/{group_id} is invalid"
            )
        expected_boxes = [item.get("box") for item in raw_accepted]
        expected_description = (
            "connected light neutral surface region detected from pixels; "
            "physical material unresolved"
            if color == "white"
            else (
                (
                    "connected dark surface region detected inside the trusted "
                    "foreground mask; physical material unresolved"
                )
                if color == "black"
                else (
                    f"connected {color} chromatic region detected from pixels; "
                    "physical material unresolved"
                )
            )
        )
        retained_boxes_are_audited = bool(group["boxes"]) and all(
            box in expected_boxes for box in group["boxes"]
        )
        if (
            str(group["family_hint"]).casefold() not in _UNCERTAIN_SEMANTICS
            or str(group["finish_hint"]).casefold() not in _UNCERTAIN_SEMANTICS
            or fusion_color_label(str(group["base_color"])) != color
            or _normalized_description(group["visual_description"])
            != expected_description
            # The second pixel-evidence pass is deliberately allowed to reject
            # individual component boxes.  Every box that survives must still
            # be present verbatim in the immutable augmentation audit; adding
            # or moving a box remains a fail-closed contract violation.
            or not retained_boxes_are_audited
        ):
            raise PaletteFusionError(
                f"accent augmentation group {view_id}/{group_id} differs "
                "from its pixel-component audit"
            )
        proof = {
            "schema_version": ACCENT_FUSION_EVIDENCE_SCHEMA_VERSION,
            "source_view_id": view_id,
            "added_group_id": group_id,
            "augmentation_audit_sha256": audit_sha256,
            "reference_image_sha256": image_sha256,
            "augmented_group_sha256": _canonical_sha256(group),
        }
        if color == "black":
            mask_sha256 = audit.get("mask_sha256")
            if (
                audit.get("masked_dark_recovery_enabled") is not True
                or not isinstance(audit.get("mask"), str)
                or not audit.get("mask")
                or not _is_sha256(mask_sha256)
            ):
                raise PaletteFusionError(
                    f"masked dark augmentation group {view_id}/{group_id} "
                    "lacks foreground-mask proof"
                )
            proof["foreground_mask_sha256"] = mask_sha256
        result[group_id] = proof
    return result


def is_verified_unresolved_pixel_chromatic_group(
    group: Mapping[str, Any],
) -> bool:
    """Return whether a canonical unresolved chromatic group is proof-bound.

    This is the shared trust boundary used by policy authoring and the
    cross-runtime policy validator.  Merely copying the synthetic description
    or assigning two different view labels is insufficient: every source must
    reproduce a pixel-augmentation group hash, and the reference images must
    have different content hashes.
    """

    return _is_verified_unresolved_pixel_group(
        group,
        association_basis=UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION,
        allowed_colors=_UNIQUE_CHROMATIC_COLORS,
    )


def is_verified_unresolved_pixel_light_neutral_group(
    group: Mapping[str, Any],
) -> bool:
    """Return whether a multiview light-neutral pixel group is proof-bound."""

    return _is_verified_unresolved_pixel_group(
        group,
        association_basis=UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION,
        allowed_colors=frozenset({"white"}),
    )


def is_verified_unresolved_pixel_masked_dark_group(
    group: Mapping[str, Any],
) -> bool:
    """Return whether a multiview dark group is foreground-mask proof-bound."""

    return _is_verified_unresolved_pixel_group(
        group,
        association_basis=UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION,
        allowed_colors=frozenset({"black"}),
        foreground_mask_required=True,
    )


def _is_verified_unresolved_pixel_group(
    group: Mapping[str, Any],
    *,
    association_basis: str,
    allowed_colors: frozenset[str],
    foreground_mask_required: bool = False,
) -> bool:
    if (
        group.get("association_basis") != association_basis
        or str(group.get("family_hint", "")).casefold()
        not in _UNCERTAIN_SEMANTICS
        or str(group.get("finish_hint", "")).casefold()
        not in _UNCERTAIN_SEMANTICS
        or group.get("singleton") is not False
    ):
        return False
    canonical_color = fusion_color_label(str(group.get("base_color", "")))
    if canonical_color not in allowed_colors:
        return False
    raw_view_ids = group.get("source_view_ids")
    raw_sources = group.get("sources")
    if (
        isinstance(raw_view_ids, (str, bytes))
        or not isinstance(raw_view_ids, Sequence)
        or any(not isinstance(view_id, str) or not view_id for view_id in raw_view_ids)
        or list(raw_view_ids) != sorted(set(raw_view_ids))
        or len(raw_view_ids) < 2
        or isinstance(raw_sources, (str, bytes))
        or not isinstance(raw_sources, Sequence)
        or len(raw_sources) != len(raw_view_ids)
        or group.get("distinct_view_count") != len(raw_view_ids)
        or group.get("source_count") != len(raw_sources)
    ):
        return False

    descriptions: set[str] = set()
    observed_views: set[str] = set()
    audit_hashes: list[str] = []
    image_hashes: list[str] = []
    group_hashes: list[str] = []
    mask_hashes: list[str] = []
    expected_proof_keys = {
        "schema_version",
        "source_view_id",
        "added_group_id",
        "augmentation_audit_sha256",
        "reference_image_sha256",
        "augmented_group_sha256",
    }
    if foreground_mask_required:
        expected_proof_keys.add("foreground_mask_sha256")
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            return False
        view_id = raw_source.get("view_id")
        local_group_id = raw_source.get("local_group_id")
        description = _unresolved_pixel_description(raw_source)
        proof = raw_source.get("accent_augmentation_evidence")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in observed_views
            or not isinstance(local_group_id, str)
            or not local_group_id
            or description is None
            or fusion_color_label(str(raw_source.get("base_color", "")))
            != canonical_color
            or not isinstance(proof, Mapping)
            or set(proof) != expected_proof_keys
            or proof.get("schema_version")
            != ACCENT_FUSION_EVIDENCE_SCHEMA_VERSION
            or proof.get("source_view_id") != view_id
            or proof.get("added_group_id") != local_group_id
            or not _is_sha256(proof.get("augmentation_audit_sha256"))
            or not _is_sha256(proof.get("reference_image_sha256"))
            or not _is_sha256(proof.get("augmented_group_sha256"))
            or (
                foreground_mask_required
                and not _is_sha256(proof.get("foreground_mask_sha256"))
            )
            or proof.get("augmented_group_sha256")
            != _canonical_sha256(_source_group_payload(raw_source))
        ):
            return False
        observed_views.add(view_id)
        descriptions.add(description)
        audit_hashes.append(str(proof["augmentation_audit_sha256"]))
        image_hashes.append(str(proof["reference_image_sha256"]))
        group_hashes.append(str(proof["augmented_group_sha256"]))
        if foreground_mask_required:
            mask_hashes.append(str(proof["foreground_mask_sha256"]))
    if (
        observed_views != set(raw_view_ids)
        or len(descriptions) != 1
        or len(set(image_hashes)) != len(image_hashes)
        or (
            foreground_mask_required
            and len(set(mask_hashes)) != len(mask_hashes)
        )
    ):
        return False
    expected_association_evidence = {
        "schema_version": UNRESOLVED_PIXEL_ASSOCIATION_EVIDENCE_SCHEMA_VERSION,
        "source_view_ids": list(raw_view_ids),
        "augmentation_audit_sha256s": sorted(audit_hashes),
        "reference_image_sha256s": sorted(image_hashes),
        "augmented_group_sha256s": sorted(group_hashes),
    }
    if foreground_mask_required:
        expected_association_evidence["foreground_mask_sha256s"] = sorted(
            mask_hashes
        )
    return group.get("association_evidence") == expected_association_evidence


def _categorical_consensus(
    sources: Sequence[Mapping[str, Any]], field: str
) -> tuple[str, list[dict[str, Any]]]:
    """Choose a stable vote; unresolved values abstain when evidence exists."""

    votes: dict[str, list[float]] = {}
    for source in sources:
        value = str(source[field])
        votes.setdefault(value, []).append(float(source["confidence"]))
    resolved_values = {
        value for value in votes if value not in _UNCERTAIN_SEMANTICS
    }
    eligible_values = resolved_values or set(votes)
    ordered = sorted(
        votes.items(),
        key=lambda item: (
            item[0] not in eligible_values,
            -len(item[1]),
            -sum(item[1]),
            item[0],
        ),
    )
    winner = ordered[0][0]
    audit = [
        {
            "value": value,
            "source_count": len(confidences),
            "confidence_sum": round(sum(confidences), 8),
        }
        for value, confidences in ordered
    ]
    return winner, audit


def _consensus_representative(
    sources: Sequence[Mapping[str, Any]],
    *,
    family_hint: str,
    finish_hint: str,
) -> Mapping[str, Any]:
    """Prefer a real source matching the canonical categorical consensus."""

    return min(
        sources,
        key=lambda source: (
            -int(str(source["family_hint"]) == family_hint),
            -int(str(source["finish_hint"]) == finish_hint),
            -float(source["confidence"]),
            str(source["view_id"]),
            str(source["local_group_id"]),
            str(source["visual_description"]),
        ),
    )


def _canonical_id(index: int, count: int) -> str:
    if count > 9999:
        raise PaletteFusionError(
            "multiview palette fusion supports at most 9999 canonical groups"
        )
    width = max(2, len(str(count)))
    return f"G{index:0{width}d}"


def fuse_multiview_palettes(
    palettes: Sequence[Mapping[str, Any]],
    *,
    augmentation_audits: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an input-order-independent union of per-view palettes.

    ``palettes`` must contain one normal ``qwen-material-palette/v1`` document
    per reference view.  The returned ``view_group_id_maps`` is the exact,
    deterministic local-to-canonical join needed by later multiview consumers.
    No source box is transformed, truncated, or copied into another view.
    """

    if isinstance(palettes, (str, bytes)) or not isinstance(palettes, Sequence):
        raise PaletteFusionError("palettes must be an array of palette objects")
    if not palettes:
        raise PaletteFusionError("palettes cannot be empty")
    if augmentation_audits is not None and (
        isinstance(augmentation_audits, (str, bytes))
        or not isinstance(augmentation_audits, Sequence)
        or len(augmentation_audits) != len(palettes)
    ):
        raise PaletteFusionError(
            "augmentation_audits must match the palette array length"
        )

    by_view: dict[str, dict[str, Any]] = {}
    augmentation_evidence_by_view: dict[str, dict[str, dict[str, Any]]] = {}
    for index, raw_palette in enumerate(palettes):
        if not isinstance(raw_palette, Mapping):
            raise PaletteFusionError(f"palettes[{index}] must be an object")
        canonical = validate_palette(raw_palette)
        view_id = canonical["source_view_id"]
        if view_id in by_view:
            raise PaletteFusionError(f"duplicate source_view_id: {view_id}")
        by_view[view_id] = canonical
        raw_audit = (
            augmentation_audits[index]
            if augmentation_audits is not None
            else None
        )
        if raw_audit is not None and not isinstance(raw_audit, Mapping):
            raise PaletteFusionError(
                f"augmentation_audits[{index}] must be an object"
            )
        augmentation_evidence_by_view[view_id] = _augmentation_evidence(
            view_id=view_id,
            palette=canonical,
            audit=raw_audit,
        )

    color_counts_by_view: dict[str, dict[str, int]] = {}
    semantic_signatures_by_color: dict[str, set[tuple[str, str]]] = {}
    signature_views_by_color: dict[
        str, dict[tuple[str, str], set[str]]
    ] = {}
    unresolved_pixel_observations_by_color: dict[
        str, list[tuple[str, str, str]]
    ] = {}
    for view_id, palette in by_view.items():
        counts: dict[str, int] = {}
        for group in palette["groups"]:
            color = fusion_color_label(str(group["base_color"]))
            counts[color] = counts.get(color, 0) + 1
            family = str(group["family_hint"])
            finish = str(group["finish_hint"])
            if (
                family not in _UNCERTAIN_SEMANTICS
                and finish not in _UNCERTAIN_SEMANTICS
            ):
                semantic_signatures_by_color.setdefault(color, set()).add(
                    (family, finish)
                )
                signature_views_by_color.setdefault(color, {}).setdefault(
                    (family, finish), set()
                ).add(view_id)
            unresolved_description = _unresolved_pixel_description(group)
            if (
                unresolved_description is not None
                and str(group["group_id"])
                in augmentation_evidence_by_view[view_id]
            ):
                unresolved_pixel_observations_by_color.setdefault(
                    color, []
                ).append(
                    (
                        view_id,
                        unresolved_description,
                        str(
                            augmentation_evidence_by_view[view_id][
                                str(group["group_id"])
                            ]["reference_image_sha256"]
                        ),
                    )
                )
        color_counts_by_view[view_id] = counts
    unique_chromatic_colors = {
        color
        for color in _UNIQUE_CHROMATIC_COLORS
        if all(counts.get(color, 0) <= 1 for counts in color_counts_by_view.values())
        and len(semantic_signatures_by_color.get(color, set())) == 1
    }
    dominant_painted_chromatic_colors: set[str] = set()
    for color in _UNIQUE_CHROMATIC_COLORS:
        if not all(
            counts.get(color, 0) <= 1 for counts in color_counts_by_view.values()
        ):
            continue
        signature_views = signature_views_by_color.get(color, {})
        if len(signature_views) < 2:
            continue
        if {finish for _family, finish in signature_views} != {"painted"}:
            continue
        ranked_support = sorted(
            (len(view_ids) for view_ids in signature_views.values()),
            reverse=True,
        )
        # Only one isolated outlier may abstain.  A 3-vs-2 split or several
        # different singleton interpretations is real ambiguity, not noise.
        if (
            len(signature_views) == 2
            and ranked_support[0] >= 2
            and ranked_support[1] == 1
        ):
            dominant_painted_chromatic_colors.add(color)
    unresolved_pixel_chromatic_colors: set[str] = set()
    for color, observations in unresolved_pixel_observations_by_color.items():
        total_color_observation_count = sum(
            counts.get(color, 0) for counts in color_counts_by_view.values()
        )
        observation_image_hashes = {
            image_sha256
            for _view_id, _description, image_sha256 in observations
        }
        if (
            all(
                counts.get(color, 0) <= 1
                for counts in color_counts_by_view.values()
            )
            and len(
                {
                    view_id
                    for view_id, _description, _image_sha256 in observations
                }
            )
            >= 2
            and len(observation_image_hashes) == len(observations)
            and len(observations) == total_color_observation_count
            and len(
                {
                    description
                    for _view_id, description, _image_sha256 in observations
                }
            )
            == 1
        ):
            unresolved_pixel_chromatic_colors.add(color)
    # A resolved Qwen black group may coexist with independently recovered
    # masked-dark evidence (for example, one oblique view names a black base
    # while the other views recover dark arms from pixels).  Keep that resolved
    # group separate, but allow the mask-bound synthetic observations to join
    # one another.  Association is applied per source below, never by colour
    # alone, so the resolved group cannot be pulled into this cluster.
    unresolved_pixel_masked_dark_colors: set[str] = set()
    dark_observations = unresolved_pixel_observations_by_color.get("black", [])
    if (
        all(
            counts.get("black", 0) <= 1
            for counts in color_counts_by_view.values()
        )
        and len({view_id for view_id, _description, _sha in dark_observations})
        >= 2
        and len({sha for _view_id, _description, sha in dark_observations})
        == len(dark_observations)
        and len(
            {
                description
                for _view_id, description, _sha in dark_observations
            }
        )
        == 1
    ):
        unresolved_pixel_masked_dark_colors.add("black")
    # A model-described white component may coexist with independently
    # recovered light-neutral pixels in other views.  Treat this exactly like
    # the masked-dark case above: join only the augmentation-bound unresolved
    # observations, and leave the resolved singleton as a separate material
    # hypothesis.  Requiring distinct source-image hashes prevents repeated
    # copies of one photograph from manufacturing multiview support.
    unresolved_pixel_light_neutral_colors: set[str] = set()
    light_neutral_observations = unresolved_pixel_observations_by_color.get(
        "white", []
    )
    if (
        all(
            counts.get("white", 0) <= 1
            for counts in color_counts_by_view.values()
        )
        and len(
            {
                view_id
                for view_id, _description, _sha in light_neutral_observations
            }
        )
        >= 2
        and len(
            {
                sha
                for _view_id, _description, sha in light_neutral_observations
            }
        )
        == len(light_neutral_observations)
        and len(
            {
                description
                for _view_id, description, _sha in light_neutral_observations
            }
        )
        == 1
    ):
        unresolved_pixel_light_neutral_colors.add("white")
    joinable_chromatic_colors = (
        unique_chromatic_colors | dominant_painted_chromatic_colors
    )

    clusters: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for view_id in sorted(by_view):
        palette = by_view[view_id]
        for group in sorted(palette["groups"], key=lambda item: item["group_id"]):
            augmentation_evidence = augmentation_evidence_by_view[view_id].get(
                str(group["group_id"])
            )
            source = _source_observation(
                view_id,
                group,
                augmentation_evidence=augmentation_evidence,
            )
            color = fusion_color_label(str(group["base_color"]))
            if (
                color in unresolved_pixel_masked_dark_colors
                and augmentation_evidence is not None
                and _unresolved_pixel_description(group) is not None
            ):
                chromatic_association = (
                    UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
                )
            elif (
                color in unresolved_pixel_light_neutral_colors
                and augmentation_evidence is not None
                and _unresolved_pixel_description(group) is not None
            ):
                chromatic_association = (
                    UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
                )
            elif color in unresolved_pixel_chromatic_colors:
                chromatic_association = (
                    UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
                    if color == "white"
                    else UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION
                )
            else:
                chromatic_association = (
                    "unique_chromatic"
                    if color in joinable_chromatic_colors
                    else None
                )
            key = _appearance_key(
                group,
                view_id=view_id,
                local_group_id=source["local_group_id"],
                chromatic_association=chromatic_association,
            )
            clusters.setdefault(key, []).append(source)

    ordered_clusters = sorted(clusters.items(), key=lambda item: item[0])
    canonical_groups: list[dict[str, Any]] = []
    view_maps: dict[str, dict[str, str]] = {view_id: {} for view_id in sorted(by_view)}
    for index, (key, raw_sources) in enumerate(ordered_clusters, start=1):
        canonical_id = _canonical_id(index, len(ordered_clusters))
        sources = sorted(
            raw_sources,
            key=lambda source: (
                source["view_id"],
                source["local_group_id"],
            ),
        )
        family_hint, family_votes = _categorical_consensus(sources, "family_hint")
        finish_hint, finish_votes = _categorical_consensus(sources, "finish_hint")
        representative = _consensus_representative(
            sources,
            family_hint=family_hint,
            finish_hint=finish_hint,
        )
        view_ids = sorted({str(source["view_id"]) for source in sources})
        confidences = [float(source["confidence"]) for source in sources]
        base_color = (
            fusion_color_label(str(representative["base_color"]))
            if key[0]
            in {
                "appearance",
                "unique_chromatic",
                "unresolved_pixel_chromatic",
                "unresolved_pixel_light_neutral",
                "unresolved_pixel_masked_dark",
            }
            else str(representative["base_color"])
        )
        canonical_group = {
                "group_id": canonical_id,
                "family_hint": family_hint,
                "base_color": base_color,
                "finish_hint": finish_hint,
                "visual_description": str(representative["visual_description"]),
                "confidence": max(confidences),
                "minimum_source_confidence": min(confidences),
                "mean_source_confidence": round(sum(confidences) / len(confidences), 8),
                "source_view_ids": view_ids,
                "distinct_view_count": len(view_ids),
                "source_count": len(sources),
                "singleton": len(view_ids) == 1,
                "representative_ref": {
                    "view_id": representative["view_id"],
                    "local_group_id": representative["local_group_id"],
                },
                "semantic_consensus": {
                    "family_hint_votes": family_votes,
                    "finish_hint_votes": finish_votes,
                },
                "sources": sources,
        }
        if key[0] in {
            "unresolved_pixel_chromatic",
            "unresolved_pixel_light_neutral",
            "unresolved_pixel_masked_dark",
        }:
            association_basis = (
                UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
                if key[0] == "unresolved_pixel_light_neutral"
                else (
                    UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
                    if key[0] == "unresolved_pixel_masked_dark"
                    else UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION
                )
            )
            canonical_group["association_basis"] = association_basis
            canonical_group["association_evidence"] = {
                "schema_version": (
                    UNRESOLVED_PIXEL_ASSOCIATION_EVIDENCE_SCHEMA_VERSION
                ),
                "source_view_ids": view_ids,
                "augmentation_audit_sha256s": sorted(
                    str(
                        source["accent_augmentation_evidence"][
                            "augmentation_audit_sha256"
                        ]
                    )
                    for source in sources
                ),
                "reference_image_sha256s": sorted(
                    str(
                        source["accent_augmentation_evidence"][
                            "reference_image_sha256"
                        ]
                    )
                    for source in sources
                ),
                "augmented_group_sha256s": sorted(
                    str(
                        source["accent_augmentation_evidence"][
                            "augmented_group_sha256"
                        ]
                    )
                    for source in sources
                ),
            }
            if (
                association_basis
                == UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
            ):
                canonical_group["association_evidence"][
                    "foreground_mask_sha256s"
                ] = sorted(
                    str(
                        source["accent_augmentation_evidence"][
                            "foreground_mask_sha256"
                        ]
                    )
                    for source in sources
                )
            verified = (
                is_verified_unresolved_pixel_light_neutral_group(canonical_group)
                if association_basis
                == UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION
                else (
                    is_verified_unresolved_pixel_masked_dark_group(canonical_group)
                    if association_basis
                    == UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION
                    else is_verified_unresolved_pixel_chromatic_group(
                        canonical_group
                    )
                )
            )
            if not verified:
                raise PaletteFusionError(
                    "unresolved pixel association failed its "
                    "hash-bound evidence contract"
                )
        canonical_groups.append(canonical_group)
        for source in sources:
            view_maps[str(source["view_id"])][str(source["local_group_id"])] = (
                canonical_id
            )

    input_group_count = sum(len(palette["groups"]) for palette in by_view.values())
    singleton_count = sum(group["singleton"] for group in canonical_groups)
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "canonical_palette": {
            "schema_version": CANONICAL_PALETTE_SCHEMA_VERSION,
            "groups": canonical_groups,
        },
        "source_views": [
            {
                "view_id": view_id,
                "local_group_count": len(by_view[view_id]["groups"]),
            }
            for view_id in sorted(by_view)
        ],
        "view_group_id_maps": {
            view_id: dict(sorted(mapping.items()))
            for view_id, mapping in sorted(view_maps.items())
        },
        "summary": {
            "input_view_count": len(by_view),
            "input_group_count": input_group_count,
            "canonical_group_count": len(canonical_groups),
            "multiview_group_count": len(canonical_groups) - singleton_count,
            "singleton_group_count": singleton_count,
            "winner_view_selected": False,
            "source_boxes_preserved": True,
        },
    }


__all__ = [
    "ACCENT_FUSION_EVIDENCE_SCHEMA_VERSION",
    "CANONICAL_PALETTE_SCHEMA_VERSION",
    "FUSION_SCHEMA_VERSION",
    "PaletteFusionError",
    "UNRESOLVED_PIXEL_ASSOCIATION_EVIDENCE_SCHEMA_VERSION",
    "UNRESOLVED_PIXEL_CHROMATIC_ASSOCIATION",
    "UNRESOLVED_PIXEL_LIGHT_NEUTRAL_ASSOCIATION",
    "UNRESOLVED_PIXEL_MASKED_DARK_ASSOCIATION",
    "fuse_multiview_palettes",
    "is_verified_unresolved_pixel_chromatic_group",
    "is_verified_unresolved_pixel_light_neutral_group",
    "is_verified_unresolved_pixel_masked_dark_group",
]
