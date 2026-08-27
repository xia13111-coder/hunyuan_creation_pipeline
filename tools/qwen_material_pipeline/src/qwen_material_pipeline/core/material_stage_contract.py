"""Version contract for hash-verified material-stage checkpoint reuse."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "qwen-material-stage-contract/v1"
PIPELINE_REVISION = "identity_parameter_evidence_collapse_recovery_20260826"


def material_stage_contract_document() -> dict[str, Any]:
    """Return the exact deterministic-stage revision accepted for resume."""

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "palette_augmentation": "masked_low_saturation_rust_completion/v1",
        "spatial_localization": (
            "unique_multiview_same_color_visual_authority/v1"
        ),
    }


__all__ = [
    "PIPELINE_REVISION",
    "SCHEMA_VERSION",
    "material_stage_contract_document",
]
