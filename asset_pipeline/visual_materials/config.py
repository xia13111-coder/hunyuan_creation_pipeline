"""Configuration contract for automatic visual-material assignment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..project_layout import SOURCE_LAYOUT


CONFIG_SCHEMA_VERSION = "qwen-auto-visual-material-bridge/v2"
QUALITY_LIGHTING_PROFILES = frozenset({"geometry", "material-neutral"})
MATERIAL_SELECTION_OBJECTIVES = frozenset(
    {"semantic_compatible_visual", "visual_similarity"}
)
MATERIAL_SELECTION_PIPELINE_MODES = frozenset({"current", "semantic_hybrid"})
MATERIAL_ASSIGNMENT_UNITS = frozenset({"palette_group", "part_id"})
MATERIAL_PARAMETER_CANDIDATE_MODES = frozenset(
    {"disabled", "evidence_gated_h0_h1"}
)
QWEN_MODEL_FAMILIES = frozenset(
    {"qwen3_5", "qwen3_vl", "openai_compatible"}
)
REMOTE_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)
_ENVIRONMENT_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LOCAL_INFERENCE_DEVICES = frozenset({"cpu", "cuda"})
NVIDIA_MATERIALS_ROOT_SCOPE = "nvidia_materials"
NVIDIA_BASE_ROOT_SCOPE = "nvidia_base"
FINAL_VISUAL_GATE_DEFAULTS: dict[str, float | int] = {
    "maximum_score_regression": 0.01,
    "maximum_group_recall_regression": 0.01,
    "maximum_group_share_error_regression": 0.01,
    "minimum_final_appearance_score": 0.62,
    "minimum_final_view_appearance_score": 0.55,
    "minimum_significant_reference_share": 0.01,
    "minimum_significant_evidence_pixels": 128,
    "maximum_policy_fallback_fraction": 0.90,
    "maximum_neutral_fallback_fraction": 0.75,
    "maximum_unresolved_entity_fraction": 0.90,
    "maximum_unresolved_face_subset_fraction": 0.50,
    "minimum_owner_local_resolved_fraction": 0.50,
    "maximum_visible_fallback_fraction": 0.20,
}
DEFAULT_CONFIG_PATH = SOURCE_LAYOUT.manual_part_id_material_config


@dataclass(frozen=True)
class VisualMaterialConfig:
    """Fully resolved local runtimes and material resources."""

    qwen_python: Path
    qwen_model_path: Path | None
    qwen_model_family: str
    qwen_model_revision: str | None
    openai_base_url: str | None
    openai_model: str | None
    openai_api_key_env: str | None
    openai_reasoning_effort: str | None
    openai_timeout_seconds: int | None
    qwen_max_new_tokens: int
    qwen_max_new_tokens_ceiling: int
    qwen_minimum_usable_palette_views: int
    qwen_minimum_usable_palette_view_ratio: float
    qwen_parallel_requests: int
    catalog: Path
    whitelist: Path
    material_root: Path
    render_resolution: int
    render_views: str
    render_rt_subframes: int
    analysis_up_axis: str
    analysis_front_axis: str
    mvinverse_mode: str
    mvinverse_python: Path
    mvinverse_repository: Path
    mvinverse_checkpoint: Path
    mvinverse_model_revision: str
    mvinverse_device: str
    mvinverse_max_side: int
    mvinverse_oom_retry_max_sides: tuple[int, ...]
    mvinverse_timeout_seconds: int
    sam3_python: Path
    sam3_repository: Path
    sam3_checkpoint: Path
    sam3_device: str
    sam3_minimum_model_score: float
    sam3_minimum_prompt_overlap: float
    sam3_maximum_image_fraction: float
    sam3_minimum_mask_pixels: int
    retrieval_python: Path
    siglip2_model_path: Path
    dinov2_model_path: Path
    retrieval_cache_dir: Path
    retrieval_observation_bank_dir: Path | None
    retrieval_device: str
    siglip_top_k: int
    retrieval_final_top_k: int
    retrieval_batch_size: int
    material_selection_pipeline_mode: str = "current"
    material_assignment_unit: str = "palette_group"
    quality_lighting_profile: str = "material-neutral"
    immutable_mdl_after_selection: bool = False
    material_parameter_candidate_mode: str = "disabled"
    exact_mdl_tournament_max_candidates: int = 12
    exact_mdl_tournament_all_groups: bool = True
    exact_mdl_tournament_minimum_score_improvement: float = 0.015
    exact_mdl_tournament_minimum_winner_margin: float = 0.005
    material_selection_objective: str = "semantic_compatible_visual"
    final_visual_gate_maximum_score_regression: float = 0.01
    final_visual_gate_maximum_group_recall_regression: float = 0.01
    final_visual_gate_maximum_group_share_error_regression: float = 0.01
    final_visual_gate_minimum_final_appearance_score: float = 0.62
    final_visual_gate_minimum_final_view_appearance_score: float = 0.55
    final_visual_gate_minimum_significant_reference_share: float = 0.01
    final_visual_gate_minimum_significant_evidence_pixels: int = 128
    final_visual_gate_maximum_policy_fallback_fraction: float = 0.90
    final_visual_gate_maximum_neutral_fallback_fraction: float = 0.75
    final_visual_gate_maximum_unresolved_entity_fraction: float = 0.90
    final_visual_gate_maximum_unresolved_face_subset_fraction: float = 0.50
    final_visual_gate_minimum_owner_local_resolved_fraction: float = 0.50
    final_visual_gate_maximum_visible_fallback_fraction: float = 0.20
    qwen_mapping_verification_views: int = 0


def read_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object with a context-rich validation error."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def write_object(path: Path, document: dict[str, Any]) -> Path:
    """Write one stable JSON object, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def canonical_sha256(document: dict[str, Any]) -> str:
    """Hash a JSON object using the pipeline's canonical serialization."""

    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_keys(
    value: dict[str, Any],
    *,
    label: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    """Reject missing and unknown fields in one versioned config object."""

    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def require_choice(
    value: Any,
    label: str,
    choices: frozenset[str],
) -> str:
    result = require_string(value, label)
    if result not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return result


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def require_at_least_two(value: Any, label: str) -> int:
    result = require_positive_int(value, label)
    if result < 2:
        raise ValueError(f"{label} must be at least 2")
    return result


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def require_unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def resolve_path(
    value: Any,
    *,
    config_dir: Path,
    label: str,
    kind: str,
    executable: bool = False,
) -> Path:
    """Resolve and validate a file-system value relative to its config."""

    raw = os.path.expandvars(require_string(value, label))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    # Keep environment-manager executable shims intact so the selected
    # interpreter continues to use its own site-packages.
    resolved = Path(os.path.abspath(path))
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def resolve_directory_target(
    value: Any,
    *,
    config_dir: Path,
    label: str,
) -> Path:
    """Resolve a writable/cache directory target without creating it."""

    raw = os.path.expandvars(require_string(value, label))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    resolved = Path(os.path.abspath(path))
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"{label} parent directory does not exist: {parent}")
    return resolved


def resolve_material_root(
    materials: dict[str, Any],
    *,
    config_dir: Path,
) -> Path:
    """Resolve a material root, enforcing an optional library-wide scope.

    Custom configurations that omit ``root_scope`` keep the historical
    directory-as-given behavior.  ``nvidia_materials`` normalizes a collection
    directory (``Base`` or ``vMaterials_2``) to its common parent and verifies
    both collections.  ``nvidia_base`` is a fail-closed production boundary:
    it always resolves to the ``Base`` collection and never permits
    ``vMaterials_2`` to enter the candidate catalog.
    """

    resolved = resolve_path(
        materials.get("material_root"),
        config_dir=config_dir,
        label="config.materials.material_root",
        kind="directory",
    )
    raw_scope = materials.get("root_scope")
    if raw_scope is None:
        return resolved
    scope = require_string(raw_scope, "config.materials.root_scope")
    if scope not in {NVIDIA_MATERIALS_ROOT_SCOPE, NVIDIA_BASE_ROOT_SCOPE}:
        raise ValueError(
            "config.materials.root_scope must be one of "
            f"{sorted({NVIDIA_MATERIALS_ROOT_SCOPE, NVIDIA_BASE_ROOT_SCOPE})}"
        )

    if scope == NVIDIA_BASE_ROOT_SCOPE:
        if resolved.name == "vMaterials_2":
            raise ValueError(
                "config.materials.root_scope='nvidia_base' rejects a "
                f"vMaterials_2 root: {resolved}"
            )
        candidate = resolved if resolved.name == "Base" else resolved / "Base"
        if not candidate.is_dir():
            raise FileNotFoundError(
                "config.materials.root_scope='nvidia_base' requires the "
                "NVIDIA Base collection; "
                f"resolved root {resolved} has no Base directory"
            )
        return candidate.resolve(strict=True)

    has_full_layout = all(
        (resolved / collection).is_dir()
        for collection in ("Base", "vMaterials_2")
    )
    candidate = (
        resolved.parent
        if not has_full_layout and resolved.name in {"Base", "vMaterials_2"}
        else resolved
    )
    missing_collections = [
        collection
        for collection in ("Base", "vMaterials_2")
        if not (candidate / collection).is_dir()
    ]
    if missing_collections:
        raise FileNotFoundError(
            "config.materials.root_scope='nvidia_materials' requires the "
            "full NVIDIA Materials root containing Base and vMaterials_2; "
            f"resolved root {candidate} is missing "
            f"{', '.join(missing_collections)}"
        )
    return candidate


def load_visual_material_config(
    config_path: str | Path | None = None,
) -> VisualMaterialConfig:
    """Load and strictly resolve the bridge's local runtime configuration."""

    raw_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    try:
        path = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"Visual-material config does not exist: {raw_path}"
        ) from exc
    document = read_object(path, "visual-material config")
    if document.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Visual-material config has an unsupported schema_version: "
            f"{document.get('schema_version')!r}"
        )
    require_keys(
        document,
        label="config",
        required=frozenset(
            {
                "schema_version",
                "qwen",
                "materials",
                "render",
                "mvinverse",
                "sam3",
                "retrieval",
            }
        ),
    )
    qwen = require_object(document.get("qwen"), "config.qwen")
    materials = require_object(document.get("materials"), "config.materials")
    render = require_object(document.get("render"), "config.render")
    final_visual_gate = require_object(
        render.get("final_visual_gate", {}),
        "config.render.final_visual_gate",
    )
    mvinverse = require_object(document.get("mvinverse"), "config.mvinverse")
    sam3 = require_object(document.get("sam3"), "config.sam3")
    retrieval = require_object(document.get("retrieval"), "config.retrieval")
    require_keys(
        qwen,
        label="config.qwen",
        required=frozenset(
            {
                "python",
                "model_family",
                "max_new_tokens",
            }
        ),
        optional=frozenset(
            {
                "model_path",
                "model_revision",
                "base_url",
                "model",
                "api_key_env",
                "reasoning_effort",
                "timeout_seconds",
                "max_new_tokens_ceiling",
                "minimum_usable_palette_views",
                "minimum_usable_palette_view_ratio",
                "mapping_verification_views",
                "parallel_requests",
            }
        ),
    )
    require_keys(
        materials,
        label="config.materials",
        required=frozenset({"catalog", "whitelist", "material_root"}),
        optional=frozenset(
            {
                "root_scope",
                "selection_pipeline_mode",
                "immutable_after_selection",
                "selection_objective",
                "assignment_unit",
                "parameter_candidate_mode",
                "exact_mdl_tournament_max_candidates",
                "exact_mdl_tournament_all_groups",
                "exact_mdl_tournament_minimum_score_improvement",
                "exact_mdl_tournament_minimum_winner_margin",
            }
        ),
    )
    require_keys(
        render,
        label="config.render",
        required=frozenset(
            {
                "resolution",
                "views",
                "rt_subframes",
                "analysis_up_axis",
                "analysis_front_axis",
            }
        ),
        optional=frozenset({"quality_lighting_profile", "final_visual_gate"}),
    )
    require_keys(
        final_visual_gate,
        label="config.render.final_visual_gate",
        required=frozenset(),
        optional=frozenset(FINAL_VISUAL_GATE_DEFAULTS),
    )
    require_keys(
        mvinverse,
        label="config.mvinverse",
        required=frozenset(
            {
                "mode",
                "python",
                "repository",
                "checkpoint",
                "model_revision",
                "device",
                "max_side",
                "oom_retry_max_sides",
                "timeout_seconds",
            }
        ),
    )
    require_keys(
        sam3,
        label="config.sam3",
        required=frozenset(
            {
                "python",
                "repository",
                "checkpoint",
                "device",
                "minimum_model_score",
                "minimum_prompt_overlap",
                "maximum_image_fraction",
                "minimum_mask_pixels",
            }
        ),
    )
    require_keys(
        retrieval,
        label="config.retrieval",
        required=frozenset(
            {
                "python",
                "siglip2_model",
                "dinov2_model",
                "cache_dir",
                "device",
                "siglip_top_k",
                "final_top_k",
                "batch_size",
            }
        ),
        optional=frozenset({"observation_bank"}),
    )
    config_dir = path.parent

    mode = require_string(mvinverse.get("mode"), "config.mvinverse.mode")
    if mode != "run":
        raise ValueError(
            "The main asset pipeline requires config.mvinverse.mode='run'; "
            "reuse is reserved for an explicitly verified standalone run"
        )
    retry_values = mvinverse.get("oom_retry_max_sides")
    if not isinstance(retry_values, list):
        raise ValueError("config.mvinverse.oom_retry_max_sides must be an array")
    retry_sides = tuple(
        require_positive_int(value, f"config.mvinverse.oom_retry_max_sides[{index}]")
        for index, value in enumerate(retry_values)
    )
    if any(left <= right for left, right in zip(retry_sides, retry_sides[1:])):
        raise ValueError(
            "config.mvinverse.oom_retry_max_sides must be strictly descending"
        )
    quality_lighting_profile = require_string(
        render.get("quality_lighting_profile", "material-neutral"),
        "config.render.quality_lighting_profile",
    )
    if quality_lighting_profile not in QUALITY_LIGHTING_PROFILES:
        raise ValueError(
            "config.render.quality_lighting_profile must be one of "
            f"{sorted(QUALITY_LIGHTING_PROFILES)}"
        )
    material_selection_objective = require_string(
        materials.get(
            "selection_objective",
            "semantic_compatible_visual",
        ),
        "config.materials.selection_objective",
    )
    if material_selection_objective not in MATERIAL_SELECTION_OBJECTIVES:
        raise ValueError(
            "config.materials.selection_objective must be one of "
            f"{sorted(MATERIAL_SELECTION_OBJECTIVES)}"
        )
    material_selection_pipeline_mode = require_choice(
        materials.get("selection_pipeline_mode", "current"),
        "config.materials.selection_pipeline_mode",
        MATERIAL_SELECTION_PIPELINE_MODES,
    )
    material_assignment_unit = require_choice(
        materials.get("assignment_unit", "palette_group"),
        "config.materials.assignment_unit",
        MATERIAL_ASSIGNMENT_UNITS,
    )
    immutable_mdl_after_selection = require_bool(
        materials.get("immutable_after_selection", False),
        "config.materials.immutable_after_selection",
    )
    material_parameter_candidate_mode = require_choice(
        materials.get("parameter_candidate_mode", "disabled"),
        "config.materials.parameter_candidate_mode",
        MATERIAL_PARAMETER_CANDIDATE_MODES,
    )
    exact_mdl_tournament_max_candidates = require_at_least_two(
        materials.get("exact_mdl_tournament_max_candidates", 12),
        "config.materials.exact_mdl_tournament_max_candidates",
    )
    if material_selection_pipeline_mode == "semantic_hybrid":
        required_hybrid_values = {
            "assignment_unit": (material_assignment_unit, "part_id"),
            "immutable_after_selection": (immutable_mdl_after_selection, False),
            "parameter_candidate_mode": (
                material_parameter_candidate_mode,
                "evidence_gated_h0_h1",
            ),
            "selection_objective": (
                material_selection_objective,
                "semantic_compatible_visual",
            ),
            "exact_mdl_tournament_max_candidates": (
                exact_mdl_tournament_max_candidates,
                3,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required_hybrid_values.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "config.materials.selection_pipeline_mode='semantic_hybrid' "
                "requires the bounded semantic/H0-H1 contract: "
                + ", ".join(mismatches)
            )
    siglip_top_k = require_at_least_two(
        retrieval.get("siglip_top_k"),
        "config.retrieval.siglip_top_k",
    )
    retrieval_final_top_k = require_at_least_two(
        retrieval.get("final_top_k"),
        "config.retrieval.final_top_k",
    )
    if retrieval_final_top_k > siglip_top_k:
        raise ValueError(
            "config.retrieval.final_top_k cannot exceed "
            "config.retrieval.siglip_top_k"
        )
    qwen_max_new_tokens = require_positive_int(
        qwen.get("max_new_tokens"),
        "config.qwen.max_new_tokens",
    )
    qwen_max_new_tokens_ceiling = require_positive_int(
        qwen.get("max_new_tokens_ceiling", qwen_max_new_tokens),
        "config.qwen.max_new_tokens_ceiling",
    )
    if qwen_max_new_tokens_ceiling < qwen_max_new_tokens:
        raise ValueError(
            "config.qwen.max_new_tokens_ceiling cannot be smaller than "
            "config.qwen.max_new_tokens"
        )
    qwen_minimum_usable_palette_views = require_positive_int(
        qwen.get("minimum_usable_palette_views", 1),
        "config.qwen.minimum_usable_palette_views",
    )
    qwen_minimum_usable_palette_view_ratio = require_unit_float(
        qwen.get("minimum_usable_palette_view_ratio", 0.0),
        "config.qwen.minimum_usable_palette_view_ratio",
    )
    qwen_mapping_verification_views = require_nonnegative_int(
        qwen.get("mapping_verification_views", 0),
        "config.qwen.mapping_verification_views",
    )
    if qwen_mapping_verification_views == 1:
        raise ValueError(
            "config.qwen.mapping_verification_views must be 0 or at least 2"
        )
    qwen_parallel_requests = require_positive_int(
        qwen.get("parallel_requests", 1),
        "config.qwen.parallel_requests",
    )
    if qwen_parallel_requests > 8:
        raise ValueError("config.qwen.parallel_requests cannot exceed 8")
    qwen_model_family = require_choice(
        qwen.get("model_family"),
        "config.qwen.model_family",
        QWEN_MODEL_FAMILIES,
    )
    if qwen_model_family == "openai_compatible":
        if qwen.get("model_path") is not None or qwen.get("model_revision") is not None:
            raise ValueError(
                "config.qwen.model_path and model_revision must be omitted for "
                "openai_compatible inference"
            )
        openai_base_url = require_string(
            qwen.get("base_url"), "config.qwen.base_url"
        ).rstrip("/")
        parsed_openai_url = urlparse(openai_base_url)
        if parsed_openai_url.scheme != "https" or not parsed_openai_url.netloc:
            raise ValueError(
                "config.qwen.base_url must be an absolute HTTPS URL"
            )
        openai_model = require_string(qwen.get("model"), "config.qwen.model")
        openai_api_key_env = require_string(
            qwen.get("api_key_env"), "config.qwen.api_key_env"
        )
        if _ENVIRONMENT_VARIABLE_RE.fullmatch(openai_api_key_env) is None:
            raise ValueError(
                "config.qwen.api_key_env must be a valid environment-variable name"
            )
        openai_reasoning_effort = require_choice(
            qwen.get("reasoning_effort", "medium"),
            "config.qwen.reasoning_effort",
            REMOTE_REASONING_EFFORTS,
        )
        openai_timeout_seconds = require_positive_int(
            qwen.get("timeout_seconds", 180),
            "config.qwen.timeout_seconds",
        )
        qwen_model_path = None
        qwen_model_revision = None
    else:
        remote_fields = {
            key
            for key in (
                "base_url",
                "model",
                "api_key_env",
                "reasoning_effort",
                "timeout_seconds",
            )
            if qwen.get(key) is not None
        }
        if remote_fields:
            raise ValueError(
                "config.qwen remote fields are only valid for "
                "model_family='openai_compatible': "
                + ", ".join(sorted(remote_fields))
            )
        qwen_model_path = resolve_path(
            qwen.get("model_path"),
            config_dir=config_dir,
            label="config.qwen.model_path",
            kind="directory",
        )
        qwen_model_revision = require_string(
            qwen.get("model_revision"),
            "config.qwen.model_revision",
        )
        openai_base_url = None
        openai_model = None
        openai_api_key_env = None
        openai_reasoning_effort = None
        openai_timeout_seconds = None

    return VisualMaterialConfig(
        qwen_python=resolve_path(
            qwen.get("python"),
            config_dir=config_dir,
            label="config.qwen.python",
            kind="file",
            executable=True,
        ),
        qwen_model_path=qwen_model_path,
        qwen_model_family=qwen_model_family,
        qwen_model_revision=qwen_model_revision,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_api_key_env=openai_api_key_env,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_timeout_seconds=openai_timeout_seconds,
        qwen_max_new_tokens=qwen_max_new_tokens,
        qwen_max_new_tokens_ceiling=qwen_max_new_tokens_ceiling,
        qwen_minimum_usable_palette_views=qwen_minimum_usable_palette_views,
        qwen_minimum_usable_palette_view_ratio=(
            qwen_minimum_usable_palette_view_ratio
        ),
        qwen_parallel_requests=qwen_parallel_requests,
        qwen_mapping_verification_views=qwen_mapping_verification_views,
        catalog=resolve_path(
            materials.get("catalog"),
            config_dir=config_dir,
            label="config.materials.catalog",
            kind="file",
        ),
        whitelist=resolve_path(
            materials.get("whitelist"),
            config_dir=config_dir,
            label="config.materials.whitelist",
            kind="file",
        ),
        material_root=resolve_material_root(materials, config_dir=config_dir),
        render_resolution=require_positive_int(
            render.get("resolution"), "config.render.resolution"
        ),
        render_views=require_string(render.get("views"), "config.render.views"),
        render_rt_subframes=require_positive_int(
            render.get("rt_subframes"), "config.render.rt_subframes"
        ),
        analysis_up_axis=require_string(
            render.get("analysis_up_axis"), "config.render.analysis_up_axis"
        ),
        analysis_front_axis=require_string(
            render.get("analysis_front_axis"), "config.render.analysis_front_axis"
        ),
        mvinverse_mode=mode,
        mvinverse_python=resolve_path(
            mvinverse.get("python"),
            config_dir=config_dir,
            label="config.mvinverse.python",
            kind="file",
            executable=True,
        ),
        mvinverse_repository=resolve_path(
            mvinverse.get("repository"),
            config_dir=config_dir,
            label="config.mvinverse.repository",
            kind="directory",
        ),
        mvinverse_checkpoint=resolve_path(
            mvinverse.get("checkpoint"),
            config_dir=config_dir,
            label="config.mvinverse.checkpoint",
            kind="directory",
        ),
        mvinverse_model_revision=require_string(
            mvinverse.get("model_revision"), "config.mvinverse.model_revision"
        ),
        mvinverse_device=require_string(
            mvinverse.get("device"), "config.mvinverse.device"
        ),
        mvinverse_max_side=require_positive_int(
            mvinverse.get("max_side"), "config.mvinverse.max_side"
        ),
        mvinverse_oom_retry_max_sides=retry_sides,
        mvinverse_timeout_seconds=require_positive_int(
            mvinverse.get("timeout_seconds"), "config.mvinverse.timeout_seconds"
        ),
        sam3_python=resolve_path(
            sam3.get("python"),
            config_dir=config_dir,
            label="config.sam3.python",
            kind="file",
            executable=True,
        ),
        sam3_repository=resolve_path(
            sam3.get("repository"),
            config_dir=config_dir,
            label="config.sam3.repository",
            kind="directory",
        ),
        sam3_checkpoint=resolve_path(
            sam3.get("checkpoint"),
            config_dir=config_dir,
            label="config.sam3.checkpoint",
            kind="file",
        ),
        sam3_device=require_choice(
            sam3.get("device"),
            "config.sam3.device",
            LOCAL_INFERENCE_DEVICES,
        ),
        sam3_minimum_model_score=require_unit_float(
            sam3.get("minimum_model_score"),
            "config.sam3.minimum_model_score",
        ),
        sam3_minimum_prompt_overlap=require_unit_float(
            sam3.get("minimum_prompt_overlap"),
            "config.sam3.minimum_prompt_overlap",
        ),
        sam3_maximum_image_fraction=require_unit_float(
            sam3.get("maximum_image_fraction"),
            "config.sam3.maximum_image_fraction",
        ),
        sam3_minimum_mask_pixels=require_positive_int(
            sam3.get("minimum_mask_pixels"),
            "config.sam3.minimum_mask_pixels",
        ),
        retrieval_python=resolve_path(
            retrieval.get("python"),
            config_dir=config_dir,
            label="config.retrieval.python",
            kind="file",
            executable=True,
        ),
        siglip2_model_path=resolve_path(
            retrieval.get("siglip2_model"),
            config_dir=config_dir,
            label="config.retrieval.siglip2_model",
            kind="directory",
        ),
        dinov2_model_path=resolve_path(
            retrieval.get("dinov2_model"),
            config_dir=config_dir,
            label="config.retrieval.dinov2_model",
            kind="directory",
        ),
        retrieval_cache_dir=resolve_directory_target(
            retrieval.get("cache_dir"),
            config_dir=config_dir,
            label="config.retrieval.cache_dir",
        ),
        retrieval_observation_bank_dir=(
            resolve_path(
                retrieval.get("observation_bank"),
                config_dir=config_dir,
                label="config.retrieval.observation_bank",
                kind="directory",
            )
            if retrieval.get("observation_bank") is not None
            else None
        ),
        retrieval_device=require_choice(
            retrieval.get("device"),
            "config.retrieval.device",
            LOCAL_INFERENCE_DEVICES,
        ),
        siglip_top_k=siglip_top_k,
        retrieval_final_top_k=retrieval_final_top_k,
        retrieval_batch_size=require_positive_int(
            retrieval.get("batch_size"),
            "config.retrieval.batch_size",
        ),
        material_selection_pipeline_mode=material_selection_pipeline_mode,
        material_assignment_unit=material_assignment_unit,
        quality_lighting_profile=quality_lighting_profile,
        immutable_mdl_after_selection=immutable_mdl_after_selection,
        material_parameter_candidate_mode=material_parameter_candidate_mode,
        exact_mdl_tournament_max_candidates=exact_mdl_tournament_max_candidates,
        exact_mdl_tournament_all_groups=require_bool(
            materials.get("exact_mdl_tournament_all_groups", True),
            "config.materials.exact_mdl_tournament_all_groups",
        ),
        exact_mdl_tournament_minimum_score_improvement=require_unit_float(
            materials.get(
                "exact_mdl_tournament_minimum_score_improvement",
                0.015,
            ),
            ("config.materials.exact_mdl_tournament_minimum_score_improvement"),
        ),
        exact_mdl_tournament_minimum_winner_margin=require_unit_float(
            materials.get(
                "exact_mdl_tournament_minimum_winner_margin",
                0.005,
            ),
            "config.materials.exact_mdl_tournament_minimum_winner_margin",
        ),
        material_selection_objective=material_selection_objective,
        final_visual_gate_maximum_score_regression=require_unit_float(
            final_visual_gate.get(
                "maximum_score_regression",
                FINAL_VISUAL_GATE_DEFAULTS["maximum_score_regression"],
            ),
            "config.render.final_visual_gate.maximum_score_regression",
        ),
        final_visual_gate_maximum_group_recall_regression=require_unit_float(
            final_visual_gate.get(
                "maximum_group_recall_regression",
                FINAL_VISUAL_GATE_DEFAULTS["maximum_group_recall_regression"],
            ),
            "config.render.final_visual_gate.maximum_group_recall_regression",
        ),
        final_visual_gate_maximum_group_share_error_regression=(
            require_unit_float(
                final_visual_gate.get(
                    "maximum_group_share_error_regression",
                    FINAL_VISUAL_GATE_DEFAULTS["maximum_group_share_error_regression"],
                ),
                (
                    "config.render.final_visual_gate."
                    "maximum_group_share_error_regression"
                ),
            )
        ),
        final_visual_gate_minimum_final_appearance_score=require_unit_float(
            final_visual_gate.get(
                "minimum_final_appearance_score",
                FINAL_VISUAL_GATE_DEFAULTS["minimum_final_appearance_score"],
            ),
            "config.render.final_visual_gate.minimum_final_appearance_score",
        ),
        final_visual_gate_minimum_final_view_appearance_score=(
            require_unit_float(
                final_visual_gate.get(
                    "minimum_final_view_appearance_score",
                    FINAL_VISUAL_GATE_DEFAULTS["minimum_final_view_appearance_score"],
                ),
                ("config.render.final_visual_gate.minimum_final_view_appearance_score"),
            )
        ),
        final_visual_gate_minimum_significant_reference_share=(
            require_unit_float(
                final_visual_gate.get(
                    "minimum_significant_reference_share",
                    FINAL_VISUAL_GATE_DEFAULTS["minimum_significant_reference_share"],
                ),
                ("config.render.final_visual_gate.minimum_significant_reference_share"),
            )
        ),
        final_visual_gate_minimum_significant_evidence_pixels=(
            require_positive_int(
                final_visual_gate.get(
                    "minimum_significant_evidence_pixels",
                    FINAL_VISUAL_GATE_DEFAULTS["minimum_significant_evidence_pixels"],
                ),
                ("config.render.final_visual_gate.minimum_significant_evidence_pixels"),
            )
        ),
        final_visual_gate_maximum_policy_fallback_fraction=require_unit_float(
            final_visual_gate.get(
                "maximum_policy_fallback_fraction",
                FINAL_VISUAL_GATE_DEFAULTS["maximum_policy_fallback_fraction"],
            ),
            "config.render.final_visual_gate.maximum_policy_fallback_fraction",
        ),
        final_visual_gate_maximum_neutral_fallback_fraction=require_unit_float(
            final_visual_gate.get(
                "maximum_neutral_fallback_fraction",
                FINAL_VISUAL_GATE_DEFAULTS["maximum_neutral_fallback_fraction"],
            ),
            "config.render.final_visual_gate.maximum_neutral_fallback_fraction",
        ),
        final_visual_gate_maximum_unresolved_entity_fraction=require_unit_float(
            final_visual_gate.get(
                "maximum_unresolved_entity_fraction",
                FINAL_VISUAL_GATE_DEFAULTS["maximum_unresolved_entity_fraction"],
            ),
            "config.render.final_visual_gate.maximum_unresolved_entity_fraction",
        ),
        final_visual_gate_maximum_unresolved_face_subset_fraction=(
            require_unit_float(
                final_visual_gate.get(
                    "maximum_unresolved_face_subset_fraction",
                    FINAL_VISUAL_GATE_DEFAULTS[
                        "maximum_unresolved_face_subset_fraction"
                    ],
                ),
                (
                    "config.render.final_visual_gate."
                    "maximum_unresolved_face_subset_fraction"
                ),
            )
        ),
        final_visual_gate_minimum_owner_local_resolved_fraction=(
            require_unit_float(
                final_visual_gate.get(
                    "minimum_owner_local_resolved_fraction",
                    FINAL_VISUAL_GATE_DEFAULTS[
                        "minimum_owner_local_resolved_fraction"
                    ],
                ),
                (
                    "config.render.final_visual_gate."
                    "minimum_owner_local_resolved_fraction"
                ),
            )
        ),
        final_visual_gate_maximum_visible_fallback_fraction=require_unit_float(
            final_visual_gate.get(
                "maximum_visible_fallback_fraction",
                FINAL_VISUAL_GATE_DEFAULTS[
                    "maximum_visible_fallback_fraction"
                ],
            ),
            (
                "config.render.final_visual_gate."
                "maximum_visible_fallback_fraction"
            ),
        ),
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "FINAL_VISUAL_GATE_DEFAULTS",
    "LOCAL_INFERENCE_DEVICES",
    "MATERIAL_SELECTION_OBJECTIVES",
    "MATERIAL_SELECTION_PIPELINE_MODES",
    "QWEN_MODEL_FAMILIES",
    "VisualMaterialConfig",
    "canonical_sha256",
    "load_visual_material_config",
    "read_object",
    "require_choice",
    "require_keys",
    "require_object",
    "require_bool",
    "require_positive_int",
    "require_string",
    "require_unit_float",
    "resolve_directory_target",
    "resolve_path",
    "write_object",
]
