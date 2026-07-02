from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "backend": {
        "name": "blender",
        "blender_executable": "blender",
        "keep_intermediate": True,
    },
    "hunyuan": {
        "endpoint": "ai3d.tencentcloudapi.com",
        "service": "ai3d",
        "version": "2025-05-13",
        "region": "ap-guangzhou",
        "secret_id_env": "TENCENTCLOUD_SECRET_ID",
        "secret_key_env": "TENCENTCLOUD_SECRET_KEY",
        "token_env": "TENCENTCLOUD_TOKEN",
        "input_url": None,
        "input_url_env": "HUNYUAN_INPUT_URL",
        "input_type": None,
        "upload_input_path": None,
        "upload_input_path_env": "HUNYUAN_UPLOAD_INPUT_PATH",
        "temp_upload": {
            "enabled": False,
            "provider": "uguu",
            "endpoint": "https://uguu.se/upload.php",
            "timeout_seconds": 300,
            "download_error_max_retries": 4,
            "download_error_retry_interval_seconds": 10,
        },
        "poll_interval_seconds": 10,
        "timeout_seconds": 3600,
        "submit_max_retries": 6,
        "submit_retry_interval_seconds": 60,
        "submit_retry_backoff_factor": 1.5,
        "submit_retry_error_codes": [
            "ResourceInsufficient",
            "InternalError",
            "FailedOperation.ResourceInsufficient",
        ],
        "normalize_result_to_source_bbox": True,
        "download_preference": ["GLB", "FBX", "ZIP", "OBJ"],
        "retopology": {
            "enabled": True,
            "polygon_type": "quadrilateral",
            "face_level": "high",
        },
        "local_postprocess": {
            "enabled": True,
        },
    },
    "cleanup": {
        "merge_distance": 0.0005,
        "degenerate_threshold": 0.00001,
        "min_component_faces": 8,
        "min_component_area_ratio": 0.00005,
        "max_component_cleanup_face_fraction": 0.05,
    },
    "source": {
        "normal_angle_degrees": 60.0,
    },
    "retopology": {
        "method": "auto",
        "target_path": None,
        "normalize_external_target_to_source_bbox": False,
        "cleanup_target_small_components": False,
        "auto_component_threshold": 32,
        "auto_boundary_edge_ratio_threshold": 0.02,
        "target_faces": 30000,
        "preserve_shape_target_face_ratio": 0.18,
        "preserve_shape_min_target_faces": 80000,
        "preserve_shape_max_target_faces": 180000,
        "preserve_shape_smooth_iterations": 0,
        "preserve_shape_shrinkwrap_iterations": 1,
        "preserve_boundary_rings": 0,
        "preserve_small_component_faces": 0,
        "voxel_size": None,
        "voxel_size_factor": 1.15,
        "adaptivity": 0.15,
        "decimate_triangulate": True,
        "decimate_after_remesh": True,
        "smooth_iterations": 2,
        "smooth_factor": 0.18,
        "shrinkwrap_iterations": 2,
        "projection_offset": 0.0,
    },
    "uv": {
        "angle_limit_degrees": 66.0,
        "island_margin": 0.02,
        "area_weight": 0.0,
    },
    "textures": {
        "enabled": True,
        "resolution": 2048,
        "transfer_max_distance": 0.004,
        "transfer_normal_dot_min": 0.1,
        "transfer_dilate_iterations": 16,
        "transfer_fill_uncovered": True,
        "transfer_fill_max_iterations": 512,
        "transfer_allow_nearest_fallback": True,
        "bake_base_color": True,
        "bake_normal": True,
        "bake_roughness": False,
        "bake_metallic": False,
        "bake_ao": False,
        "bake_emissive": False,
        "pbr_repaint": False,
        "neutral_roughness": 0.65,
        "neutral_metallic": 0.0,
    },
    "qc": {
        "fail_on_error": False,
        "adapt_thresholds_to_selected_method": True,
        "uv_overlap_grid": 128,
        "thresholds": {
            "min_faces": 1000,
            "max_faces": 60000,
            "max_projection_rms": 0.02,
            "max_projection_max": 0.08,
            "max_surface_area_relative_delta": None,
            "max_bbox_axis_relative_delta": None,
            "max_uv_overlap_ratio": 0.02,
            "max_uv_out_of_bounds_fraction": 0.0,
            "max_open_or_nonmanifold_edges": 0,
        },
    },
    "export": {
        "formats": ["glb"],
        "glb_filename": "refined_asset.glb",
        "obj_filename": "refined_asset.obj",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return deep_merge(config, loaded)


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return copy.deepcopy(config)
    return deep_merge(config, overrides)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
