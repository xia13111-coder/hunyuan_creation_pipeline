#!/usr/bin/env python3
"""Validate a collected material bundle for a regular, non-instanced USD.

The reusable audit and dependency-inspection implementation lives in
``validation_common``.  This module remains the stable regular-validator API
and command entry point.
"""

from __future__ import annotations

from qwen_material_pipeline.usd.validation_common import (
    CHECK_LABELS,
    SCHEMA_VERSION,
    Audit,
    collect_mapping,
    is_inside,
    load_json_object,
    local_asset_candidates,
    main,
    parameter_matches,
    parse_args,
    report_records,
    scan_mdl_document,
    start_isaac_if_needed,
    strip_mdl_comments,
    validate_final_bundle,
    verify_materials,
    verify_mdl_textures,
    verify_usd_dependencies,
)

# Compatibility aliases for callers of the pre-refactor helper names.  New
# code must import the public names from ``validation_common``.
_Audit = Audit
_collect_mapping = collect_mapping
_is_inside = is_inside
_load_json_object = load_json_object
_local_asset_candidates = local_asset_candidates
_parameter_matches = parameter_matches
_report_records = report_records
_scan_mdl_document = scan_mdl_document
_start_isaac_if_needed = start_isaac_if_needed
_strip_mdl_comments = strip_mdl_comments
_verify_materials = verify_materials
_verify_mdl_textures = verify_mdl_textures
_verify_usd_dependencies = verify_usd_dependencies


__all__ = [
    "CHECK_LABELS",
    "SCHEMA_VERSION",
    "validate_final_bundle",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
