#!/usr/bin/env python3
"""Run the bounded local-Qwen selector once per photo appearance component."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_material_pipeline.evidence.appearance_component_material import (
    COMPONENT_EVIDENCE_SCHEMA_VERSION,
)
from qwen_material_pipeline.qwen.local_vl import TransformersQwen3VLRunner
from qwen_material_pipeline.workflows.part_id_qwen import run_part_id_qwen_rerank


OUTPUT_SCHEMA_VERSION = "qwen-appearance-component-rerank/v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_component_qwen_rerank(
    *,
    component_evidence: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    catalog: Mapping[str, Any],
    runner: Any,
    model: str,
    output_dir: Path,
    batch_size: int = 4,
    candidate_count: int = 8,
) -> dict[str, Any]:
    if component_evidence.get("schema_version") != COMPONENT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported component evidence schema")
    # The existing bounded Qwen implementation is deliberately reused.  Its
    # entity IDs are opaque strings, so component IDs receive the same strict
    # JSON/candidate validation and actual-MDL comparison-sheet safeguards as
    # ordinary Part-ID selections.
    synthetic_evidence = dict(component_evidence)
    synthetic_evidence["schema_version"] = "qwen-part-id-reference-evidence/v1"
    raw = run_part_id_qwen_rerank(
        evidence=synthetic_evidence,
        retrieval=retrieval,
        catalog=catalog,
        runner=runner,
        model=model,
        output_dir=output_dir,
        batch_size=batch_size,
        candidate_count=candidate_count,
        allow_color_tuning=False,
        entity_label="photo-supported appearance component",
    )
    raw.pop("integrity", None)
    raw.update(
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "assignment_unit": "appearance_component",
            "part_material_groups_used": False,
            "mdl_parameter_mutation_allowed": False,
            "component_evidence_sha256": component_evidence["integrity"][
                "document_sha256"
            ],
            "selection_contract": "one_immutable_base_mdl_per_component",
        }
    )
    return {**raw, "integrity": {"document_sha256": _canonical_sha256(raw)}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="select one immutable NVIDIA Base MDL for each photo appearance component"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3.5-local-appearance-component")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = TransformersQwen3VLRunner(
        args.model_path.expanduser().resolve(strict=True),
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=512 * 512,
        max_total_pixels=4 * 512 * 512,
    )
    runner.preflight()
    try:
        result = run_component_qwen_rerank(
            component_evidence=_read(args.evidence.expanduser().resolve(strict=True), "component evidence"),
            retrieval=_read(args.retrieval.expanduser().resolve(strict=True), "component retrieval"),
            catalog=_read(args.catalog.expanduser().resolve(strict=True), "material catalog"),
            runner=runner,
            model=args.model_name,
            output_dir=args.output_dir.expanduser().resolve(),
            batch_size=args.batch_size,
            candidate_count=args.candidate_count,
        )
    finally:
        runner.unload()
    output = args.output.expanduser().resolve()
    _write(output, result)
    print(json.dumps({"output": str(output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
