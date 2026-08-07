#!/usr/bin/env python3
"""Pure-Python CLI for catalog search and Qwen material analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from qwen_material_pipeline.materials.catalog import (
    DEFAULT_MATERIAL_ROOT,
    MaterialCatalog,
    build_catalog,
)
from qwen_material_pipeline.qwen.client import QwenMaterialClient


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST = PACKAGE_DIR / "configs" / "materials" / "industrial_whitelist.json"
SUPPORTED_VIEW_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
)


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return document


def _write_object(path: str | Path, value: dict[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resolved


def _parse_view(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("view must use ID=IMAGE syntax")
    view_id, image = value.split("=", 1)
    view_id = view_id.strip()
    image = image.strip()
    if not view_id or not image:
        raise argparse.ArgumentTypeError("view ID and image cannot be empty")
    return {"id": view_id, "image": image}


def _view_id_from_filename(filename: str) -> str:
    """Build a stable user-reference ID without using reserved CAD prefixes."""

    stem = unicodedata.normalize("NFKC", Path(filename).stem).casefold()
    normalized_characters = "".join(
        character if character.isalnum() else " " for character in stem
    )
    slug = "_".join(normalized_characters.split())
    if not slug:
        digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
        slug = f"image_{digest}"
    return f"ref_{slug}"


def _views_from_directory(path: str | Path) -> list[dict[str, str]]:
    directory = Path(path).expanduser()
    if not directory.exists():
        raise ValueError(f"--view-dir does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"--view-dir must be a directory: {directory}")

    image_paths = sorted(
        (
            candidate
            for candidate in directory.iterdir()
            if candidate.is_file()
            and candidate.suffix.casefold() in SUPPORTED_VIEW_EXTENSIONS
        ),
        key=lambda candidate: (candidate.name.casefold(), candidate.name),
    )
    if not image_paths:
        extensions = ", ".join(sorted(SUPPORTED_VIEW_EXTENSIONS))
        raise ValueError(
            f"--view-dir contains no supported images ({extensions}): {directory}"
        )
    return [
        {
            "id": _view_id_from_filename(image_path.name),
            "image": str(image_path.resolve(strict=True)),
        }
        for image_path in image_paths
    ]


def _collect_input_views(
    *,
    explicit_views: Sequence[dict[str, str]],
    view_directories: Sequence[str | Path],
    registry: dict[str, Any],
    include_registry_renders: bool,
) -> list[dict[str, str]]:
    views = list(explicit_views)
    for directory in view_directories:
        views.extend(_views_from_directory(directory))
    if include_registry_renders:
        views.extend(_registry_views(registry))
    if not views:
        raise ValueError(
            "Provide --view ID=IMAGE, --view-dir DIR, or --include-registry-renders"
        )

    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for view in views:
        view_id = view["id"]
        if view_id in seen:
            duplicate_ids.add(view_id)
        seen.add(view_id)
    if duplicate_ids:
        raise ValueError(
            "Duplicate input view IDs: " + ", ".join(sorted(duplicate_ids))
        )
    return views


def _whitelist_ids(path: str | Path) -> list[str]:
    document = _read_object(path)
    values = document.get("material_ids")
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ValueError("Whitelist material_ids must be an array of strings")
    if len(set(values)) != len(values):
        raise ValueError("Whitelist contains duplicate material_ids")
    return values


def _candidate_records(
    catalog: MaterialCatalog,
    *,
    whitelist: str | Path | None,
    family: str | None,
    color: str | None,
    finish: str | None,
    query: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if whitelist:
        records = [
            catalog.get(material_id).to_dict()
            for material_id in _whitelist_ids(whitelist)
        ]
    else:
        records = catalog.search(
            family=family, color=color, finish=finish, query=query, top_k=top_k
        )
    for record in records:
        thumbnail = record.get("thumbnail_path")
        if isinstance(thumbnail, str):
            # Catalog construction has already validated this path below root.
            preview = (catalog.root / thumbnail).resolve(strict=True)
            preview.relative_to(catalog.root)
            record["thumbnail_image"] = str(preview)
    return records


def _registry_views(registry: dict[str, Any]) -> list[dict[str, str]]:
    render_set = registry.get("render_set")
    if not isinstance(render_set, dict):
        return []
    result: list[dict[str, str]] = []
    for view in render_set.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("view_id"), str):
            continue
        if isinstance(view.get("rgb"), str):
            result.append({"id": f"cad_{view['view_id']}", "image": view["rgb"]})
        if isinstance(view.get("part_ids"), str):
            result.append(
                {"id": f"part_ids_{view['view_id']}", "image": view["part_ids"]}
            )
    for index, image in enumerate(render_set.get("contact_sheets", []), start=1):
        if isinstance(image, str):
            result.append({"id": f"part_contact_{index:02d}", "image": image})
    return result


def _create_analysis_client(args: argparse.Namespace) -> Any:
    if args.backend == "transformers":
        if args.model_path is None:
            raise ValueError("--model-path is required when --backend=transformers")
        if args.base_url is not None:
            raise ValueError("--base-url is only valid for the DashScope backend")
        from qwen_material_pipeline.qwen.local_vl import LocalQwen3VLClient

        return LocalQwen3VLClient(
            model_path=args.model_path,
            model=args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            max_total_pixels=args.max_total_pixels,
            max_image_bytes=args.max_image_bytes,
            raw_output_path=args.raw_output,
        )
    if args.model_path is not None:
        raise ValueError("--model-path is only valid for the Transformers backend")
    if args.raw_output is not None:
        raise ValueError("--raw-output is only valid for the Transformers backend")
    return QwenMaterialClient(model=args.model, base_url=args.base_url)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-catalog", help="scan the local MDL library")
    build.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    build.add_argument("--output", type=Path, required=True)

    search = commands.add_parser("search", help="search an indexed material catalog")
    search.add_argument("--catalog", type=Path, required=True)
    search.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    search.add_argument("--family")
    search.add_argument("--color")
    search.add_argument("--finish")
    search.add_argument("--query")
    search.add_argument("--top-k", type=int, default=10)

    analyze = commands.add_parser("analyze", help="send multi-view evidence to Qwen")
    analyze.add_argument("--registry", type=Path, required=True)
    analyze.add_argument("--catalog", type=Path, required=True)
    analyze.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    analyze.add_argument("--view", type=_parse_view, action="append", default=[])
    analyze.add_argument(
        "--view-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "add supported images from one directory (non-recursive, sorted by "
            "filename); may be repeated"
        ),
    )
    analyze.add_argument("--include-registry-renders", action="store_true")
    analyze.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    analyze.add_argument("--no-whitelist", action="store_true")
    analyze.add_argument("--family")
    analyze.add_argument("--color")
    analyze.add_argument("--finish")
    analyze.add_argument("--query")
    analyze.add_argument("--top-k", type=int, default=30)
    analyze.add_argument("--max-assignments", type=int)
    analyze.add_argument(
        "--backend",
        choices=("dashscope", "transformers"),
        default="dashscope",
        help="remote DashScope API or local Hugging Face Transformers",
    )
    analyze.add_argument("--model")
    analyze.add_argument("--base-url")
    analyze.add_argument("--model-path", type=Path)
    analyze.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    analyze.add_argument("--device-map", default="auto")
    analyze.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    analyze.add_argument("--max-new-tokens", type=int, default=8192)
    analyze.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    analyze.add_argument("--max-total-pixels", type=int, default=16 * 1024 * 1024)
    analyze.add_argument("--max-image-bytes", type=int, default=25 * 1024 * 1024)
    analyze.add_argument(
        "--raw-output",
        type=Path,
        help="save unvalidated local model text for diagnosis",
    )
    analyze.add_argument("--dry-run", action="store_true")
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-catalog":
        catalog = build_catalog(args.material_root, args.output)
        print(json.dumps({"output": str(args.output), "count": len(catalog.materials)}))
        return 0

    catalog = MaterialCatalog.load(args.catalog, material_root=args.material_root)
    if args.command == "search":
        print(
            json.dumps(
                catalog.search(
                    family=args.family,
                    color=args.color,
                    finish=args.finish,
                    query=args.query,
                    top_k=args.top_k,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    registry = _read_object(args.registry)
    parts = registry.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Part registry contains no parts")
    views = _collect_input_views(
        explicit_views=args.view,
        view_directories=args.view_dir,
        registry=registry,
        include_registry_renders=args.include_registry_renders,
    )

    candidates = _candidate_records(
        catalog,
        whitelist=None if args.no_whitelist else args.whitelist,
        family=args.family,
        color=args.color,
        finish=args.finish,
        query=args.query,
        top_k=args.top_k,
    )
    client = _create_analysis_client(args)
    result = client.analyze(
        views=views,
        parts=parts,
        candidate_materials=candidates,
        max_assignments=args.max_assignments,
        dry_run=args.dry_run,
    )
    output = _write_object(args.output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "backend": args.backend,
                "dry_run": args.dry_run,
                "view_count": len(views),
                "part_count": len(parts),
                "candidate_count": len(candidates),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
