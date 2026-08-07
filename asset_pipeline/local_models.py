"""Resolve local model assets and build an offline SAM3D config overlay."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


LOCAL_MODELS_ONLY_ENV = "PIPELINE_LOCAL_MODELS_ONLY"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
_REMOTE_URI_PREFIXES = ("http://", "https://", "hf://")
_KNOWN_REMOTE_MODEL_IDS = ("Ruicheng/moge", "facebookresearch/dinov2")
_LOCAL_MODEL_PATH_FIELDS = {
    "repo_or_dir",
    "pretrained_model_name_or_path",
    "repo_id",
    "model_id",
    "weights",
}


class LocalModelError(RuntimeError):
    """Raised when an inference path would need a remote model download."""


@dataclass(frozen=True)
class Sam3dLocalModels:
    """All local source and weight paths used by SAM3D reconstruction."""

    single_view_root: Path
    pipeline_config: Path
    sam3_repository: Path
    sam3_checkpoint: Path
    moge_checkpoint: Path
    dinov2_repository: Path
    dinov2_checkpoint: Path

    def environment(self) -> dict[str, str]:
        return {
            "SAM3D_SINGLE_VIEW_ROOT": str(self.single_view_root),
            "SAM3D_PIPELINE_CONFIG": str(self.pipeline_config),
            "SAM3_REPOSITORY": str(self.sam3_repository),
            "SAM3_CHECKPOINT": str(self.sam3_checkpoint),
            "SAM3D_MOGE_CHECKPOINT": str(self.moge_checkpoint),
            "SAM3D_DINOV2_REPOSITORY": str(self.dinov2_repository),
            "SAM3D_DINOV2_CHECKPOINT": str(self.dinov2_checkpoint),
        }


def configure_offline_model_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Permanently disable implicit downloads in normal project inference."""

    target = os.environ if environment is None else environment
    target[LOCAL_MODELS_ONLY_ENV] = "1"
    for name, value in OFFLINE_ENVIRONMENT.items():
        # Local-only mode is a fail-closed contract, so an inherited online
        # value cannot silently weaken it.
        target[name] = value
    target.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    target.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    return {
        LOCAL_MODELS_ONLY_ENV: target[LOCAL_MODELS_ONLY_ENV],
        **{name: target[name] for name in OFFLINE_ENVIRONMENT},
        "HF_HOME": target["HF_HOME"],
        "TORCH_HOME": target["TORCH_HOME"],
    }


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        path = candidate.expanduser()
        if path.exists():
            return path.resolve()
    return None


def _required_path(
    *,
    environment: Mapping[str, str],
    variable: str,
    candidates: list[Path],
    kind: str,
    minimum_bytes: int = 1,
) -> Path:
    raw = str(environment.get(variable, "")).strip()
    path = Path(raw).expanduser().resolve() if raw else _first_existing(candidates)
    if path is None or not path.exists():
        searched = ", ".join(str(item.expanduser()) for item in candidates)
        raise LocalModelError(
            f"Local model path {variable} is missing; set it in .env. "
            f"Searched: {searched or '(no fallback candidates)'}"
        )
    if kind == "file":
        if not path.is_file():
            raise LocalModelError(f"{variable} must be a file: {path}")
        if path.stat().st_size < minimum_bytes:
            raise LocalModelError(
                f"{variable} is empty or incomplete: {path} "
                f"({path.stat().st_size} bytes)"
            )
    elif kind == "directory":
        if not path.is_dir():
            raise LocalModelError(f"{variable} must be a directory: {path}")
    else:  # pragma: no cover - internal caller contract.
        raise ValueError(f"Unsupported local model path kind: {kind}")
    return path


def _huggingface_hub_roots(environment: Mapping[str, str]) -> list[Path]:
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = str(environment.get(name, "")).strip()
        if value:
            roots.append(Path(value).expanduser())
    hf_home = Path(
        str(environment.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    roots.append(hf_home / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduplicated.append(root)
            seen.add(key)
    return deduplicated


def _moge_candidates(environment: Mapping[str, str], root: Path) -> list[Path]:
    candidates = [
        root / "checkpoints" / "moge-vitl" / "model.pt",
        root / "models" / "moge-vitl" / "model.pt",
    ]
    for hub_root in _huggingface_hub_roots(environment):
        snapshots = hub_root / "models--Ruicheng--moge-vitl" / "snapshots"
        if snapshots.is_dir():
            cached: list[tuple[float, Path]] = []
            for path in snapshots.glob("*/model.pt"):
                try:
                    if path.is_file():
                        cached.append((path.stat().st_mtime, path))
                except OSError:
                    # A Hugging Face cache can contain a broken snapshot
                    # symlink while another process is pruning blobs.
                    continue
            candidates.extend(
                path
                for _, path in sorted(
                    cached, key=lambda candidate: candidate[0], reverse=True
                )
            )
    return candidates


def resolve_sam3d_local_models(
    single_view_root: str | Path,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> Sam3dLocalModels:
    """Resolve every active SAM3D weight without contacting a model hub."""

    target = os.environ if environment is None else environment
    configure_offline_model_environment(target)
    root = Path(single_view_root).expanduser().resolve()
    torch_home = Path(str(target.get("TORCH_HOME", Path.home() / ".cache" / "torch")))
    models = Sam3dLocalModels(
        single_view_root=_required_path(
            environment=target,
            variable="SAM3D_SINGLE_VIEW_ROOT",
            candidates=[root],
            kind="directory",
        ),
        pipeline_config=_required_path(
            environment=target,
            variable="SAM3D_PIPELINE_CONFIG",
            candidates=[
                root
                / "checkpoints"
                / "sam-3d-objects"
                / "checkpoints"
                / "pipeline.yaml"
            ],
            kind="file",
        ),
        sam3_repository=_required_path(
            environment=target,
            variable="SAM3_REPOSITORY",
            candidates=[root / "submodules" / "sam3"],
            kind="directory",
        ),
        sam3_checkpoint=_required_path(
            environment=target,
            variable="SAM3_CHECKPOINT",
            candidates=[
                root / "checkpoints" / "sam3.pt",
                Path.home()
                / ".cache"
                / "modelscope"
                / "hub"
                / "models"
                / "facebook"
                / "sam3"
                / "sam3.pt",
            ],
            kind="file",
            minimum_bytes=1024 * 1024,
        ),
        moge_checkpoint=_required_path(
            environment=target,
            variable="SAM3D_MOGE_CHECKPOINT",
            candidates=_moge_candidates(target, root),
            kind="file",
            minimum_bytes=1024 * 1024,
        ),
        dinov2_repository=_required_path(
            environment=target,
            variable="SAM3D_DINOV2_REPOSITORY",
            candidates=[torch_home / "hub" / "facebookresearch_dinov2_main"],
            kind="directory",
        ),
        dinov2_checkpoint=_required_path(
            environment=target,
            variable="SAM3D_DINOV2_CHECKPOINT",
            candidates=[
                torch_home / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
            ],
            kind="file",
            minimum_bytes=1024 * 1024,
        ),
    )
    if not (models.sam3_repository / "sam3" / "model_builder.py").is_file():
        raise LocalModelError(
            f"SAM3_REPOSITORY is incomplete: {models.sam3_repository}"
        )
    sam3_bpe = models.sam3_repository / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    try:
        bpe_is_complete = sam3_bpe.is_file() and sam3_bpe.stat().st_size >= 1024
    except OSError:
        bpe_is_complete = False
    if not bpe_is_complete:
        raise LocalModelError(
            "SAM3_REPOSITORY is missing its local BPE vocabulary: " f"{sam3_bpe}"
        )
    if not (models.dinov2_repository / "hubconf.py").is_file():
        raise LocalModelError(
            "SAM3D_DINOV2_REPOSITORY is incomplete: " f"{models.dinov2_repository}"
        )
    target.update(models.environment())
    return models


def configure_discovered_sam3d_environment(
    project_root: str | Path,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Populate existing local SAM3D paths without requiring that workflow."""

    target = os.environ if environment is None else environment
    configure_offline_model_environment(target)
    project = Path(project_root).expanduser().resolve()
    configured_single = str(target.get("SAM3D_SINGLE_VIEW_ROOT", "")).strip()
    configured_multi = str(target.get("SAM3D_MULTI_VIEW_ROOT", "")).strip()
    single_root = _first_existing(
        [
            *([Path(configured_single)] if configured_single else []),
            project / "tools" / "sam3d" / "third_party" / "sam-3d-objects",
            Path.home() / "下载" / "sam-3d-objects",
        ]
    )
    multi_root = _first_existing(
        [
            *([Path(configured_multi)] if configured_multi else []),
            project / "tools" / "sam3d" / "third_party" / "sam-3d-objects-multiview",
            Path.home() / "sam-3d-objects-multiview",
        ]
    )
    if single_root is not None:
        target.setdefault("SAM3D_SINGLE_VIEW_ROOT", str(single_root))
        try:
            resolve_sam3d_local_models(single_root, environment=target)
        except LocalModelError:
            # Other workflows must remain usable without SAM3D. The SAM3D
            # entry point repeats this resolution strictly before inference.
            pass
    if multi_root is not None:
        target.setdefault("SAM3D_MULTI_VIEW_ROOT", str(multi_root))
    names = (
        "SAM3D_SINGLE_VIEW_ROOT",
        "SAM3D_MULTI_VIEW_ROOT",
        "SAM3D_PIPELINE_CONFIG",
        "SAM3_REPOSITORY",
        "SAM3_CHECKPOINT",
        "SAM3D_MOGE_CHECKPOINT",
        "SAM3D_DINOV2_REPOSITORY",
        "SAM3D_DINOV2_CHECKPOINT",
    )
    return {name: target[name] for name in names if target.get(name)}


def _resolve_pipeline_member(
    config_dir: Path,
    value: Any,
    *,
    label: str,
    minimum_bytes: int = 1,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LocalModelError(f"SAM3D pipeline field {label} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise LocalModelError(
            f"SAM3D pipeline dependency is missing or incomplete: {label}={path}"
        )
    return path


def _patch_dinov2_nodes(value: Any, models: Sam3dLocalModels) -> int:
    count = 0
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str) and target.endswith(".dit.embedder.dino.Dino"):
            value["repo_or_dir"] = str(models.dinov2_repository)
            value["source"] = "local"
            raw_kwargs = value.get("backbone_kwargs")
            kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
            kwargs["weights"] = str(models.dinov2_checkpoint)
            value["backbone_kwargs"] = kwargs
            count += 1
        for child in value.values():
            count += _patch_dinov2_nodes(child, models)
    elif isinstance(value, list):
        for child in value:
            count += _patch_dinov2_nodes(child, models)
    return count


def _remote_model_sources(value: Any, trail: tuple[str, ...] = ()) -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            issues.extend(_remote_model_sources(child, (*trail, str(key))))
        return issues
    if isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_remote_model_sources(child, (*trail, str(index))))
        return issues
    if not isinstance(value, str):
        return issues

    text = value.strip()
    lower = text.lower()
    label = ".".join(trail) or "<root>"
    if lower.startswith(_REMOTE_URI_PREFIXES) or any(
        model_id.lower() in lower for model_id in _KNOWN_REMOTE_MODEL_IDS
    ):
        issues.append(f"{label}={text}")
        return issues
    field = trail[-1].lower() if trail else ""
    if field == "source" and lower in {"github", "huggingface", "hf", "remote"}:
        issues.append(f"{label}={text}")
    elif field in _LOCAL_MODEL_PATH_FIELDS and text and not Path(text).is_absolute():
        issues.append(f"{label}={text}")
    return issues


def _assert_local_model_sources(value: Any, *, label: str) -> None:
    issues = _remote_model_sources(value)
    if issues:
        preview = ", ".join(issues[:5])
        raise LocalModelError(
            f"SAM3D local overlay contains a non-local model source in {label}: "
            f"{preview}"
        )


def materialize_sam3d_local_config(
    models: Sam3dLocalModels,
    output_dir: str | Path,
) -> Path:
    """Create an absolute, offline-only overlay without editing vendor files."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config_dir = models.pipeline_config.parent
    document = yaml.safe_load(models.pipeline_config.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LocalModelError(
            f"SAM3D pipeline config must be a YAML mapping: {models.pipeline_config}"
        )

    depth_model = document.get("depth_model")
    if not isinstance(depth_model, dict) or not isinstance(
        depth_model.get("model"), dict
    ):
        raise LocalModelError("SAM3D pipeline has no configurable depth_model.model")
    depth_model["model"]["pretrained_model_name_or_path"] = str(models.moge_checkpoint)

    dino_node_count = 0
    for key, value in list(document.items()):
        if key.endswith("_ckpt_path"):
            document[key] = str(
                _resolve_pipeline_member(
                    config_dir,
                    value,
                    label=key,
                    minimum_bytes=1024,
                )
            )
        elif key.endswith("_config_path"):
            source = _resolve_pipeline_member(config_dir, value, label=key)
            nested = yaml.safe_load(source.read_text(encoding="utf-8"))
            patched_count = _patch_dinov2_nodes(nested, models)
            _assert_local_model_sources(nested, label=source.name)
            patched_path = destination / f"{key}.{source.name}"
            patched_path.write_text(
                yaml.safe_dump(nested, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            document[key] = str(patched_path)
            dino_node_count += patched_count

    if dino_node_count < 1:
        raise LocalModelError(
            "SAM3D pipeline contains no DINOv2 encoder to bind to local weights"
        )
    _assert_local_model_sources(document, label=models.pipeline_config.name)
    overlay = destination / "pipeline.local.yaml"
    overlay.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return overlay


__all__ = [
    "LOCAL_MODELS_ONLY_ENV",
    "LocalModelError",
    "Sam3dLocalModels",
    "configure_discovered_sam3d_environment",
    "configure_offline_model_environment",
    "materialize_sam3d_local_config",
    "resolve_sam3d_local_models",
]
