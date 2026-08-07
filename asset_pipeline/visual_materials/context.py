"""Validated invocation state for one visual-material pipeline run."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import unique_path
from .config import VisualMaterialConfig
from .contracts import USD_SUFFIXES, VISUAL_INFERENCE_MODES
from .workspace import VisualMaterialWorkspace


ConfigLoader = Callable[[str | Path | None], VisualMaterialConfig]
IsaacPythonResolver = Callable[[], Path]
ReferenceParser = Callable[[Sequence[str]], tuple[tuple[str, Path], ...]]
ResumeValidator = Callable[
    [
        Path,
        tuple[tuple[str, Path], ...],
        VisualMaterialConfig,
        Path | None,
    ],
    bool,
]


@dataclass(frozen=True)
class VisualMaterialPipelineContext:
    """Inputs, runtime, output root and artifact map after fail-closed validation."""

    source: Path
    source_cad: Path | None
    references: tuple[tuple[str, Path], ...]
    foreground_annotations: Path | None
    config_path: str
    config: VisualMaterialConfig
    isaac_python: Path
    destination: Path
    partial_live_resume: bool
    inference_mode: str
    workspace: VisualMaterialWorkspace

    @classmethod
    def create(
        cls,
        *,
        source_usd: str,
        source_cad: str | None,
        references: Sequence[str],
        foreground_annotations: str | None,
        output_dir: str | None,
        config_path: str | None,
        inference_mode: str,
        default_config_path: Path,
        config_loader: ConfigLoader,
        isaac_python_resolver: IsaacPythonResolver,
        reference_parser: ReferenceParser,
        resume_validator: ResumeValidator,
    ) -> "VisualMaterialPipelineContext":
        if inference_mode not in VISUAL_INFERENCE_MODES:
            raise ValueError(
                "inference_mode must be one of "
                f"{sorted(VISUAL_INFERENCE_MODES)}, got {inference_mode!r}"
            )

        resolved_foreground: Path | None = None
        if foreground_annotations is not None:
            if inference_mode != "live":
                raise ValueError(
                    "Human SAM3 foreground annotations require "
                    "inference_mode='live'"
                )
            try:
                resolved_foreground = (
                    Path(foreground_annotations).expanduser().resolve(strict=True)
                )
            except (OSError, RuntimeError) as exc:
                raise FileNotFoundError(
                    "SAM3 foreground annotation file does not exist: "
                    f"{foreground_annotations}"
                ) from exc
            if not resolved_foreground.is_file():
                raise ValueError(
                    "SAM3 foreground annotations must be one JSON file: "
                    f"{resolved_foreground}"
                )

        try:
            source = Path(source_usd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(
                f"CAD/GLB converted USD does not exist: {source_usd}"
            ) from exc
        if not source.is_file() or source.suffix.lower() not in USD_SUFFIXES:
            raise ValueError(f"Visual-material input must be one USD file: {source}")

        resolved_source_cad: Path | None = None
        if source_cad is not None:
            try:
                resolved_source_cad = (
                    Path(source_cad).expanduser().resolve(strict=True)
                )
            except (OSError, RuntimeError) as exc:
                raise FileNotFoundError(
                    f"Manual STEP/STP source does not exist: {source_cad}"
                ) from exc
            if (
                not resolved_source_cad.is_file()
                or resolved_source_cad.suffix.lower() not in {".stp", ".step"}
            ):
                raise ValueError(
                    f"Manual material source must be STEP/STP: {resolved_source_cad}"
                )

        parsed_references = reference_parser(references)
        effective_config_path = config_path or str(default_config_path)
        config = config_loader(effective_config_path)
        isaac = isaac_python_resolver().expanduser().resolve()
        if not isaac.is_file() or not os.access(isaac, os.X_OK):
            raise FileNotFoundError(f"Isaac Sim Python is unavailable: {isaac}")

        if output_dir is None:
            destination = unique_path(
                source.parent.parent / f"{source.stem}_visual_material"
            ).resolve()
            partial_live_resume = False
        else:
            destination = Path(output_dir).expanduser().resolve()
            partial_live_resume = inference_mode == "live" and resume_validator(
                destination,
                parsed_references,
                config,
                resolved_foreground,
            )
            if destination.exists() and not partial_live_resume:
                raise FileExistsError(
                    "Visual-material output already exists; refusing stale reuse: "
                    f"{destination}"
                )
        if not partial_live_resume:
            destination.mkdir(parents=True, exist_ok=False)

        return cls(
            source=source,
            source_cad=resolved_source_cad,
            references=parsed_references,
            foreground_annotations=resolved_foreground,
            config_path=str(effective_config_path),
            config=config,
            isaac_python=isaac,
            destination=destination,
            partial_live_resume=partial_live_resume,
            inference_mode=inference_mode,
            workspace=VisualMaterialWorkspace.create(
                destination=destination,
                source=source,
            ),
        )


__all__ = ["VisualMaterialPipelineContext"]
