from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

from asset_pipeline.local_models import (
    LOCAL_MODELS_ONLY_ENV,
    OFFLINE_ENVIRONMENT,
    LocalModelError,
    configure_offline_model_environment,
    materialize_sam3d_local_config,
    resolve_sam3d_local_models,
)
from tools.sam3d import run_multiview_local


_MINIMUM_WEIGHT_BYTES = 1024 * 1024


def _write_sparse_weight(path: Path, size: int = _MINIMUM_WEIGHT_BYTES) -> Path:
    """Create a lightweight size-valid checkpoint without allocating a model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.seek(size - 1)
        handle.write(b"\0")
    return path


def _sam3d_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "sam-3d-objects"
    config_dir = root / "checkpoints" / "sam-3d-objects" / "checkpoints"
    pipeline_config = config_dir / "pipeline.yaml"
    dino_config = config_dir / "slat_generator.yaml"
    sam3_repository = root / "submodules" / "sam3"
    dino_repository = tmp_path / "torch-hub" / "facebookresearch_dinov2_main"

    sam3_repository.joinpath("sam3").mkdir(parents=True)
    sam3_repository.joinpath("sam3", "model_builder.py").write_text(
        "# local SAM3 fixture\n",
        encoding="utf-8",
    )
    _write_sparse_weight(
        sam3_repository / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        size=1024,
    )
    dino_repository.mkdir(parents=True)
    dino_repository.joinpath("hubconf.py").write_text(
        "# local DINOv2 fixture\n",
        encoding="utf-8",
    )

    dino_config.parent.mkdir(parents=True, exist_ok=True)
    dino_config.write_text(
        yaml.safe_dump(
            {
                "encoder": {
                    "_target_": "sam3d_objects.model.backbone.dit.embedder.dino.Dino",
                    "repo_or_dir": "facebookresearch/dinov2",
                    "source": "github",
                    "backbone_kwargs": {"name": "dinov2_vitl14_reg"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pipeline_config.write_text(
        yaml.safe_dump(
            {
                "depth_model": {
                    "model": {"pretrained_model_name_or_path": "Ruicheng/moge-vitl"}
                },
                "slat_generator_config_path": dino_config.name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sam3_checkpoint = _write_sparse_weight(root / "checkpoints" / "sam3.pt")
    moge_checkpoint = _write_sparse_weight(tmp_path / "models" / "moge" / "model.pt")
    dino_checkpoint = _write_sparse_weight(
        tmp_path / "torch-hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    )
    environment = {
        "SAM3D_SINGLE_VIEW_ROOT": str(root),
        "SAM3D_PIPELINE_CONFIG": str(pipeline_config),
        "SAM3_REPOSITORY": str(sam3_repository),
        "SAM3_CHECKPOINT": str(sam3_checkpoint),
        "SAM3D_MOGE_CHECKPOINT": str(moge_checkpoint),
        "SAM3D_DINOV2_REPOSITORY": str(dino_repository),
        "SAM3D_DINOV2_CHECKPOINT": str(dino_checkpoint),
    }
    return root, environment


def test_offline_environment_is_fail_closed() -> None:
    environment = {
        LOCAL_MODELS_ONLY_ENV: "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "false",
        "HF_DATASETS_OFFLINE": "no",
        "HF_HUB_DISABLE_TELEMETRY": "0",
        "HF_HOME": "/models/huggingface",
        "TORCH_HOME": "/models/torch",
    }

    resolved = configure_offline_model_environment(environment)

    assert environment[LOCAL_MODELS_ONLY_ENV] == "1"
    assert all(environment[name] == "1" for name in OFFLINE_ENVIRONMENT)
    assert resolved["HF_HOME"] == "/models/huggingface"
    assert resolved["TORCH_HOME"] == "/models/torch"


def test_resolve_sam3d_local_models_uses_only_explicit_local_assets(
    tmp_path: Path,
) -> None:
    root, environment = _sam3d_fixture(tmp_path)

    models = resolve_sam3d_local_models(root, environment=environment)

    assert models.single_view_root == root.resolve()
    assert (
        models.pipeline_config == Path(environment["SAM3D_PIPELINE_CONFIG"]).resolve()
    )
    assert models.sam3_checkpoint == Path(environment["SAM3_CHECKPOINT"]).resolve()
    assert (
        models.moge_checkpoint == Path(environment["SAM3D_MOGE_CHECKPOINT"]).resolve()
    )
    assert (
        models.dinov2_repository
        == Path(environment["SAM3D_DINOV2_REPOSITORY"]).resolve()
    )
    assert (
        models.dinov2_checkpoint
        == Path(environment["SAM3D_DINOV2_CHECKPOINT"]).resolve()
    )
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_sam3d_overlay_replaces_moge_and_dinov2_remote_sources(
    tmp_path: Path,
) -> None:
    root, environment = _sam3d_fixture(tmp_path)
    models = resolve_sam3d_local_models(root, environment=environment)

    overlay = materialize_sam3d_local_config(models, tmp_path / "overlay")
    document = yaml.safe_load(overlay.read_text(encoding="utf-8"))

    assert document["depth_model"]["model"]["pretrained_model_name_or_path"] == str(
        models.moge_checkpoint
    )
    patched_dino_config = Path(document["slat_generator_config_path"])
    patched_dino = yaml.safe_load(patched_dino_config.read_text(encoding="utf-8"))[
        "encoder"
    ]
    assert patched_dino["repo_or_dir"] == str(models.dinov2_repository)
    assert patched_dino["source"] == "local"
    assert patched_dino["backbone_kwargs"]["weights"] == str(models.dinov2_checkpoint)

    serialized = overlay.read_text(encoding="utf-8") + patched_dino_config.read_text(
        encoding="utf-8"
    )
    assert "Ruicheng/moge" not in serialized
    assert "facebookresearch/dinov2" not in serialized
    assert "http://" not in serialized
    assert "https://" not in serialized


def test_missing_explicit_weight_fails_before_any_fallback(tmp_path: Path) -> None:
    root, environment = _sam3d_fixture(tmp_path)
    missing = tmp_path / "missing" / "moge" / "model.pt"
    environment["SAM3D_MOGE_CHECKPOINT"] = str(missing)

    with pytest.raises(LocalModelError) as caught:
        resolve_sam3d_local_models(root, environment=environment)

    message = str(caught.value)
    assert "SAM3D_MOGE_CHECKPOINT" in message
    assert "is missing" in message
    assert not missing.exists()


def test_overlay_rejects_remote_source_in_any_nested_config(tmp_path: Path) -> None:
    root, environment = _sam3d_fixture(tmp_path)
    models = resolve_sam3d_local_models(root, environment=environment)
    pipeline = yaml.safe_load(models.pipeline_config.read_text(encoding="utf-8"))
    remote_config = models.pipeline_config.parent / "decoder.yaml"
    remote_config.write_text(
        yaml.safe_dump(
            {"model": {"pretrained_model_name_or_path": "example/remote-model"}}
        ),
        encoding="utf-8",
    )
    pipeline["decoder_config_path"] = remote_config.name
    models.pipeline_config.write_text(
        yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(LocalModelError, match="non-local model source"):
        materialize_sam3d_local_config(models, tmp_path / "overlay")


def test_multiview_wrapper_forces_overlay_and_forwards_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "capture.json"
    vendor_script = tmp_path / "run_inference.py"
    local_overlay = tmp_path / "pipeline.local.yaml"
    local_overlay.write_text("depth_model: local\n", encoding="utf-8")
    vendor_script.write_text(
        """
import json
import os
import sys
from pathlib import Path

calls = []


def Inference(config_path, *args, **kwargs):
    calls.append({"config_path": config_path, "args": args, "kwargs": kwargs})
    return object()


def main():
    Inference(
        "vendor-default.yaml",
        "constructor-positional",
        compile=False,
        marker="preserved",
    )
    Path(os.environ["SAM3D_FAKE_VENDOR_CAPTURE"]).write_text(
        json.dumps({"calls": calls, "argv": sys.argv}),
        encoding="utf-8",
    )
    return 23
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SAM3D_FAKE_VENDOR_CAPTURE", str(capture))
    outer_argv = ["outer-program", "--must-survive"]
    monkeypatch.setattr(sys, "argv", outer_argv)

    result = run_multiview_local.main(
        [
            "--vendor-script",
            str(vendor_script),
            "--model-config",
            str(local_overlay),
            "--",
            "--input_path",
            str(tmp_path / "images"),
            "--seed",
            "7",
        ]
    )

    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert result == 23
    assert payload["calls"] == [
        {
            "config_path": str(local_overlay.resolve()),
            "args": ["constructor-positional"],
            "kwargs": {"compile": False, "marker": "preserved"},
        }
    ]
    assert payload["argv"] == [
        str(vendor_script.resolve()),
        "--input_path",
        str(tmp_path / "images"),
        "--seed",
        "7",
    ]
    assert sys.argv is outer_argv


@pytest.mark.parametrize(
    "script_name",
    ("run_inference.py", "run_inference_weighted.py"),
)
def test_bundled_multiview_vendor_has_wrapper_static_interface(
    script_name: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = (
        project_root
        / "tools"
        / "sam3d"
        / "third_party"
        / "sam-3d-objects-multiview"
        / script_name
    )
    if not script.is_file():
        pytest.skip("optional SAM3D vendor checkout is not part of the source release")

    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    imports_inference = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "inference"
        and any(alias.name == "Inference" for alias in node.names)
        for node in tree.body
    )
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    inference_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Inference"
    ]

    assert imports_inference
    assert len(main_functions) == 1
    assert not main_functions[0].args.posonlyargs
    assert not main_functions[0].args.args
    assert inference_calls
