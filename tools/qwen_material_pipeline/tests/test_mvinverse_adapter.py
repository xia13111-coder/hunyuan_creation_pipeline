from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from qwen_material_pipeline.mvinverse.adapter import (
    MVInverseExecutionError,
    MVInverseInputError,
    SCHEMA_VERSION,
    _git_metadata,
    main,
    run_mvinverse_adapter,
)


FAKE_RUNNER = r"""#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--checkpoint-format", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--num-frames", type=int, required=True)
args = parser.parse_args()

counter = args.repo / "invocations.txt"
previous = counter.read_text(encoding="utf-8") if counter.exists() else ""
counter.write_text(previous + args.input_dir.name + "\n", encoding="utf-8")
behavior = (args.repo / "behavior.txt").read_text(encoding="utf-8").strip() if (args.repo / "behavior.txt").exists() else "success"
if behavior == "oom" and "0112" in args.input_dir.name:
    print("torch.cuda.OutOfMemoryError: CUDA out of memory", file=sys.stderr)
    raise SystemExit(1)

inputs = sorted(args.input_dir.glob("*.png"))
with Image.open(inputs[0]) as first:
    size = first.size
args.output_dir.mkdir(parents=True)
maps = (("albedo", "RGB"), ("metallic", "L"), ("roughness", "L"), ("normal", "RGB"), ("shading", "RGB"))
for index in range(args.num_frames):
    for name, mode in maps:
        if behavior == "incomplete" and not (index == 0 and name == "albedo"):
            continue
        color = (10 + index, 20, 30) if mode == "RGB" else 80 + index
        Image.new(mode, size, color).save(args.output_dir / f"{index:03d}_{name}.png")
print(json.dumps({
    "status": "SUCCESS",
    "image_shape_nchw": [args.num_frames, 3, size[1], size[0]],
    "elapsed_seconds": 0.01,
    "cuda_memory": {"max_memory_allocated_bytes": 123, "max_memory_reserved_bytes": 456},
    "offline": os.environ.get("HF_HUB_OFFLINE"),
}))
"""


def test_packaged_revision_does_not_inherit_parent_git_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "vendored" / "mvinverse"
    repo.mkdir(parents=True)
    revision = "6172ff9a437444df028ed67523badfa523173f21"
    (repo / "REVISION").write_text(revision + "\n", encoding="utf-8")

    assert _git_metadata(repo) == {
        "git_revision": revision,
        "tracked_worktree_dirty": False,
    }


def _case(
    tmp_path: Path, *, sizes: tuple[tuple[int, int], ...] = ((100, 140), (100, 140))
):
    images = []
    colors = ((220, 10, 10), (10, 220, 10), (10, 10, 220))
    # Filenames are deliberately opposite to manifest order.  The adapter must
    # preserve source_views order instead of sorting source paths.
    names = ("z_last.png", "a_first.png", "m_middle.png")
    for index, size in enumerate(sizes):
        path = tmp_path / names[index]
        Image.new("RGB", size, colors[index]).save(path)
        images.append(path)
    manifest = tmp_path / "reference_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_views": [
                    {"id": f"ref_{index}", "image": str(path)}
                    for index, path in enumerate(images)
                ],
                "view_order_semantics": "unordered_same_asset_views",
            }
        ),
        encoding="utf-8",
    )
    repo = tmp_path / "external_repo"
    module = repo / "mvinverse" / "models"
    module.mkdir(parents=True)
    (module / "mvinverse.py").write_text("class MVInverse: pass\n", encoding="utf-8")
    (repo / "LICENSE").write_text(
        "MVInverse for non-commercial purposes\n", encoding="utf-8"
    )
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER, encoding="utf-8")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"deterministic fake checkpoint")
    output = tmp_path / "output"
    return manifest, repo, runner, checkpoint, output


def _run(
    manifest: Path,
    repo: Path,
    runner: Path,
    checkpoint: Path,
    output: Path,
    **kwargs,
):
    return run_mvinverse_adapter(
        reference_manifest=manifest,
        repo=repo,
        python_executable=sys.executable,
        checkpoint=checkpoint,
        model_revision="fake-revision-001",
        output_dir=output,
        acknowledge_noncommercial=True,
        device="cpu",
        max_side=112,
        oom_retry_max_sides=(84,),
        runner_script=runner,
        **kwargs,
    )


def test_license_acknowledgement_is_required_before_external_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    with pytest.raises(MVInverseInputError, match="explicitly acknowledged"):
        run_mvinverse_adapter(
            reference_manifest=tmp_path / "missing.json",
            repo=tmp_path / "missing_repo",
            python_executable=sys.executable,
            checkpoint=tmp_path / "missing.pt",
            output_dir=output,
            acknowledge_noncommercial=False,
        )

    ledger = json.loads(
        (output / "mvinverse_inference_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "FAILED"
    assert ledger["license"]["acknowledged_noncommercial_use"] is False
    assert ledger["failure"]["phase"] == "license"


def test_success_preserves_manifest_order_and_hashes_every_output(
    tmp_path: Path,
) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)

    ledger = _run(manifest, repo, runner, checkpoint, output)

    assert ledger["schema_version"] == SCHEMA_VERSION
    assert ledger["status"] == "SUCCESS"
    assert [view["view_id"] for view in ledger["inputs"]["source_views"]] == [
        "ref_0",
        "ref_1",
    ]
    staged = ledger["preprocessing"]["attempts"][0]
    assert [item["filename"] for item in staged["images"]] == [
        "000000.png",
        "000001.png",
    ]
    assert staged["common_size"] == [70, 112]
    with Image.open(Path(staged["images"][0]["path"])) as first:
        assert first.getpixel((10, 10))[0] > 200
    with Image.open(Path(staged["images"][1]["path"])) as second:
        assert second.getpixel((10, 10))[1] > 200
    assert ledger["model"]["checkpoint"]["declared_revision"] == "fake-revision-001"
    assert len(ledger["model"]["checkpoint"]["checkpoint_sha256"]) == 64
    assert ledger["outputs"]["map_count"] == 10
    assert len(ledger["outputs"]["output_set_sha256"]) == 64
    assert all(len(record["sha256"]) == 64 for record in ledger["outputs"]["maps"])
    attempt = ledger["execution"]["attempts"][0]
    assert attempt["status"] == "SUCCESS"
    assert attempt["runner_telemetry"]["offline"] == "1"
    assert (
        attempt["runner_telemetry"]["cuda_memory"]["max_memory_allocated_bytes"] == 123
    )
    assert attempt["elapsed_seconds"] >= 0
    persisted = json.loads(
        (output / "mvinverse_inference_ledger.json").read_text(encoding="utf-8")
    )

    def absolute_paths(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [path for item in value.values() for path in absolute_paths(item)]
        if isinstance(value, list):
            return [path for item in value for path in absolute_paths(item)]
        if isinstance(value, str) and Path(value).is_absolute():
            return [value]
        return []

    assert absolute_paths(persisted) == []


def test_dry_run_stages_fixed_resolutions_but_never_launches(tmp_path: Path) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)

    ledger = _run(manifest, repo, runner, checkpoint, output, dry_run=True)

    assert ledger["status"] == "DRY_RUN"
    assert ledger["execution"]["status"] == "NOT_EXECUTED"
    assert ledger["execution"]["attempts"] == []
    assert [item["max_side"] for item in ledger["preprocessing"]["attempts"]] == [
        112,
        84,
    ]
    assert (output / "inputs_0112").is_dir()
    assert (output / "inputs_0084").is_dir()
    assert not (repo / "invocations.txt").exists()
    assert ledger["outputs"] is None


def test_reuse_revalidates_fingerprint_and_output_hashes_without_launch(
    tmp_path: Path,
) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)
    first = _run(manifest, repo, runner, checkpoint, output)
    assert (repo / "invocations.txt").read_text(encoding="utf-8").splitlines() == [
        "inputs_0112"
    ]

    reused = _run(manifest, repo, runner, checkpoint, output, reuse_existing=True)

    assert reused["status"] == "REUSED"
    assert reused["run_fingerprint"] == first["run_fingerprint"]
    assert (
        reused["outputs"]["output_set_sha256"] == first["outputs"]["output_set_sha256"]
    )
    assert reused["execution"]["status"] == "REUSED_NOT_EXECUTED"
    assert (repo / "invocations.txt").read_text(encoding="utf-8").splitlines() == [
        "inputs_0112"
    ]

    Image.new("RGB", (70, 112), "purple").save(output / "maps" / "000_albedo.png")
    with pytest.raises(MVInverseInputError, match="hashes do not match"):
        _run(manifest, repo, runner, checkpoint, output, reuse_existing=True)
    failed = json.loads(
        (output / "mvinverse_inference_ledger.json").read_text(encoding="utf-8")
    )
    assert failed["status"] == "FAILED"
    assert failed["failure"]["phase"] == "select_mode"


def test_reuse_is_portable_across_output_directories_when_pixels_are_identical(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    manifest, repo, runner, checkpoint, output = _case(first_root)
    first = _run(manifest, repo, runner, checkpoint, output)

    second_root = tmp_path / "second"
    second_root.mkdir()
    relocated_images: list[Path] = []
    first_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    for index, view in enumerate(first_manifest["source_views"]):
        source = Path(view["image"])
        relocated = second_root / f"relocated_{index}.png"
        shutil.copy2(source, relocated)
        relocated_images.append(relocated)
    relocated_manifest = second_root / "reference_manifest.json"
    relocated_manifest.write_text(
        json.dumps(
            {
                "source_views": [
                    {"id": f"ref_{index}", "image": str(path)}
                    for index, path in enumerate(relocated_images)
                ],
                "view_order_semantics": "unordered_same_asset_views",
            }
        ),
        encoding="utf-8",
    )
    relocated_output = second_root / "output"
    shutil.copytree(output, relocated_output)

    reused = _run(
        relocated_manifest,
        repo,
        runner,
        checkpoint,
        relocated_output,
        reuse_existing=True,
    )

    assert reused["status"] == "REUSED"
    assert reused["run_fingerprint"] == first["run_fingerprint"]
    assert reused["execution"]["reuse_fingerprint_validation"] == {
        "status": "PASS",
        "contract": "qwen-mvinverse-content-fingerprint/v2",
        "relocation_compatible": True,
        "source_set_sha256": first["inputs"]["source_set_sha256"],
    }
    assert (repo / "invocations.txt").read_text(encoding="utf-8").splitlines() == [
        "inputs_0112"
    ]


def test_cuda_oom_retries_only_at_next_fixed_resolution(tmp_path: Path) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)
    (repo / "behavior.txt").write_text("oom", encoding="utf-8")

    ledger = _run(manifest, repo, runner, checkpoint, output)

    attempts = ledger["execution"]["attempts"]
    assert [item["status"] for item in attempts] == ["CUDA_OOM", "SUCCESS"]
    assert [item["max_side"] for item in attempts] == [112, 84]
    assert ledger["outputs"]["preprocessing_max_side"] == 84
    assert ledger["outputs"]["preprocessed_size"] == [56, 84]
    assert (repo / "invocations.txt").read_text(encoding="utf-8").splitlines() == [
        "inputs_0112",
        "inputs_0084",
    ]


def test_incomplete_output_fails_closed_and_is_not_published(tmp_path: Path) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)
    (repo / "behavior.txt").write_text("incomplete", encoding="utf-8")

    with pytest.raises(MVInverseExecutionError, match="incomplete or contaminated"):
        _run(manifest, repo, runner, checkpoint, output)

    assert not (output / "maps").exists()
    assert not (output / ".maps.tmp").exists()
    ledger = json.loads(
        (output / "mvinverse_inference_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "FAILED"
    assert ledger["failure"]["phase"] == "verify_outputs"


def test_mismatched_preprocessed_sizes_fail_before_subprocess(tmp_path: Path) -> None:
    manifest, repo, runner, checkpoint, output = _case(
        tmp_path, sizes=((100, 140), (140, 100))
    )

    with pytest.raises(MVInverseInputError, match="preprocessed sizes differ"):
        _run(manifest, repo, runner, checkpoint, output, dry_run=True)

    assert not (repo / "invocations.txt").exists()


def test_local_huggingface_directory_is_supported_but_single_safetensors_is_not(
    tmp_path: Path,
) -> None:
    manifest, repo, runner, _checkpoint, output = _case(tmp_path)
    checkpoint_dir = tmp_path / "local_hf_model"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")
    weights = checkpoint_dir / "model.safetensors"
    weights.write_bytes(b"fake safetensors for adapter contract test")

    ledger = _run(
        manifest,
        repo,
        runner,
        checkpoint_dir,
        output,
        dry_run=True,
    )
    assert ledger["model"]["checkpoint"]["format"] == "huggingface_directory"
    assert ledger["model"]["checkpoint"]["effective_revision"] == "fake-revision-001"

    second_output = tmp_path / "single_file_output"
    with pytest.raises(MVInverseInputError, match="parent directory"):
        _run(manifest, repo, runner, weights, second_output, dry_run=True)


def test_cli_returns_input_error_and_writes_failed_ledger_without_ack(
    tmp_path: Path,
) -> None:
    manifest, repo, runner, checkpoint, output = _case(tmp_path)

    exit_code = main(
        [
            "--reference-manifest",
            str(manifest),
            "--repo",
            str(repo),
            "--python",
            sys.executable,
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--runner-script",
            str(runner),
        ]
    )

    assert exit_code == 2
    ledger = json.loads(
        (output / "mvinverse_inference_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "FAILED"
