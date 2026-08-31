from __future__ import annotations

import subprocess
from pathlib import Path

from qwen_material_pipeline.setup.material_models import (
    ENTITYSEG_FILENAME,
    ModelSetupError,
    SetupPaths,
    _checkout_repository,
    main,
    update_environment_file,
)
from qwen_material_pipeline.setup import material_models


def test_setup_paths_cover_every_downloaded_model(tmp_path: Path) -> None:
    paths = SetupPaths.from_root(tmp_path / "models")
    environment = paths.environment()

    assert environment["QWEN35_MODEL_PATH"] == str(paths.qwen_model)
    assert environment["MVINVERSE_CHECKPOINT"] == str(paths.mvinverse_model)
    assert environment["SAM3_REPOSITORY"] == str(paths.sam3_repository)
    assert environment["SAM3_CHECKPOINT"] == str(paths.sam3_checkpoint)
    assert environment["ENTITYSEG_PYTHON"] == str(paths.entityseg_python)
    assert environment["ENTITYSEG_CHECKPOINT"].endswith(ENTITYSEG_FILENAME)
    assert environment["NVIDIA_BASE_OBSERVATION_BANK"] == str(paths.observation_bank)


def test_environment_update_preserves_unmanaged_values(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text(
        "SECRET=keep-me\nQWEN35_MODEL_PATH=/old\nQWEN35_MODEL_PATH=/duplicate\n",
        encoding="utf-8",
    )

    update_environment_file(
        target,
        {
            "QWEN35_MODEL_PATH": "/new path/qwen",
            "SAM3_CHECKPOINT": "/models/sam3.pt",
        },
    )

    result = target.read_text(encoding="utf-8")
    assert "SECRET=keep-me" in result
    assert result.count("QWEN35_MODEL_PATH=") == 1
    assert 'QWEN35_MODEL_PATH="/new path/qwen"' in result
    assert "SAM3_CHECKPOINT=/models/sam3.pt" in result


def test_dry_run_does_not_download_or_write(tmp_path: Path, capsys) -> None:
    model_root = tmp_path / "models"
    env_file = tmp_path / ".env"

    assert (
        main(
            [
                "--model-root",
                str(model_root),
                "--env-file",
                str(env_file),
                "--dry-run",
            ]
        )
        == 0
    )

    assert not model_root.exists()
    assert not env_file.exists()
    assert str(model_root) in capsys.readouterr().out


def test_access_failure_stops_before_any_download(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    qwen_started = False
    sources_started = False

    def fail_access() -> None:
        raise ModelSetupError("access denied")

    def record_qwen(_paths: SetupPaths) -> None:
        nonlocal qwen_started
        qwen_started = True

    def record_sources(_paths: SetupPaths) -> None:
        nonlocal sources_started
        sources_started = True

    monkeypatch.setattr(material_models, "_check_gated_access", fail_access)
    monkeypatch.setattr(material_models, "_prepare_sources", record_sources)
    monkeypatch.setattr(material_models, "_run_qwen_setup", record_qwen)

    assert main(["--model-root", str(tmp_path / "models")]) == 1
    assert not sources_started
    assert not qwen_started
    assert "access denied" in capsys.readouterr().err


def test_source_checkout_populates_worktree_when_default_head_is_pinned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "setup-test@example.invalid"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Setup Test"], cwd=source, check=True)
    (source / "model.py").write_text("ready = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    destination = tmp_path / "checkout"
    _checkout_repository(
        repository=str(source), revision=revision, destination=destination
    )

    assert (destination / "model.py").read_text(encoding="utf-8") == "ready = True\n"
