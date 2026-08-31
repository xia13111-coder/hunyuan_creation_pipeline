#!/usr/bin/env python3
"""Prepare Blender and automatic-material models, then update ``.env``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MATERIAL_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
QWEN_SETUP = MATERIAL_PIPELINE_ROOT / "scripts" / "qwen35" / "setup_qwen35_runtime.sh"
GIB = 1024**3
_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=")
_PLAIN_ENV_VALUE = re.compile(r"[A-Za-z0-9_./:@+,-]+")

BLENDER_VERSION = "4.5.0"
BLENDER_DIRECTORY = f"blender-{BLENDER_VERSION}-linux-x64"
BLENDER_ARCHIVE = f"{BLENDER_DIRECTORY}.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
BLENDER_SHA256 = "1188b95cc12321c770b631939f7c25a096910b6f884a990bf9c0f62d52b38aec"

MVINVERSE_REVISION = "ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e"
DINOV2_REVISION = "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
ENTITYSEG_REVISION = "143d5e7e2880dcf3df2df6570984143c2e9a9b45"
SAM3_SOURCE_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
ENTITY_SOURCE_REVISION = "6e7e13ac91ef508088e1b848167c01f19b00b512"
DETECTRON2_REVISION = "e39b8d0e6a5d17f713b20061b9cfc30f92213a5a"
ENTITYSEG_FILENAME = (
    "CropFormer_model/Entity_Segmentation/CropFormer_swin_tiny_3x/"
    "CropFormer_swin_tiny_3x_5cea5e.pth"
)


class ModelSetupError(RuntimeError):
    """Raised when model setup cannot complete safely."""


@dataclass(frozen=True)
class SetupPaths:
    model_root: Path
    qwen_runtime: Path
    qwen_python: Path
    qwen_model: Path
    siglip2_model: Path
    mvinverse_model: Path
    dinov2_model: Path
    sam3_checkpoint: Path
    entityseg_checkpoint: Path
    sam3_repository: Path
    entity_repository: Path
    entityseg_cropformer_root: Path
    entityseg_config: Path
    detectron2_repository: Path
    entityseg_runtime: Path
    entityseg_python: Path
    observation_bank: Path
    retrieval_cache: Path
    blender_home: Path
    blender_bin: Path
    blender_archive: Path

    @classmethod
    def from_root(cls, root: Path) -> "SetupPaths":
        model_root = root.expanduser().resolve()
        qwen_runtime = model_root / "qwen35_4b_runtime"
        entityseg_root = model_root / "models" / "entityseg"
        sources_root = model_root / "sources"
        entity_repository = sources_root / "Entity"
        cropformer_root = entity_repository / "Entityv2" / "CropFormer"
        entityseg_runtime = model_root / "runtimes" / "entityseg"
        return cls(
            model_root=model_root,
            qwen_runtime=qwen_runtime,
            qwen_python=qwen_runtime / "env" / "bin" / "python",
            qwen_model=qwen_runtime / "model",
            siglip2_model=model_root / "models" / "siglip2-base-patch16-224",
            mvinverse_model=model_root / "models" / "mvinverse",
            dinov2_model=model_root / "models" / "dinov2-with-registers-large",
            sam3_checkpoint=model_root / "models" / "sam3" / "sam3.pt",
            entityseg_checkpoint=entityseg_root / ENTITYSEG_FILENAME,
            sam3_repository=sources_root / "sam3",
            entity_repository=entity_repository,
            entityseg_cropformer_root=cropformer_root,
            entityseg_config=(
                cropformer_root
                / "configs"
                / "entityv2"
                / "entity_segmentation"
                / "cropformer_swin_tiny_3x.yaml"
            ),
            detectron2_repository=sources_root / "detectron2",
            entityseg_runtime=entityseg_runtime,
            entityseg_python=entityseg_runtime / "bin" / "python",
            observation_bank=(
                MATERIAL_PIPELINE_ROOT / "assets" / "nvidia_base_observation_bank_v1"
            ),
            retrieval_cache=model_root / "cache" / "visual-retrieval",
            blender_home=model_root / "tools" / BLENDER_DIRECTORY,
            blender_bin=model_root / "tools" / BLENDER_DIRECTORY / "blender",
            blender_archive=model_root / "downloads" / BLENDER_ARCHIVE,
        )

    def environment(self, *, blender_bin: Path | None = None) -> dict[str, str]:
        owner_python = str(Path(sys.executable).resolve())
        return {
            "PIPELINE_LOCAL_MODELS_ONLY": "1",
            "BLENDER_BIN": str(blender_bin or self.blender_bin),
            "MODEL_CACHE_ROOT": str(self.model_root),
            "QWEN35_PYTHON": str(self.qwen_python),
            "QWEN35_MODEL_PATH": str(self.qwen_model),
            "QWEN_PYTHON": owner_python,
            "MVINVERSE_PYTHON": owner_python,
            "MVINVERSE_REPOSITORY": str(
                MATERIAL_PIPELINE_ROOT / "third_party" / "mvinverse"
            ),
            "MVINVERSE_CHECKPOINT": str(self.mvinverse_model),
            "SIGLIP2_MODEL_PATH": str(self.siglip2_model),
            "DINOV2_MODEL_PATH": str(self.dinov2_model),
            "SAM3_REPOSITORY": str(self.sam3_repository),
            "SAM3_CHECKPOINT": str(self.sam3_checkpoint),
            "ENTITYSEG_PYTHON": str(self.entityseg_python),
            "ENTITYSEG_CROPFORMER_ROOT": str(self.entityseg_cropformer_root),
            "ENTITYSEG_CONFIG": str(self.entityseg_config),
            "ENTITYSEG_CHECKPOINT": str(self.entityseg_checkpoint),
            "NVIDIA_BASE_OBSERVATION_BANK": str(self.observation_bank),
            "VISUAL_RETRIEVAL_CACHE": str(self.retrieval_cache),
        }


def _enable_explicit_downloads() -> None:
    # Normal inference remains offline. Only this explicit setup command enables
    # network access, and the mainline switches back to local-only at startup.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "0"


def _check_free_space(path: Path, minimum_free_gb: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum_free_gb * GIB:
        raise ModelSetupError(
            f"model disk has {free / GIB:.1f} GiB free; "
            f"at least {minimum_free_gb} GiB is required"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified_file(
    *, url: str, destination: Path, expected_sha256: str
) -> Path:
    """Download one resumable file and accept it only after SHA-256 validation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256_file(destination) == expected_sha256:
        return destination
    partial = destination.with_name(f"{destination.name}.partial")
    if partial.is_file() and _sha256_file(partial) == expected_sha256:
        partial.replace(destination)
        return destination

    aria2 = shutil.which("aria2c")
    if aria2 and url.startswith(("https://", "http://")):
        completed = subprocess.run(
            [
                aria2,
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=1M",
                "--max-tries=5",
                "--retry-wait=2",
                "--summary-interval=5",
                f"--dir={destination.parent}",
                f"--out={partial.name}",
                url,
            ],
            check=False,
        )
        if completed.returncode == 0:
            if partial.is_file() and _sha256_file(partial) == expected_sha256:
                partial.replace(destination)
                return destination
            partial.unlink(missing_ok=True)
            raise ModelSetupError("parallel Blender download failed SHA-256 validation")

    start = partial.stat().st_size if partial.is_file() else 0
    for attempt in range(2):
        headers = {"User-Agent": "hunyuan-asset-pipeline-setup/1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and start and attempt == 0:
                partial.unlink(missing_ok=True)
                start = 0
                continue
            raise ModelSetupError(
                f"unable to download Blender: HTTP {exc.code}"
            ) from exc
        with response:
            append = start > 0 and getattr(response, "status", None) == 206
            with partial.open("ab" if append else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        break

    actual_sha256 = _sha256_file(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ModelSetupError(
            "Blender archive failed SHA-256 validation: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(destination)
    return destination


def _safe_extract_blender(archive: Path, destination: Path) -> None:
    staging = destination.parent / f".{destination.name}.extracting-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staging_root = staging.resolve()
    try:
        with tarfile.open(archive, mode="r:xz") as bundle:
            members = bundle.getmembers()
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ModelSetupError("Blender archive contains an unsafe path")
                target = staging.joinpath(*relative.parts).resolve()
                target.relative_to(staging_root)
                if member.issym():
                    (target.parent / member.linkname).resolve().relative_to(
                        staging_root
                    )
                elif member.islnk():
                    staging.joinpath(
                        *PurePosixPath(member.linkname).parts
                    ).resolve().relative_to(staging_root)
            bundle.extractall(staging)
        extracted = staging / BLENDER_DIRECTORY
        executable = extracted / "blender"
        if not executable.is_file():
            raise ModelSetupError("Blender archive did not contain its executable")
        if destination.exists():
            shutil.rmtree(destination)
        extracted.replace(destination)
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise ModelSetupError("unable to extract the verified Blender archive") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _blender_version(executable: Path) -> str | None:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return None
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^Blender\s+(\d+\.\d+\.\d+)", completed.stdout)
    return match.group(1) if completed.returncode == 0 and match else None


def _prepare_blender(
    paths: SetupPaths,
    *,
    url: str = BLENDER_URL,
    expected_sha256: str = BLENDER_SHA256,
) -> Path:
    """Provide the same pinned portable Blender used by the full Docker image."""

    candidates = [
        os.getenv("BLENDER_BIN") or "",
        shutil.which("blender") or "",
        paths.blender_bin,
        Path("/opt/blender/blender"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        executable = Path(candidate).expanduser().absolute()
        if _blender_version(executable) == BLENDER_VERSION:
            return executable
    archive = _download_verified_file(
        url=url,
        destination=paths.blender_archive,
        expected_sha256=expected_sha256,
    )
    paths.blender_home.parent.mkdir(parents=True, exist_ok=True)
    _safe_extract_blender(archive, paths.blender_home)
    if _blender_version(paths.blender_bin) != BLENDER_VERSION:
        raise ModelSetupError(
            f"installed Blender did not report the pinned version {BLENDER_VERSION}"
        )
    return paths.blender_bin


def _check_gated_access() -> None:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except ImportError as exc:
        raise ModelSetupError(
            "huggingface-hub is missing; install the project environment first"
        ) from exc

    gated_files = (
        ("facebook/sam3", "sam3.pt", "model"),
        ("qqlu1992/Adobe_EntitySeg", ENTITYSEG_FILENAME, "dataset"),
    )
    try:
        for repository, filename, repo_type in gated_files:
            url = hf_hub_url(repository, filename, repo_type=repo_type)
            get_hf_file_metadata(url, token=True, timeout=30)
    except Exception as exc:
        raise ModelSetupError(
            "Hugging Face access is not ready. Accept access for facebook/sam3 "
            "and qqlu1992/Adobe_EntitySeg, then run `hf auth login` or set "
            "HF_TOKEN."
        ) from exc


def _git(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _checkout_repository(
    *,
    repository: str,
    revision: str,
    destination: Path,
    sparse_path: str | None = None,
) -> None:
    git_dir = destination / ".git"
    if destination.exists() and not git_dir.is_dir():
        if any(destination.iterdir()):
            raise ModelSetupError(
                f"source destination exists but is not a Git checkout: {destination}"
            )
        destination.rmdir()
    if not git_dir.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(destination),
        )
    has_checkout = (git_dir / "index").is_file()
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current == revision and has_checkout:
        return
    if has_checkout:
        tracked_changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tracked_changes:
            raise ModelSetupError(f"source checkout has local changes: {destination}")
    if sparse_path:
        _git("sparse-checkout", "init", "--cone", cwd=destination)
        _git("sparse-checkout", "set", sparse_path, cwd=destination)
    _git("fetch", "--depth=1", "origin", revision, cwd=destination)
    _git("checkout", "--detach", "FETCH_HEAD", cwd=destination)


def _prepare_sources(paths: SetupPaths) -> None:
    _checkout_repository(
        repository="https://github.com/facebookresearch/sam3.git",
        revision=SAM3_SOURCE_REVISION,
        destination=paths.sam3_repository,
    )
    _checkout_repository(
        repository="https://github.com/qqlu/Entity.git",
        revision=ENTITY_SOURCE_REVISION,
        destination=paths.entity_repository,
        sparse_path="Entityv2/CropFormer",
    )
    _checkout_repository(
        repository="https://github.com/facebookresearch/detectron2.git",
        revision=DETECTRON2_REVISION,
        destination=paths.detectron2_repository,
    )


def _entityseg_smoke(paths: SetupPaths, *, quiet: bool) -> bool:
    smoke = """
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / 'demo_cropformer'), str(root)]
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import MultiScaleDeformableAttention
print('EntitySeg runtime ready')
"""
    completed = subprocess.run(
        [
            str(paths.entityseg_python),
            "-c",
            smoke,
            str(paths.entityseg_cropformer_root),
        ],
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    return completed.returncode == 0


def _cuda_build_environment() -> dict[str, str]:
    """Select an installed CUDA toolkit compatible with the active PyTorch."""

    try:
        import torch
    except ImportError as exc:
        raise ModelSetupError(
            "PyTorch is required before setting up EntitySeg"
        ) from exc
    torch_cuda = torch.version.cuda
    if not torch_cuda:
        raise ModelSetupError("the active PyTorch build has no CUDA support")
    required_major, required_minor = (int(part) for part in torch_cuda.split(".")[:2])

    candidates: list[Path] = []
    configured = os.getenv("CUDA_HOME")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(f"/usr/local/cuda-{required_major}.{required_minor}"),
            *sorted(Path("/usr/local").glob(f"cuda-{required_major}.*")),
        ]
    )
    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(Path(nvcc_on_path).resolve().parents[1])

    compatible: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not (candidate / "bin" / "nvcc").is_file():
            continue
        seen.add(candidate)
        completed = subprocess.run(
            [str(candidate / "bin" / "nvcc"), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(r"release\s+(\d+)\.(\d+)", completed.stdout)
        if not match or int(match.group(1)) != required_major:
            continue
        minor = int(match.group(2))
        compatible.append((abs(minor - required_minor), candidate))
    if not compatible:
        raise ModelSetupError(
            f"no CUDA {required_major}.x toolkit matches PyTorch CUDA {torch_cuda}"
        )
    cuda_home = min(compatible, key=lambda item: item[0])[1]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_HOME": str(cuda_home),
            "PATH": f"{cuda_home / 'bin'}{os.pathsep}{environment.get('PATH', '')}",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _prepare_entityseg_runtime(paths: SetupPaths) -> None:
    if paths.entityseg_python.is_file() and _entityseg_smoke(paths, quiet=True):
        return
    if not paths.entityseg_python.is_file():
        paths.entityseg_runtime.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                str(paths.entityseg_runtime),
            ],
            check=True,
        )
    python = str(paths.entityseg_python)
    isolated_environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--requirement",
            str(MATERIAL_PIPELINE_ROOT / "requirements" / "entityseg.txt"),
        ],
        check=True,
        env=isolated_environment,
    )
    build_environment = _cuda_build_environment()
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--editable",
            str(paths.detectron2_repository),
        ],
        check=True,
        env=build_environment,
    )
    ops_root = (
        paths.entityseg_cropformer_root
        / "mask2former"
        / "modeling"
        / "pixel_decoder"
        / "ops"
    )
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--force-reinstall",
            str(ops_root),
        ],
        check=True,
        env={**build_environment, "FORCE_CUDA": "1"},
    )
    if not _entityseg_smoke(paths, quiet=False):
        raise ModelSetupError("EntitySeg runtime smoke test failed")


def _run_qwen_setup(paths: SetupPaths) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "ASSET_MODEL_VOLUME": str(paths.model_root),
            "QWEN35_RUNTIME_ROOT": str(paths.qwen_runtime),
            "QWEN35_ENV_DIR": str(paths.qwen_python.parents[1]),
            "QWEN35_MODEL_DIR": str(paths.qwen_model),
            "SIGLIP2_MODEL_DIR": str(paths.siglip2_model),
        }
    )
    if os.getenv("CONDA_EXE"):
        environment["CONDA_BIN"] = os.environ["CONDA_EXE"]
    subprocess.run(["bash", str(QWEN_SETUP)], check=True, env=environment)


def _download_remaining_models(paths: SetupPaths, max_workers: int) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        repo_id="maddog241/mvinverse",
        revision=MVINVERSE_REVISION,
        allow_patterns=("config.json", "model.safetensors"),
        local_dir=paths.mvinverse_model,
        max_workers=max_workers,
        token=True,
    )
    snapshot_download(
        repo_id="facebook/dinov2-with-registers-large",
        revision=DINOV2_REVISION,
        allow_patterns=("config.json", "model.safetensors", "preprocessor_config.json"),
        local_dir=paths.dinov2_model,
        max_workers=max_workers,
        token=True,
    )
    hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        revision=SAM3_REVISION,
        local_dir=paths.sam3_checkpoint.parent,
        token=True,
    )
    hf_hub_download(
        repo_id="qqlu1992/Adobe_EntitySeg",
        repo_type="dataset",
        filename=ENTITYSEG_FILENAME,
        revision=ENTITYSEG_REVISION,
        local_dir=paths.entityseg_checkpoint.parents[3],
        token=True,
    )


def _verify_downloads(paths: SetupPaths, *, blender_bin: Path) -> None:
    required = (
        blender_bin,
        paths.qwen_python,
        paths.qwen_model / "config.json",
        paths.siglip2_model / "config.json",
        paths.mvinverse_model / "config.json",
        paths.mvinverse_model / "model.safetensors",
        paths.dinov2_model / "config.json",
        paths.dinov2_model / "model.safetensors",
        paths.sam3_repository / "sam3" / "model_builder.py",
        paths.sam3_repository / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        paths.sam3_checkpoint,
        paths.entityseg_python,
        paths.entityseg_config,
        paths.detectron2_repository / "detectron2" / "__init__.py",
        paths.entityseg_checkpoint,
        paths.observation_bank / "scope_report.json",
        paths.observation_bank / "render_manifest.json",
        paths.observation_bank / "appearance_profiles.json",
        paths.observation_bank / "visual_embeddings.npz",
        paths.observation_bank / "index_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ModelSetupError("download is incomplete: " + ", ".join(missing))


def _format_env_value(value: str) -> str:
    return value if _PLAIN_ENV_VALUE.fullmatch(value) else json.dumps(value)


def update_environment_file(path: Path, values: dict[str, str]) -> None:
    path = path.expanduser().resolve()
    template = PROJECT_ROOT / ".env.example"
    if path.exists():
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    else:
        original = template.read_text(encoding="utf-8")
        mode = 0o600

    output: list[str] = []
    written: set[str] = set()
    for line in original.splitlines():
        match = _ENV_ASSIGNMENT.match(line)
        name = match.group(1) if match else None
        if name not in values:
            output.append(line)
        elif name not in written:
            output.append(f"{name}={_format_env_value(values[name])}")
            written.add(name)

    missing = [name for name in values if name not in written]
    if missing:
        if output and output[-1]:
            output.append("")
        output.append("# Automatic local runtime paths")
        output.extend(f"{name}={_format_env_value(values[name])}" for name in missing)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.chmod(temporary, mode)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    default_root = os.getenv("ASSET_MODEL_VOLUME") or os.getenv("MODEL_CACHE_ROOT")
    if not default_root:
        default_root = str(Path.home() / ".cache" / "hunyuan_asset_pipeline")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(default_root))
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--minimum-free-gb", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> SetupPaths:
    if args.max_workers < 1:
        raise ModelSetupError("--max-workers must be at least 1")
    if args.minimum_free_gb < 1:
        raise ModelSetupError("--minimum-free-gb must be at least 1")
    paths = SetupPaths.from_root(args.model_root)
    if args.dry_run:
        print(f"Model root: {paths.model_root}")
        print(f"Blender: {paths.blender_bin}")
        print(f"Source root: {paths.sam3_repository.parent}")
        print(f"Observation bank: {paths.observation_bank}")
        print(f"Environment file: {args.env_file.expanduser().resolve()}")
        return paths

    _enable_explicit_downloads()
    print("[1/6] Checking access and disk space", flush=True)
    _check_gated_access()
    _check_free_space(paths.model_root, args.minimum_free_gb)
    print(f"[2/6] Preparing Blender {BLENDER_VERSION}", flush=True)
    blender_bin = _prepare_blender(paths)
    print("[3/6] Preparing SAM3 and EntitySeg", flush=True)
    _prepare_sources(paths)
    _prepare_entityseg_runtime(paths)
    print("[4/6] Preparing Qwen3.5 and SigLIP2", flush=True)
    _run_qwen_setup(paths)
    print("[5/6] Downloading MVInverse, DINOv2, SAM3 and EntitySeg", flush=True)
    _download_remaining_models(paths, args.max_workers)
    _verify_downloads(paths, blender_bin=blender_bin)
    print("[6/6] Updating .env", flush=True)
    update_environment_file(args.env_file, paths.environment(blender_bin=blender_bin))
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run(args)
    except (ModelSetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"Runtime and models ready: {paths.model_root}")
        print(f"Updated: {args.env_file.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
