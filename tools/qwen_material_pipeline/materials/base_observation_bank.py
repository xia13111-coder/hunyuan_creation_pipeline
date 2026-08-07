#!/usr/bin/env python3
"""Build a strict NVIDIA Base-only rendered material observation bank.

The bank is deliberately separate from per-asset inference.  It scans exactly
``NVIDIA/Materials/Base``, renders every exported MDL on one fixed observation
rig in a single Isaac Sim session, then seals SigLIP2, DINOv2, color, texture,
and authored OmniPBR descriptors into a content-addressed index.

No path below ``vMaterials_2`` is accepted at any stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

from qwen_material_pipeline.materials.catalog import MaterialCatalog


BANK_SCHEMA_VERSION = "nvidia-base-material-observation-bank/v1"
RENDER_SCHEMA_VERSION = "nvidia-base-material-observation-render/v1"
INDEX_SCHEMA_VERSION = "nvidia-base-material-observation-index/v1"
SCOPE_SCHEMA_VERSION = "nvidia-base-material-scope/v1"
OBSERVATION_PROFILES = ("neutral_iso", "grazing_front", "top_soft")
OBSERVATION_RIG_ID = "uv-sphere+cylinder+plate+industrial-part/v2"
DEFAULT_BASE_ROOT = Path(
    os.environ.get(
        "VISUAL_MATERIAL_ROOT",
        str(
            Path.home()
            / "isaacsim_assets"
            / "Assets"
            / "Isaac"
            / "4.5"
            / "NVIDIA"
            / "Materials"
            / "Base"
        ),
    )
)
DEFAULT_ISAAC_PYTHON = Path(
    os.environ.get(
        "ISAAC_PYTHON",
        str(Path.home() / "isaacsim500" / "python.sh"),
    )
)
DEFAULT_SIGLIP2_MODEL = Path(
    os.environ.get(
        "SIGLIP2_MODEL",
        str(
            Path.home()
            / ".cache"
            / "qwen_material_pipeline"
            / "models"
            / "siglip2-base-patch16-224"
        ),
    )
)
DEFAULT_DINOV2_MODEL = Path(
    os.environ.get(
        "DINOV2_MODEL",
        str(
            Path.home()
            / ".cache"
            / "qwen_material_pipeline"
            / "models"
            / "dinov2-with-registers-large"
        ),
    )
)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_OMNIPBR_FLOAT_FIELDS = (
    "reflection_roughness_constant",
    "reflection_roughness_texture_influence",
    "metallic_constant",
    "metallic_texture_influence",
    "specular_level",
    "bump_factor",
    "albedo_desaturation",
    "albedo_add",
    "albedo_brightness",
)
_OMNIPBR_BOOL_FIELDS = (
    "enable_ORM_texture",
    "enable_emission",
    "project_uvw",
    "world_or_object",
)
_OMNIPBR_TEXTURE_FIELDS = (
    "diffuse_texture",
    "reflectionroughness_texture",
    "metallic_texture",
    "ORM_texture",
    "normalmap_texture",
)


class BaseObservationBankError(ValueError):
    """Raised when the Base-only bank contract cannot be satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaseObservationBankError(
            f"unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BaseObservationBankError(f"{label} must be a JSON object: {path}")
    return value


def resolve_base_root(value: Path | str) -> Path:
    """Resolve a path to the exact NVIDIA ``Base`` collection or fail closed."""

    raw = Path(value).expanduser()
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise BaseObservationBankError(
            f"material root does not exist: {raw}"
        ) from exc
    if not resolved.is_dir():
        raise BaseObservationBankError(
            f"material root must be a directory: {resolved}"
        )
    if resolved.name == "vMaterials_2":
        raise BaseObservationBankError(
            f"vMaterials_2 is forbidden by the Base-only contract: {resolved}"
        )
    candidate = resolved if resolved.name == "Base" else resolved / "Base"
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise BaseObservationBankError(
            f"NVIDIA Base collection was not found below {resolved}"
        ) from exc
    if not candidate.is_dir() or candidate.name != "Base":
        raise BaseObservationBankError(
            f"resolved material root is not the Base collection: {candidate}"
        )
    if any(part.casefold() == "vmaterials_2" for part in candidate.parts):
        raise BaseObservationBankError(
            f"resolved Base root crosses vMaterials_2: {candidate}"
        )
    return candidate


def _assert_base_catalog(catalog: MaterialCatalog) -> None:
    if catalog.root.name != "Base":
        raise BaseObservationBankError(
            f"catalog root is not NVIDIA Base: {catalog.root}"
        )
    forbidden: list[str] = []
    for record in catalog.materials:
        fields = (record.material_id, record.mdl_path, record.thumbnail_path or "")
        if any("vmaterials_2" in field.casefold() for field in fields):
            forbidden.append(record.material_id)
    if forbidden:
        raise BaseObservationBankError(
            "Base-only catalog contains forbidden vMaterials_2 entries: "
            + ", ".join(forbidden[:5])
        )


def initialize_bank(
    *,
    material_root: Path,
    output_dir: Path,
) -> tuple[MaterialCatalog, dict[str, Any]]:
    """Build the exact catalog, allowlist, and auditable scope report."""

    root = resolve_base_root(material_root)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    catalog = MaterialCatalog.scan(root)
    _assert_base_catalog(catalog)
    if not catalog.materials:
        raise BaseObservationBankError(f"Base catalog contains no exports: {root}")

    catalog_path = catalog.save(destination / "catalog.json")
    allowlist_path = _write_json(
        destination / "allowlist.json", catalog.to_full_allowlist_dict()
    )
    source_records = []
    for path in sorted(root.rglob("*.mdl")):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise BaseObservationBankError(
                f"MDL module escaped Base through a link: {path}"
            ) from exc
        source_records.append(
            {
                "path": relative,
                "sha256": _sha256_file(resolved),
            }
        )
    report = {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "scope": "nvidia_base",
        "resolved_material_root": str(root),
        "collection_name": root.name,
        "catalog": catalog_path.name,
        "catalog_sha256": _sha256_file(catalog_path),
        "allowlist": allowlist_path.name,
        "allowlist_sha256": _sha256_file(allowlist_path),
        "material_count": len(catalog.materials),
        "mdl_module_count": len(source_records),
        "mdl_sources_sha256": _canonical_sha256(source_records),
        "forbidden_vmaterials_2_count": 0,
        "exact_cover": True,
    }
    _write_json(destination / "scope_report.json", report)
    return catalog, report


def _material_slug(material_id: str, sub_identifier: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sub_identifier).strip("_.")
    readable = readable[:48] or "material"
    digest = hashlib.sha256(material_id.encode("utf-8")).hexdigest()[:16]
    return f"{readable}_{digest}"


def _relative_file(
    *,
    bank_dir: Path,
    value: Any,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(value, str) or not value:
        raise BaseObservationBankError(f"{label} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BaseObservationBankError(f"{label} is unsafe: {value!r}")
    try:
        path = (bank_dir / relative).resolve(strict=True)
        path.relative_to(bank_dir)
    except (OSError, ValueError) as exc:
        raise BaseObservationBankError(
            f"{label} is missing or outside the bank: {value!r}"
        ) from exc
    if not path.is_file():
        raise BaseObservationBankError(f"{label} is not a file: {path}")
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise BaseObservationBankError(f"{label} failed SHA-256 validation: {path}")
    return path


def _parse_omnipbr_defaults(source: str) -> dict[str, Any]:
    """Extract immutable authored values from common Base OmniPBR modules."""

    result: dict[str, Any] = {}
    color_match = re.search(
        r"\bdiffuse_color_constant\s*:\s*color\s*\(([^)]*)\)",
        source,
    )
    if color_match:
        values = [
            float(item)
            for item in _FLOAT_RE.findall(color_match.group(1))
        ]
        if len(values) >= 3:
            result["diffuse_color_constant"] = values[:3]
    for name in _OMNIPBR_FLOAT_FIELDS:
        match = re.search(
            rf"\b{re.escape(name)}\s*:\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)f?\b",
            source,
        )
        if match:
            result[name] = float(match.group(1))
    for name in _OMNIPBR_BOOL_FIELDS:
        match = re.search(
            rf"\b{re.escape(name)}\s*:\s*(true|false)\b",
            source,
        )
        if match:
            result[name] = match.group(1) == "true"
    for name in _OMNIPBR_TEXTURE_FIELDS:
        match = re.search(
            rf"\b{re.escape(name)}\s*:\s*texture_2d\s*\(\s*"
            r'(?:"([^"]+)")?',
            source,
        )
        if match:
            result[name] = match.group(1)
    result["source_kind"] = (
        "omnipbr_authored_defaults" if result else "unparsed_mdl"
    )
    return result


def _define_box(stage: Any, path: str, position: tuple[float, float, float],
                scale: tuple[float, float, float]) -> Any:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetScale(Gf.Vec3f(*scale))
    return cube.GetPrim()


def _define_uv_box(
    stage: Any,
    path: str,
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> Any:
    """Define a Cube with deterministic per-face UVs for textured MDLs."""

    from pxr import Gf, Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetScale(Gf.Vec3f(*scale))
    face_uvs = [
        Gf.Vec2f(u, v)
        for _ in range(6)
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ]
    UsdGeom.PrimvarsAPI(cube).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(face_uvs)
    return cube.GetPrim()


def _define_uv_sphere(stage: Any, path: str, center: tuple[float, float, float],
                      radius: float = 0.72, rings: int = 20,
                      segments: int = 40) -> Any:
    from pxr import Gf, Sdf, UsdGeom

    points: list[Any] = []
    st_values: list[Any] = []
    for ring in range(rings + 1):
        v = ring / rings
        phi = math.pi * v
        for segment in range(segments + 1):
            u = segment / segments
            theta = 2.0 * math.pi * u
            points.append(
                Gf.Vec3f(
                    center[0] + radius * math.sin(phi) * math.cos(theta),
                    center[1] + radius * math.sin(phi) * math.sin(theta),
                    center[2] + radius * math.cos(phi),
                )
            )
            st_values.append(Gf.Vec2f(u, 1.0 - v))
    counts: list[int] = []
    indices: list[int] = []
    st_indices: list[int] = []
    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * stride + segment
            b = a + 1
            c = a + stride + 1
            d = a + stride
            counts.append(4)
            indices.extend((a, b, c, d))
            st_indices.extend((a, b, c, d))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(st_values)
    st.SetIndices(st_indices)
    return mesh.GetPrim()


def _define_uv_cylinder(
    stage: Any,
    path: str,
    center: tuple[float, float, float],
    *,
    radius: float = 0.55,
    height: float = 1.55,
    segments: int = 40,
) -> Any:
    from pxr import Gf, Sdf, UsdGeom

    points: list[Any] = []
    st_values: list[Any] = []
    bottom = center[2] - height * 0.5
    top = center[2] + height * 0.5
    for z, v in ((bottom, 0.0), (top, 1.0)):
        for segment in range(segments + 1):
            u = segment / segments
            theta = 2.0 * math.pi * u
            points.append(
                Gf.Vec3f(
                    center[0] + radius * math.cos(theta),
                    center[1] + radius * math.sin(theta),
                    z,
                )
            )
            st_values.append(Gf.Vec2f(u, v))
    counts: list[int] = []
    indices: list[int] = []
    st_indices: list[int] = []
    stride = segments + 1
    for segment in range(segments):
        a, b = segment, segment + 1
        c, d = stride + segment + 1, stride + segment
        counts.append(4)
        indices.extend((a, b, c, d))
        st_indices.extend((a, b, c, d))
    # The observation bank focuses on the UV-bearing side response.  Thin
    # neutral top/bottom gaps avoid degenerate cap UVs dominating retrieval.
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(st_values)
    st.SetIndices(st_indices)
    return mesh.GetPrim()


def _define_preview_material(
    stage: Any, path: str, color: tuple[float, float, float], roughness: float
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _define_mdl_material(
    stage: Any,
    *,
    path: str,
    mdl_path: Path,
    sub_identifier: str,
) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl_path)), "mdl")
    shader.SetSourceAssetSubIdentifier(sub_identifier, "mdl")
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    output.SetRenderType("material")
    connectable = shader.ConnectableAPI()
    material.CreateSurfaceOutput("mdl").ConnectToSource(connectable, "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(connectable, "out")
    return material


def _set_observation_lighting(stage: Any, profile: str) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Get(stage, "/World/Lights/Dome")
    distant = UsdLux.DistantLight.Get(stage, "/World/Lights/Distant")
    key = UsdLux.SphereLight.Get(stage, "/World/Lights/Key")
    settings = {
        "neutral_iso": (420.0, 900.0, 18000.0, (4.0, -4.0, 5.5)),
        "grazing_front": (120.0, 2100.0, 36000.0, (-4.5, -3.0, 1.8)),
        "top_soft": (300.0, 500.0, 26000.0, (0.0, -1.0, 6.5)),
    }
    dome_intensity, distant_intensity, key_intensity, key_position = settings[
        profile
    ]
    dome.CreateIntensityAttr(dome_intensity)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    distant.CreateIntensityAttr(distant_intensity)
    distant.CreateAngleAttr(6.0)
    key.CreateIntensityAttr(key_intensity)
    key.CreateRadiusAttr(1.0)
    UsdGeom.XformCommonAPI(key).SetTranslate(Gf.Vec3d(*key_position))


def _mask_from_segmentation(segmentation: Mapping[str, Any]) -> Any:
    import numpy as np

    raw = np.asarray(segmentation.get("data"))
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    if raw.ndim != 2:
        raise BaseObservationBankError(
            f"invalid semantic segmentation shape: {raw.shape}"
        )
    mask = np.zeros(raw.shape, dtype=bool)
    labels_by_id = segmentation.get("info", {}).get("idToLabels", {})
    for raw_id, labels in labels_by_id.items():
        if "bank_sample" not in str(labels):
            continue
        try:
            numeric_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        mask |= raw == numeric_id
    if int(mask.sum()) < 128:
        raise BaseObservationBankError(
            "standard observation rig produced an empty semantic mask"
        )
    return mask


def render_bank(
    *,
    material_root: Path,
    output_dir: Path,
    resolution: int,
    rt_subframes: int,
    limit: int | None,
) -> dict[str, Any]:
    """Render Base materials.  This function must run inside Isaac Sim Python."""

    import numpy as np
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdShade
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.semantics import add_update_semantics

    if resolution < 256:
        raise BaseObservationBankError("observation resolution must be at least 256")
    if rt_subframes < 1:
        raise BaseObservationBankError("rt_subframes must be positive")
    bank_dir = output_dir.expanduser().resolve()
    catalog, scope = initialize_bank(
        material_root=material_root, output_dir=bank_dir
    )
    catalog_sha256 = scope["catalog_sha256"]
    manifest_path = bank_dir / "render_manifest.json"
    records_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous = _read_json(manifest_path, "render manifest")
        if (
            previous.get("schema_version") != RENDER_SCHEMA_VERSION
            or previous.get("catalog_sha256") != catalog_sha256
            or previous.get("profiles") != list(OBSERVATION_PROFILES)
            or previous.get("rig") != OBSERVATION_RIG_ID
            or previous.get("resolution") != resolution
        ):
            raise BaseObservationBankError(
                "existing render manifest does not match this catalog or rig"
            )
        for record in previous.get("materials", []):
            if isinstance(record, dict) and isinstance(
                record.get("material_id"), str
            ):
                try:
                    for observation in record.get("observations", []):
                        _relative_file(
                            bank_dir=bank_dir,
                            value=observation.get("image"),
                            label="resumed observation",
                            expected_sha256=observation.get("sha256"),
                        )
                    if "observation_source" not in record:
                        record = {
                            **record,
                            "observation_source": "standard_rtx_rig",
                        }
                    records_by_id[record["material_id"]] = record
                except BaseObservationBankError:
                    pass

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise BaseObservationBankError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Lights")

    ground = _define_box(stage, "/World/Ground", (0.0, 0.75, -0.09), (7.0, 5.5, 0.12))
    backdrop = _define_box(
        stage, "/World/Backdrop", (0.0, 2.1, 2.0), (7.0, 0.12, 4.2)
    )
    neutral = _define_preview_material(
        stage, "/World/Looks/Neutral", (0.18, 0.18, 0.18), 0.48
    )
    for prim in (ground, backdrop):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(neutral)

    proxies = [
        _define_uv_sphere(stage, "/World/Samples/Sphere", (-2.15, 0.0, 0.92)),
        _define_uv_cylinder(stage, "/World/Samples/Cylinder", (-0.70, 0.0, 0.82)),
        _define_uv_box(
            stage, "/World/Samples/Plate", (0.72, 0.0, 0.72), (1.05, 0.24, 1.35)
        ),
        _define_uv_box(
            stage, "/World/Samples/IndustrialPart", (2.05, 0.0, 0.80), (1.10, 0.72, 1.45)
        ),
    ]
    for prim in proxies:
        add_update_semantics(prim, "bank_sample", type_label="bank_sample")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateTextureFileAttr("")
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Distant")
    UsdGeom.XformCommonAPI(distant).SetRotate(
        Gf.Vec3f(35.0, -30.0, 25.0),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    UsdLux.SphereLight.Define(stage, "/World/Lights/Key")

    camera_specs = {
        "neutral_iso": ((5.8, -8.4, 4.1), (0.0, 0.0, 0.85)),
        "grazing_front": ((6.8, -5.8, 2.1), (0.0, 0.0, 0.80)),
        "top_soft": ((3.8, -4.5, 8.0), (0.0, 0.0, 0.55)),
    }
    captures: dict[str, tuple[Any, Any]] = {}
    for profile, (position, look_at) in camera_specs.items():
        camera = rep.create.camera(
            name=f"BaseBank_{profile}",
            position=position,
            look_at=look_at,
            focal_length=52.0,
            clipping_range=(0.01, 100.0),
        )
        product = rep.create.render_product(
            camera, (resolution, resolution), name=f"BaseBank_{profile}"
        )
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        segmentation = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"semanticTypes": ["bank_sample"], "colorize": False},
        )
        segmentation.attach(product)
        captures[profile] = (rgb, segmentation)

    for _ in range(4):
        omni.kit.app.get_app().update()

    selected = list(catalog.materials)
    if limit is not None:
        selected = selected[:limit]
    total = len(selected)
    for index, record in enumerate(selected, start=1):
        if record.material_id in records_by_id:
            print(
                f"[BASE-BANK] {index}/{total} reuse {record.material_id}",
                flush=True,
            )
            continue
        mdl_path, sub_identifier = catalog.resolve_material(record.material_id)
        slug = _material_slug(record.material_id, sub_identifier)
        print(
            f"[BASE-BANK] {index}/{total} begin {record.material_id}",
            flush=True,
        )
        observations: list[dict[str, Any]] = []
        mdl_source = mdl_path.read_text(encoding="utf-8", errors="replace")
        observation_source = "standard_rtx_rig"
        if "OmniPBR_Opacity" in mdl_source:
            # Isaac Sim 5.0's RTX/Replicator stack terminates Kit at native
            # level when this Base cutout module is bound to the multi-shape
            # rig.  NVIDIA ships a renderer-produced material preview for the
            # same immutable MDL.  Preserve that evidence instead of silently
            # excluding the material or replacing its opacity model.
            if record.thumbnail_path is None:
                raise BaseObservationBankError(
                    "opacity MDL has no NVIDIA official preview: "
                    f"{record.material_id}"
                )
            thumbnail = (catalog.root / record.thumbnail_path).resolve(strict=True)
            try:
                thumbnail.relative_to(catalog.root)
            except ValueError as exc:
                raise BaseObservationBankError(
                    f"opacity preview escaped Base: {thumbnail}"
                ) from exc
            with Image.open(thumbnail) as opened:
                preview = opened.convert("RGB").resize(
                    (resolution, resolution), Image.Resampling.LANCZOS
                )
            preview_pixels = np.asarray(preview, dtype=np.uint8)
            border = np.concatenate(
                (
                    preview_pixels[:8].reshape(-1, 3),
                    preview_pixels[-8:].reshape(-1, 3),
                    preview_pixels[:, :8].reshape(-1, 3),
                    preview_pixels[:, -8:].reshape(-1, 3),
                ),
                axis=0,
            )
            background = np.median(border.astype(np.float32), axis=0)
            foreground = np.linalg.norm(
                preview_pixels.astype(np.float32) - background[None, None, :],
                axis=2,
            ) >= 18.0
            if int(foreground.sum()) < 128:
                foreground = np.ones(
                    preview_pixels.shape[:2], dtype=bool
                )
            material_dir = bank_dir / "renders" / slug
            material_dir.mkdir(parents=True, exist_ok=True)
            for profile in OBSERVATION_PROFILES:
                image_path = material_dir / f"{profile}.png"
                preview.save(image_path)
                mask_path = material_dir / f"{profile}.mask.png"
                Image.fromarray(
                    foreground.astype(np.uint8) * 255, mode="L"
                ).save(mask_path)
                observations.append(
                    {
                        "profile": profile,
                        "image": image_path.relative_to(bank_dir).as_posix(),
                        "sha256": _sha256_file(image_path),
                        "mask": mask_path.relative_to(bank_dir).as_posix(),
                        "mask_sha256": _sha256_file(mask_path),
                        "foreground_pixels": int(foreground.sum()),
                    }
                )
            preview.close()
            observation_source = "nvidia_official_preview_opacity_safe"
            print(
                f"[BASE-BANK] {index}/{total} opacity-safe official preview "
                f"{record.material_id}",
                flush=True,
            )
        else:
            material = _define_mdl_material(
                stage,
                path=f"/World/Looks/M_{slug}",
                mdl_path=mdl_path,
                sub_identifier=sub_identifier,
            )
            for prim in proxies:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            for _ in range(3):
                omni.kit.app.get_app().update()
            for profile in OBSERVATION_PROFILES:
                print(
                    f"[BASE-BANK] {index}/{total} capture "
                    f"{record.material_id} profile={profile}",
                    flush=True,
                )
                _set_observation_lighting(stage, profile)
                for _ in range(2):
                    omni.kit.app.get_app().update()
                for _ in range(2):
                    rep.orchestrator.step(
                        rt_subframes=rt_subframes, delta_time=0.0
                    )
                rgb_annotator, segmentation_annotator = captures[profile]
                pixels = np.asarray(rgb_annotator.get_data())
                if pixels.ndim != 3 or pixels.shape[2] < 3:
                    raise BaseObservationBankError(
                        f"invalid RGB render for {record.material_id}: "
                        f"{pixels.shape}"
                    )
                mask = _mask_from_segmentation(
                    segmentation_annotator.get_data()
                )
                rgb_pixels = pixels[:, :, :3].astype(np.uint8)
                material_dir = bank_dir / "renders" / slug
                material_dir.mkdir(parents=True, exist_ok=True)
                image_path = material_dir / f"{profile}.png"
                Image.fromarray(rgb_pixels, mode="RGB").save(image_path)
                mask_dir = bank_dir / "masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{profile}.png"
                current_mask = (mask.astype(np.uint8) * 255)
                if mask_path.is_file():
                    with Image.open(mask_path) as opened:
                        previous_mask = np.asarray(opened.convert("L"))
                    if (
                        previous_mask.shape != current_mask.shape
                        or not np.array_equal(
                            previous_mask >= 128, current_mask >= 128
                        )
                    ):
                        raise BaseObservationBankError(
                            "observation silhouette changed for profile "
                            f"{profile}"
                        )
                else:
                    Image.fromarray(current_mask, mode="L").save(mask_path)
                observations.append(
                    {
                        "profile": profile,
                        "image": image_path.relative_to(bank_dir).as_posix(),
                        "sha256": _sha256_file(image_path),
                        "mask": mask_path.relative_to(bank_dir).as_posix(),
                        "mask_sha256": _sha256_file(mask_path),
                        "foreground_pixels": int(mask.sum()),
                    }
                )
        records_by_id[record.material_id] = {
            "material_id": record.material_id,
            "mdl_path": record.mdl_path,
            "sub_identifier": record.sub_identifier,
            "slug": slug,
            "observation_source": observation_source,
            "observations": observations,
        }
        ordered_records = [
            records_by_id[item.material_id]
            for item in catalog.materials
            if item.material_id in records_by_id
        ]
        _write_json(
            manifest_path,
            {
                "schema_version": RENDER_SCHEMA_VERSION,
                "bank_schema_version": BANK_SCHEMA_VERSION,
                "catalog_sha256": catalog_sha256,
                "scope": "nvidia_base",
                "profiles": list(OBSERVATION_PROFILES),
                "rig": OBSERVATION_RIG_ID,
                "resolution": resolution,
                "rt_subframes": rt_subframes,
                "material_count": len(catalog.materials),
                "rendered_material_count": len(ordered_records),
                "complete": len(ordered_records) == len(catalog.materials),
                "materials": ordered_records,
            },
        )
        print(
            f"[BASE-BANK] {index}/{total} rendered {record.material_id}",
            flush=True,
        )
    return _read_json(manifest_path, "render manifest")


def _appearance_statistics(image: Any, mask: Any) -> dict[str, Any]:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    valid = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    pixels = rgb[valid]
    if len(pixels) < 128:
        raise BaseObservationBankError("observation contains too few foreground pixels")
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = np.divide(
        maximum - minimum,
        np.maximum(maximum, 1e-6),
    )
    luminance = (
        0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    )
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    gradient_x = np.abs(np.diff(gray, axis=1))
    gradient_y = np.abs(np.diff(gray, axis=0))
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    texture_energy = float(
        0.5
        * (
            gradient_x[valid_x].mean()
            + gradient_y[valid_y].mean()
        )
    )
    return {
        "foreground_pixels": int(len(pixels)),
        "mean_rgb": [round(float(value), 6) for value in pixels.mean(axis=0)],
        "median_rgb": [
            round(float(value), 6) for value in np.median(pixels, axis=0)
        ],
        "std_rgb": [round(float(value), 6) for value in pixels.std(axis=0)],
        "luminance_p10_p50_p90": [
            round(float(value), 6)
            for value in np.quantile(luminance, (0.10, 0.50, 0.90))
        ],
        "mean_saturation": round(float(saturation.mean()), 6),
        "highlight_fraction": round(float((luminance >= 0.90).mean()), 6),
        "dark_fraction": round(float((luminance <= 0.10).mean()), 6),
        "texture_gradient_energy": round(texture_energy, 6),
    }


def build_index(
    *,
    material_root: Path,
    output_dir: Path,
    siglip2_model: Path,
    dinov2_model: Path,
    device: str,
    batch_size: int,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Encode all rendered observations with local SigLIP2 and DINOv2."""

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    from qwen_material_pipeline.retrieval.visual_materials import (
        _dino_tokens,
        _masked_square,
        _model_fingerprint,
        _normalize_rows,
        _siglip_image_embeddings,
        _verified_siglip2_model_identity,
    )

    bank_dir = output_dir.expanduser().resolve(strict=True)
    root = resolve_base_root(material_root)
    catalog = MaterialCatalog.load(
        bank_dir / "catalog.json", material_root=root
    )
    _assert_base_catalog(catalog)
    scope = _read_json(bank_dir / "scope_report.json", "scope report")
    render_manifest = _read_json(
        bank_dir / "render_manifest.json", "render manifest"
    )
    if render_manifest.get("schema_version") != RENDER_SCHEMA_VERSION:
        raise BaseObservationBankError("unsupported render manifest schema")
    if render_manifest.get("catalog_sha256") != scope.get("catalog_sha256"):
        raise BaseObservationBankError("render manifest catalog identity changed")
    if not render_manifest.get("complete") and not allow_incomplete:
        raise BaseObservationBankError(
            "render bank is incomplete; finish all Base materials before indexing"
        )
    record_by_id = {
        record["material_id"]: record
        for record in render_manifest.get("materials", [])
        if isinstance(record, dict) and isinstance(record.get("material_id"), str)
    }
    records = [
        record_by_id[item.material_id]
        for item in catalog.materials
        if item.material_id in record_by_id
    ]
    if not records:
        raise BaseObservationBankError("render manifest contains no materials")

    prepared: list[tuple[Any, Any]] = []
    profiles: list[dict[str, Any]] = []
    for record in records:
        observations = record.get("observations")
        if not isinstance(observations, list) or len(observations) != len(
            OBSERVATION_PROFILES
        ):
            raise BaseObservationBankError(
                f"incomplete observations for {record['material_id']}"
            )
        stats_by_profile: dict[str, Any] = {}
        observation_audit = []
        for observation in observations:
            image_path = _relative_file(
                bank_dir=bank_dir,
                value=observation.get("image"),
                label="observation image",
                expected_sha256=observation.get("sha256"),
            )
            mask_path = _relative_file(
                bank_dir=bank_dir,
                value=observation.get("mask"),
                label="observation mask",
                expected_sha256=observation.get("mask_sha256"),
            )
            canvas, mask = _masked_square(image_path, mask_path, size=224)
            prepared.append((canvas, mask))
            with Image.open(image_path) as opened_image, Image.open(
                mask_path
            ) as opened_mask:
                stats_by_profile[observation["profile"]] = _appearance_statistics(
                    opened_image, opened_mask
                )
            observation_audit.append(
                {
                    "profile": observation["profile"],
                    "image": observation["image"],
                    "sha256": observation["sha256"],
                    "mask": observation["mask"],
                    "mask_sha256": observation["mask_sha256"],
                }
            )
        mdl_path, _ = catalog.resolve_material(record["material_id"])
        source = mdl_path.read_text(encoding="utf-8", errors="replace")
        profiles.append(
            {
                "material_id": record["material_id"],
                "observation_source": record.get(
                    "observation_source", "legacy_unspecified"
                ),
                "observations": observation_audit,
                "appearance": stats_by_profile,
                "authored_mdl": _parse_omnipbr_defaults(source),
            }
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise BaseObservationBankError("CUDA indexing requested but unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    siglip2_model = siglip2_model.expanduser().resolve(strict=True)
    dinov2_model = dinov2_model.expanduser().resolve(strict=True)
    siglip_identity = _verified_siglip2_model_identity(siglip2_model)
    siglip_processor = AutoProcessor.from_pretrained(
        siglip2_model, local_files_only=True, trust_remote_code=False
    )
    siglip_model = AutoModel.from_pretrained(
        siglip2_model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    siglip_model.eval()
    siglip_views = _siglip_image_embeddings(
        siglip_model,
        siglip_processor,
        [item[0] for item in prepared],
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    views_per_material = len(OBSERVATION_PROFILES)
    siglip_embeddings = _normalize_rows(
        siglip_views.reshape(
            len(records), views_per_material, siglip_views.shape[-1]
        ).mean(axis=1)
    )
    del siglip_model
    del siglip_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dino_identity = _model_fingerprint(dinov2_model)
    dino_processor = AutoImageProcessor.from_pretrained(
        dinov2_model,
        local_files_only=True,
        trust_remote_code=False,
        backend="pil",
    )
    mean = list(getattr(dino_processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(dino_processor, "image_std", [0.229, 0.224, 0.225]))
    dino_model_instance = AutoModel.from_pretrained(
        dinov2_model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    dino_model_instance.eval()
    pooled_dino: list[Any] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        token_sets = _dino_tokens(
            dino_model_instance,
            [item[0] for item in batch],
            [item[1] for item in batch],
            device=device,
            dtype=dtype,
            mean=mean,
            std=std,
        )
        for tokens in token_sets:
            if len(tokens) == 0:
                raise BaseObservationBankError(
                    "DINOv2 observation mask produced no usable patch tokens"
                )
            pooled_dino.append(tokens.mean(axis=0))
    dino_views = _normalize_rows(np.stack(pooled_dino))
    dino_embeddings = _normalize_rows(
        dino_views.reshape(
            len(records), views_per_material, dino_views.shape[-1]
        ).mean(axis=1)
    )
    for canvas, mask in prepared:
        canvas.close()
        mask.close()
    del dino_model_instance
    del dino_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    profile_document = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "scope": "nvidia_base",
        "catalog_sha256": scope["catalog_sha256"],
        "material_count": len(records),
        "materials": profiles,
    }
    profiles_path = _write_json(
        bank_dir / "appearance_profiles.json", profile_document
    )
    npz_path = bank_dir / "visual_embeddings.npz"
    temporary_npz = bank_dir / "visual_embeddings.npz.tmp"
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(
            stream,
            material_ids=np.asarray(
                [record["material_id"] for record in records]
            ),
            siglip2=siglip_embeddings.astype(np.float16),
            dinov2=dino_embeddings.astype(np.float16),
        )
    temporary_npz.replace(npz_path)
    observation_source_counts: dict[str, int] = {}
    for profile in profiles:
        source = str(profile["observation_source"])
        observation_source_counts[source] = (
            observation_source_counts.get(source, 0) + 1
        )
    unsigned = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "bank_schema_version": BANK_SCHEMA_VERSION,
        "scope": "nvidia_base",
        "catalog_sha256": scope["catalog_sha256"],
        "render_manifest_sha256": _sha256_file(
            bank_dir / "render_manifest.json"
        ),
        "appearance_profiles": profiles_path.name,
        "appearance_profiles_sha256": _sha256_file(profiles_path),
        "visual_embeddings": npz_path.name,
        "visual_embeddings_sha256": _sha256_file(npz_path),
        "material_count": len(records),
        "complete": len(records) == len(catalog.materials),
        "profiles": list(OBSERVATION_PROFILES),
        "observation_source_counts": observation_source_counts,
        "siglip2": {
            "model": siglip_identity,
            "dimension": int(siglip_embeddings.shape[1]),
            "aggregation": "normalized_mean_of_three_masked_rig_views",
        },
        "dinov2": {
            "model": dino_identity,
            "dimension": int(dino_embeddings.shape[1]),
            "aggregation": "normalized_mean_of_foreground_patch_tokens_and_views",
        },
        "forbidden_vmaterials_2_count": 0,
    }
    manifest = {**unsigned, "manifest_sha256": _canonical_sha256(unsigned)}
    _write_json(bank_dir / "index_manifest.json", manifest)
    return manifest


def verify_bank(
    *,
    material_root: Path,
    output_dir: Path,
    require_index: bool,
) -> dict[str, Any]:
    root = resolve_base_root(material_root)
    bank_dir = output_dir.expanduser().resolve(strict=True)
    catalog = MaterialCatalog.load(
        bank_dir / "catalog.json", material_root=root
    )
    _assert_base_catalog(catalog)
    scope = _read_json(bank_dir / "scope_report.json", "scope report")
    if scope.get("scope") != "nvidia_base":
        raise BaseObservationBankError("scope report is not Base-only")
    if _sha256_file(bank_dir / "catalog.json") != scope.get("catalog_sha256"):
        raise BaseObservationBankError("catalog hash differs from scope report")
    if _sha256_file(bank_dir / "allowlist.json") != scope.get("allowlist_sha256"):
        raise BaseObservationBankError("allowlist hash differs from scope report")
    render = _read_json(bank_dir / "render_manifest.json", "render manifest")
    if not render.get("complete"):
        raise BaseObservationBankError("render bank is not complete")
    if render.get("rendered_material_count") != len(catalog.materials):
        raise BaseObservationBankError("render bank does not exactly cover catalog")
    expected_ids = [item.material_id for item in catalog.materials]
    rendered_ids = [
        record.get("material_id") for record in render.get("materials", [])
    ]
    if rendered_ids != expected_ids:
        raise BaseObservationBankError(
            "render bank material IDs/order do not exactly match the Base catalog"
        )
    observation_source_counts: dict[str, int] = {}
    checked_images = 0
    for record in render.get("materials", []):
        source = str(record.get("observation_source", "legacy_unspecified"))
        observation_source_counts[source] = (
            observation_source_counts.get(source, 0) + 1
        )
        for observation in record.get("observations", []):
            _relative_file(
                bank_dir=bank_dir,
                value=observation.get("image"),
                label="observation image",
                expected_sha256=observation.get("sha256"),
            )
            _relative_file(
                bank_dir=bank_dir,
                value=observation.get("mask"),
                label="observation mask",
                expected_sha256=observation.get("mask_sha256"),
            )
            checked_images += 1
    index_verified = False
    if require_index:
        index = _read_json(bank_dir / "index_manifest.json", "index manifest")
        unsigned = dict(index)
        expected_manifest_hash = unsigned.pop("manifest_sha256", None)
        if _canonical_sha256(unsigned) != expected_manifest_hash:
            raise BaseObservationBankError("index manifest seal is invalid")
        for name, key in (
            ("appearance_profiles", "appearance_profiles_sha256"),
            ("visual_embeddings", "visual_embeddings_sha256"),
        ):
            _relative_file(
                bank_dir=bank_dir,
                value=index.get(name),
                label=name,
                expected_sha256=index.get(key),
            )
        if not index.get("complete"):
            raise BaseObservationBankError("visual index is not complete")
        index_verified = True
    return {
        "scope": "nvidia_base",
        "material_count": len(catalog.materials),
        "checked_observation_images": checked_images,
        "render_complete": True,
        "index_verified": index_verified,
        "observation_source_counts": observation_source_counts,
        "forbidden_vmaterials_2_count": 0,
    }


def _module_environment() -> dict[str, str]:
    environment = os.environ.copy()
    tools_root = Path(__file__).resolve().parents[2]
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(tools_root)
        if not existing
        else os.pathsep.join((str(tools_root), existing))
    )
    return environment


def _run_child(python: Path, arguments: Sequence[str]) -> None:
    executable = python.expanduser().resolve(strict=True)
    command = [
        str(executable),
        "-m",
        "qwen_material_pipeline",
        "base-bank",
        *arguments,
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, env=_module_environment(), check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--material-root", type=Path, default=DEFAULT_BASE_ROOT)
    shared.add_argument("--output-dir", type=Path, required=True)

    subparsers.add_parser(
        "catalog", parents=[shared], help="build the strict Base catalog"
    )
    render = subparsers.add_parser(
        "render", parents=[shared], help="render the standard observation rig"
    )
    render.add_argument("--resolution", type=int, default=512)
    render.add_argument("--rt-subframes", type=int, default=4)
    render.add_argument("--limit", type=int)

    index = subparsers.add_parser(
        "index", parents=[shared], help="encode the rendered observation bank"
    )
    index.add_argument("--siglip2-model", type=Path, default=DEFAULT_SIGLIP2_MODEL)
    index.add_argument("--dinov2-model", type=Path, default=DEFAULT_DINOV2_MODEL)
    index.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    index.add_argument("--batch-size", type=int, default=12)
    index.add_argument("--allow-incomplete", action="store_true")

    build = subparsers.add_parser(
        "build",
        parents=[shared],
        help="catalog, render in Isaac Sim, index, and verify",
    )
    build.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    build.add_argument("--index-python", type=Path, default=Path(sys.executable))
    build.add_argument("--siglip2-model", type=Path, default=DEFAULT_SIGLIP2_MODEL)
    build.add_argument("--dinov2-model", type=Path, default=DEFAULT_DINOV2_MODEL)
    build.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    build.add_argument("--batch-size", type=int, default=12)
    build.add_argument("--resolution", type=int, default=512)
    build.add_argument("--rt-subframes", type=int, default=4)
    build.add_argument(
        "--render-chunk-size",
        type=int,
        default=40,
        help=(
            "restart Isaac after this many newly rendered materials; bounded "
            "sessions avoid long-lived RTX/MDL state accumulation"
        ),
    )
    build.add_argument("--limit", type=int)
    build.add_argument("--skip-index", action="store_true")

    verify = subparsers.add_parser(
        "verify", parents=[shared], help="verify exact coverage and hashes"
    )
    verify.add_argument("--without-index", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "catalog":
            catalog, report = initialize_bank(
                material_root=args.material_root,
                output_dir=args.output_dir,
            )
            result = {**report, "catalog_material_count": len(catalog.materials)}
        elif args.command == "render":
            app = None
            try:
                from isaacsim import SimulationApp

                app = SimulationApp(
                    {"headless": True, "create_new_stage": False}
                )
                result = render_bank(
                    material_root=args.material_root,
                    output_dir=args.output_dir,
                    resolution=args.resolution,
                    rt_subframes=args.rt_subframes,
                    limit=args.limit,
                )
            finally:
                if app is not None:
                    app.close()
        elif args.command == "index":
            result = build_index(
                material_root=args.material_root,
                output_dir=args.output_dir,
                siglip2_model=args.siglip2_model,
                dinov2_model=args.dinov2_model,
                device=args.device,
                batch_size=args.batch_size,
                allow_incomplete=args.allow_incomplete,
            )
        elif args.command == "verify":
            result = verify_bank(
                material_root=args.material_root,
                output_dir=args.output_dir,
                require_index=not args.without_index,
            )
        else:
            catalog, _scope = initialize_bank(
                material_root=args.material_root,
                output_dir=args.output_dir,
            )
            if args.render_chunk_size < 1:
                raise BaseObservationBankError(
                    "--render-chunk-size must be positive"
                )
            requested_count = min(
                len(catalog.materials),
                args.limit if args.limit is not None else len(catalog.materials),
            )
            manifest_path = args.output_dir.expanduser().resolve() / "render_manifest.json"
            rendered_count = 0
            if manifest_path.is_file():
                existing_manifest = _read_json(
                    manifest_path, "render manifest"
                )
                if (
                    existing_manifest.get("schema_version")
                    == RENDER_SCHEMA_VERSION
                    and existing_manifest.get("rig") == OBSERVATION_RIG_ID
                    and existing_manifest.get("resolution") == args.resolution
                ):
                    rendered_count = int(
                        existing_manifest.get("rendered_material_count", 0)
                    )
            while rendered_count < requested_count:
                target_count = min(
                    requested_count,
                    rendered_count + args.render_chunk_size,
                )
                render_arguments = [
                    "render",
                    "--material-root",
                    str(args.material_root),
                    "--output-dir",
                    str(args.output_dir),
                    "--resolution",
                    str(args.resolution),
                    "--rt-subframes",
                    str(args.rt_subframes),
                    "--limit",
                    str(target_count),
                ]
                _run_child(args.isaac_python, render_arguments)
                updated_manifest = _read_json(
                    manifest_path, "render manifest"
                )
                updated_count = int(
                    updated_manifest.get("rendered_material_count", 0)
                )
                if updated_count < target_count or updated_count <= rendered_count:
                    raise BaseObservationBankError(
                        "Isaac render chunk exited without reaching its "
                        f"target: before={rendered_count}, "
                        f"target={target_count}, after={updated_count}"
                    )
                rendered_count = updated_count
            if not args.skip_index:
                index_arguments = [
                    "index",
                    "--material-root",
                    str(args.material_root),
                    "--output-dir",
                    str(args.output_dir),
                    "--siglip2-model",
                    str(args.siglip2_model),
                    "--dinov2-model",
                    str(args.dinov2_model),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ]
                if args.limit is not None:
                    index_arguments.append("--allow-incomplete")
                _run_child(args.index_python, index_arguments)
            result = (
                verify_bank(
                    material_root=args.material_root,
                    output_dir=args.output_dir,
                    require_index=not args.skip_index,
                )
                if args.limit is None
                else {
                    "scope": "nvidia_base",
                    "partial_validation_material_limit": args.limit,
                    "output_dir": str(args.output_dir),
                }
            )
    except BaseException as exc:
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
