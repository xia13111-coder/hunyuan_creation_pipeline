#!/usr/bin/env python3
"""Run official GigaPose weights on one complete USD workpiece.

This adapter intentionally avoids the BOP dataset wrapper: it exports every
registered Mesh into one immutable analysis-space triangle mesh, renders the
official 162 CAD template viewpoints, and runs the official GigaPose networks
directly.  The output remains a proposal report; the main pipeline must still
verify each pose with an Isaac render before it can become a camera seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


MODEL_REVISION = "17fcf97f493f79e56a215ab10ebff16d95cfe34b"
SCHEMA_VERSION = "qwen-rigid-pose-model-proposals/v1"
TEMPLATE_INTRINSICS = np.asarray(
    ((572.4114, 0.0, 320.0), (0.0, 573.57043, 240.0), (0.0, 0.0, 1.0)),
    dtype=np.float32,
)
AXES = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Axis vector is degenerate")
    return value / norm


def _analysis_basis(up_axis: str, front_axis: str) -> np.ndarray:
    up = _normalize(np.asarray(AXES[up_axis], dtype=np.float64))
    requested_front = _normalize(np.asarray(AXES[front_axis], dtype=np.float64))
    front = _normalize(requested_front - np.dot(requested_front, up) * up)
    rear = -front
    right = _normalize(np.cross(rear, up))
    rear = _normalize(np.cross(up, right))
    return np.stack((right, rear, up), axis=1)


def _registry_bounds(registry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    boxes = [np.asarray(raw.get("world_bbox"), dtype=np.float64) for raw in registry["parts"]]
    if not boxes or any(box.shape != (2, 3) or not np.isfinite(box).all() for box in boxes):
        raise ValueError("Registry does not contain finite world bounds for every Part")
    return (
        np.min(np.stack([box[0] for box in boxes]), axis=0),
        np.max(np.stack([box[1] for box in boxes]), axis=0),
    )


def export_registered_mesh(
    *, registry_path: Path, output: Path, analysis_up_axis: str, analysis_front_axis: str
) -> dict[str, Any]:
    """Export exact registry coverage as one centered analysis-space mesh."""

    import trimesh
    from pxr import Usd, UsdGeom

    registry = _read(registry_path)
    stage = Usd.Stage.Open(str(Path(registry["asset_usd"]).resolve(strict=True)))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {registry['asset_usd']}")
    minimum, maximum = _registry_bounds(registry)
    center = 0.5 * (minimum + maximum)
    basis = _analysis_basis(analysis_up_axis, analysis_front_axis)
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for raw in registry["parts"]:
        prim_path = str(raw["prim_path"])
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Registered Part is not a Mesh: {prim_path}")
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if points is None or sum(counts) != len(indices):
            raise RuntimeError(f"Invalid Mesh topology: {prim_path}")
        matrix = xforms.GetLocalToWorldTransform(prim)
        world = np.asarray(
            [tuple(matrix.Transform(point)) for point in points], dtype=np.float64
        )
        analysis = (world - center) @ basis
        vertices.append(analysis)
        cursor = 0
        left_handed = str(mesh.GetOrientationAttr().Get()) == "leftHanded"
        determinant = float(np.linalg.det(np.asarray(matrix.ExtractRotationMatrix())))
        reverse = left_handed != (determinant < 0.0)
        for count in counts:
            face = [int(value) + offset for value in indices[cursor : cursor + count]]
            cursor += count
            if reverse:
                face.reverse()
            for index in range(1, len(face) - 1):
                triangles.append((face[0], face[index], face[index + 1]))
        offset += len(points)
    vertex_array = np.concatenate(vertices, axis=0)
    face_array = np.asarray(triangles, dtype=np.int64)
    result = trimesh.Trimesh(
        vertices=vertex_array, faces=face_array, process=False, validate=False
    )
    result.export(output)
    bounds = np.asarray(result.bounds, dtype=np.float64)
    return {
        "mesh": str(output),
        "mesh_sha256": _sha256(output),
        "vertex_count": int(len(vertex_array)),
        "triangle_count": int(len(face_array)),
        "analysis_bounds": bounds.tolist(),
        "analysis_diagonal": float(np.linalg.norm(bounds[1] - bounds[0])),
        "world_center": center.tolist(),
        "analysis_to_world_basis_columns": basis.tolist(),
        "registered_part_count": len(registry["parts"]),
    }


def _template_distance(bounds: np.ndarray) -> float:
    corners = np.asarray(
        [
            (x, y, z)
            for x in (bounds[0, 0], bounds[1, 0])
            for y in (bounds[0, 1], bounds[1, 1])
            for z in (bounds[0, 2], bounds[1, 2])
        ],
        dtype=np.float64,
    )
    radius = float(np.max(np.linalg.norm(corners, axis=1)))
    half_fov_x = math.atan(320.0 / float(TEMPLATE_INTRINSICS[0, 0]))
    half_fov_y = math.atan(240.0 / float(TEMPLATE_INTRINSICS[1, 1]))
    return max(radius / math.sin(min(half_fov_x, half_fov_y)) * 1.12, radius * 2.4)


def render_templates(
    *, mesh_path: Path, poses_path: Path, output: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    poses = np.load(poses_path).astype(np.float64)
    distance = _template_distance(np.asarray(mesh.bounds, dtype=np.float64))
    poses[:, :3, 3] *= distance / np.linalg.norm(poses[0, :3, 3])
    scene = pyrender.Scene(
        bg_color=np.asarray((0.0, 0.0, 0.0, 0.0)), ambient_light=(0.65, 0.65, 0.65)
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.62, 0.68, 0.74, 1.0),
        metallicFactor=0.05,
        roughnessFactor=0.75,
    )
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    camera = pyrender.IntrinsicsCamera(
        fx=float(TEMPLATE_INTRINSICS[0, 0]),
        fy=float(TEMPLATE_INTRINSICS[1, 1]),
        cx=float(TEMPLATE_INTRINSICS[0, 2]),
        cy=float(TEMPLATE_INTRINSICS[1, 2]),
        znear=max(1e-3, distance * 0.05),
        zfar=distance * 4.0,
    )
    scene.add(camera, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(640, 480)
    cv_to_gl = np.diag((1.0, -1.0, -1.0, 1.0))
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index, pose in enumerate(poses):
            scene.set_pose(node, pose=cv_to_gl @ pose)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = np.array(color, copy=True)
            color[:, :, 3] = np.where(depth > 0.0, 255, 0).astype(np.uint8)
            Image.fromarray(color, mode="RGBA").save(output / f"{index:06d}.png")
    finally:
        renderer.delete()
    np.save(output / "object_poses.npy", poses)
    return poses, {
        "template_count": int(len(poses)),
        "template_distance": distance,
        "intrinsics": TEMPLATE_INTRINSICS.tolist(),
        "poses_sha256": _sha256(output / "object_poses.npy"),
    }


def _bbox(mask: np.ndarray) -> list[int]:
    rows, cols = np.where(mask > 0)
    if not len(rows):
        raise ValueError("Foreground mask is empty")
    return [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]


def _load_rgba(path: Path) -> tuple[torch.Tensor, list[int]]:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    return torch.from_numpy(rgba.copy()).permute(2, 0, 1).float() / 255.0, _bbox(rgba[:, :, 3])


def _query_data(
    *, raw: dict[str, Any], manifest_path: Path, baseline: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, dict[str, Any]]:
    image_path = Path(str(raw["image"]))
    if not image_path.is_absolute():
        image_path = (manifest_path.parent / image_path).resolve(strict=True)
    confirmed = raw.get("confirmed_mask")
    mask_value = confirmed.get("path") if isinstance(confirmed, dict) else raw.get("palette_mask")
    mask_path = Path(str(mask_value))
    if not mask_path.is_absolute():
        mask_path = (manifest_path.parent / mask_path).resolve(strict=True)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None or bgr.shape[:2] != mask.shape:
        raise ValueError(f"Reference image/mask mismatch: {raw['id']}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.concatenate((rgb, np.where(mask[:, :, None] > 0, 255, 0).astype(np.uint8)), axis=2)
    rgba[mask == 0, :3] = 0
    tensor = torch.from_numpy(rgba).permute(2, 0, 1).float() / 255.0
    height, width = mask.shape
    horizontal_aperture_mm = 20.955
    focal = float(baseline.get("focal_length_mm", 45.0))
    fx = focal / horizontal_aperture_mm * width
    principal_u = float(baseline.get("principal_point_u", 0.0))
    principal_v = float(baseline.get("principal_point_v", 0.0))
    K = np.asarray(
        ((fx, 0.0, width * (0.5 + principal_u)), (0.0, fx, height * (0.5 + principal_v)), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    return tensor, torch.from_numpy((mask > 0).astype(np.float32)), torch.asarray(_bbox(mask)), K, {
        "image": str(image_path),
        "image_sha256": _sha256(image_path),
        "mask": str(mask_path),
        "mask_sha256": _sha256(mask_path),
        "intrinsics_source": "sealed_baseline_focal_plus_default_usd_aperture",
        "intrinsics": K.tolist(),
    }


def _load_model(
    *, repo: Path, checkpoint: Path, dinov2_repo: Path, device: torch.device
):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from src.models.matching import LocalSimilarity
    from src.models.network.ae_net import AENet
    from src.models.network.ist_net import ISTNet, Regressor
    from src.models.network.resnet import ResNet

    if not dinov2_repo.is_dir():
        raise FileNotFoundError(f"DINOv2 source checkout is missing: {dinov2_repo}")
    dino = torch.hub.load(
        str(dinov2_repo), "dinov2_vitl14", source="local", pretrained=False
    )
    ae = AENet(
        model_name="dinov2_vitl14",
        dinov2_model=dino,
        descriptor_size=1024,
        max_batch_size=8,
    )
    backbone = ResNet(
        {
            "n_heads": 0,
            "input_dim": 3,
            "input_size": 256,
            "initial_dim": 128,
            "block_dims": [128, 192, 256, 512],
            "descriptor_size": 256,
        }
    )
    ist = ISTNet(
        model_name="resnet",
        backbone=backbone,
        regressor=Regressor(
            descriptor_size=256,
            hidden_dim=256,
            use_tanh_act=True,
            normalize_output=True,
        ),
        max_batch_size=16,
        descriptor_size=256,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    ae.load_state_dict(
        {key.removeprefix("ae_net."): value for key, value in state.items() if key.startswith("ae_net.")},
        strict=True,
    )
    ist.load_state_dict(
        {key.removeprefix("ist_net."): value for key, value in state.items() if key.startswith("ist_net.")},
        strict=True,
    )
    ae.eval().to(device)
    ist.eval().to(device)
    return ae, ist, LocalSimilarity(k=5, sim_threshold=0.5, patch_threshold=3).to(device)


def infer(
    *,
    repo: Path,
    checkpoint: Path,
    dinov2_repo: Path,
    template_dir: Path,
    template_poses: np.ndarray,
    manifest_path: Path,
    baseline_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from src.models.poses import ObjectPoseRecovery
    from src.utils.crop import CropResizePad
    import src.megapose.utils.tensor_collection as tc

    device = torch.device("cuda")
    ae, ist, matcher = _load_model(
        repo=repo,
        checkpoint=checkpoint,
        dinov2_repo=dinov2_repo,
        device=device,
    )
    crop = CropResizePad(target_size=224)
    normalize_mean = torch.asarray((0.48145466, 0.4578275, 0.40821073))[:, None, None]
    normalize_std = torch.asarray((0.26862954, 0.26130258, 0.27577711))[:, None, None]

    template_rgba, template_boxes = zip(
        *[_load_rgba(template_dir / f"{index:06d}.png") for index in range(len(template_poses))]
    )
    rgba = torch.stack(template_rgba)
    cropped = crop(torch.asarray(template_boxes), images=rgba)
    template_rgb = (cropped["images"][:, :3] - normalize_mean) / normalize_std
    template_masks = cropped["images"][:, -1]
    with torch.inference_mode():
        template_ae = ae(template_rgb.to(device))
        template_ist = ist.forward_by_chunk(template_rgb.to(device))
    template_M = cropped["M"].float().to(device)
    template_pose_t = torch.from_numpy(template_poses).float().to(device)
    pose_recovery = ObjectPoseRecovery(
        template_K=torch.from_numpy(TEMPLATE_INTRINSICS).float()[None].to(device),
        template_Ms=template_M[None],
        template_poses=template_pose_t[None],
    ).to(device)
    manifest = _read(manifest_path)
    baseline = {str(raw["view_id"]): raw for raw in _read(baseline_path)["views"]}
    references: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for raw in manifest.get("source_views", []):
        view_id = str(raw["id"])
        if view_id not in baseline:
            continue
        query_rgba, _, query_box, query_K, provenance = _query_data(
            raw=raw, manifest_path=manifest_path, baseline=baseline[view_id]
        )
        query_cropped = crop(query_box[None], images=query_rgba[None])
        query_rgb = (query_cropped["images"][:, :3] - normalize_mean) / normalize_std
        query_mask = query_cropped["images"][:, -1]
        with torch.inference_mode():
            query_rgb_device = query_rgb.to(device)
            query_ae = ae(query_rgb_device)
            predictions = matcher.test(
                src_feats=template_ae[None],
                tar_feat=query_ae,
                src_masks=template_masks[None].to(device),
                tar_mask=query_mask.to(device),
            )
            query_ist = ist.forward_by_chunk(query_rgb_device)
            k = matcher.k
            num_patches = predictions.src_pts.shape[2]
            pred_scales = torch.zeros(1, k, num_patches, device=device)
            pred_inplanes = torch.zeros(1, k, num_patches, 2, device=device)
            for index in range(k):
                selected = predictions.id_src[:, index]
                pred_scales[:, index], pred_inplanes[:, index] = ist.inference(
                    src_feat=template_ist[selected],
                    tar_feat=query_ist,
                    src_pts=predictions.src_pts[:, index],
                    tar_pts=predictions.tar_pts[:, index],
                )
            predictions.register_tensor("relScale", pred_scales)
            predictions.register_tensor("relInplane", pred_inplanes)
            predictions = pose_recovery.forward_ransac(predictions=predictions)
            scores = torch.sum(predictions.ransac_scores, dim=2) / num_patches
            poses = pose_recovery.forward_recovery(
                tar_label=torch.ones(1, dtype=torch.long, device=device),
                tar_K=torch.from_numpy(query_K)[None].to(device),
                tar_M=query_cropped["M"].float().to(device),
                pred_src_views=predictions.id_src,
                pred_M=predictions.M.clone(),
            )
            order = torch.argsort(scores[0], descending=True)
        candidates = []
        for rank, hypothesis_index in enumerate(order.tolist(), start=1):
            pose = poses[0, hypothesis_index].detach().cpu().numpy()
            candidates.append(
                {
                    "rank": rank,
                    "model_score": float(scores[0, hypothesis_index].item()),
                    "inlier_fraction": float(scores[0, hypothesis_index].item()),
                    "template_view_id": int(predictions.id_src[0, hypothesis_index].item()),
                    "object_to_camera_rotation": pose[:3, :3].tolist(),
                    "object_to_camera_translation": pose[:3, 3].tolist(),
                }
            )
        views.append({"view_id": view_id, "candidates": candidates})
        references.append({"view_id": view_id, **provenance})
        print(f"[GIGAPOSE] {view_id} top score={candidates[0]['model_score']:.6f}", flush=True)
    if set(baseline) != {raw["view_id"] for raw in views}:
        raise ValueError("GigaPose inputs do not exactly cover the baseline cameras")
    return views, references


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"GigaPose output already exists: {output}")
    output.mkdir(parents=True)
    registry = args.registry.expanduser().resolve(strict=True)
    manifest = args.reference_manifest.expanduser().resolve(strict=True)
    baseline = args.baseline_view_specs.expanduser().resolve(strict=True)
    repo = args.gigapose_repo.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    dinov2_repo = args.dinov2_repo.expanduser().resolve(strict=True)
    revision = subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True
    ).strip()
    if revision != MODEL_REVISION:
        raise ValueError(f"Unexpected GigaPose revision: {revision}")
    mesh_path = output / "whole_asset_analysis.ply"
    mesh = export_registered_mesh(
        registry_path=registry,
        output=mesh_path,
        analysis_up_axis=args.analysis_up_axis,
        analysis_front_axis=args.analysis_front_axis,
    )
    source_poses = repo / "src/lib3d/predefined_poses/obj_poses_level1.npy"
    template_poses, templates = render_templates(
        mesh_path=mesh_path, poses_path=source_poses, output=output / "templates"
    )
    views, references = infer(
        repo=repo,
        checkpoint=checkpoint,
        dinov2_repo=dinov2_repo,
        template_dir=output / "templates",
        template_poses=template_poses,
        manifest_path=manifest,
        baseline_path=baseline,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": "GigaPose",
            "repository": str(repo),
            "repository_revision": revision,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "dinov2_repository": str(dinov2_repo),
            "paper_method": "162_template_view_retrieval_plus_patch_4DoF_recovery",
            "top_k": 5,
        },
        "inputs": {
            "registry": str(registry),
            "registry_sha256": _sha256(registry),
            "reference_manifest": str(manifest),
            "reference_manifest_sha256": _sha256(manifest),
            "baseline_view_specs": str(baseline),
            "baseline_view_specs_sha256": _sha256(baseline),
            "analysis_up_axis": args.analysis_up_axis,
            "analysis_front_axis": args.analysis_front_axis,
            "references": references,
        },
        "whole_asset_mesh": mesh,
        "templates": templates,
        "whole_asset_only": True,
        "per_mesh_transform_applied": False,
        "views": views,
    }
    _write(output / "gigapose_proposals.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--baseline-view-specs", type=Path, required=True)
    parser.add_argument("--gigapose-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-up-axis", choices=tuple(AXES), default="z")
    parser.add_argument("--analysis-front-axis", choices=tuple(AXES), default="-y")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": "PASS", "views": result["views"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
