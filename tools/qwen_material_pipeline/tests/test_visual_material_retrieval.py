from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import qwen_material_pipeline.retrieval.visual_materials as retrieval
from qwen_material_pipeline.retrieval.visual_materials import (
    BASE_BANK_INDEX_SCHEMA_VERSION,
    BASE_BANK_SCOPE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    VisualRetrievalError,
    _bank_color_vectors,
    _build_or_load_siglip_index,
    _catalog_digest,
    _descriptor_text,
    _load_catalog,
    _load_base_observation_bank,
    _load_request,
    _masked_query_rgb,
    _masked_square,
    _material_text,
    _model_fingerprint,
    _mvinverse_prior_scores,
    _normalize_rows,
    _siglip_pooled_features,
)


def test_masked_query_rgb_accepts_six_pixel_part_id_evidence() -> None:
    image = Image.new("RGB", (8, 8), (20, 40, 60))
    mask = Image.new("L", (8, 8), 0)
    for x in range(1, 7):
        mask.putpixel((x, 3), 255)

    color = _masked_query_rgb([image], [mask])

    assert color is not None
    assert np.allclose(color, np.asarray([20, 40, 60]) / 255.0)


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (40, 24),
    color: tuple[int, int, int] = (220, 20, 20),
) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_request_resolves_observation_files_without_loading_models(
    tmp_path: Path,
) -> None:
    material_root = tmp_path / "mdl"
    material_root.mkdir()
    _write_json(tmp_path / "catalog.json", {"materials": [{"material_id": "M1"}]})
    image = _write_image(tmp_path / "reference.png")
    mask = tmp_path / "mask.png"
    Image.new("L", image_size := (40, 24), 255).save(mask)
    assert image_size == Image.open(image).size
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "catalog": "catalog.json",
        "material_root": "mdl",
        "groups": [
            {
                "group_id": "G01",
                "descriptor": {"visual_description": "brushed silver metal"},
                "observations": [
                    {
                        "view_id": "front",
                        "image": "reference.png",
                        "mask": "mask.png",
                    }
                ],
            }
        ],
    }
    request_path = _write_json(tmp_path / "request.json", request)

    loaded, catalog, root, groups = _load_request(request_path)

    assert loaded["schema_version"] == REQUEST_SCHEMA_VERSION
    assert catalog == (tmp_path / "catalog.json").resolve()
    assert root == material_root.resolve()
    assert groups[0]["observations"] == [
        {
            "view_id": "front",
            "image": image.resolve(),
            "mask": mask.resolve(),
        }
    ]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([], "groups must be a non-empty"),
        (
            [
                {"group_id": "G01", "observations": []},
                {"group_id": "G01", "observations": []},
            ],
            "group IDs must be unique",
        ),
        (
            [{"group_id": "G01", "observations": "reference.png"}],
            "observations must be an array",
        ),
        (
            [{"group_id": "G01", "observations": ["not-an-object"]}],
            "observation 0 is invalid",
        ),
    ],
)
def test_load_request_rejects_ambiguous_groups(
    tmp_path: Path, groups: list, message: str
) -> None:
    (tmp_path / "mdl").mkdir()
    _write_json(tmp_path / "catalog.json", {"materials": [{"material_id": "M1"}]})
    request_path = _write_json(
        tmp_path / "request.json",
        {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "catalog": "catalog.json",
            "material_root": "mdl",
            "groups": groups,
        },
    )

    with pytest.raises(VisualRetrievalError, match=message):
        _load_request(request_path)


def test_descriptor_and_catalog_text_have_deterministic_field_order() -> None:
    assert _descriptor_text(
        {
            "group_id": "G01",
            "descriptor": {
                "roughness_hint": 0.35,
                "finish_hint": "brushed",
                "visual_description": "cool silver with directional grain",
                "family_hint": "metal",
                "ignored": "must not enter retrieval text",
            },
        }
    ) == (
        "cool silver with directional grain. metal. brushed. 0.35"
    )
    assert _descriptor_text(
        {"group_id": "G02", "descriptor": "  glossy black polymer  "}
    ) == "glossy black polymer"
    assert _descriptor_text({"group_id": "G03", "descriptor": {}}) == (
        "industrial material region G03"
    )
    assert _material_text(
        {
            "material_id": "Steel",
            "display_name": "Steel",
            "description": "Steel",
            "family": "metal",
            "keywords": ["silver", "metal"],
            "colors": ["silver"],
            "finishes": ["brushed"],
        }
    ) == "Steel. metal. silver. brushed"


def test_masked_square_neutralizes_background_and_erodes_boundaries(
    tmp_path: Path,
) -> None:
    image_path = _write_image(tmp_path / "part.png", size=(40, 24))
    mask = np.zeros((24, 40), dtype=np.uint8)
    mask[4:20, 10:30] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    canvas, eroded = _masked_square(image_path, mask_path, size=64)
    try:
        pixels = np.asarray(canvas)
        eroded_pixels = np.asarray(eroded)
        assert canvas.size == (64, 64)
        assert eroded.size == (64, 64)
        assert pixels[0, 0].tolist() == [127, 127, 127]
        assert np.any(np.all(pixels == [220, 20, 20], axis=-1))
        assert 0 < np.count_nonzero(eroded_pixels) < np.count_nonzero(
            np.any(pixels != [127, 127, 127], axis=-1)
        )
    finally:
        canvas.close()
        eroded.close()


def test_masked_square_rejects_empty_foreground(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "part.png")
    mask_path = tmp_path / "mask.png"
    Image.new("L", (40, 24), 0).save(mask_path)

    with pytest.raises(VisualRetrievalError, match="mask contains no foreground"):
        _masked_square(image_path, mask_path)


def test_masked_square_preserves_thin_black_part_evidence(tmp_path: Path) -> None:
    image_path = _write_image(
        tmp_path / "thin_black_part.png",
        size=(80, 40),
        color=(2, 2, 2),
    )
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[5:35, 39:41] = 255
    mask_path = tmp_path / "thin_mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    canvas, retained = _masked_square(image_path, mask_path, size=64)
    try:
        valid = np.asarray(retained) >= 128
        pixels = np.asarray(canvas)
        assert np.count_nonzero(valid) >= 16
        assert float(pixels[valid].mean()) < 10.0
    finally:
        canvas.close()
        retained.close()


def test_catalog_loader_indexes_every_material_and_confines_thumbnails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mdl"
    root.mkdir()
    thumbnail = _write_image(root / "steel.png")
    catalog_path = _write_json(
        tmp_path / "catalog.json",
        {
            "materials": [
                {
                    "material_id": "M_STEEL",
                    "thumbnail_path": "steel.png",
                    "display_name": "Steel",
                },
                {
                    "material_id": "M_NO_PREVIEW",
                    "display_name": "Material without a rendered preview",
                },
            ]
        },
    )

    materials = _load_catalog(catalog_path, root.resolve())
    digest, records = _catalog_digest(materials)

    assert [item["material_id"] for item in materials] == [
        "M_STEEL",
        "M_NO_PREVIEW",
    ]
    assert materials[0]["_thumbnail"] == thumbnail.resolve()
    assert materials[1]["_thumbnail"] is None
    assert len(digest) == 64
    assert [item["material_id"] for item in records] == [
        "M_STEEL",
        "M_NO_PREVIEW",
    ]
    assert records[0]["thumbnail"]["sha256"]
    assert records[1]["thumbnail"] is None

    outside = _write_image(tmp_path / "outside.png")
    _write_json(
        catalog_path,
        {
            "materials": [
                {"material_id": "ESCAPE", "thumbnail_path": outside.name}
            ]
        },
    )
    with pytest.raises(VisualRetrievalError, match="escapes the material root"):
        _load_catalog(catalog_path, root.resolve())


def test_model_fingerprint_and_row_normalization_are_content_bound(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"siglip"}', encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"weight-a")

    first = _model_fingerprint(model)
    weights.write_bytes(b"weight-b")
    second = _model_fingerprint(model)

    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]
    normalized = _normalize_rows(
        np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    )
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalized[1].tolist() == [0.0, 0.0]


def _write_tiny_base_observation_bank(
    root: Path,
    *,
    siglip_identity: dict,
    dino_identity: dict,
) -> tuple[Path, Path]:
    base = root / "Base"
    base.mkdir()
    (base / "A.mdl").write_text("export material A() = material();", encoding="utf-8")
    (base / "B.mdl").write_text("export material B() = material();", encoding="utf-8")
    material_ids = ["mdl:A.mdl#A", "mdl:B.mdl#B"]
    bank = root / "bank"
    bank.mkdir()
    source_digest, module_count = retrieval._mdl_source_digest(base)
    _write_json(
        bank / "scope_report.json",
        {
            "schema_version": BASE_BANK_SCOPE_SCHEMA_VERSION,
            "scope": "nvidia_base",
            "resolved_material_root": str(base.resolve()),
            "collection_name": "Base",
            "catalog_sha256": "a" * 64,
            "material_count": 2,
            "mdl_module_count": module_count,
            "mdl_sources_sha256": source_digest,
            "forbidden_vmaterials_2_count": 0,
            "exact_cover": True,
        },
    )
    profiles = {
        "schema_version": BASE_BANK_INDEX_SCHEMA_VERSION,
        "scope": "nvidia_base",
        "materials": [
            {
                "material_id": material_id,
                "appearance": {
                    "neutral_iso": {
                        "median_rgb": [float(index), 0.0, 0.0]
                    }
                },
                "authored_mdl": {
                    "reflection_roughness_constant": 0.2 + 0.6 * index,
                    "metallic_constant": float(index),
                },
            }
            for index, material_id in enumerate(material_ids)
        ],
    }
    profiles_path = _write_json(bank / "appearance_profiles.json", profiles)
    embeddings_path = bank / "visual_embeddings.npz"
    with embeddings_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            material_ids=np.asarray(list(reversed(material_ids))),
            siglip2=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float16),
            dinov2=np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.float16),
        )
    unsigned = {
        "schema_version": BASE_BANK_INDEX_SCHEMA_VERSION,
        "scope": "nvidia_base",
        "catalog_sha256": "a" * 64,
        "complete": True,
        "visual_embeddings": embeddings_path.name,
        "visual_embeddings_sha256": retrieval._sha256_file(embeddings_path),
        "appearance_profiles": profiles_path.name,
        "appearance_profiles_sha256": retrieval._sha256_file(profiles_path),
        "material_count": 2,
        "observation_source_counts": {"standard_rtx_rig": 2},
        "siglip2": {"model": siglip_identity, "dimension": 2},
        "dinov2": {"model": dino_identity, "dimension": 2},
        "forbidden_vmaterials_2_count": 0,
    }
    _write_json(
        bank / "index_manifest.json",
        {**unsigned, "manifest_sha256": retrieval._canonical_sha256(unsigned)},
    )
    return base, bank


def test_base_observation_bank_is_sealed_reordered_and_supplies_visual_priors(
    tmp_path: Path,
) -> None:
    siglip_identity = {"fingerprint_sha256": "s"}
    dino_identity = {"fingerprint_sha256": "d"}
    base, bank = _write_tiny_base_observation_bank(
        tmp_path,
        siglip_identity=siglip_identity,
        dino_identity=dino_identity,
    )
    material_ids = ["mdl:A.mdl#A", "mdl:B.mdl#B"]

    loaded = _load_base_observation_bank(
        bank_dir=bank,
        material_root=base,
        material_ids=material_ids,
        siglip_model_identity=siglip_identity,
        dino_model_identity=dino_identity,
    )

    assert loaded["identity"]["material_count"] == 2
    assert np.allclose(
        loaded["siglip2"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    colors = _bank_color_vectors(material_ids, loaded["profiles_by_id"])
    assert colors.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    scores, available = _mvinverse_prior_scores(
        group={
            "descriptor": {
                "roughness_hint": 0.8,
                "metallicity_hint": 1.0,
            }
        },
        material_ids=material_ids,
        profiles_by_id=loaded["profiles_by_id"],
    )
    assert available is True
    assert scores[1] > scores[0]


def test_base_observation_bank_rejects_changed_mdl_source(tmp_path: Path) -> None:
    siglip_identity = {"fingerprint_sha256": "s"}
    dino_identity = {"fingerprint_sha256": "d"}
    base, bank = _write_tiny_base_observation_bank(
        tmp_path,
        siglip_identity=siglip_identity,
        dino_identity=dino_identity,
    )
    (base / "A.mdl").write_text("changed", encoding="utf-8")

    with pytest.raises(VisualRetrievalError, match="sources changed"):
        _load_base_observation_bank(
            bank_dir=bank,
            material_root=base,
            material_ids=["mdl:A.mdl#A", "mdl:B.mdl#B"],
            siglip_model_identity=siglip_identity,
            dino_model_identity=dino_identity,
        )


def test_siglip2_pinned_identity_checks_every_file_and_exact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "siglip2"
    model.mkdir()
    files = {
        "config.json": b'{"model_type":"siglip"}',
        "model.safetensors": b"trusted weights",
        "tokenizer.model": b"trusted tokenizer",
    }
    records = []
    for relative, content in files.items():
        (model / relative).write_bytes(content)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": retrieval._sha256_file(model / relative),
            }
        )
    records.sort(key=lambda item: item["path"])
    manifest_sha256 = retrieval._canonical_sha256(records)
    monkeypatch.setattr(
        retrieval,
        "SIGLIP2_CONTENT_MANIFEST_SHA256",
        manifest_sha256,
    )
    identity = {
        "schema_version": retrieval.SIGLIP2_IDENTITY_SCHEMA_VERSION,
        "repository": retrieval.SIGLIP2_CANONICAL_REPOSITORY,
        "revision": retrieval.SIGLIP2_CANONICAL_REVISION,
        "content_manifest_sha256": manifest_sha256,
        "config_sha256": next(
            item["sha256"] for item in records if item["path"] == "config.json"
        ),
        "runtime_files": records,
    }
    _write_json(model / "checkpoint_identity.json", identity)

    assert retrieval._verify_pinned_siglip2_identity(model) == identity

    (model / "tokenizer.model").write_bytes(b"tampered tokenizer")
    with pytest.raises(VisualRetrievalError, match="size/SHA-256"):
        retrieval._verify_pinned_siglip2_identity(model)
    (model / "tokenizer.model").write_bytes(files["tokenizer.model"])

    (model / "unmanifested.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VisualRetrievalError, match="file set differs"):
        retrieval._verify_pinned_siglip2_identity(model)


class _FakeAutoProcessor:
    @classmethod
    def from_pretrained(
        cls,
        _path: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ):
        assert local_files_only is True
        assert trust_remote_code is False
        return cls()


class _FakeModel:
    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None


class _FakeAutoModel:
    @classmethod
    def from_pretrained(
        cls,
        _path: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        dtype,
    ) -> _FakeModel:
        assert local_files_only is True
        assert trust_remote_code is False
        assert dtype == "float32"
        return _FakeModel()


def test_siglip_v5_pooler_output_and_legacy_tensor_are_supported() -> None:
    class FakeTensor:
        def detach(self):
            return self

    tensor = FakeTensor()
    assert _siglip_pooled_features(tensor, "image") is tensor
    assert (
        _siglip_pooled_features(
            types.SimpleNamespace(pooler_output=tensor),
            "text",
        )
        is tensor
    )
    with pytest.raises(VisualRetrievalError, match="no pooled feature"):
        _siglip_pooled_features(types.SimpleNamespace(), "image")


def test_siglip_index_contract_keeps_thumbnailless_catalog_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "siglip2"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"weights")
    thumbnail = _write_image(tmp_path / "steel.png")
    materials = [
        {
            "material_id": "M_STEEL",
            "display_name": "Steel",
            "_thumbnail": thumbnail,
        },
        {
            "material_id": "M_TEXT_ONLY",
            "display_name": "Text-only material",
            "_thumbnail": None,
        },
    ]
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_transformers.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        retrieval,
        "_verified_siglip2_model_identity",
        _model_fingerprint,
    )
    monkeypatch.setattr(
        retrieval,
        "_siglip_text_embeddings",
        lambda _model, _processor, texts, **_kwargs: np.asarray(
            [[1.0, 0.0], [0.0, 1.0]][: len(texts)], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "_siglip_image_embeddings",
        lambda _model, _processor, images, **_kwargs: np.asarray(
            [[1.0, 0.0] for _image in images], dtype=np.float32
        ),
    )

    embeddings, manifest, _model, _processor = _build_or_load_siglip_index(
        materials=materials,
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        device="cpu",
        batch_size=2,
    )

    assert embeddings.shape == (2, 2)
    assert manifest["material_count"] == 2
    assert manifest["thumbnail_count"] == 1
    assert manifest["text_only_count"] == 1
    assert [item["material_id"] for item in manifest["catalog_records"]] == [
        "M_STEEL",
        "M_TEXT_ONLY",
    ]
    assert Path(manifest["npz"]).is_file()
