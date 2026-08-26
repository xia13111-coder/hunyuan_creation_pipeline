"""Index and search the local NVIDIA MDL material library.

The module deliberately has no third-party dependencies.  Qwen (or another
planner) may use the search result's stable ``material_id``, but filesystem
paths are always resolved again beneath a caller-controlled material root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from qwen_material_pipeline.core.paths import MODELS_ROOT
from qwen_material_pipeline.materials.semantics import (
    CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION,
    infer_catalog_surface_semantics,
    normalize_catalog_surface_semantics,
)


_DEFAULT_ISAAC_MATERIALS = (
    Path.home()
    / "isaacsim_assets"
    / "Assets"
    / "Isaac"
    / "4.5"
    / "NVIDIA"
    / "Materials"
)
DEFAULT_MATERIAL_ROOT = Path(
    os.environ.get(
        "VISUAL_MATERIAL_ROOT",
        str(
            _DEFAULT_ISAAC_MATERIALS
            if _DEFAULT_ISAAC_MATERIALS.is_dir()
            else MODELS_ROOT / "materials" / "nvidia"
        ),
    )
).expanduser()
CATALOG_SCHEMA_VERSION = 2
LEGACY_CATALOG_SCHEMA_VERSION = 1
ALLOWLIST_SCHEMA_VERSION = 1


class MaterialCatalogError(ValueError):
    """Base exception for invalid catalogs and material definitions."""


class MaterialPathError(MaterialCatalogError):
    """Raised when a catalog path escapes the configured material root."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPORT_RE = re.compile(
    r"\bexport\s+material\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE
)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
_CONST_STRING_RE = re.compile(
    r"\bconst\s+string\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE
)
_WORD_RE = re.compile(r"[a-z0-9]+")


_KNOWN_FAMILIES = {
    "carpet",
    "ceramic",
    "composite",
    "concrete",
    "fabric",
    "gems",
    "glass",
    "ground",
    "leather",
    "liquids",
    "masonry",
    "metal",
    "paint",
    "paper",
    "plaster",
    "plastic",
    "rubber",
    "stone",
    "wood",
}

_FAMILY_ALIASES = {
    "aluminium": "metal",
    "aluminum": "metal",
    "brass": "metal",
    "bronze": "metal",
    "cast iron": "metal",
    "copper": "metal",
    "iron": "metal",
    "metallic": "metal",
    "stainless": "metal",
    "stainless steel": "metal",
    "steel": "metal",
    "caoutchouc": "rubber",
    "elastomer": "rubber",
    "polymer": "plastic",
    "金属": "metal",
    "钢": "metal",
    "不锈钢": "metal",
    "铝": "metal",
    "塑料": "plastic",
    "橡胶": "rubber",
    "玻璃": "glass",
    "木材": "wood",
    "石材": "stone",
    "陶瓷": "ceramic",
    "油漆": "paint",
}

_COLOR_ALIASES = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "red": "red",
    "russet": "red",
    "orange": "orange",
    "yellow": "yellow",
    "gold": "gold",
    "golden": "gold",
    "green": "green",
    "blue": "blue",
    "brown": "brown",
    "beige": "beige",
    "tan": "tan",
    "purple": "purple",
    "violet": "purple",
    "pink": "pink",
    "cyan": "cyan",
    "turquoise": "cyan",
    "clear": "clear",
    "transparent": "clear",
    "黑": "black",
    "黑色": "black",
    "白": "white",
    "白色": "white",
    "灰": "gray",
    "灰色": "gray",
    "银": "silver",
    "银色": "silver",
    "红": "red",
    "红色": "red",
    "橙": "orange",
    "橙色": "orange",
    "黄": "yellow",
    "黄色": "yellow",
    "金色": "gold",
    "绿": "green",
    "绿色": "green",
    "蓝": "blue",
    "蓝色": "blue",
    "棕": "brown",
    "棕色": "brown",
    "紫": "purple",
    "紫色": "purple",
    "粉": "pink",
    "粉色": "pink",
    "透明": "clear",
}

_FINISH_ALIASES = {
    "matte": "matte",
    "matt": "matte",
    "gloss": "glossy",
    "glossy": "glossy",
    "shiny": "glossy",
    "polished": "polished",
    "brushed": "brushed",
    "brushing": "brushed",
    "satin": "satin",
    "rough": "rough",
    "smooth": "smooth",
    "worn": "worn",
    "weathered": "worn",
    "dirty": "dirty",
    "cracked": "cracked",
    "hammered": "hammered",
    "cast": "cast",
    "anodized": "anodized",
    "galvanized": "galvanized",
    "painted": "painted",
    "coated": "coated",
    "reflective": "reflective",
    "frosted": "frosted",
    "scratched": "scratched",
    "rust": "rusty",
    "rusted": "rusty",
    "rusty": "rusty",
    "clean": "clean",
    "new": "new",
    "哑光": "matte",
    "亚光": "matte",
    "亮光": "glossy",
    "高光": "glossy",
    "抛光": "polished",
    "拉丝": "brushed",
    "缎面": "satin",
    "粗糙": "rough",
    "光滑": "smooth",
    "磨损": "worn",
    "脏污": "dirty",
    "开裂": "cracked",
    "锤纹": "hammered",
    "铸造": "cast",
    "阳极氧化": "anodized",
    "镀锌": "galvanized",
    "喷漆": "painted",
    "涂层": "coated",
    "反光": "reflective",
    "磨砂": "frosted",
    "划痕": "scratched",
    "生锈": "rusty",
}

_QUERY_REPLACEMENTS = {
    "不锈钢": " stainless steel ",
    "拉丝": " brushed ",
    "哑光": " matte ",
    "亚光": " matte ",
    "抛光": " polished ",
    "喷漆": " painted ",
    "镀锌": " galvanized ",
    "橡胶": " rubber ",
    "塑料": " plastic ",
    "金属": " metal ",
    "玻璃": " glass ",
}


def _canonical_root(material_root: os.PathLike[str] | str) -> Path:
    try:
        root = Path(material_root).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MaterialPathError(
            f"material root does not exist: {material_root}"
        ) from exc
    if not root.is_dir():
        raise MaterialPathError(f"material root is not a directory: {root}")
    return root


def _validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise MaterialPathError(f"invalid catalog path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise MaterialPathError(f"catalog path must be a safe relative path: {value!r}")
    return relative


def _resolve_under_root(
    root: Path,
    relative_path: str,
    *,
    allowed_suffixes: Iterable[str],
) -> Path:
    relative = _validate_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise MaterialPathError(
            f"path is missing or outside material root: {relative_path!r}"
        ) from exc
    suffixes = {suffix.casefold() for suffix in allowed_suffixes}
    if not resolved.is_file() or resolved.suffix.casefold() not in suffixes:
        raise MaterialPathError(f"unexpected material file type: {relative_path!r}")
    return resolved


def stable_material_id(mdl_path: str, sub_identifier: str) -> str:
    """Return a root-independent ID for one exported MDL material."""

    relative = _validate_relative_path(mdl_path).as_posix()
    if not isinstance(sub_identifier, str) or not _IDENTIFIER_RE.fullmatch(
        sub_identifier
    ):
        raise MaterialCatalogError(f"invalid MDL sub-identifier: {sub_identifier!r}")
    return f"mdl:{relative}#{sub_identifier}"


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    mdl_path: str
    sub_identifier: str
    display_name: str
    description: str
    keywords: tuple[str, ...]
    family: str
    category_path: str
    colors: tuple[str, ...]
    finishes: tuple[str, ...]
    thumbnail_path: str | None
    surface_semantics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "mdl_path": self.mdl_path,
            "sub_identifier": self.sub_identifier,
            "display_name": self.display_name,
            "description": self.description,
            "keywords": list(self.keywords),
            "family": self.family,
            "category_path": self.category_path,
            "colors": list(self.colors),
            "finishes": list(self.finishes),
            "thumbnail_path": self.thumbnail_path,
            "surface_semantics": dict(self.surface_semantics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaterialRecord":
        if not isinstance(value, Mapping):
            raise MaterialCatalogError("material record must be a JSON object")
        try:
            keywords = _string_tuple(value.get("keywords", []), "keywords")
            colors = _string_tuple(value.get("colors", []), "colors")
            finishes = _string_tuple(value.get("finishes", []), "finishes")
            thumbnail = value.get("thumbnail_path")
            if thumbnail is not None and not isinstance(thumbnail, str):
                raise TypeError("thumbnail_path must be a string or null")
            family = _required_string(value, "family")
            raw_surface_semantics = value.get("surface_semantics")
            if raw_surface_semantics is None:
                raw_surface_semantics = infer_catalog_surface_semantics(
                    family=family,
                    tokens=_query_tokens(
                        " ".join(
                            (
                                _required_string(value, "material_id"),
                                _required_string(value, "mdl_path"),
                                _required_string(value, "sub_identifier"),
                                _required_string(value, "display_name"),
                                _optional_string(value, "description"),
                                " ".join(keywords),
                                " ".join(finishes),
                            )
                        )
                    ),
                )
            surface_semantics = normalize_catalog_surface_semantics(
                raw_surface_semantics
            )
            return cls(
                material_id=_required_string(value, "material_id"),
                mdl_path=_required_string(value, "mdl_path"),
                sub_identifier=_required_string(value, "sub_identifier"),
                display_name=_required_string(value, "display_name"),
                description=_optional_string(value, "description"),
                keywords=keywords,
                family=family,
                category_path=_optional_string(value, "category_path"),
                colors=colors,
                finishes=finishes,
                thumbnail_path=thumbnail,
                surface_semantics=surface_semantics,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterialCatalogError(f"invalid material record: {exc}") from exc


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key, "")
    if not isinstance(result, str):
        raise TypeError(f"{key} must be a string")
    return result


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(value)


class MaterialCatalog:
    """A validated, deterministic collection of exported MDL materials."""

    def __init__(
        self,
        material_root: os.PathLike[str] | str,
        materials: Sequence[MaterialRecord],
    ) -> None:
        self.root = _canonical_root(material_root)
        ordered = sorted(materials, key=lambda item: item.material_id)
        by_id: dict[str, MaterialRecord] = {}
        for record in ordered:
            expected_id = stable_material_id(record.mdl_path, record.sub_identifier)
            if record.material_id != expected_id:
                raise MaterialCatalogError(
                    f"material ID does not match path and sub-identifier: {record.material_id}"
                )
            _resolve_under_root(self.root, record.mdl_path, allowed_suffixes={".mdl"})
            if record.thumbnail_path is not None:
                _resolve_under_root(
                    self.root,
                    record.thumbnail_path,
                    allowed_suffixes={".png", ".jpg", ".jpeg", ".webp"},
                )
            try:
                normalize_catalog_surface_semantics(record.surface_semantics)
            except ValueError as exc:
                raise MaterialCatalogError(
                    "invalid catalog surface semantics for "
                    f"{record.material_id}: {exc}"
                ) from exc
            if record.material_id in by_id:
                raise MaterialCatalogError(
                    f"duplicate material ID: {record.material_id}"
                )
            by_id[record.material_id] = record
        self.materials = tuple(ordered)
        self._by_id = by_id

    @classmethod
    def scan(
        cls, material_root: os.PathLike[str] | str = DEFAULT_MATERIAL_ROOT
    ) -> "MaterialCatalog":
        root = _canonical_root(material_root)
        materials: list[MaterialRecord] = []
        for directory, subdirectories, filenames in os.walk(root, followlinks=False):
            subdirectories.sort()
            for filename in sorted(filenames):
                if not filename.casefold().endswith(".mdl"):
                    continue
                source = Path(directory, filename)
                try:
                    resolved = source.resolve(strict=True)
                    relative = resolved.relative_to(root).as_posix()
                except (FileNotFoundError, OSError, ValueError) as exc:
                    raise MaterialPathError(
                        f"unsafe MDL path encountered: {source}"
                    ) from exc
                materials.extend(_parse_mdl(resolved, relative, root))
        return cls(root, materials)

    @classmethod
    def load(
        cls,
        catalog_path: os.PathLike[str] | str,
        *,
        material_root: os.PathLike[str] | str = DEFAULT_MATERIAL_ROOT,
    ) -> "MaterialCatalog":
        try:
            with Path(catalog_path).open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise MaterialCatalogError(
                f"cannot load material catalog: {catalog_path}"
            ) from exc
        if not isinstance(document, dict):
            raise MaterialCatalogError("catalog document must be a JSON object")
        schema_version = document.get("schema_version")
        if schema_version not in {
            LEGACY_CATALOG_SCHEMA_VERSION,
            CATALOG_SCHEMA_VERSION,
        }:
            raise MaterialCatalogError(
                f"unsupported catalog schema: {schema_version!r}"
            )
        raw_materials = document.get("materials")
        if not isinstance(raw_materials, list):
            raise MaterialCatalogError("catalog materials must be a JSON array")
        # The document's material_root is informational.  It is intentionally
        # ignored so an untrusted catalog cannot choose its own whitelist.
        return cls(
            material_root, [MaterialRecord.from_dict(item) for item in raw_materials]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "surface_semantics_schema_version": (
                CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION
            ),
            # Informational only: load() deliberately trusts the caller's
            # configured root.  Keeping this relative makes the generated
            # catalog portable between machines and Isaac asset mounts.
            "material_root": ".",
            "material_count": len(self.materials),
            "materials": [record.to_dict() for record in self.materials],
        }

    def to_full_allowlist_dict(self) -> dict[str, Any]:
        """Return an exact, deterministic allowlist for this whole catalog.

        The pipeline retains the historical ``whitelist`` input name as a
        compatibility and audit boundary.  ``scope=catalog_exact`` changes its
        meaning from a hand-picked subset to every exported material that was
        validated beneath the caller-controlled material root.
        """

        return {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "name": "nvidia_full_material_catalog",
            "description": (
                "Automatically generated full allowlist containing every "
                "exported NVIDIA MDL material below the configured material root."
            ),
            "scope": "catalog_exact",
            "material_count": len(self.materials),
            "material_ids": [record.material_id for record in self.materials],
        }

    def save(self, output_path: os.PathLike[str] | str) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(
                self.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True
            )
            stream.write("\n")
        return destination

    def get(self, material_id: str) -> MaterialRecord:
        try:
            return self._by_id[material_id]
        except KeyError as exc:
            raise MaterialCatalogError(f"unknown material ID: {material_id!r}") from exc

    def resolve_material(self, material_id: str) -> tuple[Path, str]:
        """Resolve a whitelisted ID to ``(mdl_file, sub_identifier)``."""

        record = self.get(material_id)
        mdl_file = _resolve_under_root(
            self.root, record.mdl_path, allowed_suffixes={".mdl"}
        )
        return mdl_file, record.sub_identifier

    def search(
        self,
        *,
        family: str | None = None,
        color: str | None = None,
        finish: str | None = None,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise MaterialCatalogError("top_k must be a positive integer")

        family_value = _canonical_family(family) if family else None
        color_values = _canonical_filter_values(color, _COLOR_ALIASES) if color else ()
        finish_values = (
            _canonical_filter_values(finish, _FINISH_ALIASES) if finish else ()
        )
        query_tokens = _query_tokens(query or "")
        scored: list[tuple[float, str, MaterialRecord, tuple[str, ...]]] = []

        for record in self.materials:
            fields = _record_search_fields(record)
            all_tokens = fields["all"]
            matched: list[str] = []
            score = 0.0

            if family_value:
                family_tokens = _query_tokens(family_value)
                family_match = record.family == family_value or bool(
                    family_tokens and family_tokens <= all_tokens
                )
                if not family_match:
                    continue
                matched.append("family")
                score += 40.0 if record.family == family_value else 24.0

            if color_values:
                found_colors = set(record.colors) & set(color_values)
                if not found_colors:
                    found_colors = set(color_values) & all_tokens
                if not found_colors:
                    continue
                matched.append("color")
                score += 28.0 + 2.0 * len(found_colors)

            if finish_values:
                found_finishes = set(record.finishes) & set(finish_values)
                if not found_finishes:
                    found_finishes = set(finish_values) & all_tokens
                if not found_finishes:
                    continue
                matched.append("finish")
                score += 28.0 + 2.0 * len(found_finishes)

            if query_tokens:
                found_query = query_tokens & all_tokens
                if not found_query:
                    continue
                matched.append("query")
                for token in found_query:
                    if token in fields["name"]:
                        score += 8.0
                    elif token in fields["keywords"]:
                        score += 5.0
                    elif token in fields["category"]:
                        score += 4.0
                    else:
                        score += 1.0
                score += 5.0 * len(found_query) / len(query_tokens)

            scored.append((score, record.material_id, record, tuple(matched)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for score, _, record, matched in scored[:top_k]:
            result = record.to_dict()
            result["score"] = round(score, 4)
            result["matched_fields"] = list(matched)
            results.append(result)
        return results


def build_catalog(
    material_root: os.PathLike[str] | str = DEFAULT_MATERIAL_ROOT,
    output_path: os.PathLike[str] | str | None = None,
) -> MaterialCatalog:
    catalog = MaterialCatalog.scan(material_root)
    if output_path is not None:
        catalog.save(output_path)
    return catalog


def search_materials(
    catalog: MaterialCatalog,
    *,
    family: str | None = None,
    color: str | None = None,
    finish: str | None = None,
    query: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not isinstance(catalog, MaterialCatalog):
        raise TypeError("catalog must be a MaterialCatalog instance")
    return catalog.search(
        family=family, color=color, finish=finish, query=query, top_k=top_k
    )


def _parse_mdl(mdl_file: Path, relative_path: str, root: Path) -> list[MaterialRecord]:
    text = mdl_file.read_text(encoding="utf-8", errors="replace")
    masked = _mask_comments_and_strings(text)
    constants = _parse_string_constants(text, masked)
    category_path = PurePosixPath(relative_path).parent.as_posix()
    family = _infer_family(PurePosixPath(relative_path))
    records: list[MaterialRecord] = []

    for export_match in _EXPORT_RE.finditer(masked):
        sub_identifier = export_match.group(1)
        open_parenthesis = masked.find("(", export_match.start(), export_match.end())
        close_parenthesis = _find_matching(text, open_parenthesis, "(", ")")
        annotation, _ = _material_annotation(text, close_parenthesis + 1)
        display_name = _annotation_text(annotation, "display_name", constants)
        if not display_name:
            display_name = _humanize(sub_identifier)
        description = _annotation_text(annotation, "description", constants)
        keywords = tuple(
            dict.fromkeys(
                value.strip()
                for value in _annotation_strings(annotation, ("key_words", "keywords"))
                if value.strip()
            )
        )
        searchable_text = " ".join(
            (
                sub_identifier,
                display_name,
                description,
                " ".join(keywords),
                category_path,
            )
        )
        colors = tuple(sorted(_infer_values(searchable_text, _COLOR_ALIASES)))
        finishes = tuple(sorted(_infer_values(searchable_text, _FINISH_ALIASES)))
        thumbnail = _find_thumbnail(mdl_file, sub_identifier, root)
        surface_semantics = infer_catalog_surface_semantics(
            family=family,
            tokens=_query_tokens(searchable_text) | set(finishes),
        )
        records.append(
            MaterialRecord(
                material_id=stable_material_id(relative_path, sub_identifier),
                mdl_path=relative_path,
                sub_identifier=sub_identifier,
                display_name=display_name,
                description=description,
                keywords=keywords,
                family=family,
                category_path=category_path,
                colors=colors,
                finishes=finishes,
                thumbnail_path=thumbnail,
                surface_semantics=surface_semantics,
            )
        )
    return records


def _mask_comments_and_strings(text: str) -> str:
    chars = list(text)
    index = 0
    while index < len(chars):
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = len(text) if end < 0 else end
            chars[index:end] = " " * (end - index)
            index = end
        elif text.startswith("/*", index):
            end_marker = text.find("*/", index + 2)
            end = len(text) if end_marker < 0 else end_marker + 2
            chars[index:end] = " " * (end - index)
            index = end
        elif text[index] == '"':
            end = _string_end(text, index)
            chars[index:end] = " " * (end - index)
            index = end
        else:
            index += 1
    return "".join(chars)


def _string_end(text: str, start: int) -> int:
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == '"':
            return index + 1
        else:
            index += 1
    return len(text)


def _find_matching(text: str, start: int, opening: str, closing: str) -> int:
    if start < 0 or start >= len(text) or text[start] != opening:
        raise MaterialCatalogError("malformed MDL delimiter")
    depth = 0
    index = start
    while index < len(text):
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
            continue
        if text[index] == '"':
            index = _string_end(text, index)
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise MaterialCatalogError("unterminated MDL delimiter")


def _skip_trivia(text: str, index: int) -> int:
    while index < len(text):
        if text[index].isspace():
            index += 1
        elif text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
        else:
            break
    return index


def _material_annotation(text: str, index: int) -> tuple[str, int]:
    index = _skip_trivia(text, index)
    if not text.startswith("[[", index):
        return "", index
    content_start = index + 2
    cursor = content_start
    while cursor < len(text):
        if text.startswith("//", cursor):
            newline = text.find("\n", cursor + 2)
            cursor = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", cursor):
            end = text.find("*/", cursor + 2)
            cursor = len(text) if end < 0 else end + 2
            continue
        if text[cursor] == '"':
            cursor = _string_end(text, cursor)
            continue
        if text.startswith("]]", cursor):
            return text[content_start:cursor], cursor + 2
        cursor += 1
    raise MaterialCatalogError("unterminated MDL annotation block")


def _parse_string_constants(text: str, masked: str) -> dict[str, str]:
    constants: dict[str, str] = {}
    for match in _CONST_STRING_RE.finditer(masked):
        cursor = match.end()
        while cursor < len(text):
            if text[cursor] == '"':
                cursor = _string_end(text, cursor)
                continue
            if text[cursor] == ";":
                break
            cursor += 1
        values = _decode_strings(text[match.end() : cursor])
        if values:
            constants[match.group(1)] = "".join(values).strip()
    return constants


def _annotation_argument(annotation: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*\(", annotation)
    if not match:
        return ""
    opening = annotation.find("(", match.start(), match.end())
    try:
        closing = _find_matching(annotation, opening, "(", ")")
    except MaterialCatalogError:
        return ""
    return annotation[opening + 1 : closing]


def _annotation_text(annotation: str, name: str, constants: Mapping[str, str]) -> str:
    argument = _annotation_argument(annotation, name)
    strings = _decode_strings(argument)
    if strings:
        return "".join(strings).strip()
    identifier = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", argument)
    return constants.get(identifier.group(1), "") if identifier else ""


def _annotation_strings(annotation: str, names: Iterable[str]) -> list[str]:
    for name in names:
        argument = _annotation_argument(annotation, name)
        if argument:
            return _decode_strings(argument)
    return []


def _decode_strings(text: str) -> list[str]:
    decoded: list[str] = []
    for match in _STRING_RE.finditer(text):
        literal = match.group(0)
        try:
            decoded.append(json.loads(literal))
        except json.JSONDecodeError:
            # MDL and JSON string escaping overlap for the material metadata in
            # this library.  Preserve an unusual literal rather than aborting a
            # full library scan.
            decoded.append(literal[1:-1].replace(r"\"", '"').replace(r"\\", "\\"))
    return decoded


def _find_thumbnail(mdl_file: Path, sub_identifier: str, root: Path) -> str | None:
    thumbnail_dir = mdl_file.parent / ".thumbs" / "256x256"
    stems = (f"{mdl_file.name}@{sub_identifier}", mdl_file.name)
    for stem in stems:
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = thumbnail_dir / f"{stem}{suffix}"
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve(strict=True)
                return resolved.relative_to(root).as_posix()
            except (FileNotFoundError, OSError, ValueError):
                continue
    return None


def _infer_family(relative_path: PurePosixPath) -> str:
    directory_parts = [part.casefold() for part in relative_path.parts[:-1]]
    for part in reversed(directory_parts):
        if part in _KNOWN_FAMILIES:
            return part
        singular = part[:-1] if part.endswith("s") else part
        if singular in _KNOWN_FAMILIES:
            return singular
    # NVIDIA Base stores the generic Paint_* presets in Miscellaneous.  Their
    # physical family is nevertheless paint, and treating the folder label as
    # the family would make them unreachable for painted-part retrieval.
    filename_tokens = _query_tokens(relative_path.stem)
    for family in sorted(_KNOWN_FAMILIES):
        if family in filename_tokens:
            return family
    return directory_parts[-1] if directory_parts else "uncategorized"


def _humanize(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return re.sub(r"[_\-]+", " ", value).strip()


def _query_tokens(value: str) -> set[str]:
    normalized = value.casefold()
    for source, replacement in _QUERY_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    return set(_WORD_RE.findall(normalized))


def _infer_values(text: str, aliases: Mapping[str, str]) -> set[str]:
    tokens = _query_tokens(text)
    lowered = text.casefold()
    result: set[str] = set()
    for alias, canonical in aliases.items():
        alias_lower = alias.casefold()
        if alias_lower in tokens or (
            not alias_lower.isascii() and alias_lower in lowered
        ):
            result.add(canonical)
    return result


def _canonical_family(value: str) -> str:
    stripped = value.strip().casefold()
    if not stripped:
        raise MaterialCatalogError("family cannot be empty")
    if stripped in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[stripped]
    tokens = _query_tokens(stripped)
    if len(tokens) == 1:
        token = next(iter(tokens))
        return _FAMILY_ALIASES.get(token, token)
    return stripped


def _canonical_filter_values(value: str, aliases: Mapping[str, str]) -> tuple[str, ...]:
    stripped = value.strip().casefold()
    if not stripped:
        raise MaterialCatalogError("search filter cannot be empty")
    result = _infer_values(stripped, aliases)
    if not result:
        result = _query_tokens(stripped)
    return tuple(sorted(result))


def _record_search_fields(record: MaterialRecord) -> dict[str, set[str]]:
    name_tokens = _query_tokens(f"{record.sub_identifier} {record.display_name}")
    keyword_tokens = _query_tokens(" ".join(record.keywords))
    category_tokens = _query_tokens(f"{record.family} {record.category_path}")
    description_tokens = _query_tokens(record.description)
    structured = set(record.colors) | set(record.finishes)
    return {
        "name": name_tokens,
        "keywords": keyword_tokens,
        "category": category_tokens,
        "all": name_tokens
        | keyword_tokens
        | category_tokens
        | description_tokens
        | structured,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="scan MDL files and write a JSON catalog"
    )
    build.add_argument("--root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--full-allowlist-output",
        type=Path,
        help=(
            "also write a catalog-exact allowlist containing every validated "
            "export below --root"
        ),
    )

    search = subparsers.add_parser("search", help="search an existing JSON catalog")
    search.add_argument("--catalog", type=Path, required=True)
    search.add_argument("--root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    search.add_argument("--family")
    search.add_argument("--color")
    search.add_argument("--finish")
    search.add_argument("--query")
    search.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if args.command == "build":
        catalog = build_catalog(args.root, args.output)
        if args.full_allowlist_output is not None:
            args.full_allowlist_output.parent.mkdir(parents=True, exist_ok=True)
            args.full_allowlist_output.write_text(
                json.dumps(
                    catalog.to_full_allowlist_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "full_allowlist_output": (
                        str(args.full_allowlist_output)
                        if args.full_allowlist_output is not None
                        else None
                    ),
                    "count": len(catalog.materials),
                }
            )
        )
        return 0
    catalog = MaterialCatalog.load(args.catalog, material_root=args.root)
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


if __name__ == "__main__":
    raise SystemExit(main())
