#!/usr/bin/env python3

"""Convert CAD STEP/STP files to USD through the Omniverse CAD converter."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaac_sim_compat import get_simulation_app_class


CAD_SUFFIXES = {".stp", ".step"}


def log(*args):
    print(*args, flush=True)


def parse_args():
    parser = argparse.ArgumentParser("Convert STEP/STP CAD files to USD")
    parser.add_argument("--input", required=True, help="CAD file or directory")
    parser.add_argument(
        "--out-dir", required=True, help="output directory for converted USD files"
    )
    parser.add_argument(
        "--headless", action="store_true", help="run Isaac Sim headless"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing USD files"
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="CAD converter option; may be passed multiple times",
    )
    return parser.parse_args()


def parse_options(items: list[str]) -> dict[str, str]:
    options = {
        "bOptimize": "true",
        "instancing": "true",
    }
    for item in items:
        if "=" not in item:
            raise ValueError(f"converter option must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        options[key.strip()] = value.strip()
    return options


def find_cad_files(root: str) -> list[Path]:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in CAD_SUFFIXES else []
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in CAD_SUFFIXES
    )


def output_path_for(cad_file: Path, input_root: Path, out_root: Path) -> Path:
    if input_root.is_file():
        rel_stem = Path(cad_file.stem)
    else:
        rel_stem = cad_file.relative_to(input_root).with_suffix("")
    out_dir = out_root / rel_stem.parent / rel_stem.name
    return out_dir / f"{rel_stem.name}.usd"


def enable_cad_converter_extensions():
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    enable_extension = getattr(manager, "set_extension_enabled_immediate", None)
    if enable_extension is None:
        enable_extension = manager.set_extension_enabled
    for extension_name in (
        "omni.kit.converter.hoops_core",
        "omni.kit.converter.hoops",
    ):
        try:
            enable_extension(extension_name, True)
        except Exception as exc:
            log(f"[WARN] failed to enable {extension_name}: {exc}")


def get_hoops_converter():
    try:
        from omni.kit.converter.hoops_core import get_instance
    except Exception as exc:
        raise RuntimeError(
            "Omniverse HOOPS CAD converter is not available. "
            "Enable/install omni.kit.converter.hoops_core in this Isaac Sim/Kit installation."
        ) from exc
    return get_instance()


def run_async_task(coro, sim):
    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(coro)
    while not task.done():
        sim.update()
        time.sleep(0.01)
    return task.result()


def convert_one(
    converter,
    sim,
    cad_file: Path,
    usd_path: Path,
    options: dict[str, str],
    overwrite: bool,
):
    if usd_path.exists() and not overwrite:
        log(f"[Skip] exists: {usd_path}")
        return
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[CAD->USD] {cad_file} -> {usd_path}")
    result = run_async_task(
        converter.create_converter_task(str(cad_file), str(usd_path), options),
        sim,
    )
    if result is False:
        raise RuntimeError(f"CAD conversion returned false for: {cad_file}")
    if not usd_path.exists():
        candidates = sorted(usd_path.parent.glob("*.usd"))
        if candidates:
            log(f"[WARN] expected output missing, found USD file: {candidates[0]}")
        else:
            raise RuntimeError(f"CAD conversion did not create USD output: {usd_path}")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    cad_files = find_cad_files(args.input)
    if not cad_files:
        log(f"No STEP/STP files found in: {args.input}")
        return 3

    SimulationApp = get_simulation_app_class()
    sim = SimulationApp({"headless": args.headless})
    try:
        enable_cad_converter_extensions()
        converter = get_hoops_converter()
        options = parse_options(args.option)
        log(f"Found {len(cad_files)} CAD file(s)")
        for cad_file in cad_files:
            usd_path = output_path_for(cad_file, input_root, out_root)
            convert_one(converter, sim, cad_file, usd_path, options, args.overwrite)
        return 0
    finally:
        sim.close()
        log("Simulation closed.")


if __name__ == "__main__":
    sys.exit(main())
