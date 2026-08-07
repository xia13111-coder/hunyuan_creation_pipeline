"""End-to-end orchestration for hand-modelled STEP/STP assets."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from .command import LogCallback
from .jobs.cad import run_cad_to_usd_job
from .jobs.delivery import run_validate_visual_material_delivery_job
from .jobs.isaac import run_add_physics_job, run_collect_job
from .paths import (
    cad_usd_output_path,
    collected_root_usd,
    mirrored_output_parent,
    suffixed_file_path,
)
from .progress import emit_progress
from .visual_materials import (
    run_assign_visual_materials_job,
    run_final_visual_acceptance_job,
)


DEFAULT_MANUAL_SDF_RESOLUTION = 32
_PROGRESS_SCOPE = "manual_cad"
_RESTORED_HISTORICAL_BASELINE = "RESTORED_HISTORICAL_BASELINE"


def _require_publishable_visual_material_result(result: dict) -> None:
    """Reject a known-bad live Look before it can be collected as final output."""

    inference_mode = str(result.get("inference_mode") or "").strip()
    quality_status = str(result.get("visual_quality_status") or "").strip()
    if (
        inference_mode == "bundled_project"
        or quality_status == _RESTORED_HISTORICAL_BASELINE
    ):
        return

    gate_status = result.get("visual_quality_gate_status")
    normalized_gate_status = (
        str(gate_status).strip().upper() if gate_status is not None else ""
    )
    if normalized_gate_status != "PASS":
        raise RuntimeError(
            "Visual material quality gate did not authorize collection: "
            f"visual_quality_gate_status={gate_status!r}. "
            "The final output directory was not published."
        )


class _WorkflowProgress:
    """Report truthful stage boundaries without inventing duration estimates."""

    def __init__(self, log_cb: LogCallback, stages: Sequence[str]) -> None:
        self._log_cb = log_cb
        self._stages = tuple(stages)
        self._completed = 0

    @contextmanager
    def stage(self, name: str, detail: str) -> Iterator[None]:
        if self._stages[self._completed] != name:
            raise RuntimeError(f"Unexpected manual CAD progress stage: {name}")
        emit_progress(
            self._log_cb,
            scope=_PROGRESS_SCOPE,
            stage=name,
            state="start",
            current=self._completed,
            total=len(self._stages),
            unit="stage",
            detail=detail,
        )
        try:
            yield
        except BaseException as exc:
            emit_progress(
                self._log_cb,
                scope=_PROGRESS_SCOPE,
                stage=name,
                state="failed",
                current=self._completed,
                total=len(self._stages),
                unit="stage",
                detail=f"{detail}: {type(exc).__name__}: {exc}",
            )
            raise
        self._completed += 1
        emit_progress(
            self._log_cb,
            scope=_PROGRESS_SCOPE,
            stage=name,
            state="complete",
            current=self._completed,
            total=len(self._stages),
            unit="stage",
            detail=detail,
        )


def run_manual_cad_workflow(
    *,
    input_path: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    cad_usd_output_dir: str | None = None,
    cad_converter_options: Sequence[str] = (),
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "sdf",
    sdf_resolution: int = DEFAULT_MANUAL_SDF_RESOLUTION,
    headless: bool = True,
    auto_visual_materials: bool = False,
    visual_material_references: Sequence[str] = (),
    visual_material_output_dir: str | None = None,
    visual_material_config: str | None = None,
    visual_foreground_annotations: str | None = None,
    visual_inference_mode: str = "live",
    acknowledge_mvinverse_noncommercial: bool = False,
    allow_policy_material_fallback: bool = False,
    resume: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    """Run STEP/STP conversion, Physics, optional visuals, and collection.

    CAD geometry must be normalized before selecting procedural NVIDIA MDLs.
    Several library materials evaluate object-space coordinates, so changing
    units or mesh-local origins after selection changes their appearance even
    when the MDL identity and parameters remain untouched.
    """

    if sdf_resolution <= 0:
        raise ValueError("sdf_resolution must be greater than zero")
    if not isinstance(allow_policy_material_fallback, bool):
        raise TypeError("allow_policy_material_fallback must be a boolean")
    if allow_policy_material_fallback and not auto_visual_materials:
        raise ValueError(
            "allow_policy_material_fallback requires auto_visual_materials=True"
        )
    if visual_foreground_annotations is not None and not auto_visual_materials:
        raise ValueError(
            "visual_foreground_annotations requires auto_visual_materials=True"
        )
    if visual_foreground_annotations is not None and visual_inference_mode != "live":
        raise ValueError(
            "visual_foreground_annotations requires visual_inference_mode='live'"
        )

    progress_stages = ["cad", "physics"]
    if auto_visual_materials:
        progress_stages.append("visual")
    progress_stages.append("collect")
    if auto_visual_materials:
        progress_stages.extend(["delivery_validation", "final_acceptance"])
    progress = _WorkflowProgress(log_cb, progress_stages)

    with progress.stage("cad", "Convert CAD geometry to USD"):
        if resume:
            if cad_usd_output_dir is None:
                raise ValueError(
                    "Manual CAD resume requires an explicit cad_usd_output_dir"
                )
            from .jobs.cad import validate_cad_input_path

            cad_files = validate_cad_input_path(
                input_path,
                require_single=auto_visual_materials,
            )
            resumed_usd_files = [
                cad_usd_output_path(cad_file, input_path, cad_usd_output_dir)
                for cad_file in cad_files
            ]
            missing_usd_files = [
                path for path in resumed_usd_files if not Path(path).is_file()
            ]
            if missing_usd_files:
                raise FileNotFoundError(
                    "Manual CAD resume has no verified converted USD for: "
                    + ", ".join(missing_usd_files)
                )
            cad_result = {
                "input_path": input_path,
                "out_dir": cad_usd_output_dir,
                "overwrite": False,
                "converter_options": list(cad_converter_options),
                "cad_files": cad_files,
                "usd_files": resumed_usd_files,
                "resumed": True,
            }
            if log_cb is not None:
                log_cb("Reusing existing CAD-to-USD checkpoint for manual resume.")
        else:
            cad_result = run_cad_to_usd_job(
                input_path=input_path,
                out_dir=cad_usd_output_dir,
                overwrite=True,
                headless=headless,
                converter_options=cad_converter_options,
                require_single=auto_visual_materials,
                log_cb=log_cb,
            )
        if auto_visual_materials and len(cad_result["usd_files"]) != 1:
            raise RuntimeError(
                "CAD Converter result violated the single-asset material contract: "
                f"found {len(cad_result['usd_files'])} USD files"
            )
    steps = [{"step": "cad_to_usd", "result": cad_result}]

    physics_results = []
    visual_material_results = []
    visual_material_input_files: list[str] = []
    collect_results = []
    delivery_validation_results = []
    final_visual_acceptance_results = []
    physics_usd_files: list[str] = []
    with progress.stage("physics", "Author collision and rigid-body physics"):
        for usd_file in cad_result["usd_files"]:
            physics_out_dir = mirrored_output_parent(
                usd_file,
                cad_result["out_dir"],
                intermediate_output_dir,
            )
            physics_usd_file = suffixed_file_path(usd_file, physics_out_dir, "_phys")
            if resume:
                if not Path(physics_usd_file).is_file():
                    raise FileNotFoundError(
                        "Manual CAD resume has no verified physics USD for: "
                        f"{physics_usd_file}"
                    )
                physics_result = {"resumed": True}
                if log_cb is not None:
                    log_cb(f"Reusing existing physics checkpoint: {physics_usd_file}")
            else:
                physics_result = run_add_physics_job(
                    folder=usd_file,
                    out_dir=physics_out_dir,
                    material_file=material_file,
                    material=material,
                    set_mass=set_mass,
                    approx=approx,
                    sdf_resolution=sdf_resolution,
                    sdf_remesh=str(approx).strip().lower() == "sdf",
                    center_origin=True,
                    headless=headless,
                    log_cb=log_cb,
                )
            if not Path(physics_usd_file).exists():
                raise FileNotFoundError(
                    f"Physics job did not create expected USD file: {physics_usd_file}"
                )
            physics_result["source_cad_usd_file"] = usd_file
            physics_result["input_usd_file"] = usd_file
            physics_result["output_usd_file"] = physics_usd_file
            physics_results.append(physics_result)
            physics_usd_files.append(physics_usd_file)

    delivery_inputs = list(physics_usd_files)
    if auto_visual_materials:
        with progress.stage("visual", "Assign reference-driven visual materials"):
            for physics_usd_file in physics_usd_files:
                visual_material_result = run_assign_visual_materials_job(
                    source_usd=physics_usd_file,
                    source_cad=cad_result["cad_files"][0],
                    references=visual_material_references,
                    foreground_annotations=visual_foreground_annotations,
                    output_dir=visual_material_output_dir,
                    config_path=visual_material_config,
                    inference_mode=visual_inference_mode,
                    acknowledge_mvinverse_noncommercial=(
                        acknowledge_mvinverse_noncommercial
                    ),
                    allow_policy_material_fallback=allow_policy_material_fallback,
                    require_complete_coverage=True,
                    log_cb=log_cb,
                )
                _require_publishable_visual_material_result(visual_material_result)
                visual_material_results.append(visual_material_result)
                delivery_inputs[len(visual_material_results) - 1] = (
                    visual_material_result["effective_usd"]
                )
                visual_material_input_files.append(physics_usd_file)

    with progress.stage("collect", "Collect USD and referenced dependencies"):
        for delivery_input in delivery_inputs:
            collect_result = run_collect_job(
                folder=delivery_input,
                out_dir=final_output_dir,
                headless=headless,
                log_cb=log_cb,
            )
            collect_result["input_usd_file"] = delivery_input
            if auto_visual_materials:
                collect_result["collected_root_usd"] = collected_root_usd(
                    delivery_input, final_output_dir
                )
            collect_results.append(collect_result)

    if auto_visual_materials:
        with progress.stage(
            "delivery_validation", "Validate collected visual-material delivery"
        ):
            for index, visual_material_result in enumerate(visual_material_results):
                delivery_input = delivery_inputs[index]
                collected_root = collect_results[index]["collected_root_usd"]
                delivery_validation = run_validate_visual_material_delivery_job(
                    look_usd=delivery_input,
                    physics_usd=delivery_input,
                    collected_root_usd=collected_root,
                    registry=visual_material_result["rendered_registry"],
                    apply_report=visual_material_result["apply_report"],
                    bundle_root=str(Path(collected_root).parent),
                    output=str(
                        Path(visual_material_result["output_dir"])
                        / "delivery_validation.json"
                    ),
                    log_cb=log_cb,
                )
                delivery_validation_results.append(delivery_validation)

        with progress.stage(
            "final_acceptance", "Run final collected visual acceptance"
        ):
            for index, visual_material_result in enumerate(visual_material_results):
                collected_root = collect_results[index]["collected_root_usd"]
                final_visual_acceptance = run_final_visual_acceptance_job(
                    collected_usd=collected_root,
                    visual_material_result=visual_material_result,
                    log_cb=log_cb,
                )
                if (
                    final_visual_acceptance.get("state") != "COMPLETED"
                    or final_visual_acceptance.get("completion_allowed") is not True
                ):
                    raise RuntimeError(
                        "Final collected visual acceptance did not authorize "
                        "pipeline completion"
                    )
                final_visual_acceptance_results.append(final_visual_acceptance)
                collect_results[index]["final_visual_acceptance"] = (
                    final_visual_acceptance
                )

    steps.append({"step": "add_physics", "result": {"jobs": physics_results}})
    if visual_material_results:
        steps.append(
            {
                "step": "assign_visual_materials",
                "result": visual_material_results[0],
            }
        )
    steps.append({"step": "collect_usd", "result": {"jobs": collect_results}})
    if delivery_validation_results:
        steps.append(
            {
                "step": "validate_visual_material_delivery",
                "result": {"jobs": delivery_validation_results},
            }
        )
    if final_visual_acceptance_results:
        steps.append(
            {
                "step": "final_visual_acceptance",
                "result": {"jobs": final_visual_acceptance_results},
            }
        )
    return {
        "workflow": "manual_cad",
        "input_path": input_path,
        "cad_usd_output_dir": cad_result["out_dir"],
        "intermediate_output_dir": intermediate_output_dir,
        "final_output_dir": final_output_dir,
        "processed_cad_files": cad_result["cad_files"],
        "processed_usd_files": cad_result["usd_files"],
        "physics_input_files": list(cad_result["usd_files"]),
        "visual_material_input_files": visual_material_input_files,
        "visual_material_output_dir": (
            visual_material_results[0]["output_dir"]
            if visual_material_results
            else None
        ),
        "visual_material_delivery_validation": delivery_validation_results,
        "visual_material_final_acceptance": final_visual_acceptance_results,
        "completion_state": ("COMPLETED" if final_visual_acceptance_results else None),
        # Kept for compatibility with older consumers of the unified result shape.
        "processed_glb_files": [],
        "sdf_resolution": sdf_resolution,
        "steps": steps,
    }


# Backward-compatible public names used by earlier integrations.
run_manual_cad_job = run_manual_cad_workflow
run_stp_physics_job = run_manual_cad_workflow


__all__ = [
    "DEFAULT_MANUAL_SDF_RESOLUTION",
    "run_manual_cad_job",
    "run_manual_cad_workflow",
    "run_stp_physics_job",
]
