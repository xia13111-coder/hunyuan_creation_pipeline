#!/usr/bin/env python3
"""Launch a local click-by-click SAM3 foreground annotation website."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from qwen_material_pipeline.segmentation.human_foreground import (
    HumanForegroundError,
    InteractiveForegroundSession,
    parse_reference_specs,
)


# Rebound to gradio.SelectData inside ``build_demo``.  Keeping the name here
# lets this lightweight command module be imported before Gradio is required.
SelectData = Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SAM3_REPOSITORY = Path(
    os.getenv(
        "SAM3_REPOSITORY",
        str(
            PROJECT_ROOT
            / "tools"
            / "sam3d"
            / "third_party"
            / "sam-3d-objects"
            / "submodules"
            / "sam3"
        ),
    )
)
DEFAULT_SAM3_CHECKPOINT = Path(
    os.getenv(
        "SAM3_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "tools"
            / "sam3d"
            / "third_party"
            / "sam-3d-objects"
            / "checkpoints"
            / "sam3.pt"
        ),
    )
)
CLICK_IMAGE_ELEM_ID = "sam3-click-image"
# Gradio 5.49 computes SelectData coordinates as if the preview used
# ``object-fit: contain``, while its default ``scale-down`` leaves small images
# unscaled inside a larger component. Keep the painted and measured rectangles
# identical so clicks remain in source-image pixel coordinates.
CLICK_IMAGE_CSS = f"#{CLICK_IMAGE_ELEM_ID} img {{ object-fit: contain !important; }}"


def build_demo(session: InteractiveForegroundSession) -> Any:
    """Build a single-user Gradio UI around one persistent SAM3 session."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError(
            "SAM3 interactive UI requires gradio==5.49.0 in hunyuan_sam3d"
        ) from exc
    global SelectData
    SelectData = gr.SelectData

    def safe_result(action, *, include_completion: bool = True):
        try:
            preview, status = action()
            result = [
                preview,
                session.render_mask(),
                session.render_cutout(),
                status,
            ]
            if include_completion:
                result.append(session.completion_status())
            return tuple(result)
        except (HumanForegroundError, OSError, RuntimeError) as exc:
            result = [
                session.render_preview(),
                session.render_mask(),
                session.render_cutout(),
                f"❌ {exc}",
            ]
            if include_completion:
                result.append(session.completion_status())
            return tuple(result)

    def switch_view(view_id: str):
        return safe_result(
            lambda: (
                session.activate(view_id),
                session.status(view_id),
            )
        )

    def click_image(mode: str, clicked_view_id: str, evt: SelectData):
        if not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
            return safe_result(lambda: (_raise("无法读取点击坐标"), ""))
        x, y = float(evt.index[0]), float(evt.index[1])
        mode_by_label = {
            "智能前景（点外新增，点内修正）": "smart_foreground",
            "修正当前实例": "refine_active",
            "强制新增零件": "new_instance",
            "背景排除": "background",
        }

        def add_selected_point():
            interaction_mode = mode_by_label.get(mode)
            if interaction_mode is None:
                raise HumanForegroundError("点击模式无效，请重新选择")
            if clicked_view_id not in session.view_order:
                raise HumanForegroundError("点击对应的参考视角无效，请重新选择")
            if session.active_view_id != clicked_view_id:
                session.activate(clicked_view_id)
            return session.add_point(x, y, mode=interaction_mode)

        return safe_result(add_selected_point)

    def save_all():
        try:
            output = session.save()
        except (HumanForegroundError, OSError, RuntimeError) as exc:
            return f"❌ {exc}", None
        return (
            f"✅ 已保存。请同时保留 {output.name} 和同目录的 "
            f"{output.stem}_masks/；正式 STEP/STP 命令加入："
            f"--visual-foreground-annotations {output}",
            str(output),
        )

    def _raise(message: str):
        raise HumanForegroundError(message)

    with gr.Blocks(title="SAM3 人工前景点选", css=CLICK_IMAGE_CSS) as demo:
        gr.Markdown(
            "# SAM3 增量实例点选\n" "每次点击都会立即分割并更新总并集。洋红色是总边界，黄色是当前已通过实例，" "红色是仍需修正的实例。"
        )
        with gr.Row():
            view = gr.Radio(
                choices=session.view_order,
                value=session.active_view_id,
                label="参考视角",
                interactive=True,
            )
            mode = gr.Radio(
                choices=[
                    "智能前景（点外新增，点内修正）",
                    "修正当前实例",
                    "强制新增零件",
                    "背景排除",
                ],
                value="智能前景（点外新增，点内修正）",
                label="点击模式",
                interactive=True,
            )
        with gr.Row():
            image = gr.Image(
                value=session.render_preview(),
                label="点击这里；每点一次立即更新",
                type="pil",
                interactive=True,
                sources=[],
                elem_id=CLICK_IMAGE_ELEM_ID,
                height=680,
            )
            with gr.Column():
                mask = gr.Image(
                    value=session.render_mask(),
                    label="黑白总 Mask",
                    type="pil",
                    interactive=False,
                    height=330,
                )
                cutout = gr.Image(
                    value=session.render_cutout(),
                    label="透明背景抠图",
                    type="pil",
                    interactive=False,
                    height=330,
                )
        status = gr.Textbox(
            value=session.status(),
            label="当前视角状态",
            interactive=False,
        )
        completion = gr.Textbox(
            value=session.completion_status(),
            label="整体进度",
            interactive=False,
        )
        with gr.Row():
            undo = gr.Button("撤销上一次分割编辑")
            cycle_candidate = gr.Button("切换当前首次候选")
            remove_region = gr.Button("删除当前/最后实例")
            clear = gr.Button("清空当前视角")
        with gr.Row():
            confirm_view = gr.Button("确认当前视角完整", variant="primary")
            save = gr.Button("四个视角均满意，保存标注", variant="stop")
        saved_file = gr.File(label="已保存的标注 JSON", interactive=False)
        gr.Markdown(
            "默认使用“智能前景”：点在已有 mask 外会自动新增零件，点在 mask 内会细化该"
            "实例。若要扩展当前实例到 mask 外，请选“修正当前实例”；误选区域使用“背景排除”。"
        )

        outputs = [image, mask, cutout, status, completion]
        view.change(switch_view, [view], outputs)
        image.select(click_image, [mode, view], outputs)
        undo.click(
            lambda: safe_result(session.undo_point),
            outputs=outputs,
        )
        clear.click(
            lambda: safe_result(session.clear_view),
            outputs=outputs,
        )
        cycle_candidate.click(
            lambda: safe_result(session.cycle_active_candidate),
            outputs=outputs,
        )
        remove_region.click(
            lambda: safe_result(session.remove_last_region),
            outputs=outputs,
        )
        confirm_view.click(
            lambda: safe_result(session.confirm_view),
            outputs=outputs,
        )
        save.click(save_all, outputs=[status, saved_file])
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        metavar="[ID=]IMAGE",
        help="same-workpiece reference image; repeat 2..4 times",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="immutable annotation JSON written after all views are confirmed",
    )
    parser.add_argument("--repository", type=Path, default=DEFAULT_SAM3_REPOSITORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--minimum-model-score", type=float, default=0.45)
    parser.add_argument("--minimum-prompt-agreement", type=float, default=0.25)
    # Keep the interactive acceptance gate identical to the formal
    # whole-workpiece foreground runner's default.  A mask that can be saved
    # here must not become unusable merely because the long job starts later.
    parser.add_argument("--maximum-image-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-mask-pixels", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.minimum_model_score <= 1.0:
        raise ValueError("--minimum-model-score must be between zero and one")
    if not 0.0 <= args.minimum_prompt_agreement <= 1.0:
        raise ValueError("--minimum-prompt-agreement must be between zero and one")
    if not 0.0 < args.maximum_image_fraction <= 1.0:
        raise ValueError("--maximum-image-fraction must be within (0,1]")
    if args.minimum_mask_pixels < 1:
        raise ValueError("--minimum-mask-pixels must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be within 1..65535")
    references = parse_reference_specs(args.reference)
    print(
        "Loading local SAM3 interactive predictor; the first load may take a while...",
        flush=True,
    )
    session = InteractiveForegroundSession(
        references=references,
        repository=args.repository,
        checkpoint=args.checkpoint,
        output=args.output,
        device=args.device,
        minimum_model_score=args.minimum_model_score,
        minimum_prompt_agreement=args.minimum_prompt_agreement,
        maximum_image_fraction=args.maximum_image_fraction,
        minimum_mask_pixels=args.minimum_mask_pixels,
        overwrite=args.overwrite,
    )
    demo = build_demo(session)
    print(
        f"SAM3 interactive UI: http://{args.host}:{args.port}",
        flush=True,
    )
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
