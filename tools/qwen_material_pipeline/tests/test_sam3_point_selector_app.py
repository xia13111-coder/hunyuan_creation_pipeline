from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from qwen_material_pipeline.web.sam3_point_selector.app import (
    CLICK_IMAGE_CSS,
    CLICK_IMAGE_ELEM_ID,
    build_demo,
)


class _Session:
    view_order = ["front"]
    active_view_id = "front"

    def render_preview(self, *_args):
        return Image.new("RGB", (8, 8), "white")

    def render_mask(self, *_args):
        return Image.new("L", (8, 8), 0)

    def render_cutout(self, *_args):
        return Image.new("RGBA", (8, 8), (0, 0, 0, 0))

    def status(self, *_args):
        return "ok"

    def completion_status(self):
        return "0/1"

    def activate(self, *_args):
        return self.render_preview()

    def add_point(self, *_args, **_kwargs):
        return self.render_preview(), "ok"

    def undo_point(self):
        return self.render_preview(), "ok"

    def clear_view(self):
        return self.render_preview(), "ok"

    def cycle_active_candidate(self):
        return self.render_preview(), "ok"

    def remove_last_region(self):
        return self.render_preview(), "ok"

    def confirm_view(self):
        return self.render_preview(), "ok"

    def save(self):
        return Path("/tmp/annotations.json")


def test_click_image_is_locked_to_the_server_side_reference() -> None:
    pytest.importorskip("gradio")

    demo = build_demo(_Session())

    main_image = next(
        component
        for component in demo.config["components"]
        if component.get("type") == "image"
        and component["props"].get("label") == "点击这里；每点一次立即更新"
    )
    button_labels = {
        component["props"].get("value")
        for component in demo.config["components"]
        if component.get("type") == "button"
    }
    assert main_image["props"]["sources"] == []
    assert main_image["props"]["elem_id"] == CLICK_IMAGE_ELEM_ID
    assert demo.config["css"] == CLICK_IMAGE_CSS
    assert "撤销上一次分割编辑" in button_labels
    assert "结束当前实例（可选）" not in button_labels
