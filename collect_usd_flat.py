#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量对 USD 做 Collect Asset：
等价于 UI 里勾选：
  - USD Only
  - Materials Only
  - Flat Collection（纹理也全部平铺在一个目录）
用法示例：

./python.sh /home/user/ubtech/physic_setting/collect_usd_flat.py \
  --folder /home/user/hunyuan3.0_assets_creation/table_1 \
  --out-dir /home/user/hunyuan3.0_assets_creation/table_1_collect \
  --headless
"""

import os
import argparse

from omni.isaac.kit import SimulationApp

# ---------- 基础 ----------

def log(*a):
    print(*a, flush=True)

def parse_args():
    ap = argparse.ArgumentParser(
        description="批量 Collect USD（USD only + materials only + flat collection）"
    )
    ap.add_argument("--folder", required=True,
                    help="输入 USD 根目录（递归查找 .usd/.usda/.usdc）")
    ap.add_argument("--out-dir", required=True,
                    help="输出根目录（每个资产一个子目录）")
    ap.add_argument("--headless", action="store_true",
                    help="无界面运行（推荐勾上）")
    return ap.parse_args()

args = parse_args()

# 启动 Kit / Isaac
sim = SimulationApp({"headless": args.headless})

# 这里开始才能 import kit 相关的模块
import omni.usd  # noqa: E402
from omni.kit.usd.collect import Collector, FlatCollectionTextureOptions  # noqa: E402
from omni.kit.async_engine import run_coroutine  # noqa: E402


# ---------- 工具函数 ----------

def find_usd_files(root):
    usd_list = []
    for r, _, fs in os.walk(root):
        for n in fs:
            if n.lower().endswith((".usd", ".usda", ".usdc")):
                usd_list.append(os.path.join(r, n))
    return sorted(usd_list)

def build_collect_dir(usd_path: str) -> str:
    """
    根据原始 usd 路径构造一个输出目录。
    现在简单处理：用文件名的 stem 做子目录名。
    例如：
      /.../table_33_GLB_phys.usd
      -> /.../table_1_collect/table_33_GLB_phys
    """
    _, fname = os.path.split(usd_path)
    stem, _ = os.path.splitext(fname)
    return os.path.join(args.out_dir, stem)


def collect_one(usd_path: str):
    collect_dir = build_collect_dir(usd_path)
    os.makedirs(collect_dir, exist_ok=True)

    log(f"\n[Collect] {usd_path}")
    log(f"         → {collect_dir}")

    collector = Collector(
        usd_path=os.path.abspath(usd_path),
        collect_dir=os.path.abspath(collect_dir),
        # 对应 UI 勾选：
        #   - USD Only
        #   - Materials Only
        #   - Flat collection + Flat textures
        usd_only=True,
        material_only=True,
        flat_collection=True,
        texture_option=FlatCollectionTextureOptions.FLAT,
    )

    async def _do_collect():
        # 打印一下进度，防止看起来像卡死
        def on_progress(cur, total):
            log(f"    progress: {cur}/{total}")

        def on_finish():
            log("    finish callback")

        ok, root_usd = await collector.collect(
            progress_callback=on_progress,
            finish_callback=on_finish,
        )
        return ok, root_usd

    # 关键点：不要自己 asyncio.run！
    # 把协程丢给 omni.kit.async_engine 来调度
    task = run_coroutine(_do_collect())

    # 用 SimulationApp.update() 驱动 Kit 的事件循环，直到任务完成
    while not task.done():
        sim.update()

    ok, root_usd = task.result()
    collector.destroy()

    if ok:
        log(f"    ✓ success, collected root = {root_usd}")
    else:
        log(f"    ✗ collect failed for {usd_path}")


def main():
    usd_files = find_usd_files(args.folder)
    if not usd_files:
        log(f"在 {args.folder} 下没有找到 USD 文件")
        return

    log(f"在 {args.folder} 下找到 {len(usd_files)} 个 USD 文件")

    for p in usd_files:
        collect_one(p)


if __name__ == "__main__":
    try:
        main()
    finally:
        sim.close()
        log("Simulation closed.")
