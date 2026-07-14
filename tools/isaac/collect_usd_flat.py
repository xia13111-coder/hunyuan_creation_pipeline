

import os
import argparse

from isaac_sim_compat import get_simulation_app_class


SimulationApp = get_simulation_app_class()

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
import omni.usd  # noqa: E402,F401 - initializes the USD extension before collector imports
from omni.kit.usd.collect import Collector, FlatCollectionTextureOptions  # noqa: E402
from omni.kit.async_engine import run_coroutine  # noqa: E402


# ---------- 工具函数 ----------

def find_usd_files(root):
    if os.path.isfile(root):
        return [root] if root.lower().endswith((".usd", ".usda", ".usdc")) else []

    usd_list = []
    for r, _, fs in os.walk(root):
        for n in fs:
            if n.lower().endswith((".usd", ".usda", ".usdc")):
                usd_list.append(os.path.join(r, n))
    return sorted(usd_list)

def build_collect_dir(usd_path: str) -> str:

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
        usd_only=True,
        material_only=True,
        flat_collection=True,
        texture_option=FlatCollectionTextureOptions.FLAT,
    )

    async def _do_collect():
        def on_progress(cur, total):
            log(f"    progress: {cur}/{total}")

        def on_finish():
            log("    finish callback")

        ok, root_usd = await collector.collect(
            progress_callback=on_progress,
            finish_callback=on_finish,
        )
        return ok, root_usd

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
