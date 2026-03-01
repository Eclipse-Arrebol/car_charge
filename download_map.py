"""
地图下载工具 —— 生成供训练使用的真实路网 GraphML 文件
=======================================================
基于可用的代理 + OSMnx，下载指定区域主干道，合并复杂路口，
导出 .graphml 文件后在 run_training_real_map.py 中用 LOCAL_GRAPHML 加载。

用法:
    cd G:\交电demo
    python download_map.py

完成后把下面的路径填入 visualization/run_training_real_map.py:
    LOCAL_GRAPHML = "zhujiang_new_town.graphml"
"""

import os
import sys
import osmnx as ox
import matplotlib
matplotlib.use("TkAgg")          # Windows 下推荐；若弹不出窗口改为 "Agg"
import matplotlib.pyplot as plt

# ============================================================
# ★ 在这里修改目标区域 ★
# ============================================================
CENTER_POINT   = (23.1193, 113.3213)   # (纬度, 经度) — 广州珠江新城
RADIUS_METERS  = 3000                  # 抓取半径（米）
CONSOLIDATE_TOL = 15                   # 路口合并容差（米），越大节点越少
OUTPUT_FILE    = "zhujiang_new_town.graphml"

# 代理配置（与你测试可用的脚本保持一致）
PROXY = "http://127.0.0.1:7890"

# 道路等级过滤（只保留主干/次干/支路，排除小区内道路）
CUSTOM_FILTER = '["highway"~"primary|secondary|tertiary"]'
# ============================================================


def download():
    # ── 配置 osmnx ──
    ox.settings.timeout = 600
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_kwargs = {
        "proxies": {"http": PROXY, "https": PROXY}
    }

    print(f"正在抓取路网: 中心 {CENTER_POINT}, 半径 {RADIUS_METERS}m ...")
    try:
        G = ox.graph_from_point(
            CENTER_POINT,
            dist=RADIUS_METERS,
            custom_filter=CUSTOM_FILTER,
        )
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请确认代理 {PROXY} 可用且网络畅通。")
        sys.exit(1)

    print("下载成功！正在投影到平面坐标系...")
    G_proj = ox.project_graph(G)

    # ── 路口合并 & 去除死胡同 ──
    print(f"合并前节点数: {len(G_proj.nodes())}")
    G_proj = ox.consolidate_intersections(
        G_proj, rebuild_graph=True,
        tolerance=CONSOLIDATE_TOL, dead_ends=False
    )
    print(f"合并后节点数: {len(G_proj.nodes())} (大幅精简！)")

    # ── 补充速度/时间属性（可选，用于可视化和后续分析）──
    G_proj = ox.add_edge_speeds(G_proj)
    G_proj = ox.add_edge_travel_times(G_proj)

    # ── 导出 GraphML ──
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    ox.save_graphml(G_proj, filepath=out_path)
    print(f"\n路网已导出: {out_path}")
    print(f"节点数: {len(G_proj.nodes())}, 边数: {len(G_proj.edges())}")

    print("\n" + "=" * 55)
    print("下一步：在 visualization/run_training_real_map.py 中设置")
    print(f'    LOCAL_GRAPHML = r"{out_path}"')
    print("=" * 55)

    # ── 可视化预览 ──
    try:
        fig, ax = ox.plot_graph(
            G_proj,
            node_size=15,
            node_color="yellow",
            edge_color="cyan",
            edge_linewidth=1.0,
            bgcolor="black",
            show=False,
            close=False,
        )
        plt.title(f"Road Network ({len(G_proj.nodes())} nodes)", color="white", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[可视化跳过] {e}")


if __name__ == "__main__":
    download()
