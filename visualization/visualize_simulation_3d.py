"""
2D 地图仿真可视化：
- 强制显示路网地图（节点+边）
- EV 严格按路网轨迹移动（环境 shortest_path 决定）
- 清晰展示站点、目标、轨迹和实时状态

用法示例：
    python visualization/visualize_simulation_3d.py --policy dqn --steps 120
    python visualization/visualize_simulation_3d.py --policy heuristic --steps 100
    python visualization/visualize_simulation_3d.py --policy dqn --save results/sim2d_map.gif
    python visualization/visualize_simulation_3d.py --policy dqn --filter-long-edges --long-edge-quantile 0.995
"""

import os
import sys
import argparse
from typing import Dict, Tuple
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import networkx as nx

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.Traffic import TrafficPowerEnv
from env.RealTrafficEnv import RealTrafficEnv
from train import DQNAgent


def infer_2d_positions(graph):
    num_nodes = graph.number_of_nodes()
    size = int(np.sqrt(num_nodes))
    if size * size == num_nodes:
        return {i: (i % size, size - 1 - i // size) for i in range(num_nodes)}
    return nx.spring_layout(graph, seed=42)


def get_raw_positions(env) -> Dict[int, Tuple[float, float]]:
    if hasattr(env, "node_positions") and env.node_positions:
        return env.node_positions
    else:
        return infer_2d_positions(env.traffic_graph)


def build_position_normalizer(pos_raw: Dict[int, Tuple[float, float]]):
    xs = np.array([p[0] for p in pos_raw.values()], dtype=float)
    ys = np.array([p[1] for p in pos_raw.values()], dtype=float)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    margin = 0.04

    def normalize_xy(x, y):
        nx_ = (x - x_min) / x_span
        ny_ = (y - y_min) / y_span
        nx_ = margin + nx_ * (1.0 - 2.0 * margin)
        ny_ = margin + ny_ * (1.0 - 2.0 * margin)
        return nx_, ny_

    pos_norm = {n: normalize_xy(p[0], p[1]) for n, p in pos_raw.items()}
    return pos_norm, normalize_xy


def _extract_geometry_points(geom):
    if geom is None:
        return None
    if hasattr(geom, "coords"):
        return list(geom.coords)
    if hasattr(geom, "geoms"):
        pts = []
        for g in geom.geoms:
            if hasattr(g, "coords"):
                pts.extend(list(g.coords))
        return pts if len(pts) >= 2 else None
    if isinstance(geom, str):
        try:
            from shapely import wkt
            parsed = wkt.loads(geom)
            return _extract_geometry_points(parsed)
        except Exception:
            return None
    return None


def build_edge_segments(graph, pos_norm, normalize_xy, filter_long_edges=False, long_edge_quantile=0.995):
    # 先收集所有线段及其归一化长度，再按分位数过滤异常长线段（仅影响显示）。
    segments_with_len = []
    for u, v, d in graph.edges(data=True):
        points = _extract_geometry_points(d.get("geometry"))
        if points and len(points) >= 2:
            for p1, p2 in zip(points[:-1], points[1:]):
                x1, y1 = normalize_xy(float(p1[0]), float(p1[1]))
                x2, y2 = normalize_xy(float(p2[0]), float(p2[1]))
                seg_len = float(np.hypot(x2 - x1, y2 - y1))
                segments_with_len.append((((x1, y1), (x2, y2)), seg_len))
        else:
            if u in pos_norm and v in pos_norm:
                x1, y1 = pos_norm[u]
                x2, y2 = pos_norm[v]
                seg_len = float(np.hypot(x2 - x1, y2 - y1))
                segments_with_len.append((((x1, y1), (x2, y2)), seg_len))

    if not filter_long_edges or len(segments_with_len) < 10:
        return [seg for seg, _ in segments_with_len], 0, len(segments_with_len), None

    q = float(np.clip(long_edge_quantile, 0.90, 0.9999))
    lengths = np.array([l for _, l in segments_with_len], dtype=float)
    threshold = float(np.quantile(lengths, q))
    filtered = [seg for seg, l in segments_with_len if l <= threshold]
    removed = len(segments_with_len) - len(filtered)
    return filtered, removed, len(segments_with_len), threshold


def _build_edge_info_suffix(filter_long_edges, removed, total, threshold):
    if not filter_long_edges:
        return ""
    if threshold is None:
        return " | edge-filter=on (insufficient segments)"
    return f" | edge-filter=on removed={removed}/{total} thr={threshold:.4f}"


def choose_actions_heuristic(env):
    actions = {}
    for ev in env.get_pending_decision_evs():
        best_id = None
        best_score = None
        for s in env.stations:
            score = len(s.queue) + len(s.connected_evs) + s.current_price * 0.5
            if best_score is None or score < best_score:
                best_score = score
                best_id = s.id
        actions[ev.id] = best_id
    return actions


def choose_actions_random(env):
    actions = {}
    for ev in env.get_pending_decision_evs():
        actions[ev.id] = np.random.randint(0, len(env.stations))
    return actions


def choose_actions_dqn(env, agent):
    actions = {}
    urgent_evs = env.get_pending_decision_evs()
    pending_counts = {s.id: 0 for s in env.stations}

    for ev in urgent_evs:
        ev_state = env.get_graph_state_for_ev(ev, pending_counts)
        action = agent.select_action(ev_state)
        actions[ev.id] = action
        pending_counts[action] += 1
    return actions


def build_env(use_real_map=True, graphml_path=None, num_evs=20, max_nodes=9999):
    if use_real_map:
        if graphml_path is None:
            graphml_path = os.path.join(project_root, "zhujiang_new_town.graphml")

        if os.path.exists(graphml_path):
            print(f"[2D] 使用真实路网: {graphml_path}")
            return RealTrafficEnv(
                graphml_file=graphml_path,
                num_stations=2,
                num_evs=num_evs,
                max_nodes=max_nodes,
                seed=42,
            )

        print(f"[2D] 未找到 graphml，回退到基础网格: {graphml_path}")

    return TrafficPowerEnv()


def run_map_animation(
    steps=120,
    interval=250,
    policy="dqn",
    save_path=None,
    use_real_map=True,
    graphml_path=None,
    trail_len=20,
    max_nodes=9999,
    filter_long_edges=False,
    long_edge_quantile=0.995,
    show_edges=False,
    show_trails=False,
):
    env = build_env(use_real_map=use_real_map, graphml_path=graphml_path, max_nodes=max_nodes)
    graph = env.traffic_graph
    pos_raw = get_raw_positions(env)
    pos, normalize_xy = build_position_normalizer(pos_raw)

    station_id_to_node = {s.id: s.traffic_node_id for s in env.stations}

    agent = None
    if policy == "dqn":
        model_path = os.path.join(project_root, "checkpoints", "trained_dqn_real.pth")
        if not use_real_map:
            model_path = os.path.join(project_root, "checkpoints", "trained_dqn.pth")

        num_actions = getattr(env, "num_stations", 2)
        station_node_ids = getattr(env, "station_node_ids", [0, 8])
        num_nodes = getattr(env, "num_nodes", env.traffic_graph.number_of_nodes())

        agent = DQNAgent(
            num_features=15,
            num_actions=num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes,
        )
        if os.path.exists(model_path):
            agent.load_model(model_path)
            agent.epsilon = 0.02
            print(f"[2D] 使用 DQN 模型: {model_path}")
        else:
            print(f"[2D] 未找到模型，自动改用 heuristic: {model_path}")
            policy = "heuristic"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ev_trails = defaultdict(lambda: deque(maxlen=max(1, trail_len)))

    edge_collection = None
    edge_info_suffix = ""
    if show_edges:
        edges, removed_edges, total_edges, edge_threshold = build_edge_segments(
            graph,
            pos,
            normalize_xy,
            filter_long_edges=filter_long_edges,
            long_edge_quantile=long_edge_quantile,
        )
        if filter_long_edges:
            if edge_threshold is None:
                print("[2D] 长边过滤已开启，但线段数过少，未执行过滤。")
            else:
                print(
                    f"[2D] 长边过滤: 删除 {removed_edges}/{total_edges} 条线段 "
                    f"(quantile={long_edge_quantile:.4f}, threshold={edge_threshold:.4f})"
                )
        edge_collection = LineCollection(edges, colors="#9aa3ad", linewidths=1.4, alpha=0.75)
        edge_info_suffix = _build_edge_info_suffix(filter_long_edges, removed_edges, total_edges, edge_threshold)

    station_points = np.array([[pos[s.traffic_node_id][0], pos[s.traffic_node_id][1]] for s in env.stations])

    def draw_base_map():
        ax.cla()
        ax.set_facecolor("#f3f5f7")
        if edge_collection is not None:
            ax.add_collection(edge_collection)

        node_x = [pos[n][0] for n in graph.nodes()]
        node_y = [pos[n][1] for n in graph.nodes()]
        ax.scatter(node_x, node_y, s=8, c="#cbd3da", alpha=0.8, zorder=2)

        ax.scatter(
            station_points[:, 0],
            station_points[:, 1],
            s=260,
            marker="P",
            c="#21bf73",
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
            label="Charging Station",
        )

        for s in env.stations:
            sx, sy = pos[s.traffic_node_id]
            ax.text(sx, sy + 0.015, f"S{s.id}", color="#0f5132", fontsize=9, ha="center", zorder=6)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_frame(reward=0.0, info_text=""):
        draw_base_map()

        waiting = 0
        charging = 0
        moving = 0
        idle = 0

        for ev in env.evs:
            x, y = pos[ev.curr_node]
            traversed = getattr(ev, "last_traversed_nodes", None) or []
            if traversed:
                for node_id in traversed:
                    if node_id in pos:
                        tx, ty = pos[node_id]
                        ev_trails[ev.id].append((tx, ty))
            else:
                ev_trails[ev.id].append((x, y))

            if ev.status == "WAITING":
                waiting += 1
                color = "#e63946"
            elif ev.status == "CHARGING":
                charging += 1
                color = "#2a9d8f"
            elif ev.status == "MOVING_TO_CHARGE":
                moving += 1
                color = "#ff7f11"
            else:
                idle += 1
                color = "#457b9d"

            trail = ev_trails[ev.id]
            if show_trails and len(trail) >= 2:
                tx = [p[0] for p in trail]
                ty = [p[1] for p in trail]
                ax.plot(tx, ty, color="#6c757d", alpha=0.45, linewidth=1.0, zorder=3)

            ax.scatter([x], [y], s=70, c=color, edgecolors="white", linewidths=0.8, zorder=7)

            target_station = ev.assigned_station
            if target_station is None:
                target_station = ev.target_station_idx
            if target_station is not None and target_station in station_id_to_node:
                gx, gy = pos[station_id_to_node[target_station]]
                ax.scatter([gx], [gy], s=80, c="#00c853", marker="x", linewidths=2.0, zorder=8)

        for s in env.stations:
            sx, sy = pos[s.traffic_node_id]
            q = len(s.queue)
            c = len(s.connected_evs)
            ax.text(
                sx,
                sy - 0.022,
                f"Q:{q} C:{c} P:{s.current_price:.1f}",
                fontsize=8,
                color="#1b4332",
                ha="center",
                zorder=6,
            )

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#457b9d", markersize=7, label="Idle"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f11", markersize=7, label="Moving"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e63946", markersize=7, label="Waiting"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a9d8f", markersize=7, label="Charging"),
            plt.Line2D([0], [0], marker="x", color="#00c853", markersize=7, label="Goal"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.95)

        ax.set_title(
            f"2D Map EV Simulation | Step {env.time_step:03d} | Reward {reward:.1f}\n"
            f"Policy={policy} | Idle={idle} Moving={moving} Waiting={waiting} Charging={charging}\n"
            f"{info_text}{edge_info_suffix}",
            fontsize=11,
            pad=10,
        )

    def update(_frame_idx):
        if policy == "dqn" and agent is not None:
            actions = choose_actions_dqn(env, agent)
        elif policy == "random":
            actions = choose_actions_random(env)
        else:
            actions = choose_actions_heuristic(env)

        _, reward, _, info = env.step(actions)
        grid_loads = info.get("grid_loads", {})
        info_text = f"GridLoads={{{', '.join([f'{k}:{v:.1f}' for k, v in grid_loads.items()])}}}"
        draw_frame(reward=reward, info_text=info_text)
        return []

    draw_frame(reward=0.0, info_text="Warmup")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False,
    )

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        ani.save(save_path, dpi=140)
        print(f"[2D] 已保存动画: {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="2D 地图仿真可视化")
    parser.add_argument("--steps", type=int, default=120, help="动画步数")
    parser.add_argument("--interval", type=int, default=250, help="帧间隔 ms")
    parser.add_argument("--policy", choices=["dqn", "heuristic", "random"], default="dqn")
    parser.add_argument("--save", type=str, default=None, help="保存路径，如 results/sim2d_map.gif")
    parser.add_argument("--grid", action="store_true", help="使用基础网格环境（默认真实路网）")
    parser.add_argument("--graphml", type=str, default=None, help="真实路网 graphml 文件路径")
    parser.add_argument("--trail-len", type=int, default=20, help="轨迹尾迹长度（帧）")
    parser.add_argument("--max-nodes", type=int, default=9999, help="路网节点上限（默认 9999 = 使用全图）")
    parser.add_argument("--filter-long-edges", action="store_true", help="过滤可视化中的超长异常边（仅显示层）")
    parser.add_argument("--long-edge-quantile", type=float, default=0.995, help="超长边阈值分位数，默认 0.995")
    parser.add_argument("--show-edges", action="store_true", help="显示路网边线（默认不显示）")
    parser.add_argument("--show-trails", action="store_true", help="显示车辆轨迹线（默认不显示）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_map_animation(
        steps=args.steps,
        interval=args.interval,
        policy=args.policy,
        save_path=args.save,
        use_real_map=not args.grid,
        graphml_path=args.graphml,
        trail_len=args.trail_len,
        max_nodes=args.max_nodes,
        filter_long_edges=args.filter_long_edges,
        long_edge_quantile=args.long_edge_quantile,
        show_edges=args.show_edges,
        show_trails=args.show_trails,
    )
