"""
基于 OSMnx 真实路网的训练入口
================================
用法:
    cd G:\\交电demo
    python visualization/run_training_real_map.py

获取真实路网 → 先运行 python download_map.py，再将生成的 .graphml 路径填入 LOCAL_GRAPHML。
若无法联网，设 OFFLINE_FALLBACK = True 使用合成路网。
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import networkx as nx
from simEvn.RealTrafficEnv import RealTrafficEnv
from train import DQNAgent
from visualization.visualize_training import TrainingVisualizer


# ============================================================
# ★ 在这里选择路网来源（三选一）★
# ============================================================

# 方式 B：本地 GraphML 文件（填写路径后优先使用）
# 先运行项目根目录下的 download_map.py 生成文件，再填入路径
LOCAL_GRAPHML = r"G:\交电demo\zhujiang_new_town.graphml"         # 例: r"G:\交电demo\zhujiang_new_town.graphml"

# 方式 C：直接使用离线合成路网（True = 跳过所有网络请求）
OFFLINE_FALLBACK = False

# 方式 A：自动联网时的 OSM 地名（LOCAL_GRAPHML=None 且 OFFLINE_FALLBACK=False 时生效）
PLACE     = "Wuchang District, Wuhan, China"   # OSM 地名

# ── 路网规模 ──
MAX_NODES = 40      # 路网精简节点数（与评估环境保持一致）
NUM_EVS   = 20      # EV 数量（与评估环境保持一致）
NUM_STATS = 2       # 充电站数量

EPISODES         = 200  # 增加训练轮次
STEPS_PER_EP     = 100
BATCH_SIZE       = 64
# ============================================================


def run_training_real():
    # ── 根据配置选择路网来源 ──
    if LOCAL_GRAPHML is not None:
        print(f"[模式 B] 使用本地 GraphML 文件: {LOCAL_GRAPHML}")
        env = RealTrafficEnv(
            graphml_file=LOCAL_GRAPHML,
            num_stations=NUM_STATS,
            num_evs=NUM_EVS,
            max_nodes=MAX_NODES,
            seed=42,
        )
    elif OFFLINE_FALLBACK:
        print("[模式 C] 使用离线合成路网（Watts-Strogatz）")
        env = RealTrafficEnv(
            place=PLACE,
            num_stations=NUM_STATS,
            num_evs=NUM_EVS,
            max_nodes=MAX_NODES,
            seed=42,
            offline=True,
        )
    else:
        print(f"[模式 A] 自动联网下载路网: {PLACE}")
        env = RealTrafficEnv(
            place=PLACE,
            num_stations=NUM_STATS,
            num_evs=NUM_EVS,
            max_nodes=MAX_NODES,
            seed=42,
        )

    # ── DQNAgent 适配真实路网的节点数和站点 ID ──
    agent = DQNAgent(
        num_features=10,
        num_actions=env.num_stations,
        station_node_ids=env.station_node_ids,
        num_nodes_per_graph=env.num_nodes,
    )

    viz = TrainingVisualizer(save_dir="visualization/results_real")

    network_desc = (LOCAL_GRAPHML or ("offline-synthetic" if OFFLINE_FALLBACK else PLACE))
    print(f"\nStart training (network: {network_desc})")
    print(f"nodes={env.num_nodes}, stations={env.station_node_ids}, EVs={NUM_EVS}\n")

    for e in range(EPISODES):
        env.reset()
        total_reward = 0.0
        avg_queue_sum = 0.0
        overload_count = 0
        total_realized = 0.0

        for _ in range(STEPS_PER_EP):

            # --- A. 顺序决策：利用率导向奖励 + per-EV 状态 ---
            urgent_evs = [ev for ev in env.evs
                          if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)

            actions = {}
            ev_dispatch = []       # (ev, ev_state, action, shaped_reward)
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                action = agent.select_action(ev_state)
                actions[ev.id] = action

                # ── 利用率导向的塑形奖励 ──
                target_st = env.stations[action]
                other_st  = env.stations[1 - action]
                eff_target = (len(target_st.queue) + len(target_st.connected_evs)
                              + pending_counts.get(action, 0))
                eff_other  = (len(other_st.queue) + len(other_st.connected_evs)
                              + pending_counts.get(1 - action, 0))

                try:
                    dist_target = nx.shortest_path_length(
                        env.traffic_graph, ev.curr_node, target_st.traffic_node_id)
                except nx.NetworkXNoPath:
                    dist_target = 5
                try:
                    dist_other = nx.shortest_path_length(
                        env.traffic_graph, ev.curr_node, other_st.traffic_node_id)
                except nx.NetworkXNoPath:
                    dist_other = 5

                max_dist = max(dist_target + dist_other, 1)
                norm_dist_target = dist_target / max_dist * 5.0

                util_target = eff_target / max(1, target_st.num_chargers)
                util_other  = eff_other  / max(1, other_st.num_chargers)

                per_ev_r = -(util_target - util_other) * 20.0
                per_ev_r -= norm_dist_target * 1.5
                per_ev_r += (other_st.current_price - target_st.current_price) * 1.0
                if util_target >= 1.0 and util_other < 1.0:
                    per_ev_r -= 30.0

                ev_dispatch.append((ev, ev_state, action, per_ev_r))
                pending_counts[action] += 1

            next_state, reward, _, info = env.step(actions)

            # 经验存储：塑形奖励 + 全局奖励混合
            global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
            for ev, ev_state, act, per_ev_r in ev_dispatch:
                mixed_r = per_ev_r + global_bonus
                next_ev_state = env.get_graph_state_for_ev(ev)
                agent.store_transition(ev_state, act, mixed_r, next_ev_state)

            if ev_dispatch or len(agent.memory) >= BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            total_reward   += reward
            total_realized += info.get("realized_power", 0.0)
            avg_queue_sum  += sum(len(s.queue) for s in env.stations) / env.num_stations
            overload_count += int(info.get("voltage_violations", 0))

        avg_queue = avg_queue_sum / STEPS_PER_EP
        viz.add_episode_data(
            episode=e + 1,
            reward=total_reward,
            epsilon=agent.epsilon,
            avg_queue=avg_queue,
            overload_count=overload_count,
        )

        if (e + 1) % 20 == 0:
            print(f"Episode {e+1}/{EPISODES}  "
                  f"Reward={total_reward:.1f}  "
                  f"Power={total_realized:.0f}kW  "
                  f"Queue={avg_queue:.2f}  "
                  f"ε={agent.epsilon:.3f}")

    # ── 保存 ──
    viz.plot_training_curves()
    viz.save_data()
    viz.generate_summary_report()

    save_path = os.path.join(project_root, "model", "trained_dqn_real.pth")
    agent.save_model(path=save_path)
    print(f"\n训练完成！模型已保存: {save_path}")


if __name__ == "__main__":
    run_training_real()
