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
from env.RealTrafficEnv import RealTrafficEnv
from train import DQNAgent
from agents.FederatedDQN import FederatedServer, FederatedClient
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
MAX_NODES = 9999    # 使用完整路网（不精简节点）
NUM_EVS   = 20      # EV 数量（与评估环境保持一致）
NUM_STATS = 2       # 充电站数量

EPISODES         = 200  # 增加训练轮次
STEPS_PER_EP     = 100
BATCH_SIZE       = 64
FED_LOCAL_STEPS  = 1    # 减小本地训练步数，降低客户端漂移
FED_ROUNDS_PER_EP = 1   # 每个 episode 结束后执行一次联邦聚合
AGGREGATION_INTERVAL = 10  # episode 内每隔若干步执行一次聚合
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

    # ── 联邦学习架构初始化 ──
    # 全局服务器
    fed_server = FederatedServer(
        num_features=14,
        num_actions=env.num_stations,
        station_node_ids=env.station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        aggregation_momentum=0.35,
    )

    # 每个充电站区域创建一个联邦客户端
    for i in range(env.num_stations):
        client = FederatedClient(
            client_id=i,
            num_features=14,
            num_actions=env.num_stations,
            station_node_ids=env.station_node_ids,
            num_nodes_per_graph=env.num_nodes,
            proximal_mu=1e-4,
        )
        fed_server.register_client(client)

    # 初始下发全局模型
    fed_server.distribute_global_model()

    viz = TrainingVisualizer(save_dir="results/real")

    network_desc = (LOCAL_GRAPHML or ("offline-synthetic" if OFFLINE_FALLBACK else PLACE))
    print(f"\nStart federated training (network: {network_desc})")
    print(f"nodes={env.num_nodes}, stations={env.station_node_ids}, EVs={NUM_EVS}")
    print(f"FL clients={env.num_stations}, local_steps={FED_LOCAL_STEPS}\n")

    for e in range(EPISODES):
        env.reset()
        total_reward = 0.0
        avg_queue_sum = 0.0
        overload_count = 0
        total_realized = 0.0

        for _ in range(STEPS_PER_EP):

            # --- A. 顺序决策：利用率导向奖励 + per-EV 状态 + Action Mask ---
            urgent_evs = [ev for ev in env.evs
                          if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)

            actions = {}
            ev_dispatch = []       # (ev, ev_state, action, shaped_reward, action_mask, client_id)
            pending_counts = {s.id: 0 for s in env.stations}
            client_loads = {client.client_id: 0 for client in fed_server.clients}

            for ev in urgent_evs:
                ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                action_mask = env.get_action_mask(ev)

                # 选择离 EV 最近的充电站对应的客户端来做决策
                client_id = _select_client_for_ev(env, ev, client_loads)
                client = fed_server.clients[client_id]

                action = client.select_action(ev_state, action_mask=action_mask)
                actions[ev.id] = action

                # ── 利用率导向的塑形奖励 ──
                metrics = env.estimate_action_metrics(ev, action, pending_counts)
                per_ev_r = -metrics["generalized_cost"]
                per_ev_r -= 6.0 * metrics["queue_time_h"]
                per_ev_r -= 2.0 * metrics["trip_time_h"]

                ev_dispatch.append((ev, ev_state, action, per_ev_r, action_mask, client_id))
                pending_counts[action] += 1
                client_loads[client_id] += 1

            next_state, reward, _, info = env.step(actions)

            # 经验存储到各自的联邦客户端
            global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
            for ev, ev_state, act, per_ev_r, mask, cid in ev_dispatch:
                realized = info["decision_costs"].get(ev.id, {})
                realized_cost = realized.get("generalized_cost", 0.0)
                mixed_r = per_ev_r - 0.2 * realized_cost + global_bonus
                next_ev_state = env.get_graph_state_for_ev(ev)
                fed_server.clients[cid].store_transition(
                    ev_state, act, mixed_r, next_ev_state, action_mask=mask
                )

            # 各客户端本地训练
            for client in fed_server.clients:
                client.local_train(BATCH_SIZE, num_steps=1)

            # 更频繁地聚合，减少本地模型在长时间局部更新后发生漂移。
            if (_ + 1) % AGGREGATION_INTERVAL == 0:
                fed_server.aggregate()
                fed_server.distribute_global_model()

            total_reward   += reward
            total_realized += info.get("realized_power", 0.0)
            avg_queue_sum  += sum(len(s.queue) for s in env.stations) / env.num_stations
            overload_count += int(info.get("voltage_violations", 0))

        # --- 联邦聚合 (每个 episode 结束后) ---
        for _ in range(FED_ROUNDS_PER_EP):
            # 各客户端执行额外本地训练步
            for client in fed_server.clients:
                client.local_train(BATCH_SIZE, num_steps=FED_LOCAL_STEPS)
            # FedAvg 聚合
            fed_server.aggregate()
            # 下发全局模型
            fed_server.distribute_global_model()

        avg_queue = avg_queue_sum / STEPS_PER_EP
        viz.add_episode_data(
            episode=e + 1,
            reward=total_reward,
            epsilon=min(c.epsilon for c in fed_server.clients),
            avg_queue=avg_queue,
            overload_count=overload_count,
        )

        if (e + 1) % 20 == 0:
            print(f"Episode {e+1}/{EPISODES}  "
                  f"Reward={total_reward:.1f}  "
                  f"Power={total_realized:.0f}kW  "
                  f"Queue={avg_queue:.2f}  "
                  f"ε={min(c.epsilon for c in fed_server.clients):.3f}")

    # ── 保存 ──
    viz.plot_training_curves()
    viz.save_data()
    viz.generate_summary_report()

    save_path = os.path.join(project_root, "checkpoints", "trained_federated_dqn_real.pth")
    fed_server.save_global_model(path=save_path)
    print(f"\n联邦训练完成！全局模型已保存: {save_path}")


def _select_client_for_ev(env, ev, client_loads):
    """按“距离 + 当前客户端负载”联合分配决策客户端，缓解样本极端非IID。"""
    best_id = 0
    best_score = float('inf')
    for i, station in enumerate(env.stations):
        try:
            dist = nx.shortest_path_length(
                env.traffic_graph, ev.curr_node, station.traffic_node_id
            )
        except nx.NetworkXNoPath:
            dist = float('inf')

        score = client_loads.get(i, 0) * 2.0 + dist
        if score < best_score:
            best_score = score
            best_id = i
    return best_id


if __name__ == "__main__":
    run_training_real()
