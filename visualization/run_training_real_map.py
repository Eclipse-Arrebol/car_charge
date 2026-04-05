"""
基于 OSMnx 真实路网的训练入口
================================
用法:
    cd G:\\交电demo
    python visualization/run_training_real_map.py

获取真实路网 -> 先运行 python download_map.py，再将生成的 .graphml 路径填入 LOCAL_GRAPHML。
若无法联网，设 OFFLINE_FALLBACK = True 使用合成路网。
"""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.RealTrafficEnv import RealTrafficEnv
from train import DQNAgent, _print_training_progress, _finish_progress_line
from agents.FederatedDQN import FederatedServer, FederatedClient
from visualization.visualize_training import TrainingVisualizer


# ============================================================
# ★ 在这里选择路网来源（三选一）★
# ============================================================

# 方式 B：本地 GraphML 文件（填写路径后优先使用）
# 先运行项目根目录下的 download_map.py 生成文件，再填入路径
LOCAL_GRAPHML = os.path.join(project_root, "zhujiang_new_town.graphml")

# 方式 C：直接使用离线合成路网（True = 跳过所有网络请求）
OFFLINE_FALLBACK = False

# 方式 A：自动联网时的 OSM 地名（LOCAL_GRAPHML=None 且 OFFLINE_FALLBACK=False 时生效）
PLACE = "Wuchang District, Wuhan, China"

# —— 路网规模 ——
MAX_NODES = 9999
NUM_EVS = 100
NUM_STATS = 2

EPISODES = 500
STEPS_PER_EP = 1000
BATCH_SIZE = 64
STEP_LOCAL_TRAIN_STEPS = 2
FED_LOCAL_STEPS = 4
FED_ROUNDS_PER_EP = 1
AGGREGATION_INTERVAL = 10
# ============================================================


def _create_real_env(num_evs, seed):
    if LOCAL_GRAPHML is not None:
        return RealTrafficEnv(
            graphml_file=LOCAL_GRAPHML,
            num_stations=NUM_STATS,
            num_evs=num_evs,
            max_nodes=MAX_NODES,
            seed=seed,
        )
    if OFFLINE_FALLBACK:
        return RealTrafficEnv(
            place=PLACE,
            num_stations=NUM_STATS,
            num_evs=num_evs,
            max_nodes=MAX_NODES,
            seed=seed,
            offline=True,
        )
    return RealTrafficEnv(
        place=PLACE,
        num_stations=NUM_STATS,
        num_evs=num_evs,
        max_nodes=MAX_NODES,
        seed=seed,
    )


def run_training_real(
    num_evs=NUM_EVS,
    episodes=EPISODES,
    steps_per_episode=STEPS_PER_EP,
    fed_rounds_per_episode=FED_ROUNDS_PER_EP,
    batch_size=BATCH_SIZE,
    proximal_mu=0.01,
    use_dp=False,
    dp_noise_multiplier=1.0,
    dp_clip_C=1.0,
    epsilon_final=0.10,
):
    if LOCAL_GRAPHML is not None:
        print(f"[模式 B] 使用本地 GraphML 文件: {LOCAL_GRAPHML}")
    elif OFFLINE_FALLBACK:
        print("[模式 C] 使用离线合成路网（Watts-Strogatz）")
    else:
        print(f"[模式 A] 自动联网下载路网: {PLACE}")

    env0 = _create_real_env(num_evs=num_evs, seed=42)
    env1 = _create_real_env(num_evs=num_evs, seed=123)
    client_envs = [env0, env1]

    # epsilon 衰减率：确保训练结束时 ε 降到 epsilon_final
    epsilon_decay = max(0.90, epsilon_final ** (1.0 / max(episodes, 1)))
    print(f"[Epsilon] decay={epsilon_decay:.6f}，{episodes} episodes 后 ε≈{epsilon_final}")

    fed_server = FederatedServer(
        num_features=15,
        num_actions=env0.num_stations,
        station_node_ids=env0.station_node_ids,
        num_nodes_per_graph=env0.num_nodes,
        aggregation_momentum=1.0,
    )

    for i, client_env in enumerate(client_envs):
        dp_sample_rate = batch_size / 20000  # q = batch / replay_buffer_size
        client = FederatedClient(
            client_id=i,
            num_features=15,
            num_actions=client_env.num_stations,
            station_node_ids=client_env.station_node_ids,
            num_nodes_per_graph=client_env.num_nodes,
            proximal_mu=proximal_mu,
            use_dp=use_dp,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_clip_C=dp_clip_C,
            dp_sample_rate=dp_sample_rate,
        )
        client.epsilon_decay = epsilon_decay
        fed_server.register_client(client)

    fed_server.distribute_global_model()
    viz = TrainingVisualizer(save_dir="results/real")

    network_desc = (LOCAL_GRAPHML or ("offline-synthetic" if OFFLINE_FALLBACK else PLACE))
    print(f"\nStart federated training (network: {network_desc})")
    print(f"nodes={env0.num_nodes}, stations={env0.station_node_ids}, EVs/client={num_evs}")
    print(f"FL clients={len(fed_server.clients)}, local_steps={FED_LOCAL_STEPS}")
    print("client_env_seeds=[42, 123]\n")

    for e in range(episodes):
        total_reward = 0.0
        avg_queue_sum = 0.0
        overload_count = 0
        total_realized = 0.0
        episode_start_time = time.time()

        for env in client_envs:
            env.reset()

        for step_idx in range(steps_per_episode):
            total_urgent = 0

            for client, env in zip(fed_server.clients, client_envs):
                urgent_evs = [ev for ev in env.evs if ev.status == "IDLE" and ev.soc < 30.0]
                urgent_evs.sort(key=lambda ev: ev.soc)

                actions = {}
                ev_dispatch = []
                pending_counts = {s.id: 0 for s in env.stations}

                for ev in urgent_evs:
                    ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                    action_mask = env.get_action_mask(ev)
                    action = client.select_action(ev_state, action_mask=action_mask)
                    actions[ev.id] = action

                    metrics = env.estimate_action_metrics(ev, action, pending_counts)
                    per_ev_r = -metrics["generalized_cost"]
                    per_ev_r -= 6.0 * metrics["queue_time_h"]
                    per_ev_r -= 2.0 * metrics["trip_time_h"]

                    ev_dispatch.append((ev, ev_state, action, per_ev_r, action_mask))
                    pending_counts[action] += 1

                _, reward, _, info = env.step(actions)

                global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
                for ev, ev_state, act, per_ev_r, mask in ev_dispatch:
                    realized = info["decision_costs"].get(ev.id, {})
                    realized_cost = realized.get("generalized_cost", 0.0)
                    mixed_r = per_ev_r - 0.2 * realized_cost + global_bonus
                    next_ev_state = env.get_graph_state_for_ev(ev)
                    client.store_transition(
                        ev_state, act, mixed_r, next_ev_state, action_mask=mask
                    )

                client.local_train(batch_size, num_steps=STEP_LOCAL_TRAIN_STEPS)

                total_reward += reward
                total_realized += info.get("realized_power", 0.0)
                avg_queue_sum += sum(len(s.queue) for s in env.stations) / env.num_stations
                overload_count += int(info.get("voltage_violations", 0))
                total_urgent += len(urgent_evs)

            _print_training_progress(
                episode=e,
                episodes=episodes,
                step=step_idx,
                steps_per_episode=steps_per_episode,
                total_reward=total_reward,
                epsilon=min(c.epsilon for c in fed_server.clients),
                episode_start_time=episode_start_time,
                extra_metrics=(
                    f"Power={total_realized:.0f}kW "
                    f"QueueAvg={avg_queue_sum / ((step_idx + 1) * len(client_envs)):.2f} "
                    f"UrgentEVs={total_urgent}"
                ),
                update_every=25,
            )

        for _ in range(fed_rounds_per_episode):
            for client in fed_server.clients:
                client.local_train(batch_size, num_steps=FED_LOCAL_STEPS)
            fed_server.aggregate()
            fed_server.distribute_global_model()

        for client in fed_server.clients:
            client.decay_epsilon()
        _finish_progress_line()

        avg_queue = avg_queue_sum / (steps_per_episode * len(client_envs))
        viz.add_episode_data(
            episode=e + 1,
            reward=total_reward,
            epsilon=min(c.epsilon for c in fed_server.clients),
            avg_queue=avg_queue,
            overload_count=overload_count,
        )

        if (e + 1) % 20 == 0:
            print(
                f"Episode {e+1}/{episodes}  "
                f"Reward={total_reward:.1f}  "
                f"Power={total_realized:.0f}kW  "
                f"Queue={avg_queue:.2f}  "
                f"ε={min(c.epsilon for c in fed_server.clients):.3f}"
            )

    viz.plot_training_curves()
    viz.save_data()
    viz.generate_summary_report()

    save_path = os.path.join(project_root, "checkpoints", "trained_federated_dqn_real.pth")
    fed_server.save_global_model(path=save_path)
    print(f"\n联邦训练完成！全局模型已保存: {save_path}")

    if use_dp:
        print("\n[差分隐私] 训练结束后的隐私预算:")
        for client in fed_server.clients:
            budget = client.get_privacy_spent()
            if budget:
                print(f"  Client {client.client_id}: ε={budget['epsilon']:.4f}, δ={budget['delta']}")


if __name__ == "__main__":
    run_training_real()
