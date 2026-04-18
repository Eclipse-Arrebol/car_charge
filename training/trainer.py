import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.FederatedDQN import FederatedClient, FederatedServer
from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from train import _finish_progress_line, _print_training_progress
from training.config import TrainConfig
from visualization.visualize_training import TrainingVisualizer


LOCAL_GRAPHML = os.path.join(
    project_root, "map_outputs", "baseline_eps40_artifacts", "G_L0_indexed.graphml"
)
OFFLINE_FALLBACK = False
PLACE = "Wuchang District, Wuhan, China"

MAX_NODES = 200
NUM_EVS = 200
NUM_STATS = 4

EPISODES = 500
STEPS_PER_EP = 1000
BATCH_SIZE = 64
STEP_LOCAL_TRAIN_STEPS = 2
STEP_TRAIN_INTERVAL = 1
FED_LOCAL_STEPS = 4
FED_ROUNDS_PER_EP = 1
AGGREGATION_INTERVAL = 10
CHECKPOINT_INTERVAL = 20
MIXED_REWARD_SCALE = 50.0
MIXED_REWARD_MIN = -20.0
MIXED_REWARD_MAX = 5.0


class FederatedTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.fed_local_steps = getattr(cfg, "fed_local_steps", FED_LOCAL_STEPS)
        self.viz = TrainingVisualizer(save_dir="results/real")
        self.client_seeds = [42, 123]

        self._print_mode_banner()

        self.client_envs = [self._build_env(seed) for seed in self.client_seeds]
        env0 = self.client_envs[0]

        epsilon_decay = max(0.90, cfg.epsilon_final ** (1.0 / max(cfg.episodes, 1)))
        self.epsilon_decay = epsilon_decay
        print(
            f"[Epsilon] decay={epsilon_decay:.6f}，"
            f"{cfg.episodes} episodes 后 ε≈{cfg.epsilon_final}"
        )

        self.fed_server = FederatedServer(
            num_features=18,
            num_actions=env0.num_stations,
            station_node_ids=env0.station_node_ids,
            num_nodes_per_graph=env0.num_nodes,
            aggregation_momentum=1.0,
        )

        for i, client_env in enumerate(self.client_envs):
            dp_sample_rate = cfg.batch_size / 20000
            client = FederatedClient(
                client_id=i,
                num_features=18,
                num_actions=client_env.num_stations,
                station_node_ids=client_env.station_node_ids,
                num_nodes_per_graph=client_env.num_nodes,
                proximal_mu=cfg.proximal_mu,
                use_dp=cfg.use_dp,
                dp_noise_multiplier=cfg.dp_noise_multiplier,
                dp_clip_C=cfg.dp_clip_C,
                dp_sample_rate=dp_sample_rate,
            )
            client.epsilon_decay = epsilon_decay
            self.fed_server.register_client(client)

        self.fed_server.distribute_global_model()

    def _print_mode_banner(self):
        graphml_file = getattr(self.cfg, "graphml_file", LOCAL_GRAPHML)
        if graphml_file is not None:
            print(f"[模式 B] 使用本地 GraphML 文件: {graphml_file}")
            station_ids = self._resolve_station_nodes()
            if station_ids is not None:
                print(f"[模式 B] 使用显式充电站节点: {station_ids}")
        elif OFFLINE_FALLBACK:
            print("[模式 C] 使用离线合成路网（Watts-Strogatz）")
        else:
            print(f"[模式 A] 自动联网下载路网: {PLACE}")

    def _resolve_station_nodes(self):
        station_cfg = getattr(self.cfg, "station_config_file", None)
        station_key = getattr(self.cfg, "station_id_key", None)
        if not station_cfg or not station_key:
            return None
        return resolve_station_nodes(station_cfg, station_key)

    def _build_env(self, seed):
        graphml_file = getattr(self.cfg, "graphml_file", LOCAL_GRAPHML)
        station_node_ids = self._resolve_station_nodes()
        if graphml_file is not None:
            return RealTrafficEnv(
                graphml_file=graphml_file,
                num_stations=self.cfg.num_stations,
                num_evs=self.cfg.num_evs,
                max_nodes=self.cfg.max_nodes,
                seed=seed,
                station_node_ids=station_node_ids,
            )
        if OFFLINE_FALLBACK:
            return RealTrafficEnv(
                place=PLACE,
                num_stations=self.cfg.num_stations,
                num_evs=self.cfg.num_evs,
                max_nodes=self.cfg.max_nodes,
                seed=seed,
                offline=True,
            )
        return RealTrafficEnv(
            place=PLACE,
            num_stations=self.cfg.num_stations,
            num_evs=self.cfg.num_evs,
            max_nodes=self.cfg.max_nodes,
            seed=seed,
        )

    def _run_episode(self, episode_idx):
        stats = {
            "total_reward": 0.0,
            "avg_queue_sum": 0.0,
            "overload_count": 0,
            "total_realized": 0.0,
            "total_mixed_reward": 0.0,
            "total_decisions": 0,
            "episode_decision_scan_s": 0.0,
            "episode_action_build_s": 0.0,
            "episode_env_step_s": 0.0,
            "episode_store_s": 0.0,
            "episode_step_train_s": 0.0,
            "episode_fed_train_s": 0.0,
            "episode_agg_s": 0.0,
            "episode_queue_h_sum": 0.0,
            "episode_trip_h_sum": 0.0,
            "episode_service_h_sum": 0.0,
            "episode_realized_queue_h_sum": 0.0,
            "episode_realized_service_h_sum": 0.0,
            "episode_generalized_cost_sum": 0.0,
            "episode_start_time": time.time(),
        }

        for env in self.client_envs:
            env.reset()

        for step_idx in range(self.cfg.steps_per_episode):
            total_urgent = 0

            for client, env in zip(self.fed_server.clients, self.client_envs):
                t0 = time.perf_counter()
                urgent_evs = env.get_pending_decision_evs()
                stats["episode_decision_scan_s"] += time.perf_counter() - t0

                actions = {}
                ev_dispatch = []
                pending_counts = {s.id: 0 for s in env.stations}

                t0 = time.perf_counter()
                for ev in urgent_evs:
                    ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                    action_mask = env.get_action_mask(ev)
                    action = client.select_action(ev_state, action_mask=action_mask)
                    actions[ev.id] = action

                    metrics = env.estimate_action_metrics(ev, action, pending_counts)
                    per_ev_r = -(metrics["queue_time_h"] + metrics["trip_time_h"])

                    stats["episode_queue_h_sum"] += metrics["queue_time_h"]
                    stats["episode_trip_h_sum"] += metrics["trip_time_h"]
                    stats["episode_service_h_sum"] += metrics["service_time_h"]

                    ev_dispatch.append((ev, ev_state, action, per_ev_r, action_mask))
                    pending_counts[action] += 1
                stats["episode_action_build_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                _, reward, _, info = env.step(actions)
                stats["episode_env_step_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                for ev, ev_state, act, per_ev_r, mask in ev_dispatch:
                    mixed_r = per_ev_r / 2.0
                    mixed_r = max(-1.0, min(0.0, mixed_r))
                    next_ev_state = env.get_graph_state_for_ev(ev)
                    client.store_transition(
                        ev_state, act, mixed_r, next_ev_state, action_mask=mask
                    )
                    stats["total_mixed_reward"] += mixed_r
                    stats["total_decisions"] += 1
                stats["episode_store_s"] += time.perf_counter() - t0

                should_step_train = (
                    self.cfg.step_local_train_steps > 0
                    and ((step_idx + 1) % max(1, self.cfg.step_train_interval) == 0)
                )
                if should_step_train:
                    t0 = time.perf_counter()
                    client.local_train(
                        self.cfg.batch_size,
                        num_steps=self.cfg.step_local_train_steps,
                    )
                    stats["episode_step_train_s"] += time.perf_counter() - t0

                stats["total_reward"] += reward
                stats["total_realized"] += info.get("realized_power", 0.0)
                stats["avg_queue_sum"] += (
                    sum(len(s.queue) for s in env.stations) / env.num_stations
                )
                stats["overload_count"] += int(info.get("voltage_violations", 0))
                total_urgent += len(urgent_evs)

            _print_training_progress(
                episode=episode_idx,
                episodes=self.cfg.episodes,
                step=step_idx,
                steps_per_episode=self.cfg.steps_per_episode,
                total_reward=stats["total_reward"],
                epsilon=min(c.epsilon for c in self.fed_server.clients),
                episode_start_time=stats["episode_start_time"],
                extra_metrics=(
                    f"Power={stats['total_realized']:.0f}kW "
                    f"QueueAvg={stats['avg_queue_sum'] / ((step_idx + 1) * len(self.client_envs)):.2f} "
                    f"UrgentEVs={total_urgent}"
                ),
                update_every=25,
            )

        return stats

    def _federated_round(self):
        fed_train_s = 0.0
        agg_s = 0.0
        for _ in range(self.cfg.fed_rounds_per_episode):
            for client in self.fed_server.clients:
                t0 = time.perf_counter()
                client.local_train(self.cfg.batch_size, num_steps=self.fed_local_steps)
                fed_train_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            self.fed_server.aggregate()
            self.fed_server.distribute_global_model()
            agg_s += time.perf_counter() - t0

        for client in self.fed_server.clients:
            client.decay_epsilon()

        return {
            "episode_fed_train_s": fed_train_s,
            "episode_agg_s": agg_s,
        }

    def _log_episode(self, episode_idx, stats, wall_time):
        avg_queue = stats["avg_queue_sum"] / (
            self.cfg.steps_per_episode * len(self.client_envs)
        )
        avg_mixed_reward = stats["total_mixed_reward"] / max(1, stats["total_decisions"])

        self.viz.add_episode_data(
            episode=episode_idx + 1,
            reward=avg_mixed_reward,
            epsilon=min(c.epsilon for c in self.fed_server.clients),
            avg_queue=avg_queue,
            overload_count=stats["overload_count"],
        )

        should_print_profile = (
            episode_idx < 3
            or ((episode_idx + 1) % 10 == 0)
            or (episode_idx + 1 == self.cfg.episodes)
        )
        if should_print_profile:
            replay_sizes = "/".join(
                str(len(client.memory)) for client in self.fed_server.clients
            )
            avg_decisions_per_step = stats["total_decisions"] / max(
                1, self.cfg.steps_per_episode * len(self.client_envs)
            )
            avg_queue_h = stats["episode_queue_h_sum"] / max(1, stats["total_decisions"])
            avg_trip_h = stats["episode_trip_h_sum"] / max(1, stats["total_decisions"])
            avg_service_h = stats["episode_service_h_sum"] / max(1, stats["total_decisions"])
            avg_realized_queue_h = stats["episode_realized_queue_h_sum"] / max(
                1, stats["total_decisions"]
            )
            avg_realized_service_h = stats["episode_realized_service_h_sum"] / max(
                1, stats["total_decisions"]
            )
            avg_generalized_cost = stats["episode_generalized_cost_sum"] / max(
                1, stats["total_decisions"]
            )
            print(
                f"[Profile][Ep {episode_idx+1}] wall={wall_time:.1f}s "
                f"scan={stats['episode_decision_scan_s']:.1f}s "
                f"build={stats['episode_action_build_s']:.1f}s "
                f"env={stats['episode_env_step_s']:.1f}s "
                f"store={stats['episode_store_s']:.1f}s "
                f"step_train={stats['episode_step_train_s']:.1f}s "
                f"fed_train={stats['episode_fed_train_s']:.1f}s "
                f"agg={stats['episode_agg_s']:.1f}s "
                f"decisions={stats['total_decisions']} "
                f"dec/step/client={avg_decisions_per_step:.3f} "
                f"replay={replay_sizes} "
                f"avg_mixed_r={avg_mixed_reward:.3f} "
                f"avg_queue_h={avg_queue_h:.3f} "
                f"avg_trip_h={avg_trip_h:.3f} "
                f"avg_service_h={avg_service_h:.3f} "
                f"avg_real_queue_h={avg_realized_queue_h:.3f} "
                f"avg_real_service_h={avg_realized_service_h:.3f} "
                f"avg_cost={avg_generalized_cost:.1f}"
            )

        if (episode_idx + 1) % 20 == 0:
            print(
                f"Episode {episode_idx+1}/{self.cfg.episodes}  "
                f"Reward={stats['total_reward']:.1f}  "
                f"Power={stats['total_realized']:.0f}kW  "
                f"Queue={avg_queue:.2f}  "
                f"ε={min(c.epsilon for c in self.fed_server.clients):.3f}"
            )

    def _save_checkpoint(self, episode_idx):
        if self.cfg.checkpoint_interval <= 0:
            return
        if (episode_idx + 1) % self.cfg.checkpoint_interval != 0:
            return

        ckpt_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir, f"trained_federated_dqn_real_ep{episode_idx + 1}.pth"
        )
        self.fed_server.save_global_model(path=ckpt_path)
        print(f"[Checkpoint] saved: {ckpt_path}")

    def train(self):
        env0 = self.client_envs[0]
        network_desc = (
            getattr(self.cfg, "graphml_file", LOCAL_GRAPHML)
            or ("offline-synthetic" if OFFLINE_FALLBACK else PLACE)
        )
        print(f"\nStart federated training (network: {network_desc})")
        print(
            f"nodes={env0.num_nodes}, stations={env0.station_node_ids}, "
            f"EVs/client={self.cfg.num_evs}"
        )
        print(
            f"FL clients={len(self.fed_server.clients)}, local_steps={self.fed_local_steps}, "
            f"step_train_steps={self.cfg.step_local_train_steps}, "
            f"step_train_interval={self.cfg.step_train_interval}, "
            f"batch_size={self.cfg.batch_size}"
        )
        print(f"client_env_seeds={self.client_seeds}\n")

        for episode_idx in range(self.cfg.episodes):
            stats = self._run_episode(episode_idx)
            round_stats = self._federated_round()
            stats.update(round_stats)
            _finish_progress_line()
            wall_time = time.time() - stats["episode_start_time"]
            self._log_episode(episode_idx, stats, wall_time)
            self._save_checkpoint(episode_idx)

        self.viz.plot_training_curves()
        self.viz.save_data()
        self.viz.generate_summary_report()

        save_path = os.path.join(project_root, "checkpoints", "trained_federated_dqn_real.pth")
        self.fed_server.save_global_model(path=save_path)
        print(f"\n联邦训练完成！全局模型已保存: {save_path}")

        if self.cfg.use_dp:
            print("\n[差分隐私] 训练结束后的隐私预算:")
            for client in self.fed_server.clients:
                budget = client.get_privacy_spent()
                if budget:
                    print(
                        f"  Client {client.client_id}: "
                        f"ε={budget['epsilon']:.4f}, δ={budget['delta']}"
                    )


def run_training_real(
    num_evs=NUM_EVS,
    episodes=EPISODES,
    steps_per_episode=STEPS_PER_EP,
    fed_rounds_per_episode=FED_ROUNDS_PER_EP,
    batch_size=BATCH_SIZE,
    step_local_train_steps=STEP_LOCAL_TRAIN_STEPS,
    step_train_interval=STEP_TRAIN_INTERVAL,
    proximal_mu=0.01,
    use_dp=False,
    dp_noise_multiplier=1.0,
    dp_clip_C=1.0,
    epsilon_final=0.10,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    mixed_reward_scale=MIXED_REWARD_SCALE,
    mixed_reward_min=MIXED_REWARD_MIN,
    mixed_reward_max=MIXED_REWARD_MAX,
    graphml_file=LOCAL_GRAPHML,
    station_config_file=None,
    station_id_key="l0_station_nodes",
    max_nodes=MAX_NODES,
):
    cfg = TrainConfig(
        num_evs=num_evs,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        fed_rounds_per_episode=fed_rounds_per_episode,
        batch_size=batch_size,
        step_local_train_steps=step_local_train_steps,
        step_train_interval=step_train_interval,
        fed_local_steps=FED_LOCAL_STEPS,
        epsilon_final=epsilon_final,
        mixed_reward_scale=mixed_reward_scale,
        mixed_reward_min=mixed_reward_min,
        mixed_reward_max=mixed_reward_max,
        proximal_mu=proximal_mu,
        use_dp=use_dp,
        dp_noise_multiplier=dp_noise_multiplier,
        dp_clip_C=dp_clip_C,
        num_stations=NUM_STATS,
        max_nodes=max_nodes,
        graphml_file=graphml_file,
        station_config_file=station_config_file or getattr(TrainConfig(), "station_config_file", None),
        station_id_key=station_id_key,
        checkpoint_interval=checkpoint_interval,
    )
    FederatedTrainer(cfg).train()


if __name__ == "__main__":
    run_training_real()
