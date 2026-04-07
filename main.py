"""
EV 充电调度强化学习系统 — 统一入口
=====================================
用法:
    python main.py train           # 基础训练 (合成路网)
    python main.py train-real      # 联邦训练 (真实路网 + 可视化)
    python main.py train-viz       # 基础训练 + 可视化
    python main.py evaluate        # 评估已训练模型
    python main.py download-map    # 下载真实路网地图
     python visualization/visualize_simulation_3d.py --policy dqn --steps 120 --show-edges
     ssh -p 22182 root@region-9.autodl.pro
"""

import argparse
import sys
import os


def _resolve_scale(command, debug, medium):
    if debug:
        return {
            "num_evs": 10,
            "steps": 50,
            "episodes": 20,
            "fed_rounds": 5,
            "batch_size": 8,
        }
    if medium:
        return {
            "num_evs": 60,
            "steps": 200,
            "episodes": 100,
            "fed_rounds": 5,
            "batch_size": 32,
        }
    defaults = {
        "train": {
            "num_evs": 10,
            "steps": 100,
            "episodes": 800,
            "fed_rounds": 1,
            "batch_size": 64,
        },
        "train-real": {
            "num_evs": 100,
            "steps": 300,
            "episodes": 500,
            "fed_rounds": 1,
            "batch_size": 64,
        },
        "train-viz": {
            "num_evs": 10,
            "steps": 100,
            "episodes": 500,
            "fed_rounds": 1,
            "batch_size": 64,
        },
        "evaluate": {
            "num_evs": 100,
            "steps": 300,
            "episodes": 5,
            "fed_rounds": 1,
            "batch_size": 64,
        },
        "download-map": {
            "num_evs": 10,
            "steps": 50,
            "episodes": 20,
            "fed_rounds": 5,
            "batch_size": 8,
        },
    }
    return defaults[command]


def _resolve_evaluation_scale(debug, medium):
    if debug:
        return {
            "episodes": 5,
            "steps": 50,
        }
    if medium:
        return {
            "episodes": 20,
            "steps": 200,
        }
    return {
        "episodes": 50,
        "steps": 1000,
    }


def cmd_train(args):
    """基础 DQN 训练（合成路网）"""
    from train import DQNAgent
    from env.Traffic import TrafficPowerEnv
    cfg = _resolve_scale(args.command, args.debug, args.medium)

    env = TrafficPowerEnv(num_evs=cfg["num_evs"])
    agent = DQNAgent(num_features=15, num_actions=2)

    episodes = cfg["episodes"]
    steps_per_episode = cfg["steps"]
    batch_size = cfg["batch_size"]
    print("开始训练 (合成路网)...")
    for e in range(episodes):
        env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            urgent_evs = [ev for ev in env.evs if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)
            actions = {}
            ev_dispatch = []
            pending = {s.id: 0 for s in env.stations}
            for ev in urgent_evs:
                state = env.get_graph_state_for_ev(ev, pending)
                mask = env.get_action_mask(ev)
                action = agent.select_action(state, action_mask=mask)
                actions[ev.id] = action
                metrics = env.estimate_action_metrics(ev, action, pending)
                per_ev_r = -metrics["generalized_cost"]
                per_ev_r -= 6.0 * metrics["queue_time_h"]
                per_ev_r -= 2.0 * metrics["trip_time_h"]
                ev_dispatch.append((ev, state, action, per_ev_r, mask))
                pending[action] = pending.get(action, 0) + 1
            _, reward, _, info = env.step(actions)
            global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
            for ev, state, action, per_ev_r, mask in ev_dispatch:
                realized = info["decision_costs"].get(ev.id, {})
                realized_cost = realized.get("generalized_cost", 0.0)
                mixed_r = per_ev_r - 0.2 * realized_cost + global_bonus
                next_state = env.get_graph_state_for_ev(ev)
                agent.store_transition(state, action, mixed_r, next_state, action_mask=mask)
            if ev_dispatch or len(agent.memory) >= batch_size:
                agent.replay(batch_size)
            total_reward += reward
        agent.decay_epsilon()
        if (e + 1) % 50 == 0:
            print(f"Episode {e+1}/{episodes}  reward={total_reward:.1f}  ε={agent.epsilon:.3f}")
    agent.save_model()
    print("训练完成，模型已保存至 checkpoints/trained_dqn.pth")


def cmd_train_real(args):
    """联邦 DQN 训练（真实路网 + 可视化）"""
    from visualization.run_training_real_map import run_training_real
    cfg = _resolve_scale(args.command, args.debug, args.medium)
    # epsilon 目标终值：
    #   debug(20ep)  → 0.80，保持充分探索
    #   medium(100ep)→ 0.90，基本随机（episode 太少，过早利用反而变差）
    #   full(500ep)  → 0.10，500 个 episode 足够完成探索→利用过渡
    epsilon_final = 0.80 if args.debug else (0.90 if args.medium else 0.10)
    run_training_real(
        num_evs=cfg["num_evs"],
        episodes=cfg["episodes"],
        steps_per_episode=cfg["steps"],
        fed_rounds_per_episode=cfg["fed_rounds"],
        batch_size=cfg["batch_size"],
        use_dp=args.dp,
        epsilon_final=epsilon_final,
    )


def cmd_train_viz(args):
    """基础训练 + 可视化输出"""
    from visualization.run_training_with_viz import run_training
    cfg = _resolve_scale(args.command, args.debug, args.medium)
    run_training(
        episodes=cfg["episodes"],
        steps_per_episode=cfg["steps"],
        num_evs=cfg["num_evs"],
    )


def cmd_evaluate(args):
    """评估已训练模型（随机 / 贪心 / DQN / 联邦DQN 四方对比）"""
    from evaluation.run_evaluation import run_evaluation, _compare_table
    cfg = _resolve_scale(args.command, args.debug, args.medium)
    eval_cfg = _resolve_evaluation_scale(args.debug, args.medium)

    USE_REAL_MAP = True
    EPISODES = eval_cfg["episodes"]
    STEPS = eval_cfg["steps"]

    map_str = "真实路网 (珠江新城)" if USE_REAL_MAP else "3x3 人工网格"
    print(f"\n>>>> 当前评估使用的地图环境: {map_str} <<<<\n")

    print("=" * 62)
    print("  【1/4】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                   use_random=True, use_real_map=USE_REAL_MAP,
                                   num_evs=cfg["num_evs"], num_stations=4)

    print("\n")
    print("=" * 62)
    print("  【2/4】贪心策略基线")
    print("=" * 62)
    greedy_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                   use_greedy=True, use_real_map=USE_REAL_MAP,
                                   num_evs=cfg["num_evs"], num_stations=4)

    print("\n")
    print("=" * 62)
    print("  【3/4】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_dqn_real.pth" if USE_REAL_MAP else "trained_dqn.pth",
                                num_evs=cfg["num_evs"], num_stations=4)

    print("\n")
    print("=" * 62)
    print("  【4/4】联邦 DQN 策略评估")
    print("=" * 62)
    fed_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_federated_dqn_real.pth",
                                num_evs=cfg["num_evs"], num_stations=4)

    _compare_table({"Random": random_report, "Greedy": greedy_report,
                    "DQN": dqn_report, "FedDQN": fed_report})


def cmd_download_map(_args):
    """下载真实城市路网"""
    import download_map  # noqa: F401


COMMANDS = {
    "train":        cmd_train,
    "train-real":   cmd_train_real,
    "train-viz":    cmd_train_viz,
    "evaluate":     cmd_evaluate,
    "download-map": cmd_download_map,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EV 充电调度强化学习系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:15s}  {v.__doc__}" for k, v in COMMANDS.items()),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=list(COMMANDS.keys()),
        default="evaluate",
        help="要执行的操作",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--debug",
        action="store_true",
        help="启用小规模快速测试参数",
    )
    mode_group.add_argument(
        "--medium",
        action="store_true",
        help="启用中等规模测试参数",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="启用差分隐私训练 (DP-SGD)",
    )
    args = parser.parse_args()
    COMMANDS[args.command](args)
