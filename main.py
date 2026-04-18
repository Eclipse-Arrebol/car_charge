"""
EV 充电调度强化学习系统 — 统一入口
=====================================
用法:
    python main.py train-real      # 联邦训练（真实路网）
    python main.py evaluate        # 四方策略对比评估
"""

import argparse
import random

from training.config import TrainConfig, EvalConfig


def _apply_train_scale(base_cfg, scale_cfg):
    for attr in [
        "num_evs",
        "episodes",
        "steps_per_episode",
        "fed_rounds_per_episode",
        "batch_size",
        "step_local_train_steps",
        "step_train_interval",
        "fed_local_steps",
        "epsilon_final",
        "checkpoint_interval",
    ]:
        setattr(base_cfg, attr, getattr(scale_cfg, attr))
    return base_cfg


def _apply_eval_scale(base_cfg, scale_cfg):
    for attr in [
        "episodes",
        "steps_per_episode",
    ]:
        setattr(base_cfg, attr, getattr(scale_cfg, attr))
    return base_cfg


def cmd_train_real(args):
    """联邦 DQN 训练（真实路网）"""
    from training.trainer import run_training_real

    if args.graph_group == "l1":
        cfg = TrainConfig.ablation_l1()
    else:
        cfg = TrainConfig.ablation_l0()

    if args.debug:
        cfg = _apply_train_scale(cfg, TrainConfig.debug())
    elif args.quick:
        cfg = _apply_train_scale(cfg, TrainConfig.quick())
    elif args.medium:
        cfg = _apply_train_scale(cfg, TrainConfig.medium())

    cfg.use_dp = args.dp
    cfg.dp_noise_multiplier = args.dp_sigma

    run_training_real(
        num_evs=cfg.num_evs,
        episodes=cfg.episodes,
        steps_per_episode=cfg.steps_per_episode,
        fed_rounds_per_episode=cfg.fed_rounds_per_episode,
        batch_size=cfg.batch_size,
        step_local_train_steps=cfg.step_local_train_steps,
        step_train_interval=cfg.step_train_interval,
        proximal_mu=cfg.proximal_mu,
        use_dp=cfg.use_dp,
        dp_noise_multiplier=cfg.dp_noise_multiplier,
        dp_clip_C=cfg.dp_clip_C,
        epsilon_final=cfg.epsilon_final,
        checkpoint_interval=cfg.checkpoint_interval,
        mixed_reward_scale=cfg.mixed_reward_scale,
        mixed_reward_min=cfg.mixed_reward_min,
        mixed_reward_max=cfg.mixed_reward_max,
        graphml_file=cfg.graphml_file,
        station_config_file=cfg.station_config_file,
        station_id_key=cfg.station_id_key,
        max_nodes=cfg.max_nodes,
        graph_group=cfg.graph_group,
    )


def cmd_evaluate(args):
    """评估已训练模型（随机 / 贪心 / DQN / 联邦DQN 四方对比）"""
    from evaluation.run_evaluation import run_evaluation, _compare_table

    if args.graph_group == "l1":
        eval_cfg = EvalConfig.ablation_l1()
    else:
        eval_cfg = EvalConfig.ablation_l0()

    if args.debug:
        eval_cfg = _apply_eval_scale(eval_cfg, EvalConfig.debug())
    elif args.quick:
        eval_cfg = _apply_eval_scale(eval_cfg, EvalConfig.quick())
    elif args.medium:
        eval_cfg = _apply_eval_scale(eval_cfg, EvalConfig.medium())

    # 预先生成固定种子，保证四种策略在相同 episode 上评估
    rng = random.Random(eval_cfg.base_seed)
    episode_seeds = [rng.randint(0, 10000) for _ in range(eval_cfg.episodes)]

    print(f"\n>>>> 评估地图: 真实路网 (珠江新城) <<<<\n")
    print(f"[Evaluation Seeds] base_seed={eval_cfg.base_seed}, "
          f"episode_seeds={episode_seeds}")

    common = dict(
        episodes=eval_cfg.episodes,
        steps_per_episode=eval_cfg.steps_per_episode,
        use_real_map=True,
        num_evs=eval_cfg.num_evs,
        num_stations=eval_cfg.num_stations,
        episode_seeds=episode_seeds,
    )

    print("=" * 62)
    print("  【1/4】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(**common, use_random=True)

    print("\n" + "=" * 62)
    print("  【2/4】贪心策略基线")
    print("=" * 62)
    greedy_report = run_evaluation(**common, use_greedy=True)

    print("\n" + "=" * 62)
    print("  【3/4】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(**common, model_file="trained_dqn_real.pth")

    print("\n" + "=" * 62)
    print("  【4/4】联邦 DQN 策略评估")
    print("=" * 62)
    fed_model_name = f"trained_federated_dqn_real_{eval_cfg.graph_group}.pth"
    fed_report = run_evaluation(**common, model_file=fed_model_name)

    _compare_table({
        "Random": random_report,
        "Greedy": greedy_report,
        "DQN":    dqn_report,
        "FedDQN": fed_report,
    })


COMMANDS = {
    "train-real": cmd_train_real,
    "evaluate":   cmd_evaluate,
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
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--debug",  action="store_true", help="极小规模，约 2 分钟")
    scale_group.add_argument("--medium", action="store_true", help="中等规模，约 30 分钟")
    scale_group.add_argument("--quick",  action="store_true", help="快速验证，约 10 分钟")
    parser.add_argument("--dp",       action="store_true", help="启用差分隐私训练 (DP-SGD)")
    parser.add_argument("--dp-sigma", type=float, default=1.0,
                        help="DP-SGD 高斯噪声倍率 σ，默认 1.0")
    parser.add_argument(
        "--graph-group",
        choices=["l0", "l1"],
        default="l0",
        help="训练输入图表示：l0=完整真实图，l1=eps40训练图",
    )
    args = parser.parse_args()
    COMMANDS[args.command](args)
