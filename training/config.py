"""
训练与评估配置 — 所有超参数的唯一来源。

用法:
    from training.config import TrainConfig, EvalConfig

    cfg = TrainConfig()            # 完整规模（默认）
    cfg = TrainConfig.medium()     # 中等规模
    cfg = TrainConfig.debug()      # 快速调试

    eval_cfg = EvalConfig()        # 完整评估
    eval_cfg = EvalConfig.debug()  # 快速验证
"""

from dataclasses import dataclass, field
import os
from typing import Optional

# 项目根目录（相对于本文件向上两级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_STATION_CONFIG = os.path.join(_PROJECT_ROOT, "config", "stations.json")
_DEFAULT_L0_GRAPHML = os.path.join(
    _PROJECT_ROOT, "map_outputs", "baseline_eps40_artifacts", "G_L0_indexed.graphml"
)
_DEFAULT_L1_GRAPHML = os.path.join(
    _PROJECT_ROOT, "map_outputs", "baseline_eps40_artifacts", "G_L1_eps40.graphml"
)
_DEFAULT_GRAPHML = _DEFAULT_L0_GRAPHML


@dataclass
class TrainConfig:
    """联邦 DQN 训练超参数（真实路网）。"""

    # ── 规模 ──────────────────────────────────────────────────────────
    num_evs: int = 100              # 每个联邦客户端的 EV 数量
    episodes: int = 500             # 训练 episode 总数
    steps_per_episode: int = 3000    # 每 episode 的仿真步数
    fed_rounds_per_episode: int = 1 # 每 episode 结束后的联邦聚合轮数
    batch_size: int = 256           # 经验回放采样批大小

    # ── 训练节奏 ──────────────────────────────────────────────────────
    step_local_train_steps: int = 1 # 每个仿真步内的本地训练步数
    step_train_interval: int = 2    # 每隔多少仿真步触发一次本地训练
    fed_local_steps: int = 4        # episode 结束时联邦轮次内的本地训练步数
    epsilon_final: float = 0.10     # 训练结束时的目标 ε 值

    # ── 奖励归一化 ────────────────────────────────────────────────────
    mixed_reward_scale: float = 50.0
    mixed_reward_min: float = -20.0
    mixed_reward_max: float = 5.0
    reward_mode: str = "baseline"
    cheat_grid_cost_scale: float = 1.0

    # ── 联邦 / 差分隐私 ───────────────────────────────────────────────
    proximal_mu: float = 0.01       # FedProx 近端正则系数
    use_dp: bool = False            # 是否启用 DP-SGD
    dp_noise_multiplier: float = 1.0  # DP 高斯噪声倍率 σ
    dp_clip_C: float = 1.0          # 梯度裁剪范数 C

    # ── 环境 ──────────────────────────────────────────────────────────
    num_stations: int = 8           # 充电站数量
    max_nodes: int = 800            # 路网最大节点数
    graphml_file: str = field(default_factory=lambda: _DEFAULT_GRAPHML)
    station_config_file: str = field(default_factory=lambda: _DEFAULT_STATION_CONFIG)
    station_id_key: str = "l0_station_nodes"
    graph_group: str = "l0"
    train_scale: str = "full"
    respawn_after_full_charge: bool = True
    base_seed: int = 42
    output_dir: Optional[str] = None
    checkpoint_basename: Optional[str] = None

    # ── 检查点 ────────────────────────────────────────────────────────
    checkpoint_interval: int = 20   # 每隔多少 episode 保存一次检查点

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def debug(cls) -> "TrainConfig":
        """极小规模，约 2 分钟，用于验证逻辑是否跑通。"""
        return cls(
            num_evs=10, episodes=20, steps_per_episode=300,
            fed_rounds_per_episode=5, batch_size=8,
            step_local_train_steps=2, step_train_interval=1,
            epsilon_final=0.80,
        )

    @classmethod
    def quick(cls) -> "TrainConfig":
        """快速验证模式，约 10 分钟，可观察到策略是否收敛。"""
        return cls(
            num_evs=50, episodes=60, steps_per_episode=600,
            fed_rounds_per_episode=1, batch_size=32,
            step_local_train_steps=2, step_train_interval=1,
            epsilon_final=0.30,
        )

    @classmethod
    def medium(cls) -> "TrainConfig":
        """中等规模，约 30 分钟，用于快速迭代实验。"""
        return cls(
            num_evs=60, episodes=100, steps_per_episode=1200,
            fed_rounds_per_episode=5, batch_size=128,
            step_local_train_steps=1, step_train_interval=4,
            epsilon_final=0.90,
        )

    @classmethod
    def ablation(cls) -> "TrainConfig":
        """图表示消融专用档位，保留调度压力但显著低于 medium 的训练成本。"""
        return cls(
            num_evs=40, episodes=30, steps_per_episode=600,
            fed_rounds_per_episode=3, batch_size=128,
            step_local_train_steps=1, step_train_interval=4,
            epsilon_final=0.90,
        )

    @classmethod
    def ablation_l0(cls) -> "TrainConfig":
        return cls(
            graphml_file=_DEFAULT_L0_GRAPHML,
            max_nodes=9999,
            station_config_file=_DEFAULT_STATION_CONFIG,
            station_id_key="l0_station_nodes",
            graph_group="l0",
        )

    @classmethod
    def ablation_l1(cls) -> "TrainConfig":
        return cls(
            graphml_file=_DEFAULT_L1_GRAPHML,
            max_nodes=9999,
            station_config_file=_DEFAULT_STATION_CONFIG,
            station_id_key="l1_station_nodes",
            graph_group="l1",
        )


@dataclass
class EvalConfig:
    """评估超参数。"""

    # ── 规模 ──────────────────────────────────────────────────────────
    episodes: int = 50              # 评估轮数（多轮取均值更稳定）
    steps_per_episode: int = 3000    # 每轮步数
    num_evs: int = 100
    num_stations: int = 8
    max_nodes: int = 9999           # 评估时使用完整路网
    use_real_map: bool = True       # True = 真实路网；False = 3x3 合成网格

    # ── 可复现性 ──────────────────────────────────────────────────────
    base_seed: int = 20260411       # 用于生成各 episode 种子的基础随机种子

    # ── 环境 ──────────────────────────────────────────────────────────
    graphml_file: str = field(default_factory=lambda: _DEFAULT_GRAPHML)
    station_config_file: str = field(default_factory=lambda: _DEFAULT_STATION_CONFIG)
    station_id_key: str = "l0_station_nodes"
    graph_group: str = "l0"
    eval_scale: str = "full"
    respawn_after_full_charge: bool = False

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def debug(cls) -> "EvalConfig":
        return cls(episodes=5, steps_per_episode=300)

    @classmethod
    def quick(cls) -> "EvalConfig":
        return cls(episodes=5, steps_per_episode=600)

    @classmethod
    def medium(cls) -> "EvalConfig":
        return cls(episodes=20, steps_per_episode=1200)

    @classmethod
    def ablation(cls) -> "EvalConfig":
        return cls(episodes=10, steps_per_episode=600)

    @classmethod
    def ablation_l0(cls) -> "EvalConfig":
        return cls(
            graphml_file=_DEFAULT_L0_GRAPHML,
            max_nodes=9999,
            station_config_file=_DEFAULT_STATION_CONFIG,
            station_id_key="l0_station_nodes",
            graph_group="l0",
        )

    @classmethod
    def ablation_l1(cls) -> "EvalConfig":
        return cls(
            graphml_file=_DEFAULT_L0_GRAPHML,
            max_nodes=9999,
            station_config_file=_DEFAULT_STATION_CONFIG,
            station_id_key="l0_station_nodes",
            graph_group="l1",
        )
