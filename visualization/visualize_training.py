"""
训练过程可视化工具。

输出内容：
- 每轮奖励曲线
- Epsilon 衰减曲线
- 平均队列长度
- 电压越限次数
- 奖励分布图
- 分阶段奖励箱线图
- 训练数据 JSON
- 训练摘要文本
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _build_font_properties():
    windows_font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    windows_font_files = [
        os.path.join(windows_font_dir, "msyh.ttc"),
        os.path.join(windows_font_dir, "msyhbd.ttc"),
        os.path.join(windows_font_dir, "simhei.ttf"),
        os.path.join(windows_font_dir, "simsun.ttc"),
    ]

    for font_path in windows_font_files:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    available_fonts = [name for name in font_candidates if name in installed_fonts]
    if not available_fonts:
        available_fonts = ["DejaVu Sans"]

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = available_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False

    return font_manager.FontProperties(family=available_fonts[0]), available_fonts


FONT_PROP, AVAILABLE_FONTS = _build_font_properties()


class TrainingVisualizer:
    """训练过程可视化器。"""

    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.episodes = []
        self.rewards = []
        self.epsilons = []
        self.losses = []
        self.avg_queue_lengths = []
        self.grid_overloads = []

    @staticmethod
    def _apply_axis_fonts(ax):
        ax.title.set_fontproperties(FONT_PROP)
        ax.xaxis.label.set_fontproperties(FONT_PROP)
        ax.yaxis.label.set_fontproperties(FONT_PROP)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)

    def add_episode_data(
        self,
        episode,
        reward,
        epsilon,
        loss=None,
        avg_queue=None,
        overload_count=None,
    ):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)

        if loss is not None:
            self.losses.append(loss)
        if avg_queue is not None:
            self.avg_queue_lengths.append(avg_queue)
        if overload_count is not None:
            self.grid_overloads.append(overload_count)

    def plot_training_curves(self, window_size=20):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("FedDQN 训练曲线", fontsize=16, fontweight="bold", fontproperties=FONT_PROP)

        auto_window = window_size
        if len(self.rewards) <= 40:
            auto_window = min(window_size, max(5, len(self.rewards) // 3))
        auto_window = max(2, min(auto_window, max(2, len(self.rewards))))

        ax1 = axes[0, 0]
        ax1.plot(self.episodes, self.rewards, alpha=0.25, color="steelblue", label="单轮奖励")
        if len(self.rewards) >= auto_window:
            moving_avg = np.convolve(
                self.rewards,
                np.ones(auto_window) / auto_window,
                mode="valid",
            )
            ax1.plot(
                self.episodes[auto_window - 1:],
                moving_avg,
                color="navy",
                linewidth=2.5,
                label=f"滑动平均 (w={auto_window})",
            )
        ax1.set_xlabel("训练轮数 (Episode)")
        ax1.set_ylabel("每次 EV 决策平均奖励")
        ax1.set_title("训练奖励曲线")
        ax1.legend(prop=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        self._apply_axis_fonts(ax1)

        ax2 = axes[0, 1]
        ax2.plot(self.episodes, self.epsilons, color="green", linewidth=2)
        ax2.set_xlabel("训练轮数 (Episode)")
        ax2.set_ylabel("Epsilon（探索率）")
        ax2.set_title("Epsilon 衰减曲线")
        ax2.grid(True, alpha=0.3)
        self._apply_axis_fonts(ax2)

        ax3 = axes[1, 0]
        if self.avg_queue_lengths:
            ax3.plot(self.episodes, self.avg_queue_lengths, color="orange", linewidth=2)
            ax3.set_xlabel("训练轮数 (Episode)")
            ax3.set_ylabel("平均队列长度")
            ax3.set_title("充电站平均排队情况")
            ax3.grid(True, alpha=0.3)
            self._apply_axis_fonts(ax3)
        else:
            ax3.text(
                0.5,
                0.5,
                "暂无队列数据",
                ha="center",
                va="center",
                fontsize=12,
                fontproperties=FONT_PROP,
            )
            ax3.axis("off")

        ax4 = axes[1, 1]
        if self.grid_overloads:
            ax4.plot(self.episodes, self.grid_overloads, color="red", linewidth=2, marker="o")
            ax4.set_xlabel("训练轮数 (Episode)")
            ax4.set_ylabel("越限次数")
            ax4.set_title("电压越限情况")
            ax4.grid(True, alpha=0.3)
            self._apply_axis_fonts(ax4)
        else:
            ax4.text(
                0.5,
                0.5,
                "暂无电压越限数据",
                ha="center",
                va="center",
                fontsize=12,
                fontproperties=FONT_PROP,
            )
            ax4.axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"训练曲线已保存: {save_path}")
        plt.close()

    def plot_reward_distribution(self, bins=30):
        plt.figure(figsize=(10, 6))
        plt.hist(self.rewards, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            np.mean(self.rewards),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"平均值: {np.mean(self.rewards):.2f}",
        )
        plt.axvline(
            np.median(self.rewards),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数: {np.median(self.rewards):.2f}",
        )

        plt.xlabel("奖励值", fontproperties=FONT_PROP)
        plt.ylabel("频次", fontproperties=FONT_PROP)
        plt.title("训练奖励分布直方图", fontproperties=FONT_PROP)
        plt.legend(prop=FONT_PROP)
        plt.grid(True, alpha=0.3, axis="y")
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)

        save_path = os.path.join(self.save_dir, "reward_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"奖励分布图已保存: {save_path}")
        plt.close()

    def plot_learning_progress(self, milestone_interval=10):
        if len(self.rewards) < milestone_interval * 2:
            print("数据不足，无法绘制学习进度对比")
            return

        num_milestones = len(self.rewards) // milestone_interval
        stage_rewards = []
        stage_labels = []

        for i in range(num_milestones):
            start_idx = i * milestone_interval
            end_idx = (i + 1) * milestone_interval
            stage_rewards.append(self.rewards[start_idx:end_idx])
            stage_labels.append(f"{start_idx + 1}-{end_idx}轮")

        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(stage_rewards, labels=stage_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        plt.xlabel("训练阶段", fontproperties=FONT_PROP)
        plt.ylabel("奖励值", fontproperties=FONT_PROP)
        plt.title("不同训练阶段的奖励对比（箱线图）", fontproperties=FONT_PROP)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)

        save_path = os.path.join(self.save_dir, "learning_progress.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"学习进度对比图已保存: {save_path}")
        plt.close()

    def save_data(self, filename="training_data.json"):
        data = {
            "episodes": self.episodes,
            "rewards": self.rewards,
            "epsilons": self.epsilons,
            "losses": self.losses,
            "avg_queue_lengths": self.avg_queue_lengths,
            "grid_overloads": self.grid_overloads,
            "voltage_violations": self.grid_overloads,
        }

        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"训练数据已保存: {save_path}")

    def load_data(self, filename="training_data.json"):
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            print(f"文件不存在: {load_path}")
            return False

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.episodes = data["episodes"]
        self.rewards = data["rewards"]
        self.epsilons = data["epsilons"]
        self.losses = data.get("losses", [])
        self.avg_queue_lengths = data.get("avg_queue_lengths", [])
        self.grid_overloads = data.get("voltage_violations", data.get("grid_overloads", []))

        print(f"训练数据已加载: {load_path}")
        return True

    def generate_summary_report(self):
        if not self.rewards:
            print("暂无训练数据")
            return

        report = f"""
============================================================
训练摘要报告
============================================================
总训练轮数: {len(self.episodes)}
奖励统计:
  - 最高奖励: {max(self.rewards):.2f}
  - 最低奖励: {min(self.rewards):.2f}
  - 平均奖励: {np.mean(self.rewards):.2f}
  - 标准差: {np.std(self.rewards):.2f}

探索率 (Epsilon):
  - 初始值: {self.epsilons[0]:.3f}
  - 最终值: {self.epsilons[-1]:.3f}
  - 衰减率: {(self.epsilons[0] - self.epsilons[-1]) / max(self.epsilons[0], 1e-6) * 100:.1f}%

最近10轮平均奖励: {np.mean(self.rewards[-10:]):.2f}
"""

        if self.avg_queue_lengths:
            report += f"\n平均队列长度: {np.mean(self.avg_queue_lengths):.2f}\n"
        if self.grid_overloads:
            report += f"总电压越限次数: {sum(self.grid_overloads)}\n"

        report += "============================================================\n"

        print(report)

        report_path = os.path.join(self.save_dir, "training_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"摘要报告已保存: {report_path}")


if __name__ == "__main__":
    np.random.seed(42)
    viz = TrainingVisualizer()

    for episode in range(1, 51):
        base_reward = -100 + episode * 2
        reward = base_reward + np.random.randn() * 20
        epsilon = max(0.05, 1.0 * (0.995 ** episode))
        avg_queue = max(0, 5 - episode * 0.08 + np.random.randn() * 0.5)
        overload = max(0, int(10 - episode * 0.15 + np.random.randn() * 2))
        viz.add_episode_data(
            episode,
            reward,
            epsilon,
            avg_queue=avg_queue,
            overload_count=overload,
        )

    viz.plot_training_curves()
    viz.plot_reward_distribution()
    viz.plot_learning_progress()
    viz.save_data()
    viz.generate_summary_report()

    print(f"\n可用字体: {AVAILABLE_FONTS}")
    print("所有训练可视化已完成。")
