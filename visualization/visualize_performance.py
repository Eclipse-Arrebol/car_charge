"""
Analyze training metrics from saved JSON data.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    def __init__(self, data_path="results/training_data.json"):
        self.data_path = data_path
        self.data = None

    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing data file: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def compute_stats(self):
        if self.data is None:
            self.load()

        rewards = np.array(self.data.get("rewards", []), dtype=float)
        epsilons = np.array(self.data.get("epsilons", []), dtype=float)

        if rewards.size == 0:
            return {}

        slope = np.polyfit(np.arange(len(rewards)), rewards, 1)[0]
        stats = {
            "episodes": len(rewards),
            "reward_max": float(rewards.max()),
            "reward_min": float(rewards.min()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_slope": float(slope),
            "epsilon_start": float(epsilons[0]) if epsilons.size else None,
            "epsilon_end": float(epsilons[-1]) if epsilons.size else None,
        }
        return stats

    def plot_overview(self, window=10, save_path=None):
        if self.data is None:
            self.load()

        episodes = np.array(self.data.get("episodes", []))
        rewards = np.array(self.data.get("rewards", []), dtype=float)
        epsilons = np.array(self.data.get("epsilons", []), dtype=float)
        queues = np.array(self.data.get("avg_queue_lengths", []), dtype=float)
        overloads = np.array(
            self.data.get("voltage_violations", self.data.get("grid_overloads", [])),
            dtype=float,
        )

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Training Performance Overview", fontsize=14, fontweight='bold')

        # Reward + moving average
        ax = axes[0, 0]
        ax.plot(episodes, rewards, color="steelblue", alpha=0.3, label="reward")
        if rewards.size >= window:
            mv = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(episodes[window-1:], mv, color="crimson", label=f"{window}-avg")
        ax.set_title("Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Epsilon
        ax = axes[0, 1]
        ax.plot(episodes, epsilons, color="seagreen")
        ax.set_title("Epsilon")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(True, alpha=0.3)

        # Queue length
        ax = axes[1, 0]
        if queues.size:
            ax.plot(episodes, queues, color="darkorange")
            ax.set_ylabel("Avg queue")
        else:
            ax.text(0.5, 0.5, "No queue data", ha="center", va="center")
        ax.set_title("Queue length")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

        # Voltage violations (grid-side constraint indicator)
        ax = axes[1, 1]
        if overloads.size:
            ax.plot(episodes, overloads, color="firebrick")
            ax.set_ylabel("Voltage violations")
        else:
            ax.text(0.5, 0.5, "No voltage-violation data", ha="center", va="center")
        ax.set_title("Voltage violations")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    try:
        stats = analyzer.compute_stats()
        print("Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        analyzer.plot_overview()
    except FileNotFoundError as exc:
        print(exc)
        print("Run visualize_training.py to generate training_data.json first.")
