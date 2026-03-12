"""
训练过程可视化工具
可视化内容：
- 每轮奖励曲线
- Epsilon衰减曲线
- 平均奖励（滑动窗口）
- 训练损失（如果记录）
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 数据记录
        self.episodes = []
        self.rewards = []
        self.epsilons = []
        self.losses = []
        self.avg_queue_lengths = []
        self.grid_overloads = []
        
    def add_episode_data(self, episode, reward, epsilon, loss=None, 
                        avg_queue=None, overload_count=None):
        """添加每一轮的训练数据"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        
        if loss is not None:
            self.losses.append(loss)
        if avg_queue is not None:
            self.avg_queue_lengths.append(avg_queue)
        if overload_count is not None:
            self.grid_overloads.append(overload_count)
    
    def plot_training_curves(self, window_size=10):
        """绘制训练曲线（奖励、epsilon等）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN训练过程可视化', fontsize=16, fontweight='bold')
        
        # 1. 奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(self.episodes, self.rewards, alpha=0.3, color='blue', label='原始奖励')
        
        # 计算滑动平均
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax1.plot(self.episodes[window_size-1:], moving_avg, 
                    color='red', linewidth=2, label=f'{window_size}轮平均')
        
        ax1.set_xlabel('训练轮数 (Episode)')
        ax1.set_ylabel('总奖励')
        ax1.set_title('训练奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Epsilon衰减曲线
        ax2 = axes[0, 1]
        ax2.plot(self.episodes, self.epsilons, color='green', linewidth=2)
        ax2.set_xlabel('训练轮数 (Episode)')
        ax2.set_ylabel('Epsilon (探索率)')
        ax2.set_title('Epsilon衰减曲线')
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均队列长度（如果有数据）
        ax3 = axes[1, 0]
        if self.avg_queue_lengths:
            ax3.plot(self.episodes, self.avg_queue_lengths, 
                    color='orange', linewidth=2)
            ax3.set_xlabel('训练轮数 (Episode)')
            ax3.set_ylabel('平均队列长度')
            ax3.set_title('充电站平均排队情况')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '暂无队列数据', 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. 电压越限次数（如果有数据）
        ax4 = axes[1, 1]
        if self.grid_overloads:
            ax4.plot(self.episodes, self.grid_overloads, 
                    color='red', linewidth=2, marker='o')
            ax4.set_xlabel('训练轮数 (Episode)')
            ax4.set_ylabel('越限次数')
            ax4.set_title('电压越限情况')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '暂无电压越限数据', 
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
        plt.close()
        
    def plot_reward_distribution(self, bins=30):
        """绘制奖励分布直方图"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.rewards, bins=bins, color='skyblue', 
                edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(self.rewards), color='red', 
                   linestyle='--', linewidth=2, label=f'平均值: {np.mean(self.rewards):.2f}')
        plt.axvline(np.median(self.rewards), color='green', 
                   linestyle='--', linewidth=2, label=f'中位数: {np.median(self.rewards):.2f}')
        
        plt.xlabel('奖励值')
        plt.ylabel('频次')
        plt.title('训练奖励分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(self.save_dir, 'reward_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奖励分布图已保存: {save_path}")
        plt.close()
        
    def plot_learning_progress(self, milestone_interval=10):
        """绘制学习进度对比（分阶段）"""
        if len(self.rewards) < milestone_interval * 2:
            print("数据不足，无法绘制学习进度对比")
            return
        
        # 分成几个阶段
        num_milestones = len(self.rewards) // milestone_interval
        stage_rewards = []
        stage_labels = []
        
        for i in range(num_milestones):
            start_idx = i * milestone_interval
            end_idx = (i + 1) * milestone_interval
            stage_rewards.append(self.rewards[start_idx:end_idx])
            stage_labels.append(f'{start_idx+1}-{end_idx}轮')
        
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(stage_rewards, labels=stage_labels, patch_artist=True)
        
        # 美化箱线图
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        plt.xlabel('训练阶段')
        plt.ylabel('奖励值')
        plt.title('不同训练阶段的奖励对比（箱线图）')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(self.save_dir, 'learning_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习进度对比图已保存: {save_path}")
        plt.close()
        
    def save_data(self, filename='training_data.json'):
        """保存训练数据到JSON文件"""
        data = {
            'episodes': self.episodes,
            'rewards': self.rewards,
            'epsilons': self.epsilons,
            'losses': self.losses,
            'avg_queue_lengths': self.avg_queue_lengths,
            # 兼容字段：历史版本叫 grid_overloads。
            # 当前含义是：每个 episode 内累计的“电压越限次数”(voltage_violations)。
            'grid_overloads': self.grid_overloads,
            'voltage_violations': self.grid_overloads,
        }
        
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"训练数据已保存: {save_path}")
        
    def load_data(self, filename='training_data.json'):
        """从JSON文件加载训练数据"""
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            print(f"文件不存在: {load_path}")
            return False
            
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.episodes = data['episodes']
        self.rewards = data['rewards']
        self.epsilons = data['epsilons']
        self.losses = data.get('losses', [])
        self.avg_queue_lengths = data.get('avg_queue_lengths', [])
        self.grid_overloads = data.get('voltage_violations', data.get('grid_overloads', []))
        
        print(f"训练数据已加载: {load_path}")
        return True
        
    def generate_summary_report(self):
        """生成训练摘要报告"""
        if not self.rewards:
            print("暂无训练数据")
            return
        
        report = f"""
{'='*60}
训练摘要报告
{'='*60}
总训练轮数: {len(self.episodes)}
奖励统计:
  - 最高奖励: {max(self.rewards):.2f}
  - 最低奖励: {min(self.rewards):.2f}
  - 平均奖励: {np.mean(self.rewards):.2f}
  - 标准差: {np.std(self.rewards):.2f}

探索率 (Epsilon):
  - 初始值: {self.epsilons[0]:.3f}
  - 最终值: {self.epsilons[-1]:.3f}
  - 衰减率: {(self.epsilons[0] - self.epsilons[-1]) / self.epsilons[0] * 100:.1f}%

最近10轮平均奖励: {np.mean(self.rewards[-10:]):.2f}
"""
        
        if self.avg_queue_lengths:
            report += f"\n平均队列长度: {np.mean(self.avg_queue_lengths):.2f}\n"
        
        if self.grid_overloads:
            report += f"总电压越限次数: {sum(self.grid_overloads)}\n"
        
        report += f"{'='*60}\n"
        
        print(report)
        
        # 保存报告到文本文件
        report_path = os.path.join(self.save_dir, 'training_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"摘要报告已保存: {report_path}")
        

# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 创建可视化器
    viz = TrainingVisualizer()
    
    # 模拟一些训练数据
    np.random.seed(42)
    for episode in range(1, 51):
        # 模拟奖励逐渐增加（学习效果）
        base_reward = -100 + episode * 2
        reward = base_reward + np.random.randn() * 20
        
        # 模拟epsilon衰减
        epsilon = max(0.05, 1.0 * (0.995 ** episode))
        
        # 模拟队列长度逐渐减少
        avg_queue = max(0, 5 - episode * 0.08 + np.random.randn() * 0.5)
        
        # 模拟电压越限次数
        overload = max(0, int(10 - episode * 0.15 + np.random.randn() * 2))
        
        viz.add_episode_data(episode, reward, epsilon, 
                           avg_queue=avg_queue, 
                           overload_count=overload)
    
    # 生成所有可视化
    viz.plot_training_curves()
    viz.plot_reward_distribution()
    viz.plot_learning_progress()
    viz.save_data()
    viz.generate_summary_report()
    
    print("\n所有训练可视化已完成！")
