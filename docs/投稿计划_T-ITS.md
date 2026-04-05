# 投稿计划：IEEE Transactions on Intelligent Transportation Systems (T-ITS)

目标期刊：IEEE T-ITS，Q2，IF≈8  
备选期刊：Transportation Research Part C（同级别，审稿周期较短）

---

## 当前状态总结

| 模块 | 状态 | 说明 |
|------|------|------|
| 真实路网环境 | ✅ | OSMnx 珠江新城，BPR 动态拥堵，TOU 分时电价，电网建模 |
| GATv2 图注意力网络 | ✅ | FeatureEncoder 分组编码 + Dueling Head |
| 联邦训练框架 | ✅ | FedProx μ=0.01，双客户端，FedAvg 加权聚合 |
| 差分隐私 | ✅ | DP-SGD + RDPAccountant |
| 基线对比 | ⚠️ | 仅 4 个（Random/Greedy/DQN/FedDQN），需扩充 |
| 充电站数量 | ⚠️ | 仅 2 个，问题过于简单，Greedy 能与 RL 抗衡 |
| 消融实验 | ❌ | 尚未做 |
| 统计显著性 | ❌ | 尚未做（需多随机种子） |
| 文献对比方法 | ❌ | 尚未加入 |
| 论文写作 | ⚠️ | 模型章有初稿，实验章/结论章待写 |

---

## 阶段一：实验环境扩充

### 1.1 充电站增加到 4~6 个

**为什么重要**：2 个站时贪心算法（Greedy）与 RL 性能接近，无法体现图神经网络的拓扑感知优势。4~6 个站时组合复杂度提升，Greedy 退化，RL 优势明显。

**改动文件**：
- `visualization/run_training_real_map.py`：`NUM_STATS = 2` → `4`
- `evaluation/run_evaluation.py`：同步更新
- 验证 `RealTrafficEnv` 的 `station_node_ids` 自动选取高度节点逻辑是否支持 4+ 站

**验收标准**：FedDQN wait_time 领先 Greedy 差距 > 10%

---

### 1.2 加入文献对比方法（至少 2 个）

T-ITS 审稿人必然要求与已发表方法对比。建议加入：

| 方法 | 类型 | 说明 |
|------|------|------|
| **最近邻贪心（Nearest + Shortest Queue）** | 规则方法 | 综合距离和队列，比纯 Greedy 更强 |
| **DDPG / TD3** | 单机 RL | 连续动作空间的经典对比，体现联邦优势 |
| **FedAvg-DQN（无 FedProx）** | 消融基线 | 单独验证 FedProx 的贡献 |

**改动文件**：`evaluation/run_evaluation.py`，新增对应策略分支

---

### 1.3 完整规模训练

```bash
python main.py train-real        # 500 episodes，100 EVs，1000 steps
python main.py evaluate          # 50 轮均值
```

**验收标准**：结果优于 medium 档，FedDQN wait_time ≤ -30%

---

## 阶段二：消融实验

验证每个模块的独立贡献，这是 T-ITS 审稿标准要求。

| 消融组 | 说明 | 预期影响 |
|--------|------|---------|
| **A**：去掉 FeatureEncoder（原始 15 维直接进 GAT） | 验证分组编码的贡献 | wait_time 上升 |
| **B**：去掉 Dueling Head（单路 MLP 输出 Q） | 验证 Dueling 的贡献 | 训练稳定性下降 |
| **C**：FedProx → FedAvg（μ=0） | 验证近端约束的贡献 | FedDQN 性能下降 |
| **D**：完整 FedGRL | 基准 | 最优 |

**改动文件**：
- 消融 A/B：在 `GraphQNetwork` 加一个 `ablation` 模式参数
- 消融 C：`proximal_mu=0` 重新训练

---

## 阶段三：统计显著性

T-ITS 要求结果具有统计可信度。

- 使用 **5 个不同随机种子**（seed = 42, 123, 456, 789, 1024）分别训练+评估
- 报告均值 ± 标准差
- 对核心指标（wait_time、abandoned_evs）做 **t 检验**，p < 0.05

**改动文件**：`main.py` 新增 `--seed` 参数，评估脚本输出 std

---

## 阶段四：DP 权衡实验

获得 ε-性能曲线，支撑差分隐私创新点。

```bash
# σ=0.3（弱保护）
python main.py train-real --dp --dp-sigma 0.3
python main.py evaluate

# σ=1.0（中等，已有部分结果）
python main.py train-real --dp --dp-sigma 1.0
python main.py evaluate

# σ=2.0（强保护）
python main.py train-real --dp --dp-sigma 2.0
python main.py evaluate
```

**目标表格（论文 Table 2）**：

| σ | ε（隐私预算） | wait_time 变化 | abandoned_evs 变化 |
|---|-------------|---------------|-------------------|
| 无 DP | ∞ | 待填 | 待填 |
| 0.3 | 待填 | 待填 | 待填 |
| 1.0 | 待填 | +5%（已有） | +9%（已有） |
| 2.0 | 待填 | 待填 | 待填 |

**待开发**：`--dp-sigma` 命令行参数（当前需手动改代码）

---

## 阶段五：论文写作

### 结构规划

| 章节 | 内容 | 状态 |
|------|------|------|
| Abstract | 问题、方法、主要结果 | ❌ |
| 1. Introduction | 背景、动机、贡献点列表 | ❌ |
| 2. Related Work | EV 充电导航、联邦 RL、图神经网络 | ❌ |
| 3. System Model | 路网、EV、充电站、电网建模 | ⚠️ 初稿 |
| 4. FedGRL 算法 | FeatureEncoder、GATv2、FedProx、DP-SGD | ⚠️ 初稿 |
| 5. Experiments | 四方对比、消融、DP 权衡、统计分析 | ❌ |
| 6. Conclusion | 三个贡献总结、局限性、未来工作 | ❌ |

### 三个核心贡献（投稿时的卖点）

1. **负载预测感知的 GATv2 编码**：FeatureEncoder 将异质节点特征分组编码（队列/成本/空间/EV状态），结合 EMA 预测到达量，赋予模型拥堵预判能力
2. **FedProx 联邦强化学习**：多区域 EV 数据协同训练，近端约束防止客户端漂移，聚合正则化提升训练稳定性
3. **DP-SGD 差分隐私**：Opacus RDPAccountant 精确追踪隐私预算，量化 ε-性能权衡曲线

---

## 执行顺序与优先级

```
阶段一（扩充环境）
  ① 充电站 2→4 个，重跑完整规模训练
  ② 加入 Nearest+ShortestQueue 和 FedAvg-DQN 两个对比方法

阶段二（消融实验）
  ③ 4 组消融，各跑完整规模

阶段三（统计显著性）
  ④ 5 个随机种子，均值±std，t 检验

阶段四（DP 实验）
  ⑤ 加 --dp-sigma 参数，跑 σ=0.3/1.0/2.0 三组

阶段五（写作）
  ⑥ 边跑实验边写 Related Work 和 System Model
  ⑦ 实验数据齐全后写 Experiments 章
  ⑧ 最后写 Abstract 和 Introduction
```

---

## 关键文件清单

| 文件 | 当前状态 | 待修改内容 |
|------|---------|-----------|
| `env/RealTrafficEnv.py` | ✅ | 支持 4+ 充电站（验证） |
| `agents/GraphQNetwork.py` | ✅ | 加 ablation 模式参数 |
| `agents/FederatedDQN.py` | ✅ | 无需修改 |
| `evaluation/run_evaluation.py` | ✅ | 加新对比方法、多种子输出 std |
| `main.py` | ✅ | 加 `--seed`、`--dp-sigma` 参数 |
| `docs/模型章初稿.md` | ⚠️ | 补 FeatureEncoder/Dueling 描述 |
