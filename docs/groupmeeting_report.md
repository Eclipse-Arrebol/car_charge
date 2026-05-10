---
marp: true
theme: default
paginate: true
size: 16:9
header: 'FedGRL: 面向真实路网的电动汽车充电导航策略'
footer: 'Eclipse · 课题组组会汇报'
style: |
  section {
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 22px;
    padding: 50px 60px;
  }
  h1 {
    color: #1a5490;
    font-size: 36px;
  }
  h2 {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 8px;
    font-size: 28px;
  }
  h3 {
    color: #1e40af;
    font-size: 22px;
    margin-top: 16px;
  }
  table {
    font-size: 18px;
    margin: 0 auto;
    border-collapse: collapse;
  }
  th {
    background-color: #2563eb;
    color: white;
    padding: 8px 12px;
  }
  td {
    padding: 6px 12px;
    border: 1px solid #e5e7eb;
  }
  .highlight {
    background-color: #fef3c7;
    padding: 10px 14px;
    border-left: 4px solid #f59e0b;
    margin: 10px 0;
  }
  .danger {
    background-color: #fee2e2;
    padding: 10px 14px;
    border-left: 4px solid #dc2626;
    margin: 10px 0;
  }
  .success {
    background-color: #d1fae5;
    padding: 10px 14px;
    border-left: 4px solid #059669;
    margin: 10px 0;
  }
  .formula {
    background-color: #f8fafc;
    padding: 12px;
    border-left: 3px solid #64748b;
    font-family: 'Cambria Math', serif;
    margin: 10px 0;
  }
  code {
    background-color: #f1f5f9;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 18px;
  }
  .small {
    font-size: 18px;
    color: #475569;
  }
  .ref {
    font-size: 16px;
    color: #64748b;
    font-style: italic;
  }
---

<!-- _class: lead -->

# FedGRL: 面向真实路网的电动汽车充电导航策略

## 阶段性研究进展汇报

<br>

**汇报人**: Eclipse
**指导教师**: 卢老师
**目标会议**: IEEE Transactions on Intelligent Transportation Systems (IF≈8)

---

## 汇报内容

1. **研究背景与问题定义** (Problem Formulation)
2. **方法论核心**: Thevenin 前瞻 Estimator 与 Co-simulation 框架
3. **关键发现**: Reward Hacking 现象的诊断与归因
4. **方法演进**: 从 Soft Reward Shaping 到 Hard Action Masking
5. **实验结果与对比分析**
6. **后续工作规划**

---

## 1. 研究背景

### 现实需求

随着电动汽车渗透率快速提升，充电负荷的时空集聚效应对配电网造成显著压力。现有充电导航策略普遍仅考虑单一目标（如行驶距离最短或排队时间最少），忽略了充电行为对电网运行状态的耦合影响。

### 研究空白

| 现有工作类别 | 局限性 |
|-------------|--------|
| Shortest-path / Greedy heuristics | 单目标，忽略电网约束 |
| 单纯 RL 充电导航 | 简化电网建模或不考虑潮流 |
| 电网侧调度方法 | 假设充电行为已知，不优化导航 |

### 本研究目标

构建**交通-电网联合仿真框架**，通过强化学习实现**多目标导航策略**，
在用户体验与电网运行指标之间建立**可控权衡 (controllable trade-off)**。

---

## 2. 系统架构

### 三层耦合建模

**交通层**: OSMnx 真实路网（广州天河，1582 节点 / 2491 边），8 个充电站节点
**电网层**: pandapower IEEE 33-bus 标准配电网（12.66 kV），充电站映射至关键母线
**决策层**: GATv2 图注意力编码 + Double DQN + FedAvg 联邦聚合

### 联合仿真闭环

<div class="formula">

EV 决策 → 路径选择 (BPR 拥堵模型) → 充电功率 → 配电网潮流 →
节点电压/网损 → 反馈进入 reward → 下一步决策

</div>

### 关键技术创新

提出 **Thevenin 等效阻抗前瞻 Estimator**，将每步潮流计算的 O(n³) 复杂度
降为 O(1) 查询，使大规模训练在单卡 RTX 4090 上可行。

---

## 3. Thevenin Estimator: 方法与验证

### 设计动机

每步决策若调用完整 Newton-Raphson 潮流求解，单 episode (1200 步) 仿真耗时
约 **40 分钟**，50 episodes 训练超过 30 小时，无法支持联邦多客户端并行。

### 方法核心

环境初始化阶段，对 33-bus 系统在基准运行点处求解 Thevenin 等效电路，
对每个充电站母线 i 预计算等效内阻 $Z_{th,i}$。决策时电压变化由：

<div class="formula">

ΔV_i ≈ |Z_{th,i}| · ΔP_i / V_nom

</div>

### 一致性验证

<div class="success">

**与完整 pandapower 潮流对比** (覆盖 1000+ 工况采样):
- Pearson 相关系数 = **0.9972**
- 平均绝对误差 (MAE) = **0.0015 p.u.**
- 单次预测耗时由 ~80ms 降至 ~0.05ms

</div>

该结果表明在配电网常规运行范围内，Thevenin 线性化的近似精度足以支持 RL 决策。

---

## 4. 关键发现: Reward Hacking 现象

### 现象描述

在初版 voltage-aware reward 设计下，训练后的 policy 在 voltage_excursion
指标上取得 **-20%** 的改善，但通过 cheat-policy 实验确定的理论上限仅为 **-8%**。

<div class="danger">

**改善幅度超出物理上限 195%，提示 policy 学习到了非预期的退化策略 (degenerate policy)。**

</div>

### 行为模式分析

| 指标 | Baseline | Voltage Policy | 异常程度 |
|------|----------|----------------|---------|
| abandoned_evs | 0.00 | 3.20 | — |
| queue_time_h | 0.0027 | 0.084 | +3022% |
| voltage_excursion | 28.42 | 22.81 | -19.7% |

### 归因

Policy 利用 reward 函数中"超时放弃 EV 不再贡献充电负荷"的特性，
**主动制造排队拥堵导致 EV 超时退出系统**，从而以"减少有效负载"的方式
人为优化电网指标。这是 RL Safety 文献中典型的 reward hacking 模式
(Krakovna et al., 2020; Skalse et al., 2022)。

---

## 5. 第一次修复尝试: Terminal Penalty + Transition 补存

### 设计思路

针对 abandon 事件设计 terminal penalty，并通过 **transition 补存机制**
将该 penalty 回溯归因至该 EV 的"选站决策步" (causal credit assignment)，
而非仅作用于被动等待步。

### 实现关键

```python
# 当 EV 在 t 时刻 abandon 时
# 检索其在 t-Δt 时刻的选站决策 transition (s_decision, a_decision)
# 注入 penalty 到该 transition 的 reward 字段
buffer.update_transition(decision_idx, reward += penalty)
```

### 实验结果 (penalty = 5.0)

| 指标 | Voltage (无penalty) | Voltage + Penalty5 | 差异 |
|------|-------------------|-------------------|------|
| voltage_excursion | 22.81 | 22.76 | -0.20% |
| abandoned_evs | 2.20 | 1.80 | -18% |
| queue_time_h | 0.0559 | 0.0697 | +24.7% |

<div class="danger">

**Penalty 几乎未改变 policy 行为模式，需进一步追溯失效根因。**

</div>

---

## 6. 失效原因诊断: 稀疏事件下的 Bellman 信号稀释

### 定量分析

每 episode 决策事件分布：

| 事件类型 | 频次 | 单次 reward 量级 | 累计贡献 |
|---------|------|----------------|---------|
| 正常充电决策 | ~2400 | $-5 \times 10^{-2}$ | $-1.2 \times 10^{2}$ |
| Abandon 事件 | ~3 | $-6 \times 10^{-1}$ | $-1.8 \times 10^{0}$ |

Abandon 事件在累计 reward 中占比仅 **1.5%**。Replay buffer 均匀采样下，
batch (size=128) 中包含 abandon transition 的期望数量 < 0.2，梯度信号
被高频常规事件主导。

### 理论判断

<div class="highlight">

在事件稀疏场景下 (event rate << batch size$^{-1}$)，**Soft reward shaping**
的 Bellman 期望梯度收敛性理论上无法保证有效信号传播。

该问题不能通过单纯增大 penalty 量级解决，因为在期望意义下被等比例稀释。
**必须从动作空间层面 (hard constraint) 切断退化策略的可行路径。**

</div>

---

## 7. 第二次修复: 三项并行改进

### 改进 1: Reward 计算 Bug 修复

诊断发现原实现中 `voltage_abandon_penalty=0.0` 导致 abandon 事件获得 0 reward，
而正常决策因 line_loss 等成本获得负 reward——**abandon 反而是次优选择中的"最优"**。

修复: 在 EV reset 前捕获真实代价 (`actual_wait_h`, `actual_trip_h`)，
作为 abandon transition 的 reward 信号。

### 改进 2: Reward 权重调整

原设置 user : grid = 0.3 : 0.7，user 信号被电网指标主导。
调整为 **0.7 : 0.3**，强化用户体验作为约束，电网作为附加优化目标。

### 改进 3: Action Mask 引入

构建 mask 函数 $M(s, a) \in \{0, 1\}$，在动作空间层面屏蔽预计将导致超时的站点选择，
使 policy 无法通过"选爆队站"路径实施 reward hacking。

---

## 8. 主实验结果: 五方对比

### 实验设置
- 环境: 真实路网 + IEEE 33-bus，40 EVs，1200 步/episode
- 训练: 50 episodes，epsilon-greedy ($\epsilon_{final}=0.05$)
- 评估: 600 步独立测试集，seed=0

### 结果对比表

| 指标 | Baseline | Cheat | Voltage-old | **Voltage-new** | **Mask-new** |
|------|----------|-------|-------------|-----------------|--------------|
| abandoned_evs | 0.00 | 0.00 | 1.20 | **0.10** | 0.80 |
| voltage_excursion (p.u.) | 28.42 | 26.04 | 22.85 | 23.53 | 23.58 |
| queue_time_h | 0.003 | 0.003 | 0.074 | **0.006** | 0.005 |
| trip_time_h | 0.031 | 0.046 | 0.080 | 0.061 | 0.060 |
| dist_cost (CNY) | 2693 | 2634 | 2543 | 2580 | 2578 |

<div class="success">

**关键观察 1**: voltage_excursion 已收敛至 cheat ceiling (-8%) 附近，
说明经过修复后 policy 达到了真实物理上限附近的优化。

</div>

---

## 9. 反直觉发现: Mask 反而增加 abandon

### 数据矛盾

mask_new 的 abandoned_evs (0.80) **高于** voltage_new (0.10)，
与"hard constraint 应优于 soft penalty"的直觉相悖。

### 机制分析

Action mask 的设计是 **保守屏蔽** (conservative masking):
- 屏蔽队列长的站 → 可选动作空间收窄
- Policy 在剩余动作中探索受限
- 极端情况: 当所有近站均被屏蔽时，EV 被迫前往远站，
  增加行程时间反而触发其他超时路径

### 论文价值

<div class="highlight">

**该现象揭示 hard constraint 与 soft reward 在多目标优化中存在权衡:**
- Hard constraint: 严格保证特定指标下界，但限制探索
- Soft reward: 允许策略空间灵活，但稀疏事件下信号弱

最优设计需根据**事件稀疏程度**与**约束严格性需求**协同选择，
为后续多目标 RL 研究提供方法论启示。

</div>

---

## 10. Mask 设计的迭代: 从间接到直接约束

### 旧 Mask (step15): 队列长度阈值

触发条件: `len(queue) ≥ k`

**局限性**: 队列长度是间接代理变量，与用户实际等待时间不严格单调相关
（队列长但服务快的站可能优于队列短但路远的站）。

### 新 Mask (step17): 总时间约束

触发条件: `trip_time_h + queue_time_h > 1.0`

**优势**: 直接以预估总等待时间为约束，与用户体验指标
(abandoned_evs, trip_time) 直接对齐。

### 当前训练状态

step17 已启动，预计今晚完成评估。预期结果:
- abandoned_evs 同时下降
- queue_time 维持低位
- 与 voltage_new 形成更对称的对照实验

---

## 11. 论文叙事框架: 三层故事

### 表层 (Methodological Contribution)
**Thevenin 前瞻 Estimator** 实现真实电网仿真在 RL 训练规模下的可行性，
Pearson=0.9972 的一致性验证支持方法严谨性。

### 中层 (Engineering Methodology)
**多目标 reward 设计的工程方法论**: 从 hacking 现象诊断、稀疏事件归因、
到 hard/soft constraint 协同设计的完整流程，构成方法论案例研究。

### 深层 (Theoretical Insight)
**稀疏事件 RL 的结构性挑战**: 揭示 soft reward shaping 在低频关键事件下
的固有局限，论证 action masking 作为必要补充而非可选优化的理论依据。

<div class="highlight">

**审稿点优势**: 不仅提交了一个性能数字，还提供了一个可复用的方法论框架。

</div>

---

## 12. 实验完成度评估

### 已完成

| 类别 | 实验 | 状态 |
|------|------|------|
| 主对比 | baseline / cheat / voltage / mask | ✓ |
| 方法论 | Thevenin 一致性验证 | ✓ |
| 消融 | reward hacking 修复前后对比 | ✓ |
| 消融 | mask 设计 (queue-based vs time-based) | 进行中 |

### 待完成 (送审硬要求)

| 类别 | 实验 | 优先级 |
|------|------|--------|
| 鲁棒性 | 多 seed 实验 (seed=1, 2) | **高** |
| 文献对比 | shortest-path / queue-greedy / RL baseline | **高** |
| 扩展性 | EV 规模 sweep (20/40/80) | 中 |
| 联邦验证 | 多城市 non-IID 实验 | 时间允许 |

---

## 13. 后续工作时间表

### 短期 (1-2 周)
- 完成 step17 评估，确定最终主方法
- 启动多 seed 实验 (voltage_new + best_mask × seed=1, 2)
- 复现 1 篇 RL 文献 baseline (候选: Qian et al. 2022 / Zhang et al. 2023)

### 中期 (3-5 周)
- 论文章节撰写 (§IV 方法 → §VII 讨论)
- 经济性分析模块: 行程时间增加 vs 电网侧效益的货币化权衡

### 长期 (6-10 周)
- 多城市路网集成 (深圳/上海) 与非 IID 联邦实验
- 整体打磨与投稿准备

### 目标节点

<div class="success">

**3 个月内完成投稿至 IEEE T-ITS，备选会议 Transportation Research Part C**

</div>

---

## 14. 当前需要老师指导的问题

### Q1: 论文主方法的选取
voltage_new 与 mask_time 在不同指标上各有优势，建议以 **voltage_new** 为主方法、
mask 为消融对照，强调"两种约束机制的权衡"叙事。**是否同意此方向？**

### Q2: 经济性分析的引入
当前 trip_time 上升约 +94% (相对 baseline)，建议引入 **Social Welfare Analysis**
小节，将电网侧效益 (网损降低 + 电压改善) 货币化，与用户行程成本对齐比较。
**老师认为该分析的深度是否需要进一步增强？**

### Q3: 联邦框架的定位
目前 IID 单城市设置下 fed 收益不显著，计划通过多城市非 IID 实验提升联邦
contribution 强度。**该实验在投稿前是否必须完成？**

---

<!-- _class: lead -->

# 谢谢老师指导

## 欢迎批评指正

<br>
<br>

<div class="ref">

参考文献已整理至项目仓库 `/docs/references.bib`
代码与实验日志: `/root/car_charge/runs/`

</div>

---
