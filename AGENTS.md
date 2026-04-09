# FedGRL 项目交接文档

## 项目概况

- **课题**：面向真实路网的电动汽车充电导航策略（FedGRL）
- **目标**：发 IEEE T-ITS（Q2，IF≈8）
- **代码位置**：`G:\交电demo`（本地）/ `/root/car_charge`（服务器）

---

## 系统架构

| 模块 | 技术方案 | 代码位置 |
|------|----------|----------|
| 路网 | OSMnx 珠江新城真实路网（graphml） | `zhujiang_new_town.graphml` |
| 特征编码 | FeatureEncoder 分组编码（4组）+ LayerNorm | `agents/GraphQNetwork.py` |
| 决策网络 | GATv2 图注意力 + Dueling Head | `agents/GraphQNetwork.py` |
| 训练框架 | Double DQN + Action Mask + FedProx | `agents/FederatedDQN.py` |
| 联邦框架 | FedAvg 加权聚合，proximal_mu=0.01 | `agents/FederatedDQN.py` |
| 差分隐私 | DP-SGD + Opacus RDPAccountant | `agents/FederatedDQN.py` |
| 环境 | BPR 动态拥堵 + TOU 分时电价 | `env/Traffic.py`、`env/RealTrafficEnv.py` |
| 评估 | 四方对比（Random/Greedy/DQN/FedDQN） | `evaluation/run_evaluation.py` |

---

## 当前最优实验结果

**第四轮（FedProx μ=0.01，medium 模式，2 个充电站）**

| 指标 | Random | Greedy | DQN | FedDQN |
|------|--------|--------|-----|--------|
| avg_wait_time_min | 223.20 | 176.55 (-21%) | 197.10 (-12%) | **164.40 (-26%)** |
| abandoned_evs | 39.50 | 30.55 (-23%) | 33.25 (-16%) | **28.25 (-28%)** |
| total_charging_cost | 1101.83 | 1113.84 (+1%) | 1062.96 (-4%) | **1100.79 (-0%)** |

> 详细历史结果见 `docs/评估结果记录.md`

---

## 本轮会话完成的改动

| 改动 | 文件 | 说明 |
|------|------|------|
| FeatureEncoder 分组编码 | `agents/GraphQNetwork.py` | 15 维特征按语义分 4 组编码后融合 + LayerNorm |
| Dueling Network 读出头 | `agents/GraphQNetwork.py` | Q = V(s) + A(s,a) − mean(A)，梯度解耦 |
| Greedy 基线 | `evaluation/run_evaluation.py` | 最短归一化队列负载启发式 |
| FedProx μ 调整 | `run_training_real_map.py` | 1e-4 → 0.01，有效近端约束 |
| EMA 特征验证 | `env/Traffic.py` | feat[14] 已确认每步更新 |
| DP-SGD | `agents/FederatedDQN.py` | 梯度裁剪 + 高斯噪声 + RDPAccountant |
| Epsilon 衰减修复 | `run_training_real_map.py`、`main.py` | 按 episodes 动态计算 decay，medium→0.90，full→0.10 |
| 充电站 2→4 个 | `run_training_real_map.py`、`main.py`、`run_evaluation.py` | 增加问题复杂度，体现图结构优势 |

> ⚠️ 架构已改变，旧 checkpoint 不可用，必须重新训练

---

## 下一步：立即要跑的命令

### 最优先：4 站完整规模训练

```bash
# 服务器上先拉最新代码
git pull origin master

# 完整规模训练（500 episodes，100 EVs，4 个站）
python main.py train-real

# 评估
python main.py evaluate
```

训练开始时确认看到：
```
[Epsilon] decay=0.99540，500 episodes 后 ε≈0.10
nodes=xxx, stations=[...4个节点ID...], EVs/client=100
```

### 之后：DP 三组对比实验

```bash
# σ=0.3（弱保护）—— 手动改 run_training_real_map.py 中 dp_noise_multiplier=0.3
python main.py train-real --dp
python main.py evaluate

# σ=2.0（强保护）—— 手动改 dp_noise_multiplier=2.0
python main.py train-real --dp
python main.py evaluate
```

> 待办：`--dp-sigma` 做成命令行参数，见 `docs/投稿计划_T-ITS.md`

---

## 投稿 T-ITS 还需完成的工作

详细计划见 `docs/投稿计划_T-ITS.md`，核心清单：

| 任务 | 状态 |
|------|------|
| 充电站 2→4，重跑完整规模 | ⬜ **下一步** |
| DP 三组实验（σ=0.3/1.0/2.0） | ⬜ 待跑 |
| 加入文献对比方法（Nearest+SQ、FedAvg无ProX） | ⬜ 待开发 |
| 4 组消融实验 | ⬜ 待跑 |
| 5 随机种子统计显著性 | ⬜ 待跑 |
| 写论文（Related Work + System Model 可先写） | ⬜ 进行中，见 `docs/模型章初稿.md` |

---

## 关键文件一览

```
agents/
  GraphQNetwork.py     ← 网络结构（FeatureEncoder + GATv2 + Dueling）
  FederatedDQN.py      ← 联邦客户端/服务器（FedProx + DP-SGD）
env/
  Traffic.py           ← 核心仿真环境（EV、充电站、电网、奖励）
  RealTrafficEnv.py    ← 真实路网适配层
  osm_loader.py        ← OSMnx 路网加载
evaluation/
  run_evaluation.py    ← 四方评估对比
  metrics.py           ← 指标计算
visualization/
  run_training_real_map.py  ← 联邦训练主入口
main.py                ← 统一命令入口
docs/
  评估结果记录.md       ← 所有轮次实验结果
  投稿计划_T-ITS.md    ← 完整投稿路线图
  进展记录_模型优化与基线扩充.md  ← 本轮改动详情
  下一步规划.md        ← 近期任务清单
  模型章初稿.md        ← 论文模型章草稿
```

---

## 运行命令速查

```bash
# 完整规模训练（推荐）
python main.py train-real

# medium 规模（快速验证，约30min）
python main.py train-real --medium

# debug（2分钟验证逻辑）
python main.py train-real --debug

# 评估（对应规模加同档参数）
python main.py evaluate
python main.py evaluate --medium
python main.py evaluate --debug

# 带差分隐私训练
python main.py train-real --dp
```

---

## 已知问题与注意事项

1. **DQN 不稳定**：单机 DQN 在新架构下有时会崩溃（wait_time 变差），不纳入论文主线，作为消融对照即可
2. **Medium 下 epsilon 保持慢衰减**（ε_final=0.90）：100 个 episode 太少，快衰减反而变差
3. **两个联邦 client 使用同一路网不同 seed**：非真正多城市异质，后续可接入第二个城市路网作为扩展实验
4. **DP 实验结果 ε 值未记录**：上轮跑 σ=1.0 时日志被截断，需重跑时记录完整输出

---

## 交接指引

将本文档发给新窗口的 Codex，并告知：**从"跑 4 站完整规模训练"开始继续工作。**
