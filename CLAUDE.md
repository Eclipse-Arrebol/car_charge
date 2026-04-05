# FedGRL 项目交接文档

## 项目概况

- **课题**：面向真实路网的电动汽车充电导航策略（FedGRL）
- **目标**：3个月内发 IEEE T-ITS（2区，IF≈8）
- **代码位置**：`/root/car_charge`（服务器）

---

## 系统架构

| 模块 | 技术方案 | 代码位置 |
|------|----------|----------|
| 路网 | OSMnx 珠江新城真实路网（graphml） | — |
| 决策网络 | GATv2 图注意力网络 | `agents/GraphQNetwork.py` |
| 训练框架 | Double DQN + Action Mask | `agents/FederatedDQN.py` |
| 联邦框架 | FedAvg 加权聚合 | `agents/FederatedDQN.py` |
| 环境 | BPR动态拥堵 + 分时电价TOU | `env/Traffic.py`、`env/RealTrafficEnv.py` |
| 评估 | 多策略对比 | `evaluation/run_evaluation.py` |

---

## 已完成的修复（上一轮会话）

| 问题 | 修复位置 | 状态 |
|------|----------|------|
| 队列无上限导致等待28000分钟 | `env/Traffic.py` → max_queue_len=8, max_wait_time_h=4.0 | ✅ |
| reward clipping太窄 [-120,40] | `train.py`, `FederatedDQN.py` → 改为 [-500,50] | ✅ |
| epsilon衰减过快 | 改为0.9999，按episode衰减 | ✅ |
| 评估未传action_mask | `evaluation/run_evaluation.py` line 129 | ✅ |
| FedDQN共享单环境无意义 | `run_training_real_map.py` → 重构为双环境 | ✅ |
| batch_size=64在debug下空跑 | debug模式改为 batch_size=8 | ✅ |

---

## 运行命令

```bash
# debug模式（快速验证逻辑，约2分钟）
python main.py train-real --debug
python main.py evaluate --debug

# medium模式（正式实验，当前待跑）
python main.py train-real --medium
python main.py evaluate --medium
```

---

## 当前状态与待完成任务

### ⚡ 最紧急：跑 medium 训练+评估，获取可靠的三组对比数字

目标结果：

- DQN vs Random：wait_time 降低 >15%
- FedDQN vs DQN：wait_time 再降 >5%
- abandoned_evs 控制在 20~50 之间

### 之后要补的三个创新点代码

1. **滑动平均预测 arrivals**：节点特征 14→15 维（`env/Traffic.py`）
2. **Opacus 差分隐私接入**（`agents/FederatedDQN.py`）
3. **第四组对比实验**：FedGRL vs Random vs DQN vs FedDQN

---

## 已知潜在问题（medium跑完后排查）

- FedDQN 在 debug 下仍比 Random 差，可能是训练轮数不足，也可能是环境建模有 bug
- 两个联邦 client 使用同一路网不同 seed，不是真正的多城市场景，后续需要接入第二个城市路网
- `RealTrafficEnv` 内部随机源未收拢到实例级，双环境的真正独立性待完善

---

## 论文框架

- **名称**：FedGRL（Federated Graph Reinforcement Learning）
- **目标期刊**：IEEE Transactions on Intelligent Transportation Systems（T-ITS）
- **备选**：Transportation Research Part C

### 三个核心贡献

1. 负载预测感知的 GATv2 编码（EMA 抑制羊群效应）
2. 差分隐私联邦训练（Opacus DP-SGD，ε-性能权衡曲线）
3. OSMnx 真实路网验证（多城市泛化）

---

## 交接指引

将本文档发给新窗口的 Claude，并指示：**从"跑 medium 训练"开始继续工作。**
