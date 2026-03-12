import networkx as nx
import numpy as np
import random
import time
import torch
from torch_geometric.data import Data
import cvxpy as cp
import sys
import os

# 1. 获取当前文件 (Traffic.py) 的目录路径 -> .../simEvn
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取这个目录的父目录 (项目根目录) -> .../YourProjectRoot
project_root = os.path.dirname(current_dir)
# 3. 把项目根目录加入到 Python 的搜索路径中
sys.path.append(project_root)

from agents.GraphQNetwork import GraphQNetwork


# ==========================================
# 1. 车辆实体 (EV Agent)
# ==========================================
class EV:
    def __init__(self, ev_id, start_node):
        self.id = ev_id
        self.curr_node = start_node
        self.target_station_idx = None
        self.soc = random.uniform(20.0, 50.0)
        self.status = "IDLE"
        self.path = []
        self.last_traversed_nodes = []   # 本 step 内实际经过并到达的节点序列
        self.current_edge_from = None    # 当前所在边的起点
        self.current_edge_target = None  # 当前正在通过的下一跳节点
        self.remaining_edge_time_h = 0.0 # 当前边剩余通行时间 (小时)
        self.current_edge_speed_kph = 0.0
        self.assigned_station = None   # 当前路由目标站 ID
        self._decision_state = None    # 决策时刻的图状态快照 (用于延迟奖励)
        self._decision_snap = None     # 决策时刻的指标快照

        # --- 评估指标统计 ---
        self.travel_steps = 0          # 行驶步数 (MOVING_TO_CHARGE 状态)
        self.wait_steps = 0            # 排队等待步数 (WAITING 状态)
        self.charge_steps = 0          # 充电步数 (CHARGING 状态)
        self.total_fee_paid = 0.0      # 累计支付的充电费用 (CNY)
        self.total_energy_charged = 0.0 # 累计充入电量 (kWh，简化: 1步的kW=kWh)
        self.charge_sessions = 0       # 完成的充电次数

    def move(self, env, step_hours=1.0):
        """
        连续移动模型：按道路长度/限速计算边通行时间，
        每个 step 按时间预算推进，可跨多条边或停留在边中间。
        """
        self.last_traversed_nodes = []
        if self.status != "MOVING_TO_CHARGE":
            return

        remaining = max(0.0, float(step_hours))
        moved_hours = 0.0

        while remaining > 1e-9:
            # 若当前不在边上，则尝试进入下一条边
            if self.remaining_edge_time_h <= 1e-9:
                if not self.path:
                    break
                next_node = self.path[0]
                _, speed_kph, travel_time_h = env.enter_edge(self.curr_node, next_node)
                self.current_edge_from = self.curr_node
                self.current_edge_target = next_node
                self.current_edge_speed_kph = speed_kph
                self.remaining_edge_time_h = max(1e-6, travel_time_h)

            consume = min(remaining, self.remaining_edge_time_h)
            remaining -= consume
            moved_hours += consume
            self.remaining_edge_time_h -= consume

            # 当前边走完，抵达下一节点
            if self.remaining_edge_time_h <= 1e-9:
                env.leave_edge(self.current_edge_from, self.current_edge_target)
                self.curr_node = self.current_edge_target
                self.last_traversed_nodes.append(self.curr_node)
                if self.path and self.path[0] == self.current_edge_target:
                    self.path.pop(0)
                self.current_edge_from = None
                self.current_edge_target = None
                self.current_edge_speed_kph = 0.0

        # 与旧逻辑兼容：按行驶时长扣 SOC（旧版约 2 SOC / step）
        self.soc -= 2.0 * moved_hours / max(1e-6, float(step_hours))
        if self.soc < 0:
            self.soc = 0


# ==========================================
# 1.5 分时电价 (Time-of-Use Tariff)
# ==========================================
def get_tou_multiplier(time_step, steps_per_day=24):
    """
    根据时间步返回电价系数 (模拟峰谷电价)

    典型分时方案:
      谷时 (Valley): 23:00-07:00  →  系数 0.5  (鼓励夜间充电)
      平时 (Flat):   07:00-10:00, 15:00-18:00  →  系数 1.0
      峰时 (Peak):   10:00-15:00, 18:00-23:00  →  系数 1.5
    """
    hour = time_step % steps_per_day
    if hour >= 23 or hour < 7:                # 谷时
        return 0.5
    elif 10 <= hour < 15 or 18 <= hour < 23:  # 峰时
        return 1.5
    else:                                     # 平时
        return 1.0


# ==========================================
# 1.6 简化配电网模型 (Simplified Radial Distribution Network)
# ==========================================
class PowerGrid:
    """
    简化的辐射状配电网，计算电压降落和线路损耗。

    拓扑 (单馈线, 400V 低压):
        变电站 (1.0 pu)
          ├── 线路A (R=0.08Ω) ──→ 母线A (Grid_A, Station 0)
          └── 线路B (R=0.12Ω) ──→ 母线B (Grid_B, Station 1)

    物理公式 (线性化 DistFlow):
        电压降落: ΔV_pu ≈ P_kW · R / (V_nom_kV² · 1000)
        线路损耗: P_loss ≈ P_kW² · R / (V_nom_kV² · 1e6)  [kW]
    """

    def __init__(self):
        self.v_nominal_kv = 0.4              # 400V 低压配电

        # 线路阻抗 (从变压器到各站点的电缆)
        self.lines = {
            'Grid_A': {'r_ohm': 0.08},       # 距离近, 阻抗小
            'Grid_B': {'r_ohm': 0.12},       # 距离远, 阻抗大
        }

        # 电压安全限值 (国标 ±5%)
        self.v_min = 0.95
        self.v_max = 1.05

        # 上一步状态
        self.bus_voltages = {bus: 1.0 for bus in self.lines}
        self.line_losses  = {bus: 0.0 for bus in self.lines}
        self.total_loss = 0.0
        self.voltage_violations = []         # [(bus, v_pu), ...]

    def run_power_flow(self, loads):
        """
        简化潮流计算。

        Args:
            loads: {'Grid_A': P_kW, 'Grid_B': P_kW}
        Returns:
            bus_voltages: {'Grid_A': v_pu, 'Grid_B': v_pu}
        """
        v2 = self.v_nominal_kv ** 2          # kV²
        self.voltage_violations = []
        self.total_loss = 0.0

        for bus, line in self.lines.items():
            p = loads.get(bus, 0.0)
            r = line['r_ohm']

            # 电压降落 (pu)
            delta_v = p * r / (v2 * 1000)
            v_pu = 1.0 - delta_v

            # 线路损耗 (kW)
            loss = (p ** 2) * r / (v2 * 1e6)

            self.bus_voltages[bus] = round(v_pu, 4)
            self.line_losses[bus]  = round(loss, 4)
            self.total_loss += loss

            if v_pu < self.v_min or v_pu > self.v_max:
                self.voltage_violations.append((bus, v_pu))

        return self.bus_voltages


# ==========================================
# 2. 充电站实体 — 凸优化调度中心 + 电网感知
# ==========================================
class ChargingStation:
    def __init__(self, station_id, traffic_node_id, power_node_id,
                 num_chargers=3, max_charger_power=20.0, max_grid_power=50.0):
        self.id = station_id
        self.traffic_node_id = traffic_node_id
        self.power_node_id = power_node_id

        # --- 阶段二新增属性 ---
        self.num_chargers = num_chargers          # 充电桩数量
        self.max_charger_power = max_charger_power  # 单桩最大功率 (kW)
        self.max_grid_power = max_grid_power        # 变压器配额 (kW)

        self.queue = []                   # 排队等候的车（还没插枪）
        self.connected_evs = []           # 已插枪、正在充电的车

        self.base_price = 1.0
        self.current_price = 1.0

        # 上一步的优化结果，供外部读取
        self.last_power_allocation = {}   # {ev_id: power_kW}
        self.last_total_load = 0.0        # 站级总负荷

    def update_price(self, tou_multiplier=1.0):
        """
        动态电价 = 基础电价 × 分时系数 + 拥堵附加费
        同时考虑: 峰谷电价 + 站内拥堵
        """
        congestion = len(self.queue) + len(self.connected_evs)
        self.current_price = self.base_price * tou_multiplier + congestion * 0.5
        return self.current_price

    # --------------------------------------------------
    # 核心方法: 凸优化功率分配
    # --------------------------------------------------
    def optimize_power(self):
        """
        用 CVXPY 求解：在电网容量约束下，最大化总充电功率并兼顾公平。

        数学模型:
            max  Σ P_i  -  λ · Σ (P_i - P_avg)^2   (最大化 + 公平正则)

            s.t. Σ P_i  ≤  max_grid_power            (变压器安全)
                 0 ≤ P_i ≤ max_charger_power          (充电桩物理限制)
                 P_i · Δt ≤ (100 - SOC_i) · capacity  (电池保护, 简化)
        """
        n = len(self.connected_evs)
        if n == 0:
            self.last_power_allocation = {}
            self.last_total_load = 0.0
            return {}

        # --- 建立 CVXPY 变量 ---
        P = cp.Variable(n, nonneg=True)

        # --- 每辆车的上界 ---
        upper_bounds = []
        for ev in self.connected_evs:
            remaining_capacity = max(0.0, 100.0 - ev.soc)   # 还能充多少 SOC%
            # 简化: 1 kW·step ≈ 5 SOC%，则 SOC 上界对应的最大功率
            max_by_battery = remaining_capacity / 5.0 * self.max_charger_power
            ub = min(self.max_charger_power, max_by_battery)
            upper_bounds.append(ub)
        upper_bounds = np.array(upper_bounds, dtype=float)

        # --- 目标函数 ---
        # 最大化总功率 + 公平正则 (λ = 0.1)
        fairness_lambda = 0.1
        avg_power = cp.sum(P) / n
        objective = cp.Maximize(
            cp.sum(P) - fairness_lambda * cp.sum_squares(P - avg_power)
        )

        # --- 约束 ---
        constraints = [
            cp.sum(P) <= self.max_grid_power,        # 变压器容量
            P <= upper_bounds,                        # 单桩+电池上界
        ]

        # --- 求解 ---
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)

        # --- 读取结果 ---
        allocation = {}
        if P.value is not None:
            for i, ev in enumerate(self.connected_evs):
                allocated = max(0.0, float(P.value[i]))
                allocation[ev.id] = allocated
        else:
            # 求解失败 → 均分
            fair_share = self.max_grid_power / n
            for ev in self.connected_evs:
                allocation[ev.id] = min(fair_share, self.max_charger_power)

        self.last_power_allocation = allocation
        self.last_total_load = sum(allocation.values())
        return allocation

    # --------------------------------------------------
    # 每一步的调度逻辑 (替换旧的固定 +5.0 逻辑)
    # --------------------------------------------------
    def step(self, tou_multiplier=1.0):
        """
        新逻辑:
          1. 把排队的车填入空余充电桩 (插枪)
          2. 调用 optimize_power() 做功率分配
          3. 按分配的功率更新每辆车的 SOC
          4. 充满的车自动离桩
          5. 更新电价
        """
        realized_power = 0.0

        # 1. 排队 → 插枪 (填满充电桩)
        while self.queue and len(self.connected_evs) < self.num_chargers:
            ev = self.queue.pop(0)
            ev.status = "CHARGING"
            self.connected_evs.append(ev)

        # 2. 凸优化分配功率
        allocation = self.optimize_power()

        # 3. 按功率更新 SOC，充满离桩，记录费用
        finished = []
        for ev in self.connected_evs:
            power = allocation.get(ev.id, 0.0)
            soc_increment = (power / max(1e-6, self.max_charger_power)) * 5.0
            ev.soc = min(100.0, ev.soc + soc_increment)
            realized_power += power

            # 记录充电费用: 功率(kWh) × 当前电价(CNY/kWh)
            ev.total_fee_paid += power * self.current_price / max(1e-6, self.max_charger_power)
            ev.total_energy_charged += power

            if ev.soc >= 95.0:
                ev.status = "IDLE"
                ev.charge_sessions += 1
                finished.append(ev)

        for ev in finished:
            self.connected_evs.remove(ev)

        # 4. 更新动态电价 (含分时电价)
        self.update_price(tou_multiplier)

        return realized_power


# ==========================================
# 3. 环境主控 (Traffic-Power Environment)
# ==========================================
class TrafficPowerEnv:
    def __init__(self):
        self.traffic_graph = nx.grid_2d_graph(3, 3)
        self.traffic_graph = nx.convert_node_labels_to_integers(self.traffic_graph)

        self.stations = [
            ChargingStation(station_id=0, traffic_node_id=0, power_node_id='Grid_A'),
            ChargingStation(station_id=1, traffic_node_id=8, power_node_id='Grid_B')
        ]

        self.evs = []
        for i in range(10):
            start_node = random.randint(1, 7)
            self.evs.append(EV(i, start_node))

        self.power_limit = 15.0
        self.time_step = 0
        self.steps_per_day = 24               # 每天 24 步 (1步 = 1小时)
        self.step_duration_h = 1.0            # 每个 step 对应的仿真时长 (小时)
        self.bpr_alpha = 0.15
        self.bpr_beta = 4.0
        self.edge_active_counts = {}          # {(u, v): active_vehicles}

        # 电网物理模型
        self.power_grid = PowerGrid()
        self.tou_multiplier = 1.0             # 当前分时电价系数

        # 预先构建图结构
        self.edge_index = self._build_edge_index()

    def reset(self):
        """重置环境到初始状态 (用于 episodic 训练)"""
        self.stations = [
            ChargingStation(station_id=0, traffic_node_id=0, power_node_id='Grid_A'),
            ChargingStation(station_id=1, traffic_node_id=8, power_node_id='Grid_B')
        ]
        self.evs = []
        for i in range(10):
            start_node = random.randint(1, 7)
            self.evs.append(EV(i, start_node))
        self.power_limit = 15.0
        self.time_step = 0
        self.power_grid = PowerGrid()
        self.tou_multiplier = 1.0
        self.edge_active_counts = {}
        return self.get_graph_state()

    @staticmethod
    def _parse_speed_kph(speed_raw, default=50.0):
        if isinstance(speed_raw, list):
            speed_raw = speed_raw[0] if speed_raw else default
        if isinstance(speed_raw, str):
            cleaned = ''.join(ch for ch in speed_raw if (ch.isdigit() or ch == '.'))
            speed_raw = float(cleaned) if cleaned else default
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            speed = float(default)
        return max(1.0, speed)

    @staticmethod
    def _parse_lanes_count(lanes_raw, default=1.0):
        if isinstance(lanes_raw, list):
            lanes_raw = lanes_raw[0] if lanes_raw else default
        if isinstance(lanes_raw, str):
            cleaned = ''.join(ch for ch in lanes_raw if (ch.isdigit() or ch == '.'))
            lanes_raw = float(cleaned) if cleaned else default
        try:
            lanes = float(lanes_raw)
        except (TypeError, ValueError):
            lanes = float(default)
        return max(1.0, lanes)

    @staticmethod
    def _infer_capacity_per_lane(highway_type):
        # 粗粒度经验值（veh/h/lane），优先保证属性缺失时有稳定回退。
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else ""
        h = str(highway_type).lower()
        if "motorway" in h:
            return 2200.0
        if "trunk" in h:
            return 2000.0
        if "primary" in h:
            return 1800.0
        if "secondary" in h:
            return 1500.0
        if "tertiary" in h:
            return 1200.0
        if "residential" in h:
            return 800.0
        if "service" in h:
            return 600.0
        return 1000.0

    @classmethod
    def _parse_capacity_vehph(cls, attrs):
        cap_raw = attrs.get('capacity', None)
        if cap_raw is not None:
            if isinstance(cap_raw, list):
                cap_raw = cap_raw[0] if cap_raw else None
            if isinstance(cap_raw, str):
                cleaned = ''.join(ch for ch in cap_raw if (ch.isdigit() or ch == '.'))
                cap_raw = float(cleaned) if cleaned else None
            try:
                cap = float(cap_raw)
                if cap > 0:
                    return cap
            except (TypeError, ValueError):
                pass

        lanes = cls._parse_lanes_count(attrs.get('lanes', 1))
        cap_lane = cls._infer_capacity_per_lane(attrs.get('highway', ''))
        return lanes * cap_lane

    @classmethod
    def _edge_profiles_from_data(cls, edge_data):
        # MultiGraph: {key: attrs_dict}; Graph: attrs_dict
        if isinstance(edge_data, dict) and edge_data and all(isinstance(v, dict) for v in edge_data.values()):
            candidates = list(edge_data.values())
        else:
            candidates = [edge_data or {}]

        profiles = []
        for attrs in candidates:
            length = attrs.get('length', 100.0)
            if isinstance(length, list):
                length = length[0] if length else 100.0
            try:
                length_m = max(1.0, float(length))
            except (TypeError, ValueError):
                length_m = 100.0

            speed_kph = cls._parse_speed_kph(attrs.get('speed_kph', attrs.get('maxspeed', 50.0)))
            t0_h = (length_m / 1000.0) / speed_kph
            capacity_vehph = cls._parse_capacity_vehph(attrs)
            profiles.append((length_m, speed_kph, max(1e-6, t0_h), max(1.0, capacity_vehph)))

        return profiles

    def _edge_flow(self, u, v):
        # 无向路网下，双向车流合并计入 x_a
        return float(self.edge_active_counts.get((u, v), 0) + self.edge_active_counts.get((v, u), 0))

    def _bpr_time_h(self, t0_h, x_flow, c_capacity):
        ratio = max(0.0, float(x_flow)) / max(1.0, float(c_capacity) * self.step_duration_h)
        return max(1e-6, float(t0_h) * (1.0 + self.bpr_alpha * (ratio ** self.bpr_beta)))

    def _dynamic_profiles(self, u, v, add_vehicle=0.0):
        edge_data = self.traffic_graph.get_edge_data(u, v, default={})
        base_profiles = self._edge_profiles_from_data(edge_data)
        x_flow = self._edge_flow(u, v) + float(add_vehicle)

        dyn = []
        for length_m, speed_kph, t0_h, capacity_vehph in base_profiles:
            t_h = self._bpr_time_h(t0_h, x_flow, capacity_vehph)
            dyn.append((length_m, speed_kph, t_h, t0_h, x_flow, capacity_vehph))
        return dyn

    def _travel_time_weight(self, u, v, edge_data):
        # 估计路径时把“本车将进入该边”计入 x_a（add_vehicle=1）
        dyn = self._dynamic_profiles(u, v, add_vehicle=1.0)
        if dyn:
            return min(item[2] for item in dyn)
        profiles = self._edge_profiles_from_data(edge_data)
        return min((t0 for _, _, t0, _ in profiles), default=1e-6)

    def get_edge_travel_profile(self, u, v):
        dyn = self._dynamic_profiles(u, v, add_vehicle=1.0)
        if not dyn:
            return 100.0, 50.0, 0.002
        length_m, speed_kph, t_h, _, _, _ = min(dyn, key=lambda x: x[2])
        return length_m, speed_kph, t_h

    def get_edge_base_profile(self, u, v):
        edge_data = self.traffic_graph.get_edge_data(u, v, default={})
        profiles = self._edge_profiles_from_data(edge_data)
        if not profiles:
            return 100.0, 50.0, 0.002, 1000.0
        return min(profiles, key=lambda x: x[2])

    def enter_edge(self, u, v):
        length_m, speed_kph, t_h = self.get_edge_travel_profile(u, v)
        key = (u, v)
        self.edge_active_counts[key] = self.edge_active_counts.get(key, 0) + 1
        return length_m, speed_kph, t_h

    def leave_edge(self, u, v):
        if u is None or v is None:
            return
        key = (u, v)
        cur = self.edge_active_counts.get(key, 0)
        if cur <= 1:
            self.edge_active_counts.pop(key, None)
        else:
            self.edge_active_counts[key] = cur - 1

    def _build_edge_index(self):
        adj = nx.to_scipy_sparse_array(self.traffic_graph).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64))
        col = torch.from_numpy(adj.col.astype(np.int64))
        edge_index = torch.stack([row, col], dim=0)
        
        # 构建边特征矩阵 (edge_attr)
        edge_attr_list = []
        for u, v in zip(row.tolist(), col.tolist()):
            length_m, speed_kph, _, _ = self.get_edge_base_profile(u, v)
            edge_attr_list.append([float(length_m) / 1000.0, float(speed_kph) / 100.0])
            
        self.edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        return edge_index

    def get_graph_state(self):
        """返回 PyG 的 Data 对象 (全局状态, 10 维特征)"""
        num_nodes = self.traffic_graph.number_of_nodes()
        # 特征维度=10:
        #   [0] 车辆数  [1] 是否充电站  [2] 排队数  [3] 当前电价
        #   [4] 在桩车辆数  [5] 负荷率  [6] 母线电压(pu)  [7] 分时电价系数
        #   [8] 请求EV的SOC/100 (默认0, 由 get_graph_state_for_ev 设置)
        #   [9] 站到请求EV的距离反函数 (默认0, 由 get_graph_state_for_ev 设置)
        x = torch.zeros((num_nodes, 10), dtype=torch.float)

        # Feature 0: 车辆分布
        for ev in self.evs:
            x[ev.curr_node, 0] += 1.0

        # Feature 7: 分时电价系数 — 全局信号，所有节点共享
        x[:, 7] = self.tou_multiplier

        # Feature 1-6: 充电站特有信息
        for station in self.stations:
            node_idx = station.traffic_node_id
            x[node_idx, 1] = 1.0                       # 是充电站
            x[node_idx, 2] = len(station.queue)         # 排队长度
            x[node_idx, 3] = station.current_price      # 动态电价 (含TOU+拥堵)
            x[node_idx, 4] = len(station.connected_evs) # 在桩车辆数
            x[node_idx, 5] = station.last_total_load / max(1.0, station.max_grid_power)
            x[node_idx, 6] = self.power_grid.bus_voltages.get(
                station.power_node_id, 1.0)             # 母线电压

        data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return data

    def get_graph_state_for_ev(self, ev, pending_counts=None):
        """
        返回带"请求 EV 信息"和"pending 预分配"的图状态。

        Feature[8] = 请求 EV 的 SOC/100 (标识位置 + 紧迫程度)
        Feature[9] = 各站到请求 EV 的距离反函数 1/(1+d)

        Args:
            ev:              当前请求决策的 EV 对象
            pending_counts:  dict {station_id: count}，表示本步内已有多少辆
                             EV 被分配到该站（尚未更新到环境）。
                             会叠加到站节点的排队特征(feature[2])上，
                             使后续 EV 能"看到"前面 EV 的选择，避免扎堆。
        """
        data = self.get_graph_state()
        # Feature[8]: 请求 EV 的 SOC (归一化), 同时标识 EV 所在节点
        data.x[ev.curr_node, 8] = ev.soc / 100.0

        # Feature[9]: 各站节点到请求 EV 的归一化距离信息
        for station in self.stations:
            try:
                dist = nx.shortest_path_length(
                    self.traffic_graph, ev.curr_node,
                    station.traffic_node_id,
                    weight=self._travel_time_weight,
                )
            except nx.NetworkXNoPath:
                dist = self.traffic_graph.number_of_nodes()
            data.x[station.traffic_node_id, 9] = 1.0 / (1.0 + dist)

        # 将本步内的 pending 分配叠加到排队特征上
        if pending_counts:
            for station in self.stations:
                pc = pending_counts.get(station.id, 0)
                if pc > 0:
                    data.x[station.traffic_node_id, 2] += pc
        return data

    def get_action_mask(self, ev):
        """
        无效动作掩码 (Action Mask)：基于物理约束判断每个充电站是否可达。

        掩码规则 (任一条件成立则标记为无效):
          1. 不可达：EV 当前节点到该站无路径
          2. SOC 不足：EV 剩余电量不足以支撑到达该站的行驶距离
             (行驶每 step 消耗约 2 SOC，按最短路径跳数估算)
          3. SOC 上限：EV 当前 SOC >= 95%，无需充电（所有站均无效 → 回退为全有效）

        Returns:
            torch.Tensor: shape [1, num_stations]，True=有效 / False=无效
        """
        num_actions = len(self.stations)
        mask = torch.ones(1, num_actions, dtype=torch.bool)

        for i, station in enumerate(self.stations):
            # 规则 1: 路径可达性
            if not nx.has_path(self.traffic_graph, ev.curr_node, station.traffic_node_id):
                mask[0, i] = False
                continue

            # 规则 2: SOC 够不够走到目标站
            try:
                path_len = nx.shortest_path_length(
                    self.traffic_graph, ev.curr_node,
                    station.traffic_node_id,
                )
                # 每跳约消耗 2 SOC，留 2 SOC 安全余量
                soc_needed = path_len * 2.0 + 2.0
                if ev.soc < soc_needed:
                    mask[0, i] = False
                    continue
            except nx.NetworkXNoPath:
                mask[0, i] = False
                continue

        # 安全回退：若所有动作均被掩码，则全部放开（避免死锁）
        if not mask.any():
            mask.fill_(True)

        return mask

    def step(self, actions):
        self.time_step += 1
        grid_loads = {'Grid_A': 0.0, 'Grid_B': 0.0}

        # --- 1. 车辆状态流转 ---
        for ev in self.evs:
            if ev.status == "IDLE":
                ev.soc -= 0.5
                neighbors = list(self.traffic_graph.neighbors(ev.curr_node))
                if neighbors: ev.curr_node = random.choice(neighbors)

                if ev.id in actions:
                    target_id = actions[ev.id]
                    target_station = self.stations[target_id]
                    try:
                        path = nx.shortest_path(
                            self.traffic_graph,
                            source=ev.curr_node,
                            target=target_station.traffic_node_id,
                            weight=self._travel_time_weight,
                        )
                        if len(path) > 1:
                            ev.path = path[1:]
                            ev.target_station_idx = target_id
                            ev.status = "MOVING_TO_CHARGE"
                            ev.current_edge_from = None
                            ev.current_edge_target = None
                            ev.remaining_edge_time_h = 0.0
                            ev.current_edge_speed_kph = 0.0
                    except:
                        pass

            elif ev.status == "MOVING_TO_CHARGE":
                ev.travel_steps += 1
                ev.move(self, step_hours=self.step_duration_h)
                target_station = self.stations[ev.target_station_idx]
                if (
                    not ev.path
                    and ev.remaining_edge_time_h <= 1e-9
                    and ev.curr_node == target_station.traffic_node_id
                ):
                    target_station.queue.append(ev)
                    ev.status = "WAITING"

            elif ev.status == "WAITING":
                ev.wait_steps += 1

            elif ev.status == "CHARGING":
                ev.charge_steps += 1

        # --- 2. 分时电价 ---
        self.tou_multiplier = get_tou_multiplier(self.time_step, self.steps_per_day)

        # --- 3. 基础设施更新 (凸优化调度) ---
        total_realized_power = 0.0
        for station in self.stations:
            load = station.step(tou_multiplier=self.tou_multiplier)
            grid_loads[station.power_node_id] += load
            total_realized_power += load

        # --- 4. 配电网潮流计算 ---
        self.power_grid.run_power_flow(grid_loads)

        # --- 5. 计算奖励 ---
        # 设计原则：DQN 只能控制"把 EV 派去哪个站"，
        # 因此奖励应聚焦于路由质量，而非充电功率（CVXPY 负责功率分配）。
        reward = 0.0

        # ① 队列均衡惩罚：两站负载越失衡，扣分越多（鼓励分散导流）
        loads = [len(s.queue) + len(s.connected_evs) for s in self.stations]
        queue_imbalance = abs(loads[0] - loads[1])
        reward -= queue_imbalance * 3.0

        # ② 总等待惩罚：排队越多越差
        total_waiting = sum(len(s.queue) for s in self.stations)
        reward -= total_waiting * 2.0

        # ③ 充电费用惩罚：在高价站充电越多扣分越多（引导去低价站）
        for station in self.stations:
            reward -= station.current_price * len(station.connected_evs) * 0.4

        # ④ 谷时利用奖励：鼓励在电价低时保持较高利用率
        if self.tou_multiplier < 1.0:
            max_possible = sum(s.max_grid_power for s in self.stations)
            reward += (total_realized_power / max(1.0, max_possible)) * 3.0

        # ⑤ 电压越限惩罚
        reward -= len(self.power_grid.voltage_violations) * 5.0

        info = {
            "grid_loads": grid_loads,
            "realized_power": total_realized_power,
            "power_allocation": {s.id: s.last_power_allocation for s in self.stations},
            "bus_voltages": dict(self.power_grid.bus_voltages),
            "line_losses": self.power_grid.total_loss,
            "voltage_violations": len(self.power_grid.voltage_violations),
            "tou_multiplier": self.tou_multiplier,
        }

        return self.get_graph_state(), reward, False, info

    def render(self):
        # 时间与电价
        hour = self.time_step % self.steps_per_day
        tou_tag = {0.5: "谷", 1.0: "平", 1.5: "峰"}.get(self.tou_multiplier, "?")
        print(f"\n=== Step {self.time_step:03d} | {hour:02d}:00 [{tou_tag}时 x{self.tou_multiplier}] ===")

        # 电网状态
        for bus, v in self.power_grid.bus_voltages.items():
            flag = " [WARN: 越限!]" if v < self.power_grid.v_min else ""
            print(f"  {bus}: V={v:.4f} pu  Loss={self.power_grid.line_losses[bus]:.3f}kW{flag}")

        # 充电站状态
        for s in self.stations:
            connected_ids = [ev.id for ev in s.connected_evs]
            queue_ids = [ev.id for ev in s.queue]
            alloc_str = ", ".join(
                f"EV{eid}:{p:.1f}kW" for eid, p in s.last_power_allocation.items()
            )
            print(
                f"  Station {s.id}: Price={s.current_price:.1f} | "
                f"Load={s.last_total_load:.1f}/{s.max_grid_power:.0f}kW | "
                f"Connected={connected_ids} | Queue={queue_ids} | "
                f"Power=[{alloc_str}]"
            )


# ==========================================
# 4. 运行验证
# ==========================================
if __name__ == "__main__":
    env = TrafficPowerEnv()

    for t in range(50):
        actions = {}
        for ev in env.evs:
            if ev.status == "IDLE" and ev.soc < 30.0:
                actions[ev.id] = random.choice([0, 1])

        graph_state, reward, done, info = env.step(actions)

        if t == 0:
            print("\n[系统自检] GNN 输入数据格式验证:")
            print(f"节点特征矩阵 x shape: {graph_state.x.shape} (期望: [9, 9])")
            print(f"边索引 edge_index shape: {graph_state.edge_index.shape}")
            print(f"特征示例 (节点0 - Station): {graph_state.x[0]}")

        if t % 10 == 0:
            print(f"\n--- Step {t} | Reward: {reward:.2f} | Power: {info['realized_power']:.1f}kW | "
                  f"TOU: x{info['tou_multiplier']} | V-violations: {info['voltage_violations']} ---")
            env.render()