import networkx as nx
import numpy as np
import random
import time
import torch
from torch_geometric.data import Data


# ==========================================
# 1. 车辆实体 (EV Agent)
# ==========================================
class EV:
    def __init__(self, ev_id, start_node):
        self.id = ev_id
        self.curr_node = start_node
        self.target_station_idx = None
        # 让初始电量随机一点，模拟真实的参差不齐
        self.soc = random.uniform(20.0, 50.0)
        self.status = "IDLE"
        self.path = []

    def move(self):
        if self.status == "MOVING_TO_CHARGE" and self.path:
            self.curr_node = self.path.pop(0)
            self.soc -= 2.0
            if self.soc < 0: self.soc = 0


# ==========================================
# 2. 充电站实体 (Infrastructure)
# ==========================================
class ChargingStation:
    def __init__(self, station_id, traffic_node_id, power_node_id):
        self.id = station_id
        self.traffic_node_id = traffic_node_id
        self.power_node_id = power_node_id
        self.queue = []
        self.charging_vehicle = None
        self.charging_power = 10.0
        self.base_price = 1.0
        self.current_price = 1.0

    def update_price(self):
        """根据排队长度动态调整价格"""
        # 排队越长，价格越高 (拥堵费)
        congestion_surcharge = len(self.queue) * 0.5
        self.current_price = self.base_price + congestion_surcharge
        return self.current_price

    def step(self):
        current_load = 0.0
        # 1. 充电逻辑
        if self.charging_vehicle:
            self.charging_vehicle.soc += 5.0
            current_load = self.charging_power
            if self.charging_vehicle.soc >= 90.0:
                self.charging_vehicle.status = "IDLE"
                self.charging_vehicle = None
                current_load = 0.0

        # 2. 队列调度
        if self.charging_vehicle is None and self.queue:
            next_ev = self.queue.pop(0)
            self.charging_vehicle = next_ev
            next_ev.status = "CHARGING"

        # 3. 更新当前时刻的动态电价 <--- 修改：每一步都更新价格
        self.update_price()

        return current_load


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
        for i in range(5):
            start_node = random.randint(1, 7)
            self.evs.append(EV(i, start_node))

        self.power_limit = 15.0
        self.time_step = 0

        # 预先构建图结构
        self.edge_index = self._build_edge_index()  # <--- 修改：初始化时构建

    def _build_edge_index(self):
        adj = nx.to_scipy_sparse_array(self.traffic_graph).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64))
        col = torch.from_numpy(adj.col.astype(np.int64))
        return torch.stack([row, col], dim=0)

    def get_graph_state(self):
        """返回 PyG 的 Data 对象"""
        num_nodes = self.traffic_graph.number_of_nodes()
        # 特征维度=4: [车辆数, 是否充电站, 排队数, 当前电价]
        x = torch.zeros((num_nodes, 4), dtype=torch.float)

        # Feature 0: 车辆分布
        for ev in self.evs:
            x[ev.curr_node, 0] += 1.0

        # Feature 1, 2, 3: 充电站信息
        for station in self.stations:
            node_idx = station.traffic_node_id
            x[node_idx, 1] = 1.0  # 是充电站
            x[node_idx, 2] = len(station.queue)  # 排队长度
            x[node_idx, 3] = station.current_price  # <--- 修改：放入动态电价特征

        data = Data(x=x, edge_index=self.edge_index)
        return data

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
                            self.traffic_graph, source=ev.curr_node, target=target_station.traffic_node_id
                        )
                        if len(path) > 1:
                            ev.path = path[1:]
                            ev.target_station_idx = target_id
                            ev.status = "MOVING_TO_CHARGE"
                    except:
                        pass

            elif ev.status == "MOVING_TO_CHARGE":
                ev.move()
                target_station = self.stations[ev.target_station_idx]
                if not ev.path and ev.curr_node == target_station.traffic_node_id:
                    target_station.queue.append(ev)
                    ev.status = "WAITING"

        # --- 2. 基础设施更新 ---
        for station in self.stations:
            load = station.step()
            grid_loads[station.power_node_id] += load

        # --- 3. 计算奖励 ---
        reward = 0
        for node, load in grid_loads.items():
            if load > self.power_limit: reward -= 50
        total_queue = sum([len(s.queue) for s in self.stations])
        reward -= total_queue * 1.0

        info = {"grid_loads": grid_loads}

        # <--- 修改：这里直接返回图神经网络需要的 PyG Data
        return self.get_graph_state(), reward, False, info

    def render(self):
        print(f"\n=== Step {self.time_step:03d} ===")
        for s in self.stations:
            status_str = f"[In-Charge: EV-{s.charging_vehicle.id}]" if s.charging_vehicle else "[Empty]"
            # 打印当前动态电价
            print(f"Station {s.id}: Price={s.current_price:.1f} | Queue: {[e.id for e in s.queue]} | {status_str}")


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

        # 获取图数据
        graph_state, reward, done, info = env.step(actions)

        # <--- 验证点：查看图数据是否正常
        if t == 0:
            print("\n[系统自检] GNN 输入数据格式验证:")
            print(f"节点特征矩阵 x shape: {graph_state.x.shape} (期望: [9, 4])")
            print(f"边索引 edge_index shape: {graph_state.edge_index.shape} (期望: [2, Num_Edges])")
            print(f"特征示例 (节点0 - Station): {graph_state.x[0]}")
            # 应该是 [车辆数, 1.0, 排队数, 电价]

        env.render()