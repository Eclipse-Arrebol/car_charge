"""
真实路网仿真环境
=================
继承 TrafficPowerEnv，将 3×3 网格替换为 OSMnx 真实城市路网。
其余仿真逻辑（EV 移动、CVXPY 充电调度、分时电价）完全复用。

用法:
    from env.RealTrafficEnv import RealTrafficEnv
    from train import DQNAgent

    env = RealTrafficEnv(
        place="Wuchang District, Wuhan, China",
        num_stations=2,
        max_nodes=30,
        num_evs=10,
    )
    agent = DQNAgent(
        num_features=9,
        num_actions=env.num_stations,
        station_node_ids=env.station_node_ids,
        num_nodes_per_graph=env.num_nodes,
    )
"""

import random
import sys
import os

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# ── 确保项目根目录在路径中 ──
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_cur)
if _root not in sys.path:
    sys.path.insert(0, _root)

from env.Traffic import (
    TrafficPowerEnv, EV, ChargingStation, PowerGrid,
    get_tou_multiplier,
)
from env.osm_loader import load_road_network, load_road_network_by_point, load_road_network_from_file


class RealTrafficEnv(TrafficPowerEnv):
    """
    基于 OSMnx 真实路网的充电调度仿真环境。

    与 TrafficPowerEnv 的主要区别:
      - traffic_graph  : OSMnx 路网（任意节点数）而非 3×3 网格
      - station_nodes  : 由 osm_loader 自动选取的高度节点
      - num_nodes      : 暴露给外部，供 DQNAgent 初始化 GraphQNetwork

    其余方法（step / get_graph_state / reset / render）完全继承自父类，
    无需任何修改。
    """

    def __init__(
        self,
        place: str = "Wuchang District, Wuhan, China",
        num_stations: int = 2,
        num_evs: int = 10,
        max_nodes: int = 30,
        cache_dir: str = None,
        seed: int = 42,
        # 也可以用经纬度替代地名
        lat: float = None,
        lon: float = None,
        dist_m: int = 1500,
        # 离线本地文件（优先级最高）
        graphml_file: str = None,
        # True = 直接使用离线合成路网，跳过一切网络请求
        offline: bool = False,
    ):
        """
        Args:
            place:        OSM 地名（与 lat/lon 二选一）
            num_stations: 充电站数量
            num_evs:      仿真 EV 数量
            max_nodes:    路网节点上限（建议 20~60）
            cache_dir:    路网缓存目录
            seed:         随机种子
            lat/lon:      经纬度中心点（与 place 二选一）
            dist_m:       以经纬度为中心的采样半径 (米)
            graphml_file: 本地 .graphml/.osm 文件路径（无需联网，优先级最高）
            offline:      True = 跳过所有网络请求，直接生成合成路网
        """
        # ── 不调用父类 __init__，手动初始化所有属性 ──
        # （父类 __init__ 会创建 3×3 网格，我们要覆盖它）

        # 1. 加载真实路网
        if graphml_file is not None:
            graph, station_nodes, self.node_positions = load_road_network_from_file(
                filepath=graphml_file,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
            )
        elif lat is not None and lon is not None:
            graph, station_nodes, self.node_positions = load_road_network_by_point(
                lat=lat, lon=lon, dist_m=dist_m,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                offline=offline,
            )
        else:
            graph, station_nodes, self.node_positions = load_road_network(
                place=place,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                offline=offline,
            )

        self.traffic_graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.station_node_ids = station_nodes          # 供外部读取
        self.num_stations = num_stations

        # 2. 充电站（与 PowerGrid 节点映射）
        power_node_map = {i: f"Grid_{chr(65 + i)}" for i in range(num_stations)}
        self.stations = [
            ChargingStation(
                station_id=i,
                traffic_node_id=station_nodes[i],
                power_node_id=power_node_map[i],
            )
            for i in range(num_stations)
        ]

        # PowerGrid 需要扩展以支持任意数量充电站
        self.power_grid = _FlexiblePowerGrid(num_stations)

        # 3. EV 初始化（避开充电站节点）
        non_station = [n for n in graph.nodes() if n not in station_nodes]
        self.evs = []
        for i in range(num_evs):
            start = random.choice(non_station) if non_station else random.choice(list(graph.nodes()))
            self.evs.append(EV(i, start))

        # 4. 其余属性（与父类一致）
        self.power_limit = 15.0
        self.time_step = 0
        self.steps_per_day = 24
        self.step_duration_h = 1.0
        self.bpr_alpha = 0.15
        self.bpr_beta = 4.0
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        self.edge_index = self._build_edge_index()

        print(f"[RealTrafficEnv] nodes={self.num_nodes}, "
              f"station_nodes={station_nodes}, EVs={num_evs})")

    # ──────────────────────────────────────────
    # 覆盖 reset：重建 EV 时避免占用充电站节点
    # ──────────────────────────────────────────
    def reset(self):
        power_node_map = {
            i: f"Grid_{chr(65 + i)}" for i in range(self.num_stations)
        }
        self.stations = [
            ChargingStation(
                station_id=i,
                traffic_node_id=self.station_node_ids[i],
                power_node_id=power_node_map[i],
            )
            for i in range(self.num_stations)
        ]
        self.power_grid = _FlexiblePowerGrid(self.num_stations)

        non_station = [n for n in self.traffic_graph.nodes()
                       if n not in self.station_node_ids]
        num_evs = len(self.evs)          # 先记住数量再重建
        self.evs = []
        for i in range(num_evs):
            start = random.choice(non_station) if non_station else \
                    random.choice(list(self.traffic_graph.nodes()))
            self.evs.append(EV(i, start))

        self.time_step = 0
        self.step_duration_h = 1.0
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        return self.get_graph_state()


# ============================================================
# 灵活配电网模型：支持任意数量充电站
# ============================================================
class _FlexiblePowerGrid(PowerGrid):
    """
    在 PowerGrid 基础上动态生成任意数量的配电线路，
    每条线路对应一个充电站（Grid_A, Grid_B, Grid_C, ...）
    """
    _BASE_R = 0.08   # 第一条线路阻抗
    _R_STEP = 0.04   # 每条线路递增的阻抗（模拟距离递增）

    def __init__(self, num_stations: int = 2):
        # 不调用父类 __init__，直接构造 lines
        self.v_nominal_kv = 0.4
        self.v_min = 0.95
        self.v_max = 1.05

        self.lines = {
            f"Grid_{chr(65 + i)}": {
                "r_ohm": self._BASE_R + i * self._R_STEP
            }
            for i in range(num_stations)
        }

        self.bus_voltages   = {bus: 1.0  for bus in self.lines}
        self.line_losses    = {bus: 0.0  for bus in self.lines}
        self.total_loss     = 0.0
        self.voltage_violations = []
