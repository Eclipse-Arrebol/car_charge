import random, sys, os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from env.base_env import TrafficPowerEnv
from env.entities import EV
from env.charging_station import ChargingStation
from env.power_grid import PowerGrid, get_tou_multiplier, _FlexiblePowerGrid
from env.osm_loader import load_road_network, load_road_network_by_point, load_road_network_from_file


_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_cur)
if _root not in sys.path:
    sys.path.insert(0, _root)


def _safe_path_display(path: str) -> str:
    try:
        return os.path.relpath(path, _root)
    except ValueError:
        return path


class RealTrafficEnv(TrafficPowerEnv):
    def __init__(
        self,
        place: str = "Wuchang District, Wuhan, China",
        num_stations: int = 2,
        num_evs: int = 10,
        max_nodes: int = 30,
        cache_dir: str = None,
        seed: int = 42,
        lat: float = None,
        lon: float = None,
        dist_m: int = 1500,
        graphml_file: str = None,
        offline: bool = False,
        station_node_ids: list = None,
    ):
        if graphml_file is not None:
            graph, station_nodes, self.node_positions = load_road_network_from_file(
                filepath=graphml_file,
                num_stations=num_stations,
                max_nodes=max_nodes,
                cache_dir=cache_dir,
                seed=seed,
                station_node_ids=station_node_ids,
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
        self.charge_trigger_soc = 30.0
        self.station_node_ids = station_nodes
        self.num_stations = num_stations

        power_node_map = {i: f"Grid_{chr(65 + i)}" for i in range(num_stations)}
        self.stations = [
            ChargingStation(
                station_id=i,
                traffic_node_id=station_nodes[i],
                power_node_id=power_node_map[i],
            )
            for i in range(num_stations)
        ]

        self.power_grid = _FlexiblePowerGrid(num_stations)

        non_station = [n for n in graph.nodes() if n not in station_nodes]
        self.evs = []
        for i in range(num_evs):
            start = random.choice(non_station) if non_station else random.choice(list(graph.nodes()))
            self.evs.append(EV(i, start))

        self.power_limit = 15.0
        self.time_step = 0
        self.steps_per_day = 144
        self.step_duration_h = 1 / 6
        self.bpr_alpha = 0.15
        self.bpr_beta = 4.0
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0
        self.edge_index = self._build_edge_index()

        self._path_cache_step: dict = {}

        print(f"[RealTrafficEnv] nodes={self.num_nodes}, "
              f"station_nodes={station_nodes}, EVs={num_evs})")

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
        num_evs = len(self.evs)
        self.evs = []
        for i in range(num_evs):
            start = random.choice(non_station) if non_station else \
                    random.choice(list(self.traffic_graph.nodes()))
            self.evs.append(EV(i, start))

        self.time_step = 0
        self.steps_per_day = 144
        self.step_duration_h = 1 / 6
        self.edge_active_counts = {}
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0
        return self.get_graph_state()
