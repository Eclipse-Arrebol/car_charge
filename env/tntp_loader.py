import random
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .osm_loader import _select_station_nodes

MILES_TO_METERS = 1609.34


def load_tntp_network(
    net_file: str,
    num_stations: int = 3,
    seed: int = 42,
    station_node_ids: Optional[List[int]] = None,
) -> Tuple[nx.Graph, list, dict]:
    G = nx.DiGraph()

    with open(net_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("~") and not data_started:
            data_started = True
            continue
        if line.startswith("<") or line.startswith("~"):
            continue
        if not data_started:
            continue

        parts = line.split(";")[0].split()
        if len(parts) < 8:
            continue

        u = int(parts[0])
        v = int(parts[1])
        capacity = float(parts[2])
        length_miles = float(parts[3])
        free_flow_time_h = float(parts[4])

        length_m = max(1.0, length_miles * MILES_TO_METERS)
        speed_kph = max(1.0, (length_miles / max(1e-6, free_flow_time_h)) * MILES_TO_METERS)

        G.add_node(u)
        G.add_node(v)

        G.add_edge(u, v,
                   length=length_m,
                   speed_kph=speed_kph,
                   capacity=capacity,
                   weight=free_flow_time_h)

    G_undirected = G.to_undirected()

    if not nx.is_connected(G_undirected):
        components = sorted(nx.connected_components(G_undirected), key=len, reverse=True)
        largest = G_undirected.subgraph(components[0]).copy()
    else:
        largest = G_undirected

    largest = nx.convert_node_labels_to_integers(largest)

    np.random.seed(seed)
    random.seed(seed)

    if station_node_ids is not None:
        missing = [n for n in station_node_ids if n not in largest.nodes()]
        if missing:
            raise ValueError(f"station_node_ids 不在图中: {missing}")
        stations = station_node_ids[:num_stations]
    else:
        stations = _select_station_nodes(largest, num_stations, seed)

    pos_layout = nx.spring_layout(largest, seed=seed, k=1.5)
    positions = {}
    for node, (x, y) in pos_layout.items():
        lon = 114.2 + (x + 1) / 2 * 0.3
        lat = 30.4 + (y + 1) / 2 * 0.3
        largest.nodes[node]["x"] = str(lon)
        largest.nodes[node]["y"] = str(lat)
        positions[node] = (lon, lat)

    print(f"[TNTP] nodes={largest.number_of_nodes()}, edges={largest.number_of_edges()}, "
          f"stations={stations}")

    return largest, stations, positions
