import json
import os
from collections import Counter

import networkx as nx
import osmnx as ox


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baseline_eps40_artifacts")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

L0_SOURCE_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_mainroads.graphml")
L1_GRAPH_PATH = os.path.join(BASELINE_DIR, "G_L1_eps40.graphml")
L1_MAPPING_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_eps40_mapping.json")

L0_INDEXED_PATH = os.path.join(BASELINE_DIR, "G_L0_indexed.graphml")
L0_RELABEL_PATH = os.path.join(BASELINE_DIR, "l0_relabel_map.json")
NODE_MAP_PATH = os.path.join(BASELINE_DIR, "node_map_eps40.json")
EDGE_MAP_PATH = os.path.join(BASELINE_DIR, "edge_map_eps40.json")
ANCHOR_PATH = os.path.join(BASELINE_DIR, "cluster_anchors_eps40.json")
STATIONS_PATH = os.path.join(CONFIG_DIR, "stations.json")

NUM_STATIONS = 4
SELECTION_SEED = 42


def _safe_display_path(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


def _normalize_id(value) -> str:
    return str(value)


def _load_connected_graph(path: str) -> nx.MultiGraph:
    graph = ox.load_graphml(path)
    if not isinstance(graph, nx.MultiGraph):
        graph = nx.MultiGraph(graph)
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    return graph


def _relabel_l0(graph: nx.MultiGraph):
    nodes_sorted = sorted(graph.nodes(), key=lambda item: _normalize_id(item))
    raw_to_l0 = {_normalize_id(old): idx for idx, old in enumerate(nodes_sorted)}
    l0_to_raw = {idx: _normalize_id(old) for idx, old in enumerate(nodes_sorted)}
    relabeled = nx.relabel_nodes(graph, {old: raw_to_l0[_normalize_id(old)] for old in nodes_sorted})
    return relabeled, raw_to_l0, l0_to_raw


def _select_station_nodes(G: nx.Graph, l0_to_l1: dict, num_stations: int):
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    sorted_nodes = sorted(nodes, key=lambda n: (-degrees[n], int(n)))
    min_dist = max(2, G.number_of_nodes() // (num_stations * 3))

    selected = []
    selected_l1 = []
    for candidate in sorted_nodes:
        projected = l0_to_l1.get(int(candidate))
        if projected is None or projected in selected_l1:
            continue
        too_close = any(
            nx.shortest_path_length(G, candidate, s) < min_dist
            for s in selected
            if nx.has_path(G, candidate, s)
        )
        if too_close:
            continue
        selected.append(int(candidate))
        selected_l1.append(int(projected))
        if len(selected) == num_stations:
            break

    if len(selected) < num_stations:
        for candidate in sorted_nodes:
            projected = l0_to_l1.get(int(candidate))
            if projected is None or projected in selected_l1 or int(candidate) in selected:
                continue
            selected.append(int(candidate))
            selected_l1.append(int(projected))
            if len(selected) == num_stations:
                break

    if len(selected) != num_stations:
        raise RuntimeError("Unable to select enough station nodes with unique L1 projections.")

    return selected, selected_l1


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    l0_raw = _load_connected_graph(L0_SOURCE_PATH)
    l0_indexed, raw_to_l0, l0_to_raw = _relabel_l0(l0_raw)
    ox.save_graphml(l0_indexed, L0_INDEXED_PATH)

    with open(L1_MAPPING_PATH, "r", encoding="utf-8") as fh:
        l1_mapping = json.load(fh)

    raw_to_train = {str(k): int(v) for k, v in l1_mapping["raw_to_train_node"].items()}
    indexed_node_map = {
        int(l0_idx): int(raw_to_train[raw_id])
        for raw_id, l0_idx in raw_to_l0.items()
        if raw_id in raw_to_train
    }
    with open(NODE_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(indexed_node_map, fh, ensure_ascii=False, indent=2)

    converted_edge_map = []
    for item in l1_mapping.get("edge_mapping", []):
        raw_path = [str(node) for node in item.get("raw_path", [])]
        indexed_path = [int(raw_to_l0[node]) for node in raw_path if node in raw_to_l0]
        converted_edge_map.append(
            {
                "train_u": int(item["train_u"]),
                "train_v": int(item["train_v"]),
                "raw_path": indexed_path,
                "raw_edge_sequence": item.get("raw_edge_sequence", []),
                "collapsed_segment_count": int(item.get("collapsed_segment_count", 1)),
            }
        )
    with open(EDGE_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(converted_edge_map, fh, ensure_ascii=False, indent=2)

    converted_anchors = {}
    for l1_node, item in l1_mapping.get("train_representative_nodes", {}).items():
        raw_node = str(item["raw_node"])
        converted_anchors[str(l1_node)] = {
            "topo_node": int(item["topo_node"]),
            "raw_node": raw_node,
            "l0_node": int(raw_to_l0[raw_node]),
        }
    with open(ANCHOR_PATH, "w", encoding="utf-8") as fh:
        json.dump(converted_anchors, fh, ensure_ascii=False, indent=2)

    with open(L0_RELABEL_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "raw_to_l0": raw_to_l0,
                "l0_to_raw": {str(k): v for k, v in l0_to_raw.items()},
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    station_l0, station_l1 = _select_station_nodes(l0_indexed, indexed_node_map, NUM_STATIONS)
    collisions = [node for node, cnt in Counter(station_l1).items() if cnt > 1]

    station_pairs = []
    for station_id, (l0_node, l1_node) in enumerate(zip(station_l0, station_l1)):
        station_pairs.append(
            {
                "station_id": station_id,
                "l0_node": int(l0_node),
                "l1_node": int(l1_node),
                "raw_l0_node": l0_to_raw[int(l0_node)],
            }
        )

    stations_config = {
        "selection_rule": {
            "description": "High-degree road nodes with spacing constraint on L0 and unique L1 projection.",
            "num_stations": NUM_STATIONS,
            "seed": SELECTION_SEED,
            "spacing_rule": "shortest-path hop distance >= floor(|V_L0| / (3 * num_stations))",
            "projection_rule": "pi(S_L0) must be injective for the chosen station set",
        },
        "graphs": {
            "l0_graph_path": os.path.relpath(L0_INDEXED_PATH, PROJECT_ROOT),
            "l1_graph_path": os.path.relpath(L1_GRAPH_PATH, PROJECT_ROOT),
            "node_map_path": os.path.relpath(NODE_MAP_PATH, PROJECT_ROOT),
            "edge_map_path": os.path.relpath(EDGE_MAP_PATH, PROJECT_ROOT),
            "anchor_path": os.path.relpath(ANCHOR_PATH, PROJECT_ROOT),
        },
        "l0_station_nodes": station_l0,
        "l1_station_nodes": station_l1,
        "station_pairs": station_pairs,
        "projection_collisions": collisions,
        "notes": [
            "Action index order is shared between L0 and L1.",
            "All evaluation for the ablation should be performed on the L0 graph.",
        ],
    }
    with open(STATIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stations_config, fh, ensure_ascii=False, indent=2)

    print(f"Saved L0 indexed graph: {_safe_display_path(L0_INDEXED_PATH)}")
    print(f"Saved node map: {_safe_display_path(NODE_MAP_PATH)}")
    print(f"Saved edge map: {_safe_display_path(EDGE_MAP_PATH)}")
    print(f"Saved anchor map: {_safe_display_path(ANCHOR_PATH)}")
    print(f"Saved station config: {_safe_display_path(STATIONS_PATH)}")
    print(f"L0 stations: {station_l0}")
    print(f"L1 stations: {station_l1}")
    print(f"Projection collisions: {collisions}")


if __name__ == "__main__":
    main()
