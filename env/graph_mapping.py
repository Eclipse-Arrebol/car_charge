import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BASELINE_DIR = os.path.join(_PROJECT_ROOT, "map_outputs", "baseline_eps40_artifacts")
_DEFAULT_STATION_CONFIG = os.path.join(_PROJECT_ROOT, "config", "stations.json")


def load_station_config(path: Optional[str] = None) -> dict:
    path = path or _DEFAULT_STATION_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_station_nodes(path: Optional[str], key: str) -> Optional[List[int]]:
    if not path:
        return None
    data = load_station_config(path)
    values = data.get(key)
    if values is None:
        raise KeyError(f"Station config does not contain key '{key}'")
    return [int(v) for v in values]


class MappingService:
    """Runtime access layer for pi / rho mappings used by L0-L1 ablation experiments."""

    def __init__(
        self,
        node_map_path: Optional[str] = None,
        edge_map_path: Optional[str] = None,
        anchor_path: Optional[str] = None,
    ):
        node_map_path = node_map_path or os.path.join(_DEFAULT_BASELINE_DIR, "node_map_eps40.json")
        edge_map_path = edge_map_path or os.path.join(_DEFAULT_BASELINE_DIR, "edge_map_eps40.json")
        anchor_path = anchor_path or os.path.join(_DEFAULT_BASELINE_DIR, "cluster_anchors_eps40.json")

        with open(node_map_path, "r", encoding="utf-8") as fh:
            raw_node_map = json.load(fh)
        with open(edge_map_path, "r", encoding="utf-8") as fh:
            raw_edge_map = json.load(fh)
        with open(anchor_path, "r", encoding="utf-8") as fh:
            raw_anchor_map = json.load(fh)

        self.node_map: Dict[int, int] = {int(k): int(v) for k, v in raw_node_map.items()}
        self.inverse_node_map: Dict[int, List[int]] = defaultdict(list)
        for l0_node, l1_node in self.node_map.items():
            self.inverse_node_map[l1_node].append(l0_node)

        self.edge_map: Dict[Tuple[int, int], dict] = {}
        for item in raw_edge_map:
            u = int(item["train_u"])
            v = int(item["train_v"])
            key = (u, v) if u <= v else (v, u)
            self.edge_map[key] = item

        self.anchor_map: Dict[int, dict] = {int(k): v for k, v in raw_anchor_map.items()}

    def project(self, l0_node: int) -> int:
        return self.node_map[int(l0_node)]

    def cluster_of(self, l0_node: int) -> int:
        return self.project(l0_node)

    def lift(self, l1_edge: Tuple[int, int]) -> List[int]:
        u, v = l1_edge
        key = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
        item = self.edge_map.get(key)
        if item is None:
            raise KeyError(f"No lifted edge path for L1 edge {key}")
        return [int(n) for n in item.get("raw_path", [])]

    def representative_anchor(self, l1_node: int) -> Optional[int]:
        item = self.anchor_map.get(int(l1_node))
        if item is None:
            return None
        return int(item.get("l0_node", item.get("raw_node", item.get("indexed_l0_node", -1))))
