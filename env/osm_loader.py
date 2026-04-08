"""
OSMnx 真实路网加载工具
======================
支持两种模式：
  1. 联网模式：从 OpenStreetMap 下载真实路网（需要网络）
  2. 离线模式：生成具有真实路网特征的合成图（无需网络，用于无法联网环境）

用法示例:
    from env.osm_loader import load_road_network

    # 联网模式
    graph, station_nodes, node_positions = load_road_network(
        place="Wuhan, China", num_stations=2, max_nodes=40
    )

    # 离线模式（自动降级）
    graph, station_nodes, node_positions = load_road_network(
        place="Wuhan", num_stations=2, max_nodes=40, offline=True
    )
"""

import os
import random
import pickle
import numpy as np
import networkx as nx

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


# ============================================================
# 核心加载函数
# ============================================================
def load_road_network(
    place: str = "Wuchang District, Wuhan, China",
    num_stations: int = 2,
    max_nodes: int = 40,
    cache_dir: str = None,
    seed: int = 42,
    offline: bool = False,
):
    """
    加载路网。联网失败时自动降级到离线合成模式。

    Args:
        place:        OSM 地名字符串
        num_stations: 充电站数量
        max_nodes:    精简后的节点上限
        cache_dir:    本地缓存目录
        seed:         随机种子
        offline:      True 直接使用离线合成路网，跳过网络请求

    Returns:
        graph         : nx.Graph，节点 ID 从 0 开始
        station_nodes : List[int]，充电站节点索引
        positions     : Dict[int, (float, float)]，节点坐标
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache"
        )
    os.makedirs(cache_dir, exist_ok=True)

    safe_place = place.replace(",", "").replace(" ", "_")
    cache_file = os.path.join(
        cache_dir, f"{safe_place}_n{max_nodes}_s{num_stations}.pkl"
    )

    # ── 读缓存 ──
    if os.path.exists(cache_file):
        print(f"[OSMnx] 从缓存加载路网: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # ── 离线模式 / 无法联网 → 合成路网 ──
    if offline or not HAS_OSMNX:
        return _build_synthetic_network(
            num_nodes=max_nodes, num_stations=num_stations,
            seed=seed, cache_file=cache_file
        )

    # ── 联网下载（依次尝试多个 Overpass 服务器，中国用户可能需要镜像）──
    # 可在 run_training_real_map.py 中将 offline=True 切换到离线模式
    # 也可提供 graphml_file 路径从本地文件加载（见下面的 load_road_network_from_file）
    OVERPASS_SERVERS = [
        "https://overpass.kumi.systems/api/",       # 全球 CDN 镜像（推荐）
        "https://maps.mail.ru/osm/tools/overpass/api/",  # 俄罗斯镜像
        "https://overpass-api.de/api/",              # 官方（德国）
    ]

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 20  # 每个服务器最多等 20 秒，快速失败

    G_raw = None
    for server in OVERPASS_SERVERS:
        try:
            print(f"[OSMnx] 尝试下载路网: {place}  (服务器: {server})")
            ox.settings.overpass_url = server
            G_raw = ox.graph_from_place(place, network_type="drive")
            print(f"[OSMnx] 下载成功 ✓")
            break
        except Exception as e:
            print(f"[OSMnx] 服务器不可用 ({type(e).__name__})，尝试下一个…")
            continue

    if G_raw is None:
        print("[OSMnx] 所有服务器均不可用，切换到离线合成路网。")
        print("        若要使用真实路网，请参考 docs/项目文件说明.md 中的离线获取方法。")
        return _build_synthetic_network(
            num_nodes=max_nodes, num_stations=num_stations,
            seed=seed, cache_file=cache_file
        )

    # ── 联网成功：简化处理 ──
    G_undirected = ox.convert.to_undirected(G_raw)
    G_connected = G_undirected.subgraph(
        max(nx.connected_components(G_undirected), key=len)
    ).copy()

    if G_connected.number_of_nodes() > max_nodes:
        G_connected = _sample_nodes(G_connected, max_nodes, seed)

    mapping = {old: new for new, old in enumerate(sorted(G_connected.nodes()))}
    G_final = nx.relabel_nodes(G_connected, mapping)
    if not nx.is_connected(G_final):
        G_final = G_final.subgraph(
            max(nx.connected_components(G_final), key=len)
        ).copy()
        G_final = nx.convert_node_labels_to_integers(G_final)

    positions = {
        n: (d.get("x", 0.0), d.get("y", 0.0))
        for n, d in G_final.nodes(data=True)
    }
    station_nodes = _select_station_nodes(G_final, num_stations, seed)
    print(f"[OSMnx] Network nodes: {G_final.number_of_nodes()}, station_nodes: {station_nodes}")

    result = (G_final, station_nodes, positions)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


# ============================================================
# 从本地文件加载路网（GraphML / OSM XML）
# ============================================================
def load_road_network_from_file(
    filepath: str,
    num_stations: int = 2,
    max_nodes: int = 40,
    cache_dir: str = None,
    seed: int = 42,
) -> tuple:
    """
    从本地 .graphml 或 .osm 文件加载真实路网，无需联网。
    推荐通过项目根目录的 download_map.py 生成 .graphml 文件。
    返回格式与 load_road_network 完全一致。
    """
    if not HAS_OSMNX:
        raise ImportError("请先安装 osmnx: pip install osmnx")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"路网文件不存在: {filepath}")

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
        )
    os.makedirs(cache_dir, exist_ok=True)

    fname = os.path.splitext(os.path.basename(filepath))[0]
    cache_file = os.path.join(cache_dir, f"local_{fname}_n{max_nodes}_s{num_stations}.pkl")

    if os.path.exists(cache_file):
        print(f"[LocalFile] 从缓存加载路网: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    ext = os.path.splitext(filepath)[1].lower()
    print(f"[LocalFile] 正在加载本地路网文件: {filepath}")

    if ext == ".graphml":
        G_raw = ox.load_graphml(filepath)
    elif ext in (".osm", ".xml"):
        G_raw = ox.graph_from_xml(filepath)
    else:
        raise ValueError(f"不支持的文件格式 '{ext}'，请使用 .graphml 或 .osm")

    G_undirected = ox.convert.to_undirected(G_raw)
    # convert_node_labels_to_integers 兼容 consolidate_intersections 产生的 frozenset 节点 ID
    G_undirected = nx.convert_node_labels_to_integers(G_undirected)

    if not nx.is_connected(G_undirected):
        G_undirected = G_undirected.subgraph(
            max(nx.connected_components(G_undirected), key=len)
        ).copy()
        G_undirected = nx.convert_node_labels_to_integers(G_undirected)

    n_total = G_undirected.number_of_nodes()
    print(f"[LocalFile] 原始路网节点: {n_total}, 边: {G_undirected.number_of_edges()}")

    # 只在节点数大于目标时精简，节点太少时直接用全图
    if n_total > max_nodes:
        G_final = _sample_nodes(G_undirected, max_nodes, seed)
        G_final = nx.convert_node_labels_to_integers(G_final)
    else:
        G_final = G_undirected

    if G_final.number_of_nodes() < num_stations:
        raise RuntimeError(
            f"路网节点数({G_final.number_of_nodes()})少于充电站数({num_stations})。\n"
            "请增大 RADIUS_METERS 或减小 CONSOLIDATE_TOL 后重新运行 download_map.py。"
        )

    positions = {n: (d.get("x", 0.0), d.get("y", 0.0))
                 for n, d in G_final.nodes(data=True)}
    station_nodes = _select_station_nodes(G_final, num_stations, seed)

    print(f"[LocalFile] 精简后节点: {G_final.number_of_nodes()}, "
          f"边: {G_final.number_of_edges()}, 充电站: {station_nodes}")

    result = (G_final, station_nodes, positions)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


# ============================================================
# 离线合成路网（模拟真实城市路网特征）
# ============================================================
def _build_synthetic_network(
    num_nodes: int = 30,
    num_stations: int = 2,
    seed: int = 42,
    cache_file: str = None,
):
    """Watts-Strogatz 小世界合成路网，无需联网。"""
    np.random.seed(seed)
    print(f"[Offline] Building synthetic urban network (nodes={num_nodes})")

    G = _make_urban_graph(num_nodes, seed)

    # 确保连通
    if not nx.is_connected(G):
        G = G.subgraph(
            max(nx.connected_components(G), key=len)
        ).copy()
    G = nx.convert_node_labels_to_integers(G)

    # 生成节点坐标（模拟经纬度，用于可视化）
    pos_layout = nx.spring_layout(G, seed=seed, k=1.5)
    # 映射到武汉市坐标范围附近（经度114.2~114.5, 纬度30.4~30.7）
    positions = {}
    for node, (x, y) in pos_layout.items():
        lon = 114.2 + (x + 1) / 2 * 0.3
        lat = 30.4  + (y + 1) / 2 * 0.3
        positions[node] = (lon, lat)

    # 把坐标写入节点属性（与 osmnx 格式一致）
    for node, (lon, lat) in positions.items():
        G.nodes[node]["x"] = lon
        G.nodes[node]["y"] = lat

    station_nodes = _select_station_nodes(G, num_stations, seed)
    print(f"[Offline] Synthetic network: nodes={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, station_nodes={station_nodes}")

    result = (G, station_nodes, positions)
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"[Offline] Network cached: {cache_file}")

    return result


def _make_urban_graph(n: int, seed: int) -> nx.Graph:
    """Watts-Strogatz 小世界图：k=4邻居，p=0.2重连概率"""
    k = min(4, n - 1)
    return nx.watts_strogatz_graph(n, k=k, p=0.2, seed=seed)


# ============================================================
# 内部工具函数
# ============================================================
def _sample_nodes(G: nx.Graph, target_n: int, seed: int) -> nx.Graph:
    """
    从大图中采样 target_n 个节点，保证子图连通。
    策略：从度数最大节点出发 BFS，优先扩展度数大的邻居，
    直到收集满 target_n 个节点。
    KMeans 子图法在稀疏路网中会产生孤立节点，故不采用。
    """
    if G.number_of_nodes() <= target_n:
        return G.copy()

    # 起点：全图度数最大的节点（主干道交叉口）
    start = max(G.nodes(), key=lambda n: G.degree(n))

    visited = [start]
    seen = {start}
    queue = [start]

    while queue and len(visited) < target_n:
        node = queue.pop(0)
        # 按度数降序扩展邻居，优先走主干道
        neighbors = sorted(G.neighbors(node),
                           key=lambda n: G.degree(n), reverse=True)
        for nb in neighbors:
            if nb not in seen:
                seen.add(nb)
                visited.append(nb)
                queue.append(nb)
                if len(visited) >= target_n:
                    break

    subG = G.subgraph(visited).copy()
    return nx.convert_node_labels_to_integers(subG)


def _select_station_nodes(G: nx.Graph, num_stations: int, seed: int) -> list:
    """选充电站：优先高度节点，互相保持距离"""
    random.seed(seed)
    nodes = list(G.nodes())
    degrees = dict(G.degree())
    sorted_nodes = sorted(nodes, key=lambda n: degrees[n], reverse=True)

    selected = []
    min_dist = max(2, G.number_of_nodes() // (num_stations * 3))
    for candidate in sorted_nodes:
        if not selected:
            selected.append(candidate)
        else:
            too_close = any(
                nx.shortest_path_length(G, candidate, s) < min_dist
                for s in selected
                if nx.has_path(G, candidate, s)
            )
            if not too_close:
                selected.append(candidate)
        if len(selected) == num_stations:
            break

    remaining = [n for n in nodes if n not in selected]
    while len(selected) < num_stations and remaining:
        selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))

    return selected


# ============================================================
# 经纬度方式（备用，同样支持离线降级）
# ============================================================
def load_road_network_by_point(
    lat: float, lon: float,
    dist_m: int = 1500,
    num_stations: int = 2,
    max_nodes: int = 40,
    cache_dir: str = None,
    seed: int = 42,
    offline: bool = False,
):
    """
    以经纬度为中心下载路网，不可用时自动降级离线模式。

    示例（武汉光谷）:
        graph, stations, pos = load_road_network_by_point(
            lat=30.5083, lon=114.4142, dist_m=1500
        )
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache"
        )
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"point_{lat}_{lon}_{dist_m}_n{max_nodes}.pkl"
    )

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    if offline or not HAS_OSMNX:
        return _build_synthetic_network(
            num_nodes=max_nodes, num_stations=num_stations,
            seed=seed, cache_file=cache_file
        )

    try:
        print(f"[OSMnx] 下载坐标 ({lat}, {lon}) 半径 {dist_m}m 路网...")
        ox.settings.use_cache = True
        ox.settings.log_console = False
        G_raw = ox.graph_from_point((lat, lon), dist=dist_m, network_type="drive")
        G_raw = ox.project_graph(G_raw)
    except Exception as e:
        print(f"[OSMnx] Download failed, switching to offline mode. Reason: {e}")
        return _build_synthetic_network(
            num_nodes=max_nodes, num_stations=num_stations,
            seed=seed, cache_file=cache_file
        )

    G_undirected = ox.convert.to_undirected(G_raw)
    G_connected = G_undirected.subgraph(
        max(nx.connected_components(G_undirected), key=len)
    ).copy()
    if G_connected.number_of_nodes() > max_nodes:
        G_connected = _sample_nodes(G_connected, max_nodes, seed)

    mapping = {old: new for new, old in enumerate(sorted(G_connected.nodes()))}
    G_final = nx.relabel_nodes(G_connected, mapping)
    if not nx.is_connected(G_final):
        G_final = G_final.subgraph(
            max(nx.connected_components(G_final), key=len)
        ).copy()
        G_final = nx.convert_node_labels_to_integers(G_final)

    positions = {n: (d.get("x", 0), d.get("y", 0))
                 for n, d in G_final.nodes(data=True)}
    station_nodes = _select_station_nodes(G_final, num_stations, seed)

    result = (G_final, station_nodes, positions)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result
