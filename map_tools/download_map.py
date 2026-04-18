import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 珠江新城中心坐标，半径 4000m 覆盖天河核心区
CENTER_LAT = 23.1197
CENTER_LON = 113.3227
RADIUS_M = 4000
NETWORK_TYPE = "drive"
TIMEOUT = 60
MAIN_ROAD_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|'
    'motorway_link|trunk_link|primary_link|secondary_link"]'
    '["access"!~"private"]'
    '["motor_vehicle"!~"no"]'
    '["service"!~"parking|parking_aisle|driveway|private"]'
)
OVERPASS_URLS = [
    "https://overpass-api.de/api/",
    "https://overpass.kumi.systems/api/",
    "https://maps.mail.ru/osm/tools/overpass/api/",
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_mainroads.graphml")
PREVIEW_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_mainroads_preview.png")


def render_preview(graph, output_path: str) -> None:
    fig, ax = ox.plot_graph(
        graph,
        node_size=8,
        node_color="#d62828",
        node_alpha=0.85,
        edge_linewidth=0.7,
        edge_color="#1d3557",
        edge_alpha=0.75,
        bgcolor="white",
        figsize=(14, 14),
        show=False,
        close=False,
    )
    ax.set_title("Guangzhou Tianhe Main Roads (EV-drivable filter)", fontsize=15, pad=14)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def download_graph():
    last_error = None
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for overpass_url in OVERPASS_URLS:
        try:
            print(f"Trying Overpass server: {overpass_url}")
            ox.settings.overpass_url = overpass_url
            ox.settings.requests_timeout = TIMEOUT
            ox.settings.requests_kwargs = {"proxies": {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}}

            print(
                f"Downloading road network from point "
                f"({CENTER_LAT}, {CENTER_LON}), radius={RADIUS_M}m, network_type={NETWORK_TYPE}"
            )
            print(f"Using main-road filter: {MAIN_ROAD_FILTER}")
            graph = ox.graph_from_point(
                (CENTER_LAT, CENTER_LON),
                dist=RADIUS_M,
                network_type=NETWORK_TYPE,
                custom_filter=MAIN_ROAD_FILTER,
            )
            print("Converting graph to undirected")
            graph = ox.convert.to_undirected(graph)

            print("Extracting largest connected component")
            largest_component = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_component).copy()

            print(f"Saving graphml to: {OUTPUT_PATH}")
            ox.save_graphml(graph, OUTPUT_PATH)
            print(f"Rendering preview image to: {PREVIEW_PATH}")
            render_preview(graph, PREVIEW_PATH)

            print(f"Downloaded graph successfully from: {overpass_url}")
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")
            print(f"Saved to: {OUTPUT_PATH}")
            print(f"Preview saved to: {PREVIEW_PATH}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Failed with {overpass_url}: {exc}", file=sys.stderr)

    print("All Overpass servers failed.", file=sys.stderr)
    if last_error is not None:
        print(f"Last error: {last_error}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download_graph()
