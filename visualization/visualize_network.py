"""
Visualize the traffic graph state (vehicles, stations, queues, prices).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

from simEvn.Traffic import TrafficPowerEnv


class NetworkVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def _infer_grid_positions(graph):
        num_nodes = graph.number_of_nodes()
        size = int(np.sqrt(num_nodes))
        if size * size == num_nodes:
            return {i: (i % size, size - 1 - i // size) for i in range(num_nodes)}
        return nx.spring_layout(graph, seed=42)

    def plot_state(self, env, show=True, save_path=None, title=None):
        data = env.get_graph_state()
        graph = env.traffic_graph

        vehicles = data.x[:, 0].cpu().numpy()
        is_station = data.x[:, 1].cpu().numpy().astype(bool)
        queue_len = data.x[:, 2].cpu().numpy()
        prices = data.x[:, 3].cpu().numpy()

        pos = self._infer_grid_positions(graph)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_axis_off()

        # Draw edges
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.4, width=1.0)

        # Base nodes colored by vehicle count
        node_sizes = 250 + vehicles * 220 + queue_len * 120
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=vehicles,
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=max(1, vehicles.max())
        )

        # Overlay stations with price-based color
        station_nodes = [i for i, v in enumerate(is_station) if v]
        if station_nodes:
            station_prices = prices[station_nodes]
            station_sizes = 400 + queue_len[station_nodes] * 220
            station_plot = nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                nodelist=station_nodes,
                node_size=station_sizes,
                node_color=station_prices,
                cmap=plt.cm.Oranges,
                node_shape='s',
                vmin=max(0.0, station_prices.min()),
                vmax=max(1.0, station_prices.max())
            )
        else:
            station_plot = None

        # Labels for stations
        station_labels = {
            s.traffic_node_id: f"S{s.id}\nQ:{len(s.queue)}\nP:{s.current_price:.1f}"
            for s in env.stations
        }
        nx.draw_networkx_labels(graph, pos, labels=station_labels, font_size=8, ax=ax)

        # Colorbars
        cbar1 = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Vehicle count")
        if station_plot is not None:
            cbar2 = fig.colorbar(station_plot, ax=ax, fraction=0.046, pad=0.12)
            cbar2.set_label("Station price")

        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.set_title("Traffic-Power Graph State", fontsize=12, fontweight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    env = TrafficPowerEnv()
    viz = NetworkVisualizer()
    viz.plot_state(env)
