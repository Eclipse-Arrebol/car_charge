from abc import ABC, abstractmethod
import random as rng

from train import DQNAgent


class BaseStrategy(ABC):
    @abstractmethod
    def select_action(self, env, ev, pending_counts) -> int:
        """给定环境、EV、当前 pending 分配，返回动作（站点索引）。"""

    @abstractmethod
    def name(self) -> str:
        pass


class RandomStrategy(BaseStrategy):
    def select_action(self, env, ev, pending_counts) -> int:
        action_mask = env.get_action_mask(ev)
        valid = [i for i in range(len(env.stations)) if action_mask[0, i].item()]
        return rng.choice(valid) if valid else 0

    def name(self) -> str:
        return "random"


class GreedyStrategy(BaseStrategy):
    def select_action(self, env, ev, pending_counts) -> int:
        action_mask = env.get_action_mask(ev)
        best_action = None
        best_score = float("inf")
        for i, station in enumerate(env.stations):
            if not action_mask[0, i].item():
                continue
            load = (
                len(station.queue)
                + len(station.connected_evs)
                + pending_counts.get(station.id, 0)
            ) / max(station.num_chargers, 1)
            if load < best_score:
                best_score = load
                best_action = i
        if best_action is None:
            valid = [i for i in range(len(env.stations)) if action_mask[0, i].item()]
            best_action = rng.choice(valid) if valid else 0
        return best_action

    def name(self) -> str:
        return "greedy"


class DQNStrategy(BaseStrategy):
    def __init__(self, model_path: str, env):
        self.agent = DQNAgent(
            num_features=18,
            num_actions=env.num_stations,
            station_node_ids=env.station_node_ids,
            num_nodes_per_graph=env.num_nodes,
        )
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.02

    def select_action(self, env, ev, pending_counts) -> int:
        ev_state = env.get_graph_state_for_ev(ev, pending_counts)
        action_mask = env.get_action_mask(ev)
        return self.agent.select_action(ev_state, action_mask=action_mask)

    def name(self) -> str:
        return "dqn"


class FedDQNStrategy(DQNStrategy):
    def name(self) -> str:
        return "feddqn"
