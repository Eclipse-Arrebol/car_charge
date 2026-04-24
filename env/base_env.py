import networkx as nx
import numpy as np
import random
import torch
from torch_geometric.data import Data

from env.entities import EV
from env.power_grid import PowerGrid, get_tou_multiplier
from env.charging_station import ChargingStation


NODE_FEATURE_DIM = 18


class TrafficPowerEnv:
    def __init__(self, num_evs=10, respawn_after_full_charge=True):
        self.traffic_graph = nx.grid_2d_graph(3, 3)
        self.traffic_graph = nx.convert_node_labels_to_integers(self.traffic_graph)
        self.num_evs = num_evs
        self.respawn_after_full_charge = respawn_after_full_charge
        self.charge_trigger_soc = 30.0

        self.stations = [
            ChargingStation(station_id=0, traffic_node_id=0, power_node_id="Grid_A",
                            respawn_after_full_charge=self.respawn_after_full_charge),
            ChargingStation(station_id=1, traffic_node_id=8, power_node_id="Grid_B",
                            respawn_after_full_charge=self.respawn_after_full_charge)
        ]

        self.evs = []
        for i in range(self.num_evs):
            start_node = random.randint(1, 7)
            self.evs.append(EV(i, start_node))

        self.power_limit = 15.0
        self.time_step = 0
        self.steps_per_day = 144
        self.step_duration_h = 1 / 6
        self.bpr_alpha = 0.15
        self.bpr_beta = 4.0
        self.edge_active_counts = {}
        self.edge_step_counts = {}
        self.edge_peak_counts = {}

        self.power_grid = PowerGrid()
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0

        self.edge_index = self._build_edge_index()
        self._path_cache_step: dict = {}

    def _reset_mask_stats_and_print(self):
        if hasattr(self, "_mask_stats") and self._mask_stats.get("total", 0) > 0:
            total = self._mask_stats["total"]
            all_masked = self._mask_stats["all_masked_before_fallback"]
            partial = self._mask_stats["partial_masked"]
            fallback = self._mask_stats["fallback"]
            print(
                f"[mask_stats] total={total} "
                f"all_masked_before_fallback={all_masked} "
                f"partial_masked={partial} "
                f"fallback={fallback} "
                f"all_masked_ratio={all_masked / max(total, 1):.4f} "
                f"partial_masked_ratio={partial / max(total, 1):.4f}"
            )
        self._mask_stats = {
            "total": 0,
            "all_masked_before_fallback": 0,
            "partial_masked": 0,
            "fallback": 0,
        }

    def reset(self):
        self._reset_mask_stats_and_print()
        self.stations = [
            ChargingStation(station_id=0, traffic_node_id=0, power_node_id="Grid_A",
                            respawn_after_full_charge=self.respawn_after_full_charge),
            ChargingStation(station_id=1, traffic_node_id=8, power_node_id="Grid_B",
                            respawn_after_full_charge=self.respawn_after_full_charge)
        ]
        self.evs = []
        for i in range(self.num_evs):
            start_node = random.randint(1, 7)
            self.evs.append(EV(i, start_node))
        self.power_limit = 15.0
        self.time_step = 0
        self.power_grid = PowerGrid()
        self.tou_multiplier = 1.0
        self.price_noise = 0.0
        self.prev_total_load = 0.0
        self.edge_active_counts = {}
        self.edge_step_counts = {}
        self.edge_peak_counts = {}
        return self.get_graph_state()

    def _reset_ev_charging_attempt(self, ev):
        ev.status = "IDLE"
        ev.target_station_idx = None
        ev.assigned_station = None
        ev.path = []
        ev.current_edge_from = None
        ev.current_edge_target = None
        ev.remaining_edge_time_h = 0.0
        ev.wait_time_h = 0.0
        ev.current_edge_speed_kph = 0.0
        ev.charge_decision_pending = ev.soc < self.charge_trigger_soc
        ev.low_soc_triggered = ev.soc < self.charge_trigger_soc
        ev.remaining_replans = 1

    def should_request_charge_decision(self, ev):
        if ev.status != "IDLE":
            return False

        if ev.charge_decision_pending:
            return True

        if ev.soc >= self.charge_trigger_soc:
            ev.low_soc_triggered = False
            return False

        if not ev.low_soc_triggered:
            ev.low_soc_triggered = True
            ev.charge_decision_pending = True
            return True

        return False

    def get_pending_decision_evs(self):
        pending_evs = [ev for ev in self.evs if self.should_request_charge_decision(ev)]
        pending_evs.sort(key=lambda ev: ev.soc)
        return pending_evs

    def _find_best_station_metrics(self, ev):
        action_mask = self.get_action_mask(ev)
        best_station_id = None
        best_metrics = None
        for station in self.stations:
            if not action_mask[0, station.id].item():
                continue
            metrics = self.estimate_action_metrics(ev, station.id)
            if best_metrics is None or metrics["generalized_cost"] < best_metrics["generalized_cost"]:
                best_station_id = station.id
                best_metrics = metrics
        return best_station_id, best_metrics

    def _should_replan_in_transit(self, ev, target_station):
        return False   # 禁用 replan 支线,避免绕过 pending_counts 的 mask
        # 以下原代码保留不删
        if ev.remaining_replans <= 0:
            return False
        if ev.remaining_edge_time_h > 1e-9:
            return False

        current_metrics = self.estimate_action_metrics(ev, target_station.id)
        queue_exploded = len(target_station.queue) >= max(1, int(0.8 * target_station.max_queue_len))
        wait_too_high = current_metrics["queue_time_h"] >= 0.75 * target_station.max_wait_time_h
        if not (queue_exploded or wait_too_high):
            return False

        best_station_id, best_metrics = self._find_best_station_metrics(ev)
        if best_station_id is None or best_station_id == target_station.id:
            return False

        much_lower_queue = best_metrics["queue_time_h"] + 0.5 < current_metrics["queue_time_h"]
        meaningfully_lower_cost = (
            best_metrics["generalized_cost"] + 10.0 < current_metrics["generalized_cost"]
        )
        return much_lower_queue or meaningfully_lower_cost

    @staticmethod
    def _parse_speed_kph(speed_raw, default=50.0):
        if isinstance(speed_raw, list):
            speed_raw = speed_raw[0] if speed_raw else default
        if isinstance(speed_raw, str):
            cleaned = "".join(ch for ch in speed_raw if (ch.isdigit() or ch == "."))
            speed_raw = float(cleaned) if cleaned else default
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            speed = float(default)
        return max(1.0, speed)

    @staticmethod
    def _parse_lanes_count(lanes_raw, default=1.0):
        if isinstance(lanes_raw, list):
            lanes_raw = lanes_raw[0] if lanes_raw else default
        if isinstance(lanes_raw, str):
            cleaned = "".join(ch for ch in lanes_raw if (ch.isdigit() or ch == "."))
            lanes_raw = float(cleaned) if cleaned else default
        try:
            lanes = float(lanes_raw)
        except (TypeError, ValueError):
            lanes = float(default)
        return max(1.0, lanes)

    @staticmethod
    def _infer_capacity_per_lane(highway_type):
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else ""
        h = str(highway_type).lower()
        if "motorway" in h:
            return 2200.0
        if "trunk" in h:
            return 2000.0
        if "primary" in h:
            return 1800.0
        if "secondary" in h:
            return 1500.0
        if "tertiary" in h:
            return 1200.0
        if "residential" in h:
            return 800.0
        if "service" in h:
            return 600.0
        return 1000.0

    @classmethod
    def _parse_capacity_vehph(cls, attrs):
        cap_raw = attrs.get("capacity", None)
        if cap_raw is not None:
            if isinstance(cap_raw, list):
                cap_raw = cap_raw[0] if cap_raw else None
            if isinstance(cap_raw, str):
                cleaned = "".join(ch for ch in cap_raw if (ch.isdigit() or ch == "."))
                cap_raw = float(cleaned) if cleaned else None
            try:
                cap = float(cap_raw)
                if cap > 0:
                    return cap
            except (TypeError, ValueError):
                pass

        lanes = cls._parse_lanes_count(attrs.get("lanes", 1))
        cap_lane = cls._infer_capacity_per_lane(attrs.get("highway", ""))
        return lanes * cap_lane

    @classmethod
    def _edge_profiles_from_data(cls, edge_data):
        if isinstance(edge_data, dict) and edge_data and all(isinstance(v, dict) for v in edge_data.values()):
            candidates = list(edge_data.values())
        else:
            candidates = [edge_data or {}]

        profiles = []
        for attrs in candidates:
            length = attrs.get("length", 100.0)
            if isinstance(length, list):
                length = length[0] if length else 100.0
            try:
                length_m = max(1.0, float(length))
            except (TypeError, ValueError):
                length_m = 100.0

            speed_kph = cls._parse_speed_kph(attrs.get("speed_kph", attrs.get("maxspeed", 50.0)))
            t0_h = (length_m / 1000.0) / speed_kph
            capacity_vehph = cls._parse_capacity_vehph(attrs)
            profiles.append((length_m, speed_kph, max(1e-6, t0_h), max(1.0, capacity_vehph)))

        return profiles

    def _edge_flow(self, u, v):
        return float(self.edge_active_counts.get((u, v), 0) + self.edge_active_counts.get((v, u), 0))

    def _bpr_time_h(self, t0_h, x_flow, c_capacity):
        ratio = max(0.0, float(x_flow)) / max(1.0, float(c_capacity) * self.step_duration_h)
        return max(1e-6, float(t0_h) * (1.0 + self.bpr_alpha * (ratio ** self.bpr_beta)))

    def _dynamic_profiles(self, u, v, add_vehicle=0.0):
        edge_data = self.traffic_graph.get_edge_data(u, v, default={})
        base_profiles = self._edge_profiles_from_data(edge_data)
        x_flow = self._edge_flow(u, v) + float(add_vehicle)

        dyn = []
        for length_m, speed_kph, t0_h, capacity_vehph in base_profiles:
            t_h = self._bpr_time_h(t0_h, x_flow, capacity_vehph)
            dyn.append((length_m, speed_kph, t_h, t0_h, x_flow, capacity_vehph))
        return dyn

    def _travel_time_weight(self, u, v, edge_data):
        dyn = self._dynamic_profiles(u, v, add_vehicle=1.0)
        if dyn:
            return min(item[2] for item in dyn)
        profiles = self._edge_profiles_from_data(edge_data)
        return min((t0 for _, _, t0, _ in profiles), default=1e-6)

    def get_edge_travel_profile(self, u, v):
        dyn = self._dynamic_profiles(u, v, add_vehicle=1.0)
        if not dyn:
            return 100.0, 50.0, 0.002
        length_m, speed_kph, t_h, _, _, _ = min(dyn, key=lambda x: x[2])
        return length_m, speed_kph, t_h

    def get_edge_base_profile(self, u, v):
        edge_data = self.traffic_graph.get_edge_data(u, v, default={})
        profiles = self._edge_profiles_from_data(edge_data)
        if not profiles:
            return 100.0, 50.0, 0.002, 1000.0
        return min(profiles, key=lambda x: x[2])

    def enter_edge(self, u, v):
        length_m, speed_kph, t_h = self.get_edge_travel_profile(u, v)
        key = (u, v)
        self.edge_active_counts[key] = self.edge_active_counts.get(key, 0) + 1
        self.edge_step_counts[key] = self.edge_step_counts.get(key, 0) + 1
        self.edge_peak_counts[key] = max(
            self.edge_peak_counts.get(key, 0),
            self.edge_active_counts[key],
        )
        return length_m, speed_kph, t_h

    def leave_edge(self, u, v):
        if u is None or v is None:
            return
        key = (u, v)
        cur = self.edge_active_counts.get(key, 0)
        if cur <= 1:
            self.edge_active_counts.pop(key, None)
        else:
            self.edge_active_counts[key] = cur - 1

    def _build_edge_index(self):
        adj = nx.to_scipy_sparse_array(self.traffic_graph).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64))
        col = torch.from_numpy(adj.col.astype(np.int64))
        edge_index = torch.stack([row, col], dim=0)

        edge_attr_list = []
        for u, v in zip(row.tolist(), col.tolist()):
            length_m, speed_kph, _, _ = self.get_edge_base_profile(u, v)
            edge_attr_list.append([float(length_m) / 1000.0, float(speed_kph) / 100.0])

        self.edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        return edge_index

    def _estimate_ev_station_metrics(self, ev, station, pending_counts=None):
        pending = 0 if pending_counts is None else pending_counts.get(station.id, 0)
        if not hasattr(self, "_path_cache_step"):
            self._path_cache_step = {}
        cache_key = (ev.curr_node, station.traffic_node_id)
        cached = self._path_cache_step.get(cache_key)
        if cached is not None:
            trip_time_h, trip_dist_km = cached
        else:
            try:
                path = nx.shortest_path(
                    self.traffic_graph,
                    source=ev.curr_node,
                    target=station.traffic_node_id,
                    weight=self._travel_time_weight,
                )
                trip_time_h = 0.0
                trip_dist_km = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    length_m, _, travel_h = self.get_edge_travel_profile(u, v)
                    trip_time_h += travel_h
                    trip_dist_km += length_m / 1000.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                trip_time_h = 24.0
                trip_dist_km = 999.0
            self._path_cache_step[cache_key] = (trip_time_h, trip_dist_km)

        queue_time_h = station.estimate_queue_wait_hours(incoming_count=pending)
        charge_time_h = station.estimate_charge_time_hours(ev)
        service_time_h = queue_time_h + charge_time_h
        charge_cost = station.estimate_charge_cost(ev)
        travel_energy_cost = trip_dist_km * ev.drive_kwh_per_km * station.current_price
        generalized_cost = ev.time_value * (trip_time_h + service_time_h) + charge_cost + travel_energy_cost

        estimated_station_load_kw_after = None
        estimated_grid_load_ratio_after = None
        estimated_voltage_pu_after = None
        current_voltage_pu = None
        estimated_voltage_delta_pu = None
        thevenin_r_ohm = None

        try:
            ev_power_kw = float(station._cc_cv_power(ev))
            estimated_station_load_kw_after = min(
                float(station.last_total_load) + ev_power_kw,
                float(station.max_grid_power),
            )
            estimated_grid_load_ratio_after = (
                estimated_station_load_kw_after / max(1.0, float(station.max_grid_power))
            )

            v_nominal_kv = getattr(self.power_grid, "v_nominal_kv", None)
            power_bus_idx = getattr(station, "power_bus_idx", None)
            has_thevenin = (
                power_bus_idx is not None
                and hasattr(self.power_grid, "get_bus_thevenin_resistance")
                and hasattr(self.power_grid, "get_last_bus_voltage")
                and v_nominal_kv is not None
            )
            if has_thevenin:
                thevenin_r_ohm = float(
                    self.power_grid.get_bus_thevenin_resistance(power_bus_idx)
                )
                current_voltage_pu = float(
                    self.power_grid.get_last_bus_voltage(power_bus_idx)
                )
                estimated_voltage_delta_pu = (
                    (ev_power_kw / 1000.0)
                    * thevenin_r_ohm
                    / (float(v_nominal_kv) ** 2)
                )
                estimated_voltage_pu_after = current_voltage_pu - estimated_voltage_delta_pu
            else:
                bus = getattr(station, "power_node_id", None)
                line = getattr(self.power_grid, "lines", {}).get(bus) if bus is not None else None
                if line is not None and v_nominal_kv is not None:
                    r_ohm = float(line["r_ohm"])
                    v2 = float(v_nominal_kv) ** 2
                    estimated_voltage_delta_pu = ev_power_kw * r_ohm / (v2 * 1000.0)
                    current_voltage_pu = self.power_grid.bus_voltages.get(bus, 1.0)
                    estimated_voltage_pu_after = current_voltage_pu - estimated_voltage_delta_pu
        except Exception:
            estimated_station_load_kw_after = None
            estimated_grid_load_ratio_after = None
            estimated_voltage_pu_after = None
            current_voltage_pu = None
            estimated_voltage_delta_pu = None
            thevenin_r_ohm = None

        return {
            "trip_time_h": trip_time_h,
            "trip_dist_km": trip_dist_km,
            "queue_time_h": queue_time_h,
            "charge_time_h": charge_time_h,
            "service_time_h": service_time_h,
            "charge_cost": charge_cost,
            "travel_energy_cost": travel_energy_cost,
            "generalized_cost": generalized_cost,
            "price": station.current_price,
            "grid_load_ratio": station.last_total_load / max(1.0, station.max_grid_power),
            "estimated_station_load_kw_after": estimated_station_load_kw_after,
            "estimated_grid_load_ratio_after": estimated_grid_load_ratio_after,
            "estimated_voltage_pu_after": estimated_voltage_pu_after,
            "current_voltage_pu": current_voltage_pu,
            "estimated_voltage_delta_pu": estimated_voltage_delta_pu,
            "thevenin_r_ohm": thevenin_r_ohm,
        }

    def estimate_action_metrics(self, ev, station_id, pending_counts=None):
        station = self.stations[station_id]
        return self._estimate_ev_station_metrics(ev, station, pending_counts=pending_counts)

    def get_graph_state(self):
        num_nodes = self.traffic_graph.number_of_nodes()
        x = torch.zeros((num_nodes, NODE_FEATURE_DIM), dtype=torch.float)

        for ev in self.evs:
            x[ev.curr_node, 0] += 1.0

        x[:, 7] = self.tou_multiplier
        x[:, 13] = self.price_noise

        for station in self.stations:
            node_idx = station.traffic_node_id
            x[node_idx, 1] = 1.0
            x[node_idx, 2] = len(station.queue)
            x[node_idx, 3] = station.current_price
            x[node_idx, 4] = len(station.connected_evs)
            x[node_idx, 5] = station.last_total_load / max(1.0, station.max_grid_power)
            x[node_idx, 6] = self.power_grid.bus_voltages.get(
                station.power_node_id, 1.0)
            x[node_idx, 14] = station.predicted_arrivals
            queue_wait_ratio = station.estimate_queue_wait_hours() / max(1e-6, station.max_wait_time_h)
            station_pressure = (
                len(station.queue) + len(station.connected_evs) + station.predicted_arrivals
            ) / max(1.0, station.max_queue_len + station.num_chargers)
            x[node_idx, 15] = min(2.0, queue_wait_ratio)
            x[node_idx, 16] = station_pressure

        data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return data

    def get_graph_state_for_ev(self, ev, pending_counts=None):
        data = self.get_graph_state()
        data.x[ev.curr_node, 8] = ev.soc / 100.0

        for station in self.stations:
            metrics = self._estimate_ev_station_metrics(ev, station, pending_counts=pending_counts)
            data.x[station.traffic_node_id, 9] = 1.0 / (1.0 + metrics["trip_time_h"])
            data.x[station.traffic_node_id, 10] = metrics["trip_time_h"]
            data.x[station.traffic_node_id, 11] = metrics["service_time_h"]
            data.x[station.traffic_node_id, 12] = metrics["generalized_cost"] / 100.0
            data.x[station.traffic_node_id, 15] = min(
                2.0, metrics["queue_time_h"] / max(1e-6, station.max_wait_time_h)
            )
            pending = 0 if pending_counts is None else pending_counts.get(station.id, 0)
            data.x[station.traffic_node_id, 16] = (
                len(station.queue) + len(station.connected_evs) + station.predicted_arrivals + pending
            ) / max(1.0, station.max_queue_len + station.num_chargers)
            data.x[station.traffic_node_id, 17] = float(
                max(0, station.num_chargers - len(station.connected_evs)) / station.num_chargers
            )

        if pending_counts:
            for station in self.stations:
                pc = pending_counts.get(station.id, 0)
                if pc > 0:
                    data.x[station.traffic_node_id, 2] += pc
        return data

    def get_action_mask(self, ev, pending_counts=None):
        num_actions = len(self.stations)
        mask = torch.ones(1, num_actions, dtype=torch.bool)
        enable_timeout_mask = getattr(self, "enable_queue_timeout_mask", False)

        for i, station in enumerate(self.stations):
            if not nx.has_path(self.traffic_graph, ev.curr_node, station.traffic_node_id):
                mask[0, i] = False
                continue

            try:
                metrics = self._estimate_ev_station_metrics(ev, station)
                soc_needed = metrics["trip_dist_km"] * ev.drive_kwh_per_km / max(1e-6, ev.battery_capacity_kwh) * 100.0 + 2.0
                if ev.soc < soc_needed or metrics["trip_time_h"] >= 24.0:
                    mask[0, i] = False
                    continue
                if enable_timeout_mask:
                    incoming = 0 if pending_counts is None else pending_counts.get(station.id, 0)
                    # 新增:硬容量比例 mask(每个充电桩平均被挤 1.5 辆以上则拒绝)
                    load_ratio = (
                        len(station.queue) + len(station.connected_evs) + incoming
                    ) / max(1, station.num_chargers)
                    capacity_threshold = getattr(self, "queue_timeout_mask_capacity_ratio", 1.5)

                    if load_ratio > capacity_threshold:
                        mask[0, i] = False
                        continue

                    # 原有 timeout 判断保持
                    est_queue_h = station.estimate_queue_wait_hours(incoming_count=incoming)
                    est_total_h = metrics["trip_time_h"] + est_queue_h
                    safety_margin_h = getattr(self, "queue_timeout_mask_safety_margin_h", 0.5)
                    max_wait_time_h = getattr(station, "max_wait_time_h", 4.0)
                    if est_total_h >= max_wait_time_h - safety_margin_h:
                        mask[0, i] = False
                        continue
            except nx.NetworkXNoPath:
                mask[0, i] = False
                continue

        # 调试用统计，后续可删
        if not hasattr(self, "_mask_stats"):
            self._mask_stats = {
                "total": 0,
                "all_masked_before_fallback": 0,
                "partial_masked": 0,
                "fallback": 0,
            }

        self._mask_stats["total"] += 1

        valid_count = int(mask.sum().item())
        num_actions = len(self.stations)

        if valid_count == 0:
            self._mask_stats["all_masked_before_fallback"] += 1
            self._mask_stats["fallback"] += 1
        elif valid_count < num_actions:
            self._mask_stats["partial_masked"] += 1

        if not mask.any():
            mask.fill_(True)

        return mask

    def step(self, actions):
        self.time_step += 1
        self.edge_step_counts = {}
        self.edge_peak_counts = {}
        self._path_cache_step: dict = {}
        abandoned_this_step = []
        for ev in self.evs:
            ev.just_abandoned_this_step = False
        grid_loads = {station.power_node_id: 0.0 for station in self.stations}
        decision_metrics = {}
        pending_counts = {s.id: 0 for s in self.stations}
        arrivals_this_step = {s.id: 0 for s in self.stations}

        urgent_evs = [ev for ev in self.evs if ev.id in actions and ev.status == "IDLE"]
        urgent_evs.sort(key=lambda item: item.soc)
        for ev in urgent_evs:
            station_id = actions[ev.id]
            ev.charge_decision_pending = False
            metrics = self.estimate_action_metrics(ev, station_id, pending_counts=pending_counts)
            decision_metrics[ev.id] = metrics
            pending_counts[station_id] += 1

        for ev in self.evs:
            if ev.status == "IDLE":
                ev.soc -= 0.5
                if ev.id in actions:
                    target_id = actions[ev.id]
                    target_station = self.stations[target_id]
                    try:
                        path = nx.shortest_path(
                            self.traffic_graph,
                            source=ev.curr_node,
                            target=target_station.traffic_node_id,
                            weight=self._travel_time_weight,
                        )
                        if len(path) > 1:
                            ev.path = path[1:]
                            ev.target_station_idx = target_id
                            ev.status = "MOVING_TO_CHARGE"
                            ev.current_edge_from = None
                            ev.current_edge_target = None
                            ev.remaining_edge_time_h = 0.0
                            ev.current_edge_speed_kph = 0.0
                        else:
                            if len(target_station.queue) >= target_station.max_queue_len:
                                self._reset_ev_charging_attempt(ev)
                            else:
                                target_station.queue.append(ev)
                                ev.target_station_idx = target_id
                                ev.status = "WAITING"
                                ev.wait_time_h = 0.0
                                arrivals_this_step[target_station.id] += 1
                    except Exception:
                        self._reset_ev_charging_attempt(ev)
                else:
                    neighbors = list(self.traffic_graph.neighbors(ev.curr_node))
                    if neighbors:
                        ev.curr_node = random.choice(neighbors)

            elif ev.status == "MOVING_TO_CHARGE":
                ev.travel_steps += 1
                ev.move(self, step_hours=self.step_duration_h)
                target_station = self.stations[ev.target_station_idx]
                if (
                    ev.curr_node != target_station.traffic_node_id
                    and self._should_replan_in_transit(ev, target_station)
                ):
                    ev.remaining_replans -= 1
                    self._reset_ev_charging_attempt(ev)
                    ev.remaining_replans = 0
                    continue
                if (
                    not ev.path
                    and ev.remaining_edge_time_h <= 1e-9
                    and ev.curr_node == target_station.traffic_node_id
                ):
                    if len(target_station.queue) >= target_station.max_queue_len:
                        self._reset_ev_charging_attempt(ev)
                    else:
                        target_station.queue.append(ev)
                        ev.status = "WAITING"
                        ev.wait_time_h = 0.0
                        arrivals_this_step[target_station.id] += 1

            elif ev.status == "WAITING":
                ev.wait_steps += 1
                ev.wait_time_h += self.step_duration_h
                target_station = None
                if ev.target_station_idx is not None:
                    target_station = self.stations[ev.target_station_idx]
                if target_station is not None and ev.wait_time_h >= target_station.max_wait_time_h:
                    if ev in target_station.queue:
                        target_station.queue.remove(ev)
                    ev.abandoned_charge_count += 1
                    ev.just_abandoned_this_step = True
                    abandoned_this_step.append(ev.id)
                    self._reset_ev_charging_attempt(ev)

            elif ev.status == "CHARGING":
                ev.charge_steps += 1
                ev.charge_time_h += self.step_duration_h

        for station in self.stations:
            station.update_arrival_prediction(arrivals_this_step.get(station.id, 0))

        self.tou_multiplier = get_tou_multiplier(self.time_step, self.steps_per_day)
        self.price_noise = random.uniform(-0.1, 0.1)

        total_realized_power = 0.0
        for station in self.stations:
            load = station.step(tou_multiplier=self.tou_multiplier, price_noise=self.price_noise, step_duration_h=self.step_duration_h)
            grid_loads[station.power_node_id] += load
            total_realized_power += load

        self.power_grid.run_power_flow(grid_loads)
        voltage_excursion = sum(
            max(0.0, self.power_grid.v_min - float(v_pu))
            for v_pu in self.power_grid.bus_voltages.values()
        )
        min_voltage_bus, min_voltage_pu = min(
            self.power_grid.bus_voltages.items(),
            key=lambda item: item[1],
        )

        user_cost = sum(m["generalized_cost"] for m in decision_metrics.values())
        queue_cost = sum(m["queue_time_h"] for m in decision_metrics.values())
        grid_cost = sum(st.last_billing_price * st.last_total_load * self.step_duration_h for st in self.stations) + 20.0 * self.power_grid.total_loss
        fluct_cost = (total_realized_power - self.prev_total_load) ** 2
        voltage_penalty = 10.0 * len(self.power_grid.voltage_violations)

        reward = -(
            0.08 * user_cost +
            10.0 * queue_cost +
            0.03 * grid_cost +
            0.01 * fluct_cost +
            voltage_penalty
        )
        self.prev_total_load = total_realized_power

        info = {
            "grid_loads": grid_loads,
            "realized_power": total_realized_power,
            "power_allocation": {s.id: s.last_power_allocation for s in self.stations},
            "bus_voltages": dict(self.power_grid.bus_voltages),
            "line_losses": self.power_grid.total_loss,
            "voltage_violations": len(self.power_grid.voltage_violations),
            "voltage_excursion": voltage_excursion,
            "min_voltage_bus": min_voltage_bus,
            "min_voltage_pu": min_voltage_pu,
            "runpp_call_count": getattr(self.power_grid, "runpp_call_count", 0),
            "runpp_total_time_s": getattr(self.power_grid, "runpp_total_time_s", 0.0),
            "runpp_last_time_s": getattr(self.power_grid, "last_runpp_time_s", 0.0),
            "tou_multiplier": self.tou_multiplier,
            "price_noise": self.price_noise,
            "step_duration_h": self.step_duration_h,
            "abandoned_this_step": abandoned_this_step,
            "decision_costs": decision_metrics,
            "active_edge_flows": dict(self.edge_active_counts),
            "active_edge_vehicle_count": int(sum(self.edge_active_counts.values())),
            "edge_entries_this_step": dict(self.edge_step_counts),
            "peak_edge_flows_this_step": dict(self.edge_peak_counts),
            "avg_decision_trip_time_h": (
                sum(m["trip_time_h"] for m in decision_metrics.values()) / max(1, len(decision_metrics))
            ),
            "objective_terms": {
                "user_cost": user_cost,
                "queue_cost": queue_cost,
                "grid_cost": grid_cost,
                "fluct_cost": fluct_cost,
                "voltage_penalty": voltage_penalty,
            },
        }

        return self.get_graph_state(), reward, False, info

    def render(self):
        hour = self.time_step % self.steps_per_day
        tou_tag = {0.5: "谷", 1.0: "平", 1.5: "峰"}.get(self.tou_multiplier, "?")
        print(f"\n=== Step {self.time_step:03d} | {hour:02d}:00 [{tou_tag}时 x{self.tou_multiplier}] ===")

        for bus, v in self.power_grid.bus_voltages.items():
            flag = " [WARN: 越限!]" if v < self.power_grid.v_min else ""
            print(f"  {bus}: V={v:.4f} pu  Loss={self.power_grid.line_losses[bus]:.3f}kW{flag}")

        for s in self.stations:
            connected_ids = [ev.id for ev in s.connected_evs]
            queue_ids = [ev.id for ev in s.queue]
            alloc_str = ", ".join(
                f"EV{eid}:{p:.1f}kW" for eid, p in s.last_power_allocation.items()
            )
            print(
                f"  Station {s.id}: Price={s.current_price:.1f} | "
                f"Load={s.last_total_load:.1f}/{s.max_grid_power:.0f}kW | "
                f"Connected={connected_ids} | Queue={queue_ids} | "
                f"Power=[{alloc_str}]"
            )
