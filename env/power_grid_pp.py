import copy
import time

import pandapower as pp


IEEE33_STATION_BUSES = {
    0: 6,
    1: 9,
    2: 12,
    3: 18,
    4: 22,
    5: 25,
    6: 17,
    7: 33,
}


# Baran-Wu IEEE 33-bus radial feeder line data: from_bus, to_bus, R(ohm), X(ohm).
# Bus numbers follow the conventional 1-based IEEE 33-bus notation.
IEEE33_LINE_DATA = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]


# Standard IEEE 33-bus load data in kW/kVAr. Slack bus 1 has no load.
IEEE33_LOAD_DATA = {
    2: (100, 60),
    3: (90, 40),
    4: (120, 80),
    5: (60, 30),
    6: (60, 20),
    7: (200, 100),
    8: (200, 100),
    9: (60, 20),
    10: (60, 20),
    11: (45, 30),
    12: (60, 35),
    13: (60, 35),
    14: (120, 80),
    15: (60, 10),
    16: (60, 20),
    17: (60, 20),
    18: (90, 40),
    19: (90, 40),
    20: (90, 40),
    21: (90, 40),
    22: (90, 40),
    23: (90, 50),
    24: (420, 200),
    25: (420, 200),
    26: (60, 25),
    27: (60, 25),
    28: (60, 20),
    29: (120, 70),
    30: (200, 600),
    31: (150, 70),
    32: (210, 100),
    33: (60, 40),
}


class PPPowerGrid33:
    """Pandapower IEEE 33-bus distribution grid with the legacy PowerGrid interface."""

    _THEVENIN_CACHE = None

    def __init__(self, station_bus_map=None, compute_thevenin=True):
        self.v_nominal_kv = 12.66
        self.v_min = 0.95
        self.v_max = 1.05
        self.station_bus_map = dict(station_bus_map or IEEE33_STATION_BUSES)
        self.station_power_nodes = {
            station_id: f"Bus_{bus}"
            for station_id, bus in self.station_bus_map.items()
        }
        self.power_node_to_bus = {
            power_node: bus
            for station_id, power_node in self.station_power_nodes.items()
            for bus in [self.station_bus_map[station_id]]
        }
        self.base_net = self._build_ieee33_net()
        self.net = copy.deepcopy(self.base_net)
        self.bus_lookup = {
            int(str(row["name"]).split("_")[1]): idx
            for idx, row in self.net.bus.iterrows()
        }
        self.line_lookup = {}
        for idx, row in self.net.line.iterrows():
            from_bus = int(self.net.bus.at[row.from_bus, "name"].split("_")[1])
            to_bus = int(self.net.bus.at[row.to_bus, "name"].split("_")[1])
            self.line_lookup[f"{from_bus}-{to_bus}"] = idx

        self.bus_voltages = {f"Bus_{i}": 1.0 for i in range(1, 34)}
        self.line_losses = {}
        self.total_loss = 0.0
        self.voltage_violations = []
        self.runpp_call_count = 0
        self.runpp_total_time_s = 0.0
        self.last_runpp_time_s = 0.0
        self.thevenin_r_ohm = {}
        if compute_thevenin:
            if PPPowerGrid33._THEVENIN_CACHE is None:
                PPPowerGrid33._THEVENIN_CACHE = self._compute_thevenin_resistances()
            self.thevenin_r_ohm = dict(PPPowerGrid33._THEVENIN_CACHE)

    @staticmethod
    def _build_ieee33_net():
        net = pp.create_empty_network(sn_mva=10.0)
        bus_indices = {}
        for bus in range(1, 34):
            bus_indices[bus] = pp.create_bus(
                net,
                vn_kv=12.66,
                name=f"Bus_{bus}",
                type="b",
            )

        pp.create_ext_grid(
            net,
            bus=bus_indices[1],
            vm_pu=1.0,
            va_degree=0.0,
            name="Slack_Bus_1",
        )

        for bus, (p_kw, q_kvar) in IEEE33_LOAD_DATA.items():
            pp.create_load(
                net,
                bus=bus_indices[bus],
                p_mw=p_kw / 1000.0,
                q_mvar=q_kvar / 1000.0,
                name=f"Load_{bus}",
            )

        for from_bus, to_bus, r_ohm, x_ohm in IEEE33_LINE_DATA:
            pp.create_line_from_parameters(
                net,
                from_bus=bus_indices[from_bus],
                to_bus=bus_indices[to_bus],
                length_km=1.0,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=0.0,
                max_i_ka=0.4,
                name=f"Line_{from_bus}_{to_bus}",
            )
        return net

    def _net_with_loads(self, loads):
        net = copy.deepcopy(self.base_net)
        for power_node, load_kw in loads.items():
            bus_num = self._resolve_bus_number(power_node)
            if bus_num is None:
                continue
            bus_idx = self.bus_lookup[bus_num]
            pp.create_load(
                net,
                bus=bus_idx,
                p_mw=float(load_kw) / 1000.0,
                q_mvar=0.0,
                name=f"EV_Load_{power_node}",
            )
        return net

    def _resolve_bus_number(self, power_node):
        if isinstance(power_node, int):
            return power_node
        if power_node in self.power_node_to_bus:
            return self.power_node_to_bus[power_node]
        if isinstance(power_node, str) and power_node.startswith("Bus_"):
            return int(power_node.split("_", 1)[1])
        return None

    def get_station_power_node(self, station_id):
        return self.station_power_nodes[station_id]

    def _runpp(self, net):
        t0 = time.perf_counter()
        pp.runpp(net, algorithm="bfsw", calculate_voltage_angles=False)
        elapsed = time.perf_counter() - t0
        self.runpp_call_count += 1
        self.runpp_total_time_s += elapsed
        self.last_runpp_time_s = elapsed

    def run_power_flow(self, loads):
        """Run pandapower AC load flow. Loads are kW keyed by Bus_N or station power node."""
        self.net = self._net_with_loads(loads)
        self._runpp(self.net)

        self.bus_voltages = {}
        self.voltage_violations = []
        for bus_idx, row in self.net.res_bus.iterrows():
            bus_num = int(self.net.bus.at[bus_idx, "name"].split("_")[1])
            key = f"Bus_{bus_num}"
            vm_pu = float(row.vm_pu)
            self.bus_voltages[key] = round(vm_pu, 6)
            if vm_pu < self.v_min or vm_pu > self.v_max:
                self.voltage_violations.append((key, vm_pu))

        self.line_losses = {}
        for line_idx, row in self.net.res_line.iterrows():
            line_name = self.net.line.at[line_idx, "name"].replace("Line_", "").replace("_", "-")
            self.line_losses[line_name] = round(float(row.pl_mw) * 1000.0, 6)
        self.total_loss = float(self.net.res_line.pl_mw.sum() * 1000.0)
        return self.bus_voltages

    def optimize_power(self, requested_loads):
        """Compatibility placeholder: station-level allocation still lives in ChargingStation."""
        return dict(requested_loads)

    def _compute_thevenin_resistances(self):
        result = {}
        base_net = copy.deepcopy(self.base_net)
        self._runpp(base_net)
        slack_v = self.v_nominal_kv * 1000.0
        perturb_mw = 1.0

        for bus_num, bus_idx in self.bus_lookup.items():
            if bus_num == 1:
                result[bus_num] = 0.0
                continue
            test_net = copy.deepcopy(self.base_net)
            pp.create_load(
                test_net,
                bus=bus_idx,
                p_mw=perturb_mw,
                q_mvar=0.0,
                name=f"Thevenin_Test_{bus_num}",
            )
            self._runpp(test_net)
            dv_pu = (
                float(base_net.res_bus.at[bus_idx, "vm_pu"])
                - float(test_net.res_bus.at[bus_idx, "vm_pu"])
            )
            # For a small active-power perturbation, R_th ~= dV_phase^2 / dP_3phase.
            r_ohm = max(0.0, dv_pu * (slack_v ** 2) / (perturb_mw * 1e6))
            result[bus_num] = r_ohm
        return result

    def get_bus_thevenin_resistance(self, bus_idx):
        bus_num = self._resolve_bus_number(bus_idx)
        if bus_num is None:
            raise KeyError(f"Unknown bus identifier: {bus_idx}")
        return self.thevenin_r_ohm[bus_num]

    def get_last_bus_voltage(self, bus_idx):
        bus_num = self._resolve_bus_number(bus_idx)
        if bus_num is None:
            raise KeyError(f"Unknown bus identifier: {bus_idx}")
        return self.bus_voltages.get(f"Bus_{bus_num}", 1.0)


__all__ = [
    "IEEE33_STATION_BUSES",
    "PPPowerGrid33",
]
