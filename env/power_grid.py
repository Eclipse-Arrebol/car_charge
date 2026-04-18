import numpy as np


def get_tou_multiplier(time_step, steps_per_day=24):
    hour = time_step % steps_per_day
    if hour >= 23 or hour < 7:
        return 0.5
    elif 10 <= hour < 15 or 18 <= hour < 23:
        return 1.5
    else:
        return 1.0


class PowerGrid:
    def __init__(self):
        self.v_nominal_kv = 0.4
        self.lines = {
            "Grid_A": {"r_ohm": 0.08},
            "Grid_B": {"r_ohm": 0.12},
        }
        self.v_min = 0.95
        self.v_max = 1.05
        self.bus_voltages = {bus: 1.0 for bus in self.lines}
        self.line_losses = {bus: 0.0 for bus in self.lines}
        self.total_loss = 0.0
        self.voltage_violations = []

    def run_power_flow(self, loads):
        v2 = self.v_nominal_kv ** 2
        self.voltage_violations = []
        self.total_loss = 0.0

        for bus, line in self.lines.items():
            p = loads.get(bus, 0.0)
            r = line["r_ohm"]
            delta_v = p * r / (v2 * 1000)
            v_pu = 1.0 - delta_v
            loss = (p ** 2) * r / (v2 * 1e6)

            self.bus_voltages[bus] = round(v_pu, 4)
            self.line_losses[bus] = round(loss, 4)
            self.total_loss += loss

            if v_pu < self.v_min or v_pu > self.v_max:
                self.voltage_violations.append((bus, v_pu))

        return self.bus_voltages


class _FlexiblePowerGrid(PowerGrid):
    _BASE_R = 0.08
    _R_STEP = 0.04

    def __init__(self, num_stations: int = 2):
        super().__init__()
        self.lines = {
            f"Grid_{chr(65 + i)}": {"r_ohm": self._BASE_R + i * self._R_STEP}
            for i in range(num_stations)
        }
        self.bus_voltages = {bus: 1.0 for bus in self.lines}
        self.line_losses = {bus: 0.0 for bus in self.lines}
