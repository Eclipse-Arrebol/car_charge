import random


class EV:
    def __init__(self, ev_id, start_node):
        self.id = ev_id
        self.curr_node = start_node
        self.target_station_idx = None
        self.soc = random.uniform(20.0, 50.0)
        self.target_soc = 0.95
        self.battery_capacity_kwh = 60.0
        self.charge_efficiency = 0.92
        self.time_value = 35.0
        self.drive_kwh_per_km = 0.18
        self.status = "IDLE"
        self.path = []
        self.last_traversed_nodes = []
        self.current_edge_from = None
        self.current_edge_target = None
        self.remaining_edge_time_h = 0.0
        self.current_edge_speed_kph = 0.0
        self.low_soc_triggered = False
        self.charge_decision_pending = False
        self.remaining_replans = 1
        self.assigned_station = None
        self._decision_state = None
        self._decision_snap = None

        self.travel_steps = 0
        self.wait_steps = 0
        self.charge_steps = 0
        self.travel_time_h = 0.0
        self.wait_time_h = 0.0
        self.charge_time_h = 0.0
        self.total_fee_paid = 0.0
        self.total_energy_charged = 0.0
        self.charge_sessions = 0
        self.abandoned_charge_count = 0

    def move(self, env, step_hours=1.0):
        self.last_traversed_nodes = []
        if self.status != "MOVING_TO_CHARGE":
            return

        remaining = max(0.0, float(step_hours))
        moved_hours = 0.0

        while remaining > 1e-9:
            if self.remaining_edge_time_h <= 1e-9:
                if not self.path:
                    break
                next_node = self.path[0]
                _, speed_kph, travel_time_h = env.enter_edge(self.curr_node, next_node)
                self.current_edge_from = self.curr_node
                self.current_edge_target = next_node
                self.current_edge_speed_kph = speed_kph
                self.remaining_edge_time_h = max(1e-6, travel_time_h)

            consume = min(remaining, self.remaining_edge_time_h)
            remaining -= consume
            moved_hours += consume
            self.remaining_edge_time_h -= consume

            if self.remaining_edge_time_h <= 1e-9:
                env.leave_edge(self.current_edge_from, self.current_edge_target)
                self.curr_node = self.current_edge_target
                self.last_traversed_nodes.append(self.curr_node)
                if self.path and self.path[0] == self.current_edge_target:
                    self.path.pop(0)
                self.current_edge_from = None
                self.current_edge_target = None
                self.current_edge_speed_kph = 0.0

        self.soc -= 2.0 * moved_hours / max(1e-6, float(step_hours))
        self.travel_time_h += moved_hours
        if self.soc < 0:
            self.soc = 0
