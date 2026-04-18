import numpy as np
import random


class ChargingStation:
    def __init__(self, station_id, traffic_node_id, power_node_id,
                 num_chargers=4, max_charger_power=20.0, max_grid_power=50.0):
        self.id = station_id
        self.traffic_node_id = traffic_node_id
        self.power_node_id = power_node_id
        self.num_chargers = num_chargers
        self.max_charger_power = max_charger_power
        self.max_grid_power = max_grid_power
        self.max_queue_len = 20
        self.max_wait_time_h = 4.0

        self.queue = []
        self.connected_evs = []

        self.base_price = 1.0
        self.current_price = 1.0
        self.service_price_markup = 0.15
        self.price_noise = 0.0

        self.last_power_allocation = {}
        self.last_total_load = 0.0
        self.last_billing_price = self.current_price
        self.predicted_arrivals = 0.0
        self.arrival_ema_alpha = 0.3

    def update_price(self, tou_multiplier=1.0, price_noise=0.0):
        congestion = len(self.queue) + len(self.connected_evs)
        self.price_noise = float(price_noise)
        energy_price = self.base_price * tou_multiplier * (1.0 + self.price_noise)
        congestion_markup = 0.08 * congestion
        self.current_price = max(
            0.1,
            energy_price + self.service_price_markup + congestion_markup,
        )
        return self.current_price

    def estimate_queue_wait_hours(self, incoming_count=0):
        total_waiting = len(self.queue) + max(0, int(incoming_count))
        if total_waiting <= 0:
            return 0.0

        residual_times = []
        for ev in self.connected_evs:
            power = max(
                1e-6,
                self.last_power_allocation.get(ev.id, self.max_charger_power * 0.7)
            )
            target_energy = max(
                0.0,
                (ev.target_soc - ev.soc) / 100.0 * ev.battery_capacity_kwh
            )
            residual_times.append(target_energy / max(1e-6, ev.charge_efficiency * power))

        residual_head = min(residual_times) if residual_times else 0.0
        avg_charge_h = self.estimate_average_charge_hours()
        queue_rounds = total_waiting / max(1, self.num_chargers)
        return residual_head + queue_rounds * avg_charge_h

    def estimate_average_charge_hours(self):
        samples = self.queue + self.connected_evs
        if not samples:
            return 0.7

        hours = []
        for ev in samples:
            remaining_energy = max(
                0.0,
                (ev.target_soc - ev.soc) / 100.0 * ev.battery_capacity_kwh
            )
            hours.append(
                remaining_energy / max(1e-6, ev.charge_efficiency * self.max_charger_power)
            )
        return max(0.1, float(np.mean(hours)))

    def estimate_charge_time_hours(self, ev):
        remaining_energy = max(
            0.0,
            (ev.target_soc - ev.soc) / 100.0 * ev.battery_capacity_kwh
        )
        return remaining_energy / max(1e-6, ev.charge_efficiency * self.max_charger_power)

    def estimate_charge_cost(self, ev):
        remaining_energy = max(
            0.0,
            (ev.target_soc - ev.soc) / 100.0 * ev.battery_capacity_kwh
        )
        billed_energy = remaining_energy / max(1e-6, ev.charge_efficiency)
        return self.current_price * billed_energy

    def update_arrival_prediction(self, arrivals_this_step):
        arrivals = max(0.0, float(arrivals_this_step))
        self.predicted_arrivals = (
            self.arrival_ema_alpha * arrivals
            + (1.0 - self.arrival_ema_alpha) * self.predicted_arrivals
        )
        return self.predicted_arrivals

    def optimize_power(self):
        n = len(self.connected_evs)
        if n == 0:
            self.last_power_allocation = {}
            self.last_total_load = 0.0
            return {}

        upper_bounds = np.array([
            min(self.max_charger_power,
                max(0.0, (ev.target_soc - ev.soc) / 100.0 * ev.battery_capacity_kwh) /
                max(1e-6, ev.charge_efficiency))
            for ev in self.connected_evs
        ], dtype=float)

        alloc = np.zeros(n, dtype=float)
        remaining = float(self.max_grid_power)
        active = np.ones(n, dtype=bool)

        while active.any():
            n_active = int(active.sum())
            fair = remaining / n_active
            capped = active & (upper_bounds <= fair)
            if not capped.any():
                alloc[active] = fair
                break
            alloc[capped] = upper_bounds[capped]
            remaining -= float(upper_bounds[capped].sum())
            active &= ~capped
            if remaining <= 1e-9:
                break

        allocation = {ev.id: float(alloc[i]) for i, ev in enumerate(self.connected_evs)}
        self.last_power_allocation = allocation
        self.last_total_load = sum(allocation.values())
        return allocation

    def step(self, tou_multiplier=1.0, price_noise=0.0, step_duration_h=1.0):
        realized_power = 0.0

        while self.queue and len(self.connected_evs) < self.num_chargers:
            ev = self.queue.pop(0)
            ev.status = "CHARGING"
            self.connected_evs.append(ev)

        allocation = self.optimize_power()

        self.last_billing_price = self.current_price
        finished = []
        for ev in self.connected_evs:
            power = allocation.get(ev.id, 0.0)
            energy_kwh = power * step_duration_h
            soc_increment = (energy_kwh * ev.charge_efficiency / ev.battery_capacity_kwh) * 100.0
            ev.soc = min(100.0, ev.soc + soc_increment)
            realized_power += power
            ev.total_fee_paid += power * self.current_price * step_duration_h
            ev.total_energy_charged += energy_kwh

            if ev.soc >= 95.0:
                ev.status = "IDLE"
                ev.charge_sessions += 1
                ev.low_soc_triggered = False
                ev.charge_decision_pending = False
                ev.remaining_replans = 1
                ev.soc = random.uniform(20.0, 50.0)
                finished.append(ev)

        for ev in finished:
            self.connected_evs.remove(ev)

        self.update_price(tou_multiplier, price_noise=price_noise)
        return realized_power
