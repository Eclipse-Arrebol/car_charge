import numpy as np
import random


class ChargingStation:
    def __init__(self, station_id, traffic_node_id, power_node_id,
                 num_chargers=4, max_charger_power=20.0, max_grid_power=50.0,
                 respawn_after_full_charge=True):
        self.id = station_id
        self.traffic_node_id = traffic_node_id
        self.power_node_id = power_node_id
        self.num_chargers = num_chargers
        self.max_charger_power = max_charger_power
        self.max_grid_power = max_grid_power
        self.respawn_after_full_charge = respawn_after_full_charge
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
        """
        估算平均单次充电时长（小时）。
        CC 阶段按额定功率，CV 阶段功率取额定值的 50% 作为保守估计。
        """
        samples = self.queue + self.connected_evs
        if not samples:
            return 0.7

        hours = []
        for ev in samples:
            hours.append(self.estimate_charge_time_hours(ev))
        return max(0.1, float(np.mean(hours)))

    def estimate_charge_time_hours(self, ev):
        """
        估算单辆 EV 完成充电所需时长（小时），基于 CC-CV 曲线。
        CC 段用额定功率，CV 段用额定功率的 50% 均值近似。
        """
        soc = ev.soc
        target = ev.target_soc
        eff = max(1e-6, ev.charge_efficiency)
        cap = ev.battery_capacity_kwh

        if soc >= target:
            return 0.0

        if soc < self._CC_CV_THRESHOLD:
            cc_energy = (self._CC_CV_THRESHOLD - soc) / 100.0 * cap
            cc_hours = cc_energy / max(1e-6, eff * self.max_charger_power)
        else:
            cc_hours = 0.0

        cv_soc_start = max(soc, self._CC_CV_THRESHOLD)
        if cv_soc_start < target:
            cv_energy = (target - cv_soc_start) / 100.0 * cap
            cv_avg_power = max(self._CV_FLOOR_KW, self.max_charger_power * 0.5)
            cv_hours = cv_energy / max(1e-6, eff * cv_avg_power)
        else:
            cv_hours = 0.0

        return cc_hours + cv_hours

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

    # CC-CV 充电曲线阈值（与 target_soc 对应）
    _CC_CV_THRESHOLD = 80.0   # 低于此 SOC 为恒流（CC）阶段
    _CV_FLOOR_KW     = 1.0    # CV 阶段最低功率，确保充电能抵达目标 SOC

    def _cc_cv_power(self, ev):
        """
        CC-CV 两段式充电功率（模拟电池物理特性）：
          - CC 阶段 (SOC < 80%)：以额定功率全速充
          - CV 阶段 (80% ≤ SOC < target)：功率线性 taper 至 CV_FLOOR_KW
        返回值为单桩请求功率（kW），尚未考虑站级总功率约束。
        """
        soc = ev.soc
        if soc < self._CC_CV_THRESHOLD:
            return self.max_charger_power
        cv_range = max(1.0, ev.target_soc - self._CC_CV_THRESHOLD)
        ratio = max(0.0, (ev.target_soc - soc) / cv_range)
        return max(self._CV_FLOOR_KW, self.max_charger_power * ratio)

    def optimize_power(self):
        """
        站级功率分配：
          1. 每辆 EV 按 CC-CV 曲线计算各自请求功率
          2. 若总请求超出电网上限，等比例降额（所有桩同比缩水）
        """
        n = len(self.connected_evs)
        if n == 0:
            self.last_power_allocation = {}
            self.last_total_load = 0.0
            return {}

        requested = np.array([self._cc_cv_power(ev) for ev in self.connected_evs], dtype=float)
        total_requested = requested.sum()
        scale = min(1.0, self.max_grid_power / max(1e-6, total_requested))
        alloc = requested * scale

        allocation = {ev.id: float(alloc[i]) for i, ev in enumerate(self.connected_evs)}
        self.last_power_allocation = allocation
        self.last_total_load = float(alloc.sum())
        return allocation

    def step(self, tou_multiplier=1.0, price_noise=0.0, step_duration_h=1.0):
        realized_power = 0.0

        while self.queue and len(self.connected_evs) < self.num_chargers:
            ev = self.queue.pop(0)
            ev.status = "CHARGING"
            ev.charge_started_count += 1
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
                if self.respawn_after_full_charge:
                    ev.soc = random.uniform(20.0, 50.0)
                finished.append(ev)

        for ev in finished:
            self.connected_evs.remove(ev)

        self.update_price(tou_multiplier, price_noise=price_noise)
        return realized_power
