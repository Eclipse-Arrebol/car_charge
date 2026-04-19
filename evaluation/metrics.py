"""
评价指标计算模块
================
用户侧指标 (User-side):
  - 行驶时间成本        (CNY/veh)
  - 充电时间成本        (CNY/veh)
  - 等待时间成本        (CNY/veh)
  - 充电费用            (CNY/veh)
  - 总充电成本          (CNY/veh)
  - 平均等待时间        (min)
  - 排队数量            (辆)

电网侧指标 (Grid-side):
  - 配电网购电成本      (CNY)
  - 累计电压偏移        (p.u.)
"""

import numpy as np


# ==========================================
# 成本参数 (可按实际情况调整)
# ==========================================
class CostParams:
    """集中管理所有单价参数，方便后续灵活调整"""
    VOTT = 30.0            # 时间价值 (CNY/h)  — Value Of Travel Time
    STEP_DURATION_H = 1/6  # 每步代表的时间 (h)，当前 1步=10分钟
    GRID_BUY_PRICE = 0.6   # 电网购电基准价格 (CNY/kWh)


# ==========================================
# 1. 用户侧指标
# ==========================================
class UserMetrics:
    """
    从 EV 对象列表中提取用户侧评价指标。

    所有"成本"指标单位: CNY/veh (每辆车的平均成本)
    时间价值法: 成本 = 步数 × 每步时长(h) × VOTT(CNY/h)
    """

    def __init__(self, params=None):
        self.p = params or CostParams()

    def compute(self, evs, stations=None, avg_evs_in_line=None, end_evs_in_line=None):
        """
        Args:
            evs:      List[EV]  — 环境中的所有 EV 对象
            stations: List[ChargingStation]  — 可选，用于计算当前排队数量
            avg_evs_in_line: float | None   — 过程平均排队数（由 Evaluator 累计）
            end_evs_in_line: float | None   — 结束时瞬时排队数（由 Evaluator 提供）
        Returns:
            dict  — 各项指标
        """
        n = len(evs)
        if n == 0:
            return self._empty()

        # --- 汇总每辆车的原始步数 ---
        travel_steps = np.array([ev.travel_steps for ev in evs], dtype=float)
        wait_steps   = np.array([ev.wait_steps   for ev in evs], dtype=float)
        charge_steps = np.array([ev.charge_steps  for ev in evs], dtype=float)
        travel_hours = np.array([getattr(ev, "travel_time_h", ev.travel_steps * self.p.STEP_DURATION_H) for ev in evs], dtype=float)
        wait_hours   = wait_steps * self.p.STEP_DURATION_H
        charge_hours = np.array([getattr(ev, "charge_time_h", ev.charge_steps * self.p.STEP_DURATION_H) for ev in evs], dtype=float)
        fees_paid    = np.array([ev.total_fee_paid for ev in evs], dtype=float)
        sessions     = np.array([ev.charge_sessions for ev in evs], dtype=float)
        started      = np.array([getattr(ev, "charge_started_count", 0) for ev in evs], dtype=float)
        abandoned    = np.array([getattr(ev, "abandoned_charge_count", 0) for ev in evs], dtype=float)

        h = self.p.STEP_DURATION_H
        vott = self.p.VOTT

        # --- 各项时间成本 (CNY/veh) ---
        travel_cost  = travel_hours * vott                # 行驶时间成本
        charge_cost  = charge_hours * vott                # 充电时间成本
        wait_cost    = wait_hours   * vott                # 等待时间成本

        # --- 充电费用 (CNY/veh) ---
        charging_fee = fees_paid                          # 已在 station.step 中累计

        # --- 总充电成本 ---
        total_cost = travel_cost + charge_cost + wait_cost + charging_fee

        # --- 平均等待时间 (min) ---
        # 仅统计有过充电行为的车
        active_mask = wait_steps > 0
        if active_mask.any():
            avg_wait_min = float(np.mean(wait_hours[active_mask]) * 60)
        else:
            avg_wait_min = 0.0

        # --- 排队数量 ---
        # evs_in_line 默认改为“过程平均排队数”；
        # 若未提供累积值，则回退为当前时刻瞬时值（兼容旧调用）。
        current_evs_in_line = 0
        if stations:
            current_evs_in_line = sum(len(s.queue) for s in stations)

        if avg_evs_in_line is None:
            evs_in_line = float(current_evs_in_line)
        else:
            evs_in_line = float(avg_evs_in_line)

        if end_evs_in_line is None:
            end_evs_in_line = float(current_evs_in_line)

        incomplete_evs = sum(1 for ev in evs if getattr(ev, "status", "IDLE") != "IDLE")

        return {
            # 每辆车的平均值
            "travel_time_cost_per_veh":   float(np.mean(travel_cost)),
            "charge_time_cost_per_veh":   float(np.mean(charge_cost)),
            "wait_time_cost_per_veh":     float(np.mean(wait_cost)),
            "charging_fee_per_veh":       float(np.mean(charging_fee)),
            "total_charging_cost_per_veh": float(np.mean(total_cost)),
            # 物理指标
            "avg_wait_time_min":          avg_wait_min,
            "evs_in_line":                evs_in_line,
            "end_evs_in_line":            float(end_evs_in_line),
            # 汇总
            "started_charge_sessions":    int(started.sum()),
            "completed_charge_sessions":  int(sessions.sum()),
            "total_energy_charged_kwh":   float(sum(ev.total_energy_charged for ev in evs)),
            "abandoned_evs":              int(abandoned.sum()),
            "incomplete_evs":             int(incomplete_evs),
        }

    @staticmethod
    def _empty():
        return {k: 0.0 for k in [
            "travel_time_cost_per_veh", "charge_time_cost_per_veh",
            "wait_time_cost_per_veh", "charging_fee_per_veh",
            "total_charging_cost_per_veh", "avg_wait_time_min",
            "evs_in_line", "end_evs_in_line",
            "started_charge_sessions", "completed_charge_sessions",
            "total_energy_charged_kwh", "abandoned_evs", "incomplete_evs",
        ]}


# ==========================================
# 2. 电网侧指标
# ==========================================
class GridMetrics:
    """
    从 PowerGrid 和 step info 中提取电网侧评价指标。
    """

    def __init__(self, params=None):
        self.p = params or CostParams()
        # 累计值 (需要跨多步累加)
        self._accum_voltage_deviation = 0.0  # Σ |V - 1.0|
        self._accum_grid_cost = 0.0          # Σ (购电量 × 分时电价)
        self._accum_loss = 0.0               # Σ 线路损耗
        self._step_count = 0

    def reset(self):
        self._accum_voltage_deviation = 0.0
        self._accum_grid_cost = 0.0
        self._accum_loss = 0.0
        self._step_count = 0

    def update(self, info, power_grid):
        """
        每步调用一次，累计电网侧数据。

        Args:
            info:       env.step() 返回的 info dict
            power_grid: env.power_grid 对象
        """
        self._step_count += 1

        # 累计电压偏移: Σ |V_bus - 1.0|  (所有母线)
        for bus, v_pu in power_grid.bus_voltages.items():
            self._accum_voltage_deviation += abs(v_pu - 1.0)

        # 配电网购电成本: 总功率(kWh) × 分时电价系数 × 基准电价
        realized = info.get("realized_power", 0.0)
        tou = info.get("tou_multiplier", 1.0)
        self._accum_grid_cost += realized * self.p.STEP_DURATION_H * tou * self.p.GRID_BUY_PRICE

        # 累计线路损耗
        self._accum_loss += info.get("line_losses", power_grid.total_loss)

    def compute(self):
        """
        Returns:
            dict — 电网侧评价指标
        """
        return {
            "distribution_network_cost_cny": round(self._accum_grid_cost, 2),
            "accumulated_voltage_excursion_pu": round(self._accum_voltage_deviation, 4),
            "total_line_losses_kwh": round(self._accum_loss, 4),
            "evaluation_steps": self._step_count,
        }


# ==========================================
# 3. 综合评估器
# ==========================================
class Evaluator:
    """
    综合评估器：整合用户侧与电网侧指标。
    在评估 episode 的 step 循环中 call update()，结束后 call report()。
    """

    def __init__(self, params=None):
        self.params = params or CostParams()
        self.user_metrics = UserMetrics(self.params)
        self.grid_metrics = GridMetrics(self.params)
        self._queue_sum = 0.0
        self._queue_steps = 0
        self._last_queue = 0.0

    def reset(self):
        self.grid_metrics.reset()
        self._queue_sum = 0.0
        self._queue_steps = 0
        self._last_queue = 0.0

    def update(self, info, power_grid, stations=None):
        """每步调用"""
        self.grid_metrics.update(info, power_grid)
        if stations is not None:
            current_queue = float(sum(len(s.queue) for s in stations))
            self._queue_sum += current_queue
            self._queue_steps += 1
            self._last_queue = current_queue

    def report(self, evs, stations=None, verbose=True):
        """
        生成完整评估报告。

        Args:
            evs:      环境中的 EV 列表
            stations: 充电站列表
            verbose:  是否打印
        Returns:
            dict — 所有指标合并
        """
        avg_queue = self._queue_sum / max(1, self._queue_steps)
        user = self.user_metrics.compute(
            evs,
            stations,
            avg_evs_in_line=avg_queue,
            end_evs_in_line=self._last_queue,
        )
        grid = self.grid_metrics.compute()
        result = {**user, **grid}

        if verbose:
            self._print_report(result)

        return result

    @staticmethod
    def _print_report(r):
        print("\n" + "=" * 62)
        print("              模 型 评 估 报 告")
        print("=" * 62)

        print("\n【用户侧指标 User-side】")
        print(f"  行驶时间成本  (Travel Time Cost) :  {r['travel_time_cost_per_veh']:.2f}  CNY/veh")
        print(f"  充电时间成本  (Charge Time Cost) :  {r['charge_time_cost_per_veh']:.2f}  CNY/veh")
        print(f"  等待时间成本  (Wait Time Cost)   :  {r['wait_time_cost_per_veh']:.2f}  CNY/veh")
        print(f"  充电费用      (Charging Fee)     :  {r['charging_fee_per_veh']:.2f}  CNY/veh")
        print(f"  总充电成本    (Total Cost)       :  {r['total_charging_cost_per_veh']:.2f}  CNY/veh")
        print(f"  平均等待时间  (Avg Wait Time)    :  {r['avg_wait_time_min']:.1f}   min")
        print(f"  平均排队数量  (Avg EVs in Line)  :  {r['evs_in_line']:.2f}   辆")
        print(f"  终点排队数量  (End EVs in Line)  :  {r['end_evs_in_line']:.2f}   辆")

        print("\n【电网侧指标 Grid-side】")
        print(f"  配电网成本    (DN Cost)          :  {r['distribution_network_cost_cny']:.2f}  CNY")
        print(f"  累计电压偏移  (Voltage Excursion):  {r['accumulated_voltage_excursion_pu']:.4f}  p.u.")
        print(f"  累计线路损耗  (Line Losses)      :  {r['total_line_losses_kwh']:.4f}  kWh")

        print("\n【汇总】")
        print(f"  评估步数: {r['evaluation_steps']}  |  "
              f"开始充电次数: {r['started_charge_sessions']}  |  "
              f"完成充电次数: {r['completed_charge_sessions']}  |  "
              f"总充电量: {r['total_energy_charged_kwh']:.1f} kWh")
        print(f"  放弃充电车辆(累计事件): {r['abandoned_evs']}  |  "
              f"未完成充电车辆: {r['incomplete_evs']}")
        print("=" * 62)
