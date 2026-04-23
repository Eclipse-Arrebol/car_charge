import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.power_grid_pp import IEEE33_STATION_BUSES, PPPowerGrid33


def _print_station_voltages(title, grid, loads):
    voltages = grid.run_power_flow(loads)

    station_voltages = {}
    print(title)
    for station_id, bus in IEEE33_STATION_BUSES.items():
        key = f"Bus_{bus}"
        station_voltages[station_id] = voltages[key]
        print(f"  station={station_id} bus={bus:>2} voltage_pu={voltages[key]:.6f}")

    values = list(station_voltages.values())
    spread = max(values) - min(values)
    lowest_station = min(station_voltages, key=station_voltages.get)
    print(f"\nVoltage spread across station buses: {spread:.6f}")
    print(
        f"Lowest station voltage: station={lowest_station}, "
        f"bus={IEEE33_STATION_BUSES[lowest_station]}, "
        f"voltage_pu={station_voltages[lowest_station]:.6f}"
    )
    return station_voltages, spread, lowest_station


def main():
    grid = PPPowerGrid33()

    print("IEEE 33-bus pandapower smoke test")
    base_station_voltages, base_spread, base_lowest_station = _print_station_voltages(
        "\nStation bus voltages under IEEE 33 base load + 0 EV station load:",
        grid,
        {},
    )

    loads = {
        grid.get_station_power_node(station_id): 50.0
        for station_id in IEEE33_STATION_BUSES
    }
    station_voltages, spread, lowest_station = _print_station_voltages(
        "\nStation bus voltages under IEEE 33 base load + 8 x 50 kW EV load:",
        grid,
        loads,
    )

    print("\nThevenin equivalent resistance by station bus:")
    for station_id, bus in IEEE33_STATION_BUSES.items():
        r_ohm = grid.get_bus_thevenin_resistance(bus)
        print(f"  station={station_id} bus={bus:>2} r_th_ohm={r_ohm:.6f}")

    print("\nSmoke expectations:")
    print(f"  spread >= 0.03: {spread >= 0.03}")
    print(
        "  lowest bus is one of {17, 33}: "
        f"{IEEE33_STATION_BUSES[lowest_station] in {17, 33}}"
    )


if __name__ == "__main__":
    main()
