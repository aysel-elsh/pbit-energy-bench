"""Power measurement: an INA219 on the Pi's 5 V supply, or a mock for laptops.

Wiring (high-side): USB 5 V -> INA219 VIN+ -> VIN- -> Pi 5 V. The Pi reads the
INA219 over I2C (default address 0x40). Common breakouts use a 0.1 ohm shunt.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

__all__ = [
    "shunt_voltage_from_raw",
    "bus_voltage_from_raw",
    "INA219",
    "MockSensor",
    "PowerLogger",
    "energy_joules",
]

_REG_SHUNT = 0x01
_REG_BUS = 0x02


def shunt_voltage_from_raw(raw: int) -> float:
    """Shunt register (signed 16-bit, 10 uV/LSB) to volts."""
    if raw & 0x8000:
        raw -= 1 << 16
    return raw * 10e-6


def bus_voltage_from_raw(raw: int) -> float:
    """Bus register (bits 15..3, 4 mV/LSB) to volts."""
    return (raw >> 3) * 4e-3


def _swap16(x: int) -> int:
    # SMBus words are little-endian; the INA219 sends big-endian.
    return ((x & 0xFF) << 8) | (x >> 8)


class INA219:
    """Minimal INA219 reader using the power-on default configuration
    (32 V range, +-320 mV shunt, 12-bit, continuous)."""

    def __init__(self, bus: int = 1, address: int = 0x40, shunt_ohms: float = 0.1):
        from smbus2 import SMBus  # imported here so laptops don't need it

        self._bus = SMBus(bus)
        self.address = address
        self.shunt_ohms = shunt_ohms

    def _read(self, reg: int) -> int:
        return _swap16(self._bus.read_word_data(self.address, reg))

    def read_power(self) -> float:
        """Instantaneous load power in watts."""
        current = shunt_voltage_from_raw(self._read(_REG_SHUNT)) / self.shunt_ohms
        return bus_voltage_from_raw(self._read(_REG_BUS)) * current


class MockSensor:
    """Stand-in for development off the Pi: idle power plus a term that
    follows this process's CPU use. Numbers are NOT real measurements."""

    def __init__(self, idle_w: float = 2.7, busy_w: float = 2.0):
        self.idle_w, self.busy_w = idle_w, busy_w
        self._last = (time.perf_counter(), time.process_time())

    def read_power(self) -> float:
        now = (time.perf_counter(), time.process_time())
        wall, cpu = now[0] - self._last[0], now[1] - self._last[1]
        self._last = now
        util = min(1.0, cpu / wall) if wall > 0 else 0.0
        return self.idle_w + self.busy_w * util


@dataclass
class PowerLogger:
    """Samples a sensor on a background thread while a ``with`` block runs."""

    sensor: object
    interval_s: float = 0.005
    samples: list = field(default_factory=list)  # (t, watts)

    def __enter__(self) -> "PowerLogger":
        self.samples.clear()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.sensor.read_power()))
            time.sleep(self.interval_s)

    def mean_power(self) -> float:
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else 0.0


def energy_joules(samples: list[tuple[float, float]]) -> float:
    """Trapezoid-rule integral of (time, watts) samples."""
    return sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(samples, samples[1:]))
