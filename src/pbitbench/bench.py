"""Run the benchmark: ``python -m pbitbench --sizes 32 64 128 --sensor mock``."""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from .mlp import MLP
from .pbits import PatternRBM
from .power import INA219, MockSensor, PowerLogger
from .task import bit_accuracy, make_task

FIELDS = [
    "model", "n", "k", "sensor", "batch", "seconds", "inferences", "idle_w", "avg_w",
    "net_energy_per_inference_j", "time_per_inference_s", "bit_accuracy", "macs_per_inference",
]


def _measure(fn, logger: PowerLogger, duration_s: float, batch: int) -> tuple[float, int]:
    """Call ``fn`` repeatedly for ``duration_s``; return (elapsed, inferences)."""
    count = 0
    with logger:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration_s:
            fn()
            count += batch
        elapsed = time.perf_counter() - t0
    return elapsed, count


def run(sizes, sensor_name="mock", duration_s=5.0, idle_s=3.0, batch=32, flip_prob=0.1,
        seed=0, out=sys.stdout):
    sensor = INA219() if sensor_name == "ina219" else MockSensor()
    rng = np.random.default_rng(seed)
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()

    idle = PowerLogger(sensor)
    with idle:
        time.sleep(idle_s)
    idle_w = idle.mean_power()

    for n in sizes:
        task = make_task(n, flip_prob=flip_prob, seed=seed + n)
        test_x, test_y = task.sample(512, rng)
        rbm = PatternRBM(task.patterns)
        mlp = MLP(n, hidden=n, rng=rng).fit(task, rng)
        work_x, _ = task.sample(batch, rng)

        models = {
            "pbit_rbm": (lambda: rbm.recall(work_x, rng), rbm.recall(test_x, rng), rbm.macs_per_item()),
            "mlp": (lambda: mlp.predict(work_x), mlp.predict(test_x), mlp.macs_per_item()),
        }
        for name, (fn, pred, macs) in models.items():
            logger = PowerLogger(sensor)
            elapsed, count = _measure(fn, logger, duration_s, batch)
            avg_w = logger.mean_power()
            writer.writerow({
                "model": name, "n": n, "k": task.k, "sensor": sensor_name, "batch": batch,
                "seconds": round(elapsed, 3), "inferences": count,
                "idle_w": round(idle_w, 4), "avg_w": round(avg_w, 4),
                "net_energy_per_inference_j": (avg_w - idle_w) * elapsed / count,
                "time_per_inference_s": elapsed / count,
                "bit_accuracy": round(bit_accuracy(pred, test_y), 4),
                "macs_per_inference": macs,
            })
            out.flush()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pbitbench", description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--sensor", choices=["mock", "ina219"], default="mock")
    p.add_argument("--duration", type=float, default=5.0, help="seconds per model per size")
    p.add_argument("--idle", type=float, default=3.0, help="seconds of idle baseline")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--flip-prob", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    a = p.parse_args(argv)
    run(a.sizes, a.sensor, a.duration, a.idle, a.batch, a.flip_prob, a.seed, a.out)
    return 0
