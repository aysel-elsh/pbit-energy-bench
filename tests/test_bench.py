import csv
import io
import time

import numpy as np
import pytest

from pbitbench import MLP, PatternRBM, bit_accuracy, make_task, pbit
from pbitbench.bench import run
from pbitbench.power import (MockSensor, PowerLogger, bus_voltage_from_raw, energy_joules,
                             shunt_voltage_from_raw)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_task_noise_level(rng):
    task = make_task(64, k=4, flip_prob=0.1, seed=0)
    noisy, clean = task.sample(2000, rng)
    assert noisy.shape == clean.shape == (2000, 64)
    assert np.mean(noisy != clean) == pytest.approx(0.1, abs=0.01)


def test_pbit_statistics(rng):
    # mean of a p-bit is tanh(field)
    for f in (-1.0, 0.0, 0.5):
        m = pbit(np.full(50_000, f, dtype=np.float32), rng)
        assert m.mean() == pytest.approx(np.tanh(f), abs=0.02)


def test_rbm_denoises(rng):
    task = make_task(64, k=4, flip_prob=0.1, seed=0)
    noisy, clean = task.sample(300, rng)
    rbm = PatternRBM(task.patterns)
    assert bit_accuracy(noisy, clean) < 0.92
    assert bit_accuracy(rbm.recall(noisy, rng), clean) > 0.98


def test_mlp_denoises(rng):
    task = make_task(32, k=3, flip_prob=0.1, seed=0)
    mlp = MLP(32, 32, rng).fit(task, rng, steps=1500)
    noisy, clean = task.sample(300, rng)
    assert bit_accuracy(mlp.predict(noisy), clean) > 0.98


def test_ina219_conversions():
    assert shunt_voltage_from_raw(0x0FA0) == pytest.approx(0.04)    # 4000 * 10 uV
    assert shunt_voltage_from_raw(0xF060) == pytest.approx(-0.04)   # negative, two's complement
    assert bus_voltage_from_raw(5000 << 3) == pytest.approx(20.0)   # 5000 * 4 mV


def test_energy_integration():
    samples = [(0.0, 2.0), (1.0, 2.0), (2.0, 4.0)]
    assert energy_joules(samples) == pytest.approx(2.0 + 3.0)


def test_power_logger_collects():
    with PowerLogger(MockSensor(), interval_s=0.001) as log:
        time.sleep(0.05)
    assert len(log.samples) > 2
    assert 2.7 <= log.mean_power() <= 4.7


def test_run_writes_csv():
    buf = io.StringIO()
    run([16], duration_s=0.2, idle_s=0.1, batch=8, out=buf)
    rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert {r["model"] for r in rows} == {"pbit_rbm", "mlp"}
    assert all(float(r["bit_accuracy"]) > 0.9 for r in rows)
