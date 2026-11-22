# pbit-energy-bench

**Does probabilistic-bit (p-bit) sampling actually save energy on small hardware?**

Thermodynamic and probabilistic computing companies (e.g. Extropic) claim that sampling with p-bits, units that flip randomly biased by their input, can do probabilistic AI work for far less energy than conventional matrix math. Those claims are about *custom chips*. This project asks a narrower, testable question:

> On a Raspberry Pi, at what problem size (if any) does p-bit sampling, simulated in software, beat a small conventional neural net on **measured energy per inference** at equal accuracy?

**Status: work in progress.** The harness, models and tests are done. Real INA219 measurements on a Pi are next. No results are claimed yet.

## How it works

| Part | What it is |
|---|---|
| **Task** (`task.py`) | Recover one of K stored ±1 patterns of length N from a copy with 10% of bits flipped |
| **p-bit model** (`pbits.py`) | A bipartite energy-based model (RBM) recalled by **block Gibbs sampling** with p-bits, `m = sign(tanh(βI) − u)` |
| **Baseline** (`mlp.py`) | A small deterministic MLP denoiser (N → N → N), trained with SGD |
| **Power** (`power.py`) | INA219 high-side current sensor on the Pi's 5 V supply, sampled on a background thread; idle baseline subtracted |
| **Benchmark** (`bench.py`) | Sweeps N, runs each model for a fixed time, writes CSV: energy/inference, time/inference, accuracy, MACs |

Both models are vectorised matrix code on the same CPU, so the comparison is about the algorithms, not Python loop overhead.

## Run it

```bash
pip install -e '.[dev]'          # add ',pi' on the Raspberry Pi for smbus2
pytest

# Laptop (mock sensor, numbers are NOT real measurements):
python -m pbitbench --sizes 32 64 128 256

# Raspberry Pi with an INA219 at 0x40:
python -m pbitbench --sensor ina219 --sizes 32 64 128 256 512 --duration 30 --out results.csv
```

## Hardware

- Raspberry Pi (4 or 5)
- INA219 breakout (0.1 Ω shunt), on I2C bus 1, address 0x40
- High-side wiring: USB 5 V → VIN+ → VIN− → Pi 5 V (a USB breakout board makes this easy)

## Plan

- [x] Shared task, p-bit RBM, MLP baseline, INA219 reader, CSV benchmark, tests
- [ ] Collect real measurements on a Pi 4 and Pi 5
- [ ] Repeat runs and add confidence intervals
- [ ] Accuracy-matched comparison (sweep Gibbs sweeps and MLP width)
- [ ] Harder tasks where sampling should matter more (multi-modal / generative)
- [ ] Compare against Extropic's [`thrml`](https://github.com/extropic-ai/thrml) where it runs

## References

- K. Y. Camsari et al., "Stochastic p-Bits for Invertible Logic," *Phys. Rev. X* 7, 031014 (2017)
- Extropic, `thrml`: block Gibbs sampling for probabilistic graphical models

## License

MIT. By [Aysel Elshenawy](https://github.com/aysel-elsh). Developed with AI assistance (Claude).
