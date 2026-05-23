# Electric Load Forecasting with Machine Learning

Published: 2025-10-10
Medium: [https://medium.com/@kyle-t-jones/electric-load-forecasting-with-machine-learning-68d7bef4f641](https://medium.com/@kyle-t-jones/electric-load-forecasting-with-machine-learning-68d7bef4f641)

## Business context

My family slept in the living room, by our natural gas fireplace during Winter Storm Uri in February 2021. Electricity demand surged to unprecedented levels while generation capacity plummeted. ERCOT, the grid balancing authority, struggled with load forecasts that failed to capture the extreme weather's compound effects on both demand and supply. Prices spiked to the regulatory cap of $9,000 per megawatt-hour, rolling blackouts affected 4.5 million homes, and the economic damage exceeded $200 billion. Yikes!

Modern load forecasting combines time-honored statistical methods with cutting-edge machine learning, leveraging massive public datasets that were unimaginable a decade ago. The EIA's Form 930 provides hourly balancing authority data. NOAA delivers weather forecasts with unprecedented accuracy. The EAGLE-I system tracks outages in near real-time. Together, these sources enable forecasting systems that outperform traditional approaches by 30--50% while using only publicly available data.

This project shows how to build a load forecasting system from scratch, using EIA generation data.

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — cyclical time features (sin/cos hour and day). Reference PyO3 benchmark: **comparable (see `benchmark_rust.py`)** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p electric_load_forecasting_with_machine_learning_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).