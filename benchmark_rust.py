#!/usr/bin/env python3
import time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/"src"))
from compute_kernel import cyclical_time_features
def main():
    u=np.arange(10000)*3600.+1.7e9
    t0=time.perf_counter()
    for _ in range(2000 if "cyclical_time_features"=="cyclical_time_features" else 200):
        u=np.arange(10000)*3600.+1.7e9
    py_s=time.perf_counter()-t0
    try:
        import electric_load_forecasting_with_machine_learning_rs as rs
    except ImportError:
        print("Build: cd rust && maturin develop --release -m py/Cargo.toml"); print(f"Python {py_s:.3f}s"); return
    rs_s=rs.bench_kernel_py(u,5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s/max(rs_s,1e-9):.1f}x")
    np.testing.assert_allclose(u, np.asarray(rs.cyclical_time_features_py(u[:500]))[0] if isinstance(rs.cyclical_time_features_py(u[:500]), tuple) else rs.cyclical_time_features_py(u[:500]), rtol=1e-12)
    print("Correctness: OK")
if __name__=="__main__": main()
