#!/usr/bin/env python3
import time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/"src"))
from compute_kernel import cyclical_time_features
def main():
    u=np.ascontiguousarray(np.arange(10000,dtype=float)*3600.+1.7e9)
    t0=time.perf_counter()
    for _ in range(2000):
        cyclical_time_features(u)
    py_s=time.perf_counter()-t0
    try:
        import electric_load_forecasting_with_machine_learning_rs as rs
    except ImportError:
        print("Build: cd rust && maturin develop --release -m py/Cargo.toml"); print(f"Python {py_s:.3f}s"); return
    rs_s=rs.bench_kernel_py(u,2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s/max(rs_s,1e-9):.1f}x")
    py_out=cyclical_time_features(u[:500])
    rs_out=rs.cyclical_time_features_py(np.ascontiguousarray(u[:500],dtype=float))
    for a,b in zip(py_out, rs_out):
        np.testing.assert_allclose(a, np.asarray(b), rtol=1e-12)
    print("Correctness: OK")
if __name__=="__main__": main()
