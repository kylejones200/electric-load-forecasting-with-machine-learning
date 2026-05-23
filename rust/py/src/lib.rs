use electric_load_forecasting_with_machine_learning_core::cyclical_time_features;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn cyclical_time_features_py<'py>(py: Python<'py>, unix_secs: PyReadonlyArray1<f64>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let c = cyclical_time_features(unix_secs.as_slice()?);
    Ok((c.hour_sin.into_pyarray(py), c.hour_cos.into_pyarray(py), c.day_sin.into_pyarray(py), c.day_cos.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (unix_secs, iterations=10_000))]
fn bench_kernel_py(unix_secs: PyReadonlyArray1<f64>, iterations: usize) -> PyResult<f64> {
    let u = unix_secs.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations { let _ = cyclical_time_features(&u); }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn electric_load_forecasting_with_machine_learning_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cyclical_time_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
