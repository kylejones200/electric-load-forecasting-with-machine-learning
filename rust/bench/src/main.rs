use electric_load_forecasting_with_machine_learning_core::cyclical_time_features;
fn main() { let u: Vec<f64>=(0..10000).map(|i| 1.7e9 + i as f64 * 3600.0).collect(); for _ in 0..5000 { let _=cyclical_time_features(&u); } }
