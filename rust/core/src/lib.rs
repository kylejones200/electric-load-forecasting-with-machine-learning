//! Cyclical time features from Unix timestamps (seconds).

const SECS_PER_HOUR: f64 = 3600.0;
const SECS_PER_DAY: f64 = 86400.0;

#[derive(Debug, Clone, PartialEq)]
pub struct CyclicalFeatures {
    pub hour_sin: Vec<f64>,
    pub hour_cos: Vec<f64>,
    pub day_sin: Vec<f64>,
    pub day_cos: Vec<f64>,
}

pub fn cyclical_time_features(unix_secs: &[f64]) -> CyclicalFeatures {
    let mut hour_sin = Vec::with_capacity(unix_secs.len());
    let mut hour_cos = Vec::with_capacity(unix_secs.len());
    let mut day_sin = Vec::with_capacity(unix_secs.len());
    let mut day_cos = Vec::with_capacity(unix_secs.len());

    for &ts in unix_secs {
        let hour = (ts / SECS_PER_HOUR) % 24.0;
        let day_of_year = (ts / SECS_PER_DAY) % 365.25;
        let h_angle = 2.0 * std::f64::consts::PI * hour / 24.0;
        let d_angle = 2.0 * std::f64::consts::PI * day_of_year / 365.25;
        hour_sin.push(h_angle.sin());
        hour_cos.push(h_angle.cos());
        day_sin.push(d_angle.sin());
        day_cos.push(d_angle.cos());
    }

    CyclicalFeatures {
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
    }
}
