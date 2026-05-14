"""
Python code extracted from 01_load_forecasting_machine_learning_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s', stream=sys.stderr, force=True)
logger = logging.getLogger(__name__)

class EIAParquetService:
    """Service for handling the massive EIA electricity parquet dataset."""

    def __init__(self, parquet_path: str='ELEC.parquet'):
        """Initialize EIA parquet service.
        
        Args:
            parquet_path: Path to the ELEC.parquet file.
        """
        self.parquet_path = parquet_path
        self.raw_data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and parse the EIA parquet data."""
        try:
            logger.info(f'Loading EIA parquet data from {self.parquet_path}')
            self.raw_data = pd.read_parquet(self.parquet_path)
            logger.info(f'Loaded {len(self.raw_data)} EIA records')
        except Exception as e:
            logger.error(f'Failed to load EIA parquet data: {e}', exc_info=True)
            self.raw_data = pd.DataFrame()

    def search_series(self, query: str, limit: int=50) -> List[Dict[str, Any]]:
        """Search for series matching a query."""
        if self.raw_data is None or self.raw_data.empty:
            return []
        results = []
        column_name = self.raw_data.columns[0]
        for i, row in self.raw_data.iterrows():
            if len(results) >= limit:
                break
            json_str = row[column_name]
            if query.lower() in json_str.lower():
                try:
                    parsed = json.loads(json_str)
                    results.append({'series_id': parsed.get('series_id', ''), 'name': parsed.get('name', ''), 'units': parsed.get('units', ''), 'geography': parsed.get('geography', ''), 'start': parsed.get('start', ''), 'end': parsed.get('end', ''), 'data_points': len(parsed.get('data', []))})
                except json.JSONDecodeError:
                    continue
        return results

    def get_time_series_data(self, series_id: str) -> Dict[str, Any]:
        """Get complete time series data for a specific series."""
        if self.raw_data is None or self.raw_data.empty:
            return {}
        column_name = self.raw_data.columns[0]
        for i, row in self.raw_data.iterrows():
            json_str = row[column_name]
            try:
                parsed = json.loads(json_str)
                if parsed.get('series_id') == series_id:
                    data_points = parsed.get('data', [])
                    dates = []
                    values = []
                    for point in data_points:
                        if len(point) >= 2:
                            date_str = point[0]
                            value = point[1]
                            if len(date_str) == 6:
                                year = int(date_str[:4])
                                month = int(date_str[4:])
                                dates.append(f'{year}-{month:02d}-01')
                                pd.concat([values, value])
                    return {'series_id': series_id, 'name': parsed.get('name', ''), 'units': parsed.get('units', ''), 'dates': dates, 'values': values, 'metadata': {'lat': parsed.get('lat'), 'lon': parsed.get('lon'), 'geography': parsed.get('geography'), 'start': parsed.get('start'), 'end': parsed.get('end')}}
            except json.JSONDecodeError:
                continue
        return {}
eia_service = EIAParquetService('ELEC.parquet')
results = eia_service.search_series('generation CAL', limit=10)
for result in results:
    logger.info(f"Series: {result['series_id']}")
    logger.info(f"Name: {result['name']}")
    logger.info(f"Data points: {result['data_points']}")
    logger.info('')
if results:
    series_data = eia_service.get_time_series_data(results[0]['series_id'])
    logger.info(f"Retrieved {len(series_data['values'])} monthly data points")
    logger.info(f"Date range: {series_data['dates'][0]} to {series_data['dates'][-1]}")
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

def prepare_features(load_data: pd.DataFrame, lookback_days: int=90) -> pd.DataFrame:
    """Prepare feature dataset for modeling.
    
    Args:
        load_data: DataFrame with columns [ts_utc, mw, ba]
        lookback_days: Number of days of historical data to use.
        
    Returns:
        DataFrame with engineered features.
    """
    df = load_data.copy()
    df['ts_utc'] = pd.to_datetime(df['ts_utc'])
    df = df.sort_values('ts_utc').reset_index(drop=True)
    df['hour'] = df['ts_utc'].dt.hour
    df['dow'] = df['ts_utc'].dt.dayofweek + 1
    df['month'] = df['ts_utc'].dt.month
    df['day_of_year'] = df['ts_utc'].dt.dayofyear
    df['is_weekend'] = (df['dow'] >= 6).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['mw_lag1'] = df['mw'].shift(1)
    df['mw_lag24'] = df['mw'].shift(24)
    df['mw_lag168'] = df['mw'].shift(168)
    df['mw_ma24'] = df['mw'].rolling(window=24, min_periods=1).mean()
    df['mw_ma168'] = df['mw'].rolling(window=168, min_periods=1).mean()
    df['mw_std24'] = df['mw'].rolling(window=24, min_periods=1).std()
    df['temperature'] = 70 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + 10 * np.sin(2 * np.pi * df['hour'] / 24)
    df['temp_squared'] = df['temperature'] ** 2
    df['cooling_degree_days'] = np.maximum(df['temperature'] - 65, 0)
    df['heating_degree_days'] = np.maximum(55 - df['temperature'], 0)
    df['is_holiday'] = 0
    return df
sample_data = pd.DataFrame({'ts_utc': pd.date_range('2024-01-01', periods=24 * 90, freq='H'), 'mw': np.random.normal(25000, 5000, 24 * 90), 'ba': 'CAL-ALL'})
features_df = prepare_features(sample_data)
logger.info(f'Original columns: {len(sample_data.columns)}')
logger.info(f'Feature columns: {len(features_df.columns)}')
logger.info(f'\nNew features: {list(features_df.columns[3:])}')
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import mlflow
import mlflow.sklearn

def train_arima_model(df: pd.DataFrame, ba: str) -> Optional[str]:
    """Train auto_arima baseline model.
    
    Args:
        df: Feature DataFrame.
        ba: Balancing authority code.
        
    Returns:
        Model URI if successful, None otherwise.
    """
    ts_data = df[['ts_utc', 'mw']].copy()
    ts_data = ts_data.dropna().sort_values('ts_utc')
    if len(ts_data) < 168:
        logger.info(f'Insufficient data: {len(ts_data)} records')
        return None
    ts_data.set_index('ts_utc', inplace=True)
    ts_series = ts_data['mw']
    with mlflow.start_run(run_name=f'arima_{ba}'):
        logger.info(f'Training auto_arima model for {ba}...')
        model = auto_arima(ts_series, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=True, start_P=0, start_Q=0, max_P=2, max_Q=2, m=24, stepwise=True, suppress_warnings=True, error_action='ignore')
        fitted_values = model.fittedvalues()
        actual_values = ts_series[fitted_values.index]
        mape = mean_absolute_percentage_error(actual_values, fitted_values)
        mae = mean_absolute_error(actual_values, fitted_values)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('mae', mae)
        mlflow.log_param('model_type', 'auto_arima')
        mlflow.log_param('order', str(model.order))
        mlflow.log_param('seasonal_order', str(model.seasonal_order))
        model_info = mlflow.sklearn.log_model(model, 'model', registered_model_name=f'leap_{ba}_arima')
        logger.info(f'ARIMA model trained: MAPE={mape:.4f}, Order={model.order}')
        return model_info.model_uri
sample_features = prepare_features(sample_data)
model_uri = train_arima_model(sample_features, 'CAL-ALL')
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def train_lightgbm_model(df: pd.DataFrame, ba: str) -> Optional[str]:
    """Train LightGBM advanced model.
    
    Args:
        df: Feature DataFrame.
        ba: Balancing authority code.
        
    Returns:
        Model URI if successful, None otherwise.
    """
    feature_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168', 'mw_std24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'dow', 'month', 'is_weekend', 'is_holiday', 'temperature', 'temp_squared', 'cooling_degree_days', 'heating_degree_days']
    model_df = df[feature_cols + ['mw']].dropna()
    if len(model_df) < 168:
        logger.info(f'Insufficient data: {len(model_df)} records')
        return None
    X = model_df[feature_cols]
    y = model_df['mw']
    with mlflow.start_run(run_name=f'lightgbm_{ba}'):
        tscv = TimeSeriesSplit(n_splits=5)
        model = LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=8, num_leaves=31, random_state=42, verbose=-1)
        cv_predictions = np.full(len(y), np.nan)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = (X.iloc[train_idx], X.iloc[val_idx])
            y_train, y_val = (y.iloc[train_idx], y.iloc[val_idx])
            model.fit(X_train, y_train)
            cv_predictions[val_idx] = model.predict(X_val)
        valid_mask = ~np.isnan(cv_predictions)
        mape = mean_absolute_percentage_error(y[valid_mask], cv_predictions[valid_mask])
        mae = mean_absolute_error(y[valid_mask], cv_predictions[valid_mask])
        rmse = np.sqrt(mean_squared_error(y[valid_mask], cv_predictions[valid_mask]))
        model.fit(X, y)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_param('model_type', 'lightgbm')
        mlflow.log_param('n_estimators', 500)
        mlflow.log_param('learning_rate', 0.1)
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        mlflow.log_text(importance_df.to_string(), 'feature_importance.txt')
        model_info = mlflow.sklearn.log_model(model, 'model', registered_model_name=f'leap_{ba}_lightgbm')
        logger.info(f'LightGBM trained: MAPE={mape:.4f}')
        logger.info(f'\nTop 5 features:')
        logger.info(importance_df.head())
        return model_info.model_uri
model_uri_lgbm = train_lightgbm_model(sample_features, 'CAL-ALL')

def generate_forecast(model, df: pd.DataFrame, horizon_hours: int=24) -> List[float]:
    """Generate multi-hour forecast using trained model.
    
    Args:
        model: Trained LightGBM model.
        df: Historical data with features.
        horizon_hours: Number of hours to forecast.
        
    Returns:
        List of forecast values.
    """
    feature_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168', 'mw_std24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'dow', 'month', 'is_weekend', 'is_holiday', 'temperature', 'temp_squared', 'cooling_degree_days', 'heating_degree_days']
    last_row = df.dropna(subset=feature_cols).iloc[-1].copy()
    forecasts = []
    for i in range(horizon_hours):
        future_time = pd.to_datetime(last_row['ts_utc']) + timedelta(hours=i + 1)
        hour = future_time.hour
        day_of_year = future_time.dayofyear
        last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        last_row['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        last_row['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        last_row['dow'] = future_time.dayofweek + 1
        last_row['month'] = future_time.month
        last_row['is_weekend'] = int(last_row['dow'] >= 6)
        last_row['temperature'] = 70 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + 10 * np.sin(2 * np.pi * hour / 24)
        last_row['temp_squared'] = last_row['temperature'] ** 2
        last_row['cooling_degree_days'] = max(last_row['temperature'] - 65, 0)
        last_row['heating_degree_days'] = max(55 - last_row['temperature'], 0)
        X = last_row[feature_cols].values.reshape(1, -1)
        forecast = model.predict(X)[0]
        pd.concat([forecasts, forecast])
        last_row['mw_lag1'] = forecast
        last_row['mw_ma24'] = 0.95 * last_row['mw_ma24'] + 0.05 * forecast
    return forecasts
forecasts_48h = generate_forecast(model, sample_features, horizon_hours=48)
logger.info('48-Hour Load Forecast:')
for i, forecast in enumerate(forecasts_48h):
    logger.info(f'Hour {i + 1}: {forecast:,.0f} MW')

def apply_scenario(df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Apply scenario adjustments to feature DataFrame.
    
    Args:
        df: Base features DataFrame.
        scenario_id: Scenario identifier.
        
    Returns:
        Modified DataFrame with scenario adjustments.
    """
    scenario_df = df.copy()
    if scenario_id == 'hot_weather':
        scenario_df['temperature'] += 15
        scenario_df['temp_squared'] = scenario_df['temperature'] ** 2
        scenario_df['cooling_degree_days'] = np.maximum(scenario_df['temperature'] - 65, 0)
        scenario_df['heating_degree_days'] = np.maximum(55 - scenario_df['temperature'], 0)
    elif scenario_id == 'high_growth':
        lag_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168']
        for col in lag_cols:
            if col in scenario_df.columns:
                scenario_df[col] *= 1.05
    elif scenario_id == 'demand_response':
        peak_hours = scenario_df['hour'].between(16, 21)
        scenario_df.loc[peak_hours, 'mw_lag1'] *= 0.9
        scenario_df.loc[peak_hours, 'mw_lag24'] *= 0.9
    elif scenario_id == 'major_outage':
        scenario_df['mw_lag1'] *= 0.8
        scenario_df['mw_ma24'] *= 0.85
    return scenario_df
scenarios = ['baseline', 'hot_weather', 'high_growth', 'demand_response']
scenario_forecasts = {}
for scenario in scenarios:
    scenario_features = np.where(scenario == 'baseline', sample_features.copy(), apply_scenario(sample_features, scenario))
    forecasts = generate_forecast(model, scenario_features, horizon_hours=24)
    scenario_forecasts[scenario] = forecasts
logger.info('Peak Load Comparison (Hour 18):')
for scenario, forecasts in scenario_forecasts.items():
    peak_load = forecasts[17]
    logger.info(f'{scenario:20s}: {peak_load:>8,.0f} MW')
baseline_peak = scenario_forecasts['baseline'][17]
for scenario in scenarios[1:]:
    scenario_peak = scenario_forecasts[scenario][17]
    diff_pct = (scenario_peak / baseline_peak - 1) * 100
    logger.info(f'  {scenario} vs baseline: {diff_pct:+.1f}%')

class OutageImpactAnalyzer:
    """Analyze how outages affect load forecasts."""

    def __init__(self, outage_service):
        self.outage_service = outage_service

    def get_outage_adjustment_factor(self, state: str, timestamp: datetime) -> float:
        """Calculate load adjustment factor based on current outages.
        
        Args:
            state: State code (e.g., 'Texas')
            timestamp: Current timestamp
            
        Returns:
            Adjustment factor (1.0 = no adjustment, 0.9 = 10% reduction)
        """
        year = timestamp.year
        outage_data = self.outage_service.get_outage_data(year=year, state=state, start_date=timestamp.strftime('%Y-%m-%d'), end_date=timestamp.strftime('%Y-%m-%d'), limit=100)
        if not outage_data:
            return 1.0
        total_customers_out = sum((r['customers_out'] for r in outage_data))
        total_customers = sum((r.get('total_customers', 0) for r in outage_data if r.get('total_customers')))
        if total_customers == 0:
            return 1.0
        outage_rate = total_customers_out / total_customers
        adjustment_factor = 1.0 - outage_rate * 0.8
        return max(0.5, adjustment_factor)

    def adjust_forecast_for_outages(self, forecast: List[float], state: str, base_time: datetime) -> List[float]:
        """Adjust forecast based on expected or ongoing outages.
        
        Args:
            forecast: Base forecast values
            state: State code
            base_time: Base timestamp for forecast
            
        Returns:
            Adjusted forecast values
        """
        adjusted_forecast = []
        for i, forecast_value in enumerate(forecast):
            forecast_time = base_time + timedelta(hours=i)
            adjustment = self.get_outage_adjustment_factor(state, forecast_time)
            adjusted_forecast.append(forecast_value * adjustment)
        return adjusted_forecast
'Service for handling the massive EIA electricity parquet dataset.'

def __init__(self, parquet_path: str='ELEC.parquet'):
    """Initialize EIA parquet service.
    
    Args:
        parquet_path: Path to the ELEC.parquet file.
    """
    self.parquet_path = parquet_path
    self.raw_data = None
    self._load_data()

def _load_data(self) -> None:
    """Load and parse the EIA parquet data."""
    try:
        logger.info(f'Loading EIA parquet data from {self.parquet_path}')
        self.raw_data = pd.read_parquet(self.parquet_path)
        logger.info(f'Loaded {len(self.raw_data)} EIA records')
    except Exception as e:
        logger.error(f'Failed to load EIA parquet data: {e}', exc_info=True)
        self.raw_data = pd.DataFrame()

def search_series(self, query: str, limit: int=50) -> List[Dict[str, Any]]:
    """Search for series matching a query."""
    if self.raw_data is None or self.raw_data.empty:
        return []
    results = []
    column_name = self.raw_data.columns[0]
for i, row in self.raw_data.iterrows():
    if len(results) >= limit:
        break
    json_str = row[column_name]
    if query.lower() in json_str.lower():
        try:
            parsed = json.loads(json_str)
            results.append({'series_id': parsed.get('series_id', ''), 'name': parsed.get('name', ''), 'units': parsed.get('units', ''), 'geography': parsed.get('geography', ''), 'start': parsed.get('start', ''), 'end': parsed.get('end', ''), 'data_points': len(parsed.get('data', []))})
        except json.JSONDecodeError:
            continue

def get_time_series_data(self, series_id: str) -> Dict[str, Any]:
    """Get complete time series data for a specific series."""
    if self.raw_data is None or self.raw_data.empty:
        return {}
    column_name = self.raw_data.columns[0]
    for i, row in self.raw_data.iterrows():
        json_str = row[column_name]
        try:
            pass
        except Exception:
            pass
            parsed = json.loads(json_str)
            if parsed.get('series_id') == series_id:
                pass
                dates = []
                values = []
                for point in data_points:
                    if len(point) >= 2:
                        date_str = point[0]
                        value = point[1]
if len(date_str) == 6:
    year = int(date_str[:4])
    month = int(date_str[4:])
    dates.append(f'{year}-{month:02d}-01')
    pd.concat([values, value])
logger.info(f"Series: {result['series_id']}")
logger.info(f"Name: {result['name']}")
logger.info(f"Data points: {result['data_points']}")
logger.info('')
series_data = eia_service.get_time_series_data(results[0]['series_id'])
logger.info(f"Retrieved {len(series_data['values'])} monthly data points")
logger.info(f"Date range: {series_data['dates'][0]} to {series_data['dates'][-1]}")
'Prepare feature dataset for modeling.\n\nArgs:\n    load_data: DataFrame with columns [ts_utc, mw, ba]\n    lookback_days: Number of days of historical data to use.\n    \nReturns:\n    DataFrame with engineered features.\n'
df = load_data.copy()
df['ts_utc'] = pd.to_datetime(df['ts_utc'])
df = df.sort_values('ts_utc').reset_index(drop=True)
df['hour'] = df['ts_utc'].dt.hour
df['dow'] = df['ts_utc'].dt.dayofweek + 1
df['month'] = df['ts_utc'].dt.month
df['day_of_year'] = df['ts_utc'].dt.dayofyear
df['is_weekend'] = (df['dow'] >= 6).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['mw_lag1'] = df['mw'].shift(1)
df['mw_lag24'] = df['mw'].shift(24)
df['mw_lag168'] = df['mw'].shift(168)
df['mw_ma24'] = df['mw'].rolling(window=24, min_periods=1).mean()
df['mw_ma168'] = df['mw'].rolling(window=168, min_periods=1).mean()
df['mw_std24'] = df['mw'].rolling(window=24, min_periods=1).std()
df['temperature'] = 70 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + 10 * np.sin(2 * np.pi * df['hour'] / 24)
df['temp_squared'] = df['temperature'] ** 2
df['cooling_degree_days'] = np.maximum(df['temperature'] - 65, 0)
df['heating_degree_days'] = np.maximum(55 - df['temperature'], 0)
df['is_holiday'] = 0
return df
ts_data = df[['ts_utc', 'mw']].copy()
ts_data = ts_data.dropna().sort_values('ts_utc')
if len(ts_data) < 168:
    logger.info(f'Insufficient data: {len(ts_data)} records')
    return None
ts_data.set_index('ts_utc', inplace=True)
ts_series = ts_data['mw']
with mlflow.start_run(run_name=f'arima_{ba}'):
    pass
logger.info(f'Training auto_arima model for {ba}...')
fitted_values = model.fittedvalues()
mape = mean_absolute_percentage_error(actual_values, fitted_values)
model_info = mlflow.sklearn.log_model(model, 'model', registered_model_name=f'leap_{ba}_arima')
feature_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168', 'mw_std24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'dow', 'month', 'is_weekend', 'is_holiday', 'temperature', 'temp_squared', 'cooling_degree_days', 'heating_degree_days']
model_df = df[feature_cols + ['mw']].dropna()
if len(model_df) < 168:
    logger.info(f'Insufficient data: {len(model_df)} records')
    return None
X = model_df[feature_cols]
y = model_df['mw']
with mlflow.start_run(run_name=f'lightgbm_{ba}'):
    pass
tscv = TimeSeriesSplit(n_splits=5)
cv_predictions = np.full(len(y), np.nan)
valid_mask = ~np.isnan(cv_predictions)
importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
model_info = mlflow.sklearn.log_model(model, 'model', registered_model_name=f'leap_{ba}_lightgbm')
'Generate multi-hour forecast using trained model.\n\nArgs:\n    model: Trained LightGBM model.\n    df: Historical data with features.\n    horizon_hours: Number of hours to forecast.\n    \nReturns:\n    List of forecast values.\n'
feature_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168', 'mw_std24', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'dow', 'month', 'is_weekend', 'is_holiday', 'temperature', 'temp_squared', 'cooling_degree_days', 'heating_degree_days']
last_row = df.dropna(subset=feature_cols).iloc[-1].copy()
forecasts = []
for i in range(horizon_hours):
    pass
future_time = pd.to_datetime(last_row['ts_utc']) + timedelta(hours=i + 1)
last_row['temperature'] = 70 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + 10 * np.sin(2 * np.pi * hour / 24)
X = last_row[feature_cols].values.reshape(1, -1)
forecast = model.predict(X)[0]
last_row['mw_lag1'] = forecast
return forecasts
logger.info(f'Hour {i + 1}: {forecast:,.0f} MW')
'Apply scenario adjustments to feature DataFrame.\n\nArgs:\n    df: Base features DataFrame.\n    scenario_id: Scenario identifier.\n    \nReturns:\n    Modified DataFrame with scenario adjustments.\n'
scenario_df = df.copy()
if scenario_id == 'hot_weather':
    pass
scenario_df['temperature'] += 15
lag_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168']
peak_hours = scenario_df['hour'].between(16, 21)
scenario_df['mw_lag1'] *= 0.8
return scenario_df
scenario_features = np.where(scenario == 'baseline', sample_features.copy(), apply_scenario(sample_features, scenario))
forecasts = generate_forecast(model, scenario_features, horizon_hours=24)
scenario_forecasts[scenario] = forecasts
peak_load = forecasts[17]
logger.info(f'{scenario:20s}: {peak_load:>8,.0f} MW')
scenario_peak = scenario_forecasts[scenario][17]
diff_pct = (scenario_peak / baseline_peak - 1) * 100
logger.info(f'  {scenario} vs baseline: {diff_pct:+.1f}%')
'Analyze how outages affect load forecasts.'

def __init__(self, outage_service):
    self.outage_service = outage_service

def get_outage_adjustment_factor(self, state: str, timestamp: datetime) -> float:
    """Calculate load adjustment factor based on current outages.
    
    Args:
        state: State code (e.g., 'Texas')
        timestamp: Current timestamp
        
    Returns:
        Adjustment factor (1.0 = no adjustment, 0.9 = 10% reduction)
    """
year = timestamp.year
total_customers_out = sum((r['customers_out'] for r in outage_data))
adjustment_factor = 1.0 - outage_rate * 0.8

def adjust_forecast_for_outages(self, forecast: List[float], state: str, base_time: datetime) -> List[float]:
    """Adjust forecast based on expected or ongoing outages.
    
    Args:
        forecast: Base forecast values
        state: State code
        base_time: Base timestamp for forecast
        
    Returns:
        Adjusted forecast values
    """
    adjusted_forecast = []
    for i, forecast_value in enumerate(forecast):
        forecast_time = base_time + timedelta(hours=i)
        adjustment = self.get_outage_adjustment_factor(state, forecast_time)
        adjusted_forecast.append(forecast_value * adjustment)
    return adjusted_forecast
