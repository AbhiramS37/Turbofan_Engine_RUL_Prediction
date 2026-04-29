# feature_engineering.py
import pandas as pd
import numpy as np
import json, os

ROLLING_WINDOWS = [5, 10, 20]
EWM_SPAN        = 10
SLOPE_WINDOW    = 10
RUL_CAP         = 125

_sensor_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/low_var_sensors.json")

def get_low_var_sensors():
    if os.path.exists(_sensor_file):
        with open(_sensor_file) as f:
            return json.load(f)
    return []

def get_sensor_cols(df):
    low_var = get_low_var_sensors()
    return [c for c in df.columns if c.startswith("sensor_") and c not in low_var]

def _rolling_slope(series, window):
    def slope(v):
        n = len(v)
        if n < 2:
            return 0.0
        return (v[-1] - v[0]) / (n - 1)
    return series.rolling(window, min_periods=2).apply(slope, raw=True).fillna(0)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw df — engine_id, cycle, op_settings, sensors
    Output: fully engineered feature df
    Features:
      - cycle (raw — strong RUL signal)
      - rolling mean + std at windows 5, 10, 20
      - EWM (span=10)
      - slope over 10 cycles
      - cummax + cummin (long-term drift)
    """
    df = df.copy()

    # Drop low-variance sensors
    low_var = get_low_var_sensors()
    df = df.drop(columns=[c for c in low_var if c in df.columns])
    sensor_cols = get_sensor_cols(df)

    # Sort before all rolling ops
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    #Rolling mean + std at multiple windows
    for window in ROLLING_WINDOWS:
        for col in sensor_cols:
            grp = df.groupby("engine_id")[col]
            df[f"{col}_mean_w{window}"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_std_w{window}"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

    #EWM
    for col in sensor_cols:
        df[f"{col}_ewm"] = df.groupby("engine_id")[col].transform(
            lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean()
        )

    #Slope
    for col in sensor_cols:
        df[f"{col}_slope"] = df.groupby("engine_id")[col].transform(
            lambda x: _rolling_slope(x, SLOPE_WINDOW)
        )

    #Cumulative max + min (captures long-term drift)
    for col in sensor_cols:
        grp = df.groupby("engine_id")[col]
        df[f"{col}_cummax"] = grp.transform("cummax")
        df[f"{col}_cummin"] = grp.transform("cummin")

    return df