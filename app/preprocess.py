# preprocess.py
import pandas as pd
import numpy as np
import json, os
from feature_eng import engineer_features, RUL_CAP

file_path = "../data/train_FD001.txt"

df = pd.read_csv(file_path, sep=r"\s+", header=None)
df = df.dropna(axis=1)

columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = columns
print("Raw shape:", df.shape)

#Compute and save low-variance sensors from actual data 
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
stds = df[sensor_cols].std()
print("\nSensor std values:")
print(stds.sort_values().to_string())

LOW_VAR_SENSORS = stds[stds < 0.01].index.tolist()
print(f"\nDropping low-variance sensors: {LOW_VAR_SENSORS}")

os.makedirs("../models", exist_ok=True)
with open("../models/low_var_sensors.json", "w") as f:
    json.dump(LOW_VAR_SENSORS, f)
print("Saved low_var_sensors.json")

#RUL
max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]
df = df.merge(max_cycle, on="engine_id")
df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CAP)
df = df.drop(columns=["max_cycle"])  # drop immediately — prevents leakage

#Feature engineering
df = engineer_features(df)

print("\nAfter feature engineering:", df.shape)
print(df["RUL"].describe())

df.to_csv("../data/processed.csv", index=False)
print("Saved processed.csv")