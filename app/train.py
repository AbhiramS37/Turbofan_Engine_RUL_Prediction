# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib, os

modeltype = "rf"   # "rf" | "xg" | "lg"

df = pd.read_csv("../data/processed.csv")

# Drop only max_cycle — keep cycle as a feature (strong RUL signal)
if "max_cycle" in df.columns:
    print("WARNING: dropping 'max_cycle'")
    df = df.drop(columns=["max_cycle"])

#Engine-level split
engines = df["engine_id"].unique()
train_engines, test_engines = train_test_split(engines, test_size=0.2, random_state=42)

train_df = df[df["engine_id"].isin(train_engines)]
test_df  = df[df["engine_id"].isin(test_engines)]

# Keep cycle as feature — drop only RUL and engine_id
drop_cols = ["RUL", "engine_id"]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["RUL"]
X_test  = test_df.drop(columns=drop_cols)
y_test  = test_df["RUL"]

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}")

#Scaling
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.1, random_state=42
)

#Models
if modeltype == "rf":
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=4,
        max_features=0.6,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

elif modeltype == "xg":
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=1.0,
        min_child_weight=3,
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )

elif modeltype == "lg":
    import lightgbm as lgb
    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=50,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(200),
        ],
    )

#Evaluate
y_pred = model.predict(X_test_s).clip(0, 125)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"\nRMSE of {modeltype}: {rmse:.4f}")

#Save
os.makedirs("../models", exist_ok=True)
joblib.dump(model,  f"../models/{modeltype}.pkl")
joblib.dump(scaler, f"../models/{modeltype}_scaler.pkl")
print(f"{modeltype} model + scaler saved!")