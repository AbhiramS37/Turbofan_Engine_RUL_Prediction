# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from io import BytesIO

#Import shared feature engineering 
sys.path.append(os.path.dirname(__file__))
from feature_eng import engineer_features

#Load model + scaler
model  = joblib.load("../models/rf.pkl")
scaler = joblib.load("../models/rf_scaler.pkl")

st.set_page_config(page_title="RUL Prediction System", layout="wide")
st.title("🔧 RUL Prediction System")
st.write("Upload raw engine sensor CSV")

#Expected raw input columns
RAW_COLS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}"     for i in range(1, 22)]
)

with st.expander("ℹ️ Expected CSV format (click to expand)"):
    st.write("Your CSV must have these columns ")
    st.code(", ".join(RAW_COLS))
    

# File upload
uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file is not None:

    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Input Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    # Validate columns 
    missing_cols = [c for c in RAW_COLS if c not in raw_df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    # 
    with st.spinner("Running feature engineering..."):

        # drop RUL if present (not needed for inference)
        infer_df = raw_df.drop(columns=["RUL"], errors="ignore").copy()

        # apply the SAME feature engineering as training
        infer_df = engineer_features(infer_df)

        # drop non-feature columns
        drop_cols = ["engine_id", "cycle", "RUL"]
        X = infer_df.drop(columns=[c for c in drop_cols if c in infer_df.columns])

        # align columns exactly to what scaler expects
        expected_cols = scaler.feature_names_in_
        missing_feats = set(expected_cols) - set(X.columns)
        extra_feats   = set(X.columns) - set(expected_cols)
        if extra_feats:
            X = X.drop(columns=list(extra_feats))
        X = X.reindex(columns=expected_cols, fill_value=0)

        # scale + predict
        X_scaled = scaler.transform(X)
        preds     = model.predict(X_scaled).clip(0, 125)

    infer_df["Predicted_RUL"] = preds.round(2)

    # Per-engine summary (last cycle = current state)
    engine_summary = (
        infer_df.sort_values(["engine_id", "cycle"])
                .groupby("engine_id")
                .last()
                .reset_index()[["engine_id", "cycle", "Predicted_RUL"]]
    )
    engine_summary.columns = ["Engine ID", "Last Cycle", "Current RUL"]
    engine_summary["Status"] = engine_summary["Current RUL"].apply(
        lambda r: "🔴 Critical" if r < 30 else ("🟡 Warning" if r < 60 else "🟢 Healthy")
    )
    engine_summary = engine_summary.sort_values("Current RUL").reset_index(drop=True)

    
    st.subheader("📊Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Engines",            len(engine_summary))
    c2.metric("🔴 Critical  (RUL < 30)",  int((engine_summary["Current RUL"] < 30).sum()))
    c3.metric("🟡 Warning   (RUL < 60)",  int(((engine_summary["Current RUL"] >= 30) & (engine_summary["Current RUL"] < 60)).sum()))
    c4.metric("🟢 Healthy   (RUL ≥ 60)",  int((engine_summary["Current RUL"] >= 60).sum()))

    
    tab1, tab2 = st.tabs(["🚨 Critical Engines", "📋 All Engines"])

    def to_excel(df):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False)
        return buf.getvalue()

    
    with tab1:
        critical = engine_summary[engine_summary["Current RUL"] < 30].reset_index(drop=True)
        if critical.empty:
            st.success("✅ No critical engines!")
        else:
            st.error(f"⚠️ {len(critical)} engine(s) need immediate attention")
            st.dataframe(critical, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.download_button("📥 Download Excel", to_excel(critical),
                               "critical_engines.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            c2.download_button("📥 Download CSV", critical.to_csv(index=False).encode(),
                               "critical_engines.csv", "text/csv")

    
    with tab2:
        status_filter = st.multiselect(
            "Filter by status",
            ["🔴 Critical", "🟡 Warning", "🟢 Healthy"],
            default=["🔴 Critical", "🟡 Warning", "🟢 Healthy"],
        )
        filtered = engine_summary[engine_summary["Status"].isin(status_filter)]
        st.dataframe(filtered, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download Excel", to_excel(filtered),
                           "engine_summary.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c2.download_button("📥 Download CSV", filtered.to_csv(index=False).encode(),
                           "engine_summary.csv", "text/csv")

   