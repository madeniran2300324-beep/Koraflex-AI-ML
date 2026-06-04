"""Day 5/9 — Fraud indicators + analytics dashboard."""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="KoraFlex — Fraud Analytics", layout="wide")
st.title("KoraFlex · Fraud Analytics")

client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGODB_DB", "koraflex")]

scores = list(db.fraud_scores.find().sort("created_at", -1).limit(5000))
if not scores:
    st.info("No fraud scores yet.")
    st.stop()

df = pd.DataFrame([{
    "created_at": s["created_at"],
    "final_score": s["final_score"],
    "decision": s["decision"],
    "amount": s["amount"],
    "latency_ms": s.get("latency_ms", 0),
    "model_version": s.get("model_version", "n/a"),
} for s in scores])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg risk", f"{df['final_score'].mean():.1f}")
c2.metric("Blocked", int((df["decision"] == 'block').sum()))
c3.metric("Review", int((df["decision"] == 'review').sum()))
c4.metric("p95 latency (ms)", f"{df['latency_ms'].quantile(0.95):.0f}")

st.plotly_chart(
    px.histogram(df, x="final_score", nbins=40, color="decision",
                 title="Risk score distribution"),
    use_container_width=True,
)
st.plotly_chart(
    px.scatter(df, x="created_at", y="final_score", color="decision",
               size="amount", title="Risk over time"),
    use_container_width=True,
)

# Feedback-driven accuracy
fb = list(db.fraud_feedback.find().limit(2000))
if fb:
    fb_df = pd.DataFrame(fb)
    joined = df.merge(
        pd.DataFrame({
            "transaction_id": [f["transaction_id"] for f in fb],
            "is_fraud": [f["is_fraud"] for f in fb],
        }),
        left_on="created_at", right_on="transaction_id", how="inner",
    )
    st.subheader("Confusion (labeled)")
    pred = (fb_df["is_fraud"] == True).astype(int)
    st.write("Labeled feedback count:", len(fb_df))
