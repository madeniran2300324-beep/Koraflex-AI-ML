"""Day 2 — Streamlit data-quality dashboard."""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="KoraFlex — Data Quality", layout="wide")
st.title("KoraFlex · Data Quality")

client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGODB_DB", "koraflex")]

runs = list(db.data_quality_runs.find().sort("created_at", -1).limit(200))
if not runs:
    st.info("No validation runs yet. POST records to /v1/data-quality/validate.")
    st.stop()

df = pd.DataFrame([{
    "created_at": r["created_at"],
    "schema": r["schema"],
    "total": r["report"]["total_records"],
    "passed": r["report"]["passed"],
    "failed": r["report"]["failed"],
    "pass_rate": r["report"]["pass_rate"],
} for r in runs])

c1, c2, c3 = st.columns(3)
c1.metric("Avg pass rate", f"{df['pass_rate'].mean():.1%}")
c2.metric("Records validated", int(df["total"].sum()))
c3.metric("Failures", int(df["failed"].sum()))

st.plotly_chart(
    px.line(df.sort_values("created_at"), x="created_at", y="pass_rate", color="schema",
            title="Pass rate over time"),
    use_container_width=True,
)

st.subheader("Latest issues")
latest = runs[0]["report"].get("issues", [])
if latest:
    st.dataframe(pd.DataFrame(latest))
else:
    st.success("No issues in the latest run.")
