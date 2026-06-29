import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from groq import Groq

st.set_page_config(page_title="Pollution Analysis Engine", page_icon="🌍", layout="wide")

st.title("🌍 Pollution Analysis Engine")
st.markdown("**AI-powered air quality analysis for Indian cities**")

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/pollution_data.csv")

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Filters")
cities   = st.sidebar.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())
filtered = df[df["City"].isin(cities)]

# ── Metrics ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
city_avg = filtered.groupby("City")["AQI"].mean()
col1.metric("🏙️ Cities Analyzed", len(cities))
col2.metric("📊 Avg AQI",         round(filtered["AQI"].mean(), 1))
col3.metric("🚨 Worst City",      city_avg.idxmax() if len(city_avg) else "N/A")
col4.metric("✅ Best City",       city_avg.idxmin() if len(city_avg) else "N/A")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Average AQI by City")
    avg = filtered.groupby("City")["AQI"].mean().sort_values(ascending=False)
    st.bar_chart(avg)

with col_b:
    st.subheader("📈 AQI Trend (5 Days)")
    trend = filtered.pivot_table(index=filtered.groupby("City").cumcount(), columns="City", values="AQI")
    st.line_chart(trend)

# ── Data Table ───────────────────────────────────────────────────────────────
st.subheader("📋 Raw Data")
st.dataframe(filtered, use_container_width=True)

# ── AI Insights ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 AI Insights (Groq LLaMA 3.3)")

api_key = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")

if api_key and st.button("Generate AI Insights"):
    client = Groq(api_key=api_key)
    for city in cities:
        row = filtered[filtered.City == city][["AQI","PM2.5","PM10"]].mean()
        with st.spinner(f"Analyzing {city}..."):
            prompt = f"City: {city}, AQI: {row['AQI']:.1f}, PM2.5: {row['PM2.5']:.1f}. Give 3-point health advisory in 60 words."
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                max_tokens=150
            )
            st.info(f"**{city}** — {res.choices[0].message.content.strip()}")
