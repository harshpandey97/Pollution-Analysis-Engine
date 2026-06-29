import os
import pandas as pd
from groq import Groq

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")
client  = Groq(api_key=API_KEY)
MODEL   = "llama-3.3-70b-versatile"

# ── Load Data ────────────────────────────────────────────────────────────────
df       = pd.read_csv("data/pollution_data.csv")
city_avg = df.groupby("City")[["AQI","PM2.5","PM10"]].mean().round(1)

# ── AI Insights ──────────────────────────────────────────────────────────────
def get_insights(city: str, aqi: float, pm25: float, pm10: float) -> str:
    prompt = f"""You are an environmental health expert.
City: {city}
AQI: {aqi} | PM2.5: {pm25} µg/m³ | PM10: {pm10} µg/m³

Give a 3-point response:
1. Health Risk level
2. Who is most affected
3. One actionable recommendation for citizens

Keep it concise, under 80 words."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# ── Main ──────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("   🤖 AI POLLUTION INSIGHTS — GROQ LLaMA 3.3")
print("="*55)

for city, row in city_avg.iterrows():
    print(f"\n🏙️  {city} (AQI: {row['AQI']})")
    print("-"*40)
    insight = get_insights(city, row["AQI"], row["PM2.5"], row["PM10"])
    print(insight)

print("\n" + "="*55)
