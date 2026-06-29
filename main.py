import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/pollution_data.csv")

# ── AQI Category Color Map ───────────────────────────────────────────────────
COLOR_MAP = {
    "Satisfactory": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Poor": "#8f3f97",
    "Very Poor": "#7e0023",
}

def get_color(category):
    return COLOR_MAP.get(category, "#999999")

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("   POLLUTION ANALYSIS ENGINE — REPORT")
print("="*55)

city_avg = df.groupby("City")[["AQI", "PM2.5", "PM10"]].mean().round(1)
city_avg = city_avg.sort_values("AQI", ascending=False)

print("\n📊 City-wise Average AQI (Jan 2024):\n")
print(f"{'City':<15} {'Avg AQI':<12} {'PM2.5':<10} {'PM10'}")
print("-"*50)
for city, row in city_avg.iterrows():
    print(f"{city:<15} {row['AQI']:<12} {row['PM2.5']:<10} {row['PM10']}")

worst = city_avg["AQI"].idxmax()
best  = city_avg["AQI"].idxmin()
print(f"\n🚨 Most Polluted : {worst} (AQI: {city_avg.loc[worst,'AQI']})")
print(f"✅ Cleanest City : {best}  (AQI: {city_avg.loc[best,'AQI']})")
print("="*55)

# ── Plot 1: Bar Chart ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("🌍 India Air Pollution Analysis — Jan 2024", fontsize=16, fontweight="bold")

cities = city_avg.index.tolist()
colors = [get_color(df[df.City==c]["Category"].mode()[0]) for c in cities]

axes[0,0].bar(cities, city_avg["AQI"], color=colors, edgecolor="black", linewidth=0.5)
axes[0,0].set_title("Average AQI by City")
axes[0,0].set_ylabel("AQI")
axes[0,0].tick_params(axis="x", rotation=15)
for i, v in enumerate(city_avg["AQI"]):
    axes[0,0].text(i, v+3, str(v), ha="center", fontsize=9, fontweight="bold")

# ── Plot 2: PM2.5 vs PM10 ────────────────────────────────────────────────────
x = range(len(cities))
axes[0,1].bar([i-0.2 for i in x], city_avg["PM2.5"], width=0.4, label="PM2.5", color="#e74c3c")
axes[0,1].bar([i+0.2 for i in x], city_avg["PM10"],  width=0.4, label="PM10",  color="#3498db")
axes[0,1].set_title("PM2.5 vs PM10 Comparison")
axes[0,1].set_xticks(list(x))
axes[0,1].set_xticklabels(cities, rotation=15)
axes[0,1].set_ylabel("µg/m³")
axes[0,1].legend()

# ── Plot 3: Trend Line ───────────────────────────────────────────────────────
for city in cities:
    city_df = df[df.City == city].reset_index(drop=True)
    axes[1,0].plot(city_df.index, city_df["AQI"], marker="o", label=city)
axes[1,0].set_title("AQI Trend (5 Days)")
axes[1,0].set_xlabel("Day")
axes[1,0].set_ylabel("AQI")
axes[1,0].legend(fontsize=8)

# ── Plot 4: Pie Chart ────────────────────────────────────────────────────────
cat_counts = df["Category"].value_counts()
pie_colors = [get_color(c) for c in cat_counts.index]
axes[1,1].pie(cat_counts, labels=cat_counts.index, colors=pie_colors,
              autopct="%1.1f%%", startangle=140)
axes[1,1].set_title("AQI Category Distribution")

plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/pollution_report.png", dpi=150, bbox_inches="tight")
print("\n📁 Chart saved → output/pollution_report.png")
plt.show()
