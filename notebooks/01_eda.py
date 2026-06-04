# %% [markdown]
# # Exploratory Data Analysis — Jordanian Mobile Market
#
# This notebook explores the cleaned dataset of second-hand mobile phone listings
# from the Jordanian market (sourced from OpenSooq/السوق المفتوح).

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import CLEANED_DATA_PATH, REPORTS_DIR

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# Use sample data if cleaned_data.csv is not available
csv_path = CLEANED_DATA_PATH if CLEANED_DATA_PATH.exists() else Path(__file__).resolve().parent.parent / "data/processed/sample_data.csv"
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head(10)

# %% [markdown]
# ## 1. Price Distribution
#
# The price distribution is right-skewed — most phones are in the 100–600 JOD range,
# with a long tail of premium devices. A log transform normalizes this for modeling.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
sns.histplot(df["price_jd"], bins=40, kde=True, ax=axes[0], color="#58A6FF")
axes[0].set_title("Price Distribution (Linear Scale)")
axes[0].set_xlabel("Price (JOD)")
axes[0].set_ylabel("Count")
axes[0].axvline(df["price_jd"].mean(), color="red", linestyle="--", label=f"Mean: {df['price_jd'].mean():.0f}")
axes[0].axvline(df["price_jd"].median(), color="green", linestyle="--", label=f"Median: {df['price_jd'].median():.0f}")
axes[0].legend()

# Log scale
sns.histplot(np.log1p(df["price_jd"]), bins=40, kde=True, ax=axes[1], color="#58A6FF")
axes[1].set_title("Price Distribution (Log Scale)")
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_price_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Price by Brand (Violin Plot)
#
# Apple dominates the premium segment with the highest median prices.
# Samsung spans a wide range (budget A-series to flagship S-series).
# Xiaomi, Tecno, and Infinix cluster in the budget segment.

# %%
top_brands = df["brand"].value_counts().head(6).index
df_top = df[df["brand"].isin(top_brands)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.violinplot(x="brand", y="price_jd", data=df_top, ax=ax, palette="Set2", inner="box")
ax.set_title("Price Distribution by Brand (Top 6)")
ax.set_xlabel("Brand")
ax.set_ylabel("Price (JOD)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_brand_violin.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Price by Condition
#
# As expected, new phones command the highest prices, followed by excellent, good, and poor condition.
# The spread within each condition is significant — condition alone doesn't determine price.

# %%
fig, ax = plt.subplots(figsize=(10, 6))
order = ["جديد", "مستعمل - ممتاز", "مستعمل - جيد", "مستعمل - سيئ"]
existing = [c for c in order if c in df["condition"].unique()]
sns.barplot(x="condition", y="price_jd", data=df, order=existing, ax=ax,
            palette="viridis", errorbar="sd", capsize=0.1)
ax.set_title("Average Price by Condition (with Std Dev)")
ax.set_xlabel("Condition")
ax.set_ylabel("Average Price (JOD)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_condition_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Price vs Storage (Scatter + Trend)
#
# Higher storage generally means higher price, but the relationship is noisy
# because storage correlates with model tier (Pro/Max models have more storage).

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x="storage_gb", y="price_jd", data=df, ax=ax, alpha=0.5, hue="brand", palette="tab10")
sns.regplot(x="storage_gb", y="price_jd", data=df, ax=ax, scatter=False, color="red", label="Trend")
ax.set_title("Price vs Storage Capacity")
ax.set_xlabel("Storage (GB)")
ax.set_ylabel("Price (JOD)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_storage_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Missing Value Heatmap
#
# Shows which columns have missing data and whether missingness is random or patterned.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax, cmap="YlOrRd")
ax.set_title("Missing Value Heatmap")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_missing_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Print null counts
print("\nNull counts per column:")
print(df.isnull().sum().to_string())

# %% [markdown]
# ## 6. Top 20 Most Common Series
#
# iPhone models dominate the Jordanian second-hand market, reflecting
# strong brand preference in the region.

# %%
fig, ax = plt.subplots(figsize=(14, 6))
top_series = df["series"].value_counts().head(20)
sns.barplot(x=top_series.index, y=top_series.values, ax=ax, palette="coolwarm")
ax.set_title("Top 20 Most Common Series")
ax.set_xlabel("Series")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "eda_top_series.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Correlation Matrix of Numeric Features
#
# Shows linear relationships between numeric features.
# Storage and phone age are the strongest numeric predictors.

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nCorrelation with price_jd:")
    print(corr["price_jd"].sort_values(ascending=False).to_string())

# %% [markdown]
# ## 8. Before/After Cleaning — Sample Listings
#
# Shows how raw Arabic titles are transformed into structured data.

# %%
sample_titles = [
    "ايفون 15 برو ماكس 256 مستعمل ممتاز بدل",
    "سامسونج جالاكسي S24 الترا 512 جديد",
    "ايفون 13 برو 128 مستعمل جيد جداً",
    "شاومي ريدمي نوت 13 256 مستعمل ممتاز",
    "هواوي نوفا 12 256 مستعمل - حالة ممتازة",
    "ايفون 14 بلس 128 كفالة",
    "سامسونج A54 128 مستعمل",
    "ايفون 16 برو ماكس 512 كرتون مسكر",
]

from src.cleaning import normalize_arabic, extract_series, normalize_condition

print(f"{'Raw Title':<55} {'Brand':<10} {'Series':<25} {'Condition'}")
print("─" * 115)
for title in sample_titles:
    brand = "Apple" if any(k in title for k in ["ايفون", "iphone"]) else "Samsung" if any(k in title for k in ["سامسونج", "جالاكسي"]) else "Other"
    series = extract_series(title, brand)
    cond = normalize_condition(title)
    print(f"{title:<55} {brand:<10} {series:<25} {cond}")

# %% [markdown]
# ## Summary of Findings
#
# 1. **Price distribution** is right-skewed → log-transform is appropriate for modeling
# 2. **Apple dominates** the premium segment; Samsung spans budget to flagship
# 3. **Condition matters** but has high variance — it's one signal among many
# 4. **Storage correlates** with price but is confounded with model tier
# 5. **iPhone models** are the most common in the Jordanian second-hand market
# 6. **Arabic text normalization** is critical — the same phone can be written many ways
# 7. **Missing data** is concentrated in optional fields (color, model_name) — price and brand are nearly complete
