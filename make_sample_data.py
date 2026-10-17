"""Generates sample_data.csv (synthetic, for demo only)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 300
regions = ["North", "South", "East", "West"]
categories = {"Electronics": (8000, 40000), "Clothing": (500, 4000),
              "Groceries": (100, 1500), "Books": (150, 1200)}
cat = rng.choice(list(categories), n)
df = pd.DataFrame({
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "region": rng.choice(regions, n),
    "category": cat,
    "units": rng.integers(1, 8, n),
})
df["unit_price"] = [int(rng.integers(*categories[c])) for c in cat]
df["revenue"] = df["units"] * df["unit_price"]
df["customer_rating"] = np.round(rng.uniform(2.5, 5.0, n), 1)
df = df.sort_values("order_date").reset_index(drop=True)
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
df.to_csv("sample_data.csv", index=False)
print(df.shape)
