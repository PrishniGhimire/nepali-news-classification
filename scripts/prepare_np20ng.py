import sys
import pandas as pd

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src, usecols=["category", "content"]).dropna()
df = df.rename(columns={"content": "text", "category": "label"})
per_class = n // df["label"].nunique()
parts = [g.sample(min(len(g), per_class), random_state=42) for _, g in df.groupby("label")]
df = pd.concat(parts)
df.to_csv(out, index=False)
print(df["label"].value_counts())