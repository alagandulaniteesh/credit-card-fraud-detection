import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "dataset/processed/creditcard_clean.csv"
)

normal = df[df["Class"] == 0].sample(
    5, random_state=42
)

fraud = df[df["Class"] == 1].sample(
    5, random_state=42
)

sample = pd.concat([normal, fraud])

sample = sample.sample(
    frac=1, random_state=42
)

sample.drop(columns=["Class"]).to_csv(
    "sample_transactions.csv",
    index=False
)

print("Sample CSV created successfully!")
print("Transactions:", len(sample))
print("Actual normal:", len(normal))
print("Actual fraud:", len(fraud))
# Save actual labels separately for evaluation
sample[["Class"]].to_csv(
    "sample_actual_labels.csv",
    index=False
)

print("Actual labels saved successfully!")