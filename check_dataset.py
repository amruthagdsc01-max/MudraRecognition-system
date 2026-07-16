import pandas as pd

df = pd.read_csv("dataset/dataset.csv")

print("Dataset shape:", df.shape)
print()

print("Labels:")
print(df["label"].value_counts())
print()

print("Last 10 rows:")
print(df.tail(10))