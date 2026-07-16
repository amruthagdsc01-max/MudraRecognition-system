import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("dataset/dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# --------------------------
# Features and Labels
# --------------------------

X = df.drop("label", axis=1)
y = df["label"]

print("\nClasses Found:")
print(y.unique())

# --------------------------
# Train-Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# --------------------------
# Train KNN
# --------------------------

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------
# Save Model
# --------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(knn, "models/knn_model.pkl")

print("\nModel saved successfully!")