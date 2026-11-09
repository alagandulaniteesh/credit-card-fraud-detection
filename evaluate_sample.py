import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

predictions = pd.read_csv("fraud_detection_results.csv")
actual = pd.read_csv("sample_actual_labels.csv")

y_true = actual["Class"]
y_pred = predictions["Prediction"]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Accuracy:", accuracy_score(y_true, y_pred))