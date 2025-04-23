import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import load_data, preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

from src.preprocess import tokenize_text

tokens = tokenize_text("Some sample text")


# Load and preprocess
data = load_data('data/tested.csv')
data = preprocess_data(data)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = train_model(X_train, y_train, model_name="rf")
results = evaluate_model(model, X_test, y_test)

print("\nModel Performance:")
for metric, value in results.items():
    print(f"{metric}: {value}")

# --- Optional Cross-Validation Test ---
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

print("\nRunning 5-Fold Cross-Validation:")
cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(cv_model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Average accuracy:", cv_scores.mean())

