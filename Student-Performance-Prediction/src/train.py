import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from pipeline import preprocessor
import joblib

# Load data
df = pd.read_csv("data/student.csv", sep=";")

# Target
df["pass"] = (df["G3"] >= 10).astype(int)

# Features
X = df.drop(["G3","pass"], axis=1)
y = df["pass"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "models/student_model.pkl")

print("Model saved successfully!")