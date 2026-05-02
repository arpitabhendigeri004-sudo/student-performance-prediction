import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Generate Synthetic Data
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "attendance": np.random.randint(40, 100, n),
    "study_hours": np.random.randint(1, 10, n),
    "quiz_score": np.random.randint(30, 100, n),
    "assignment_score": np.random.randint(30, 100, n)
})

# Rule-based target (simulation)
data["pass"] = (
    (data["attendance"] > 60) &
    (data["quiz_score"] > 50) &
    (data["assignment_score"] > 50)
).astype(int)

# Step 2: Split Data
X = data.drop("pass", axis=1)
y = data["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Step 5: Save Model
joblib.dump(model, "models/model.pkl")

print("Model saved!")