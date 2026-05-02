from src.preprocessing import load_data, clean_data, split_data, scale_data, split_train_test
from src.mode1 import train_model

# Load data
data = load_data()

# Preprocess
data = clean_data(data)
X, y = split_data(data)
X_scaled = scale_data(X)
X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

# Train model
model = train_model(X_train, y_train)
import joblib

joblib.dump(model, "model.pkl")

# Predict
y_pred = model.predict(X_test)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_confusion_matrix(y_test, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=1,
        linecolor='black',
        annot_kws={"size": 14, "weight": "bold"}
    )

    plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)

    # Labels (customize if needed)
    plt.xticks([0.5, 1.5], ["Normal", "Attack"])
    plt.yticks([0.5, 1.5], ["Normal", "Attack"], rotation=0)

    # Accuracy text
    plt.text(
        0.5, -0.2,
        f"Accuracy: {acc:.2f}",
        ha='center',
        va='center',
        transform=plt.gca().transAxes,
        fontsize=12
    )

    plt.tight_layout()

    # Save image (optional)
    plt.savefig("confusion_matrix.png", dpi=300)

    plt.show()