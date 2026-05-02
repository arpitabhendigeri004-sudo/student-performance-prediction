from src.preprocessing import load_data, clean_data, split_data, scale_data, split_train_test

# Load data
data = load_data()

# Clean data
data = clean_data(data)

# Split
X, y = split_data(data)

# Scale
X_scaled = scale_data(X)

# Train-test split
X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

print("🎯 Preprocessing Complete!")
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))