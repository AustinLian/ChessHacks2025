import numpy as np

path = r"training/data/processed/sf_supervised_dataset.npz"

data = np.load(path)

print("Keys:", data.files)

X = data["X"]
y_policy = data["y_policy"]
y_value = data["y_value"]
y_result = data["y_result"]

print("X shape:", X.shape)
print("y_policy shape:", y_policy.shape)
print("y_value shape:", y_value.shape)
print("y_result shape:", y_result.shape)

# Inspect a few entries
print("\nSample X[0] min/max:", X[0].min(), X[0].max())
print("Sample move index y_policy[0]:", y_policy[0])
print("Sample SF value y_value[0]:", y_value[0])
print("Sample result-from-side-to-move y_result[0]:", y_result[0])

print("\nDataset appears to load successfully.")
