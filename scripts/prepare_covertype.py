import os
import numpy as np
import pandas as pd

RAW_FILE = "data/raw/covtype.data.gz"
OUTPUT_DIR = "data/processed"

RANDOM_SEED = 2026

CONTINUOUS_FEATURES = [
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

ASPECT_FEATURE = "Aspect"

WILDERNESS_FEATURES = [
    f"Wilderness_Area{i}" for i in range(1, 5)
]

SOIL_FEATURES = [
    f"Soil_Type{i}" for i in range(1, 41)
]

LABEL_FEATURE = "Cover_Type"

ALL_COLUMNS = (
    ["Elevation", "Aspect", "Slope",
     "Horizontal_Distance_To_Hydrology",
     "Vertical_Distance_To_Hydrology",
     "Horizontal_Distance_To_Roadways",
     "Hillshade_9am", "Hillshade_Noon",
     "Hillshade_3pm",
     "Horizontal_Distance_To_Fire_Points"]
    + WILDERNESS_FEATURES
    + SOIL_FEATURES
    + [LABEL_FEATURE]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading original Covertype dataset...")

df = pd.read_csv(
    RAW_FILE,
    header=None,
    names=ALL_COLUMNS,
    compression="gzip"
)

print("Original observations:", len(df))

# Separate the target label from the clustering features.
labels = df[LABEL_FEATURE].to_numpy(dtype=np.int32)

# Standardise the nine non-circular continuous features.
continuous = df[CONTINUOUS_FEATURES].to_numpy(dtype=np.float64)

means = continuous.mean(axis=0)
stds = continuous.std(axis=0, ddof=0)

if np.any(stds == 0):
    raise ValueError("A continuous feature has zero standard deviation.")

continuous_scaled = (continuous - means) / stds

# Convert Aspect into two circular features.
aspect_radians = np.deg2rad(
    df[ASPECT_FEATURE].to_numpy(dtype=np.float64)
)

aspect_sin = np.sin(aspect_radians)
aspect_cos = np.cos(aspect_radians)

aspect_features = np.column_stack((aspect_sin, aspect_cos))

# Retain binary indicator features as 0/1.
binary_features = df[
    WILDERNESS_FEATURES + SOIL_FEATURES
].to_numpy(dtype=np.float64)

# Final feature order: 9 continuous + 2 aspect + 44 binary.
X = np.column_stack((
    continuous_scaled,
    aspect_features,
    binary_features
))

if X.shape != (581012, 55):
    raise ValueError(f"Unexpected feature matrix shape: {X.shape}")

if not np.isfinite(X).all():
    raise ValueError("Prepared features contain non-finite values.")

print("Prepared feature matrix:", X.shape)
print("Feature count:", X.shape[1])
print("Label count:", len(labels))

# One fixed permutation creates nested, reproducible subsets.
rng = np.random.default_rng(RANDOM_SEED)
permutation = rng.permutation(len(X))

datasets = {
    "100k": 100_000,
    "500k": 500_000,
    "full": len(X),
}

for name, size in datasets.items():
    indices = permutation[:size]

    X_subset = X[indices]
    y_subset = labels[indices]

    feature_path = os.path.join(
        OUTPUT_DIR, f"covertype_{name}.csv"
    )

    label_path = os.path.join(
        OUTPUT_DIR, f"covertype_{name}_labels.csv"
    )

    # Headerless files simplify C++ input parsing.
    np.savetxt(
        feature_path,
        X_subset,
        delimiter=",",
        fmt="%.10g"
    )

    np.savetxt(
        label_path,
        y_subset,
        fmt="%d"
    )

    print(f"\nCreated {name} dataset")
    print("Observations:", size)
    print("Feature file:", feature_path)
    print("Label file:", label_path)
    print("Label distribution:")

    unique, counts = np.unique(y_subset, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Cover type {label}: {count}")

print("\nPREPROCESSING COMPLETE")
