import pandas as pd

file_path = "data/raw/covtype.data.gz"

continuous_features = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

wilderness_features = [
    f"Wilderness_Area{i}" for i in range(1, 5)
]

soil_features = [
    f"Soil_Type{i}" for i in range(1, 41)
]

columns = (
    continuous_features
    + wilderness_features
    + soil_features
    + ["Cover_Type"]
)

print("Reading original UCI Covertype dataset...")

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    compression="gzip"
)

print("\n========== DATASET VALIDATION ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Expected rows: 581012")
print("Expected columns: 55")

print("\n========== MISSING VALUES ==========")
print("Total missing values:", int(df.isna().sum().sum()))

print("\n========== INDICATOR VALIDATION ==========")
indicator_features = wilderness_features + soil_features

invalid_values = {}
for col in indicator_features:
    values = set(df[col].dropna().unique())
    if not values.issubset({0, 1}):
        invalid_values[col] = sorted(values)

print("Number of indicator features:", len(indicator_features))
print("Invalid binary columns:", invalid_values)

print("\n========== COVER TYPE DISTRIBUTION ==========")
print(df["Cover_Type"].value_counts().sort_index())

print("\n========== CONTINUOUS FEATURE SUMMARY ==========")
print(df[continuous_features].describe().round(2).to_string())

print("\n========== TARGET VALIDATION ==========")
print("Unique cover types:", sorted(df["Cover_Type"].unique()))

print("\n========== VALIDATION RESULT ==========")

valid = (
    len(df) == 581012
    and len(df.columns) == 55
    and df.isna().sum().sum() == 0
    and len(invalid_values) == 0
    and set(df["Cover_Type"].unique()) == set(range(1, 8))
)

print("PASS" if valid else "CHECK REQUIRED")
