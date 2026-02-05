"""
Prepare dataset for modeling:
- Loads cleaned CSV data
- One-hot encoding categorical features
- Handling datetime features
- Scaling numerical features
- Split into training (80%) and test (20%) sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ================================================================
# STEP 0: LOAD CLEANED DATAFRAME
# ================================================================

CLEANED_DATA_PATH = Path("Cleaned_House_Data.csv")

print("="*70)
print("PREPARING DATASET FOR MODELING")
print("="*70)

# FIX: Load the file instead of relying on a memory variable
if CLEANED_DATA_PATH.exists():
    df = pd.read_csv(CLEANED_DATA_PATH)
    # Convert 'Date Sold' back to datetime since CSV loads it as a string
    if 'Date Sold' in df.columns:
        df['Date Sold'] = pd.to_datetime(df['Date Sold'])
    print(f"\nLoaded cleaned data. Shape: {df.shape}")
else:
    print(f"ERROR: {CLEANED_DATA_PATH} not found! Run cleandata.py first.")
    raise FileNotFoundError("Cleaned CSV file is missing.")

# ================================================================
# STEP 1: IDENTIFY TARGET AND FEATURES
# ================================================================

target_col = "Price" # Identified from your exploration output
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Target identified: '{target_col}'")
print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

# ================================================================
# STEP 2: ONE-HOT ENCODING
# ================================================================

# Identify categorical columns (excluding datetime)
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"\nEncoding categories: {categorical_cols}")

# get_dummies is the most stable method for one-hot encoding in your environment
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ================================================================
# STEP 3: HANDLE DATETIME FEATURES
# ================================================================

datetime_cols = X_encoded.select_dtypes(include=["datetime64"]).columns.tolist()

for col in datetime_cols:
    X_encoded[f"{col}_Year"] = X_encoded[col].dt.year
    X_encoded[f"{col}_Month"] = X_encoded[col].dt.month
    X_encoded = X_encoded.drop(columns=[col])
    print(f"Extracted features from '{col}'")

# ================================================================
# STEP 4: SCALE NUMERICAL FEATURES
# ================================================================

# Scale all numeric columns (excluding the dummy variables created above)
numeric_cols = ['Size', 'Bedrooms', 'Bathrooms', 'Year Built', 'Property_Age']
scaler = StandardScaler()

X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
print(f"\nScaled numerical features: {numeric_cols}")

# ================================================================
# STEP 5: TRAIN/TEST SPLIT (80-20)
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nSplit complete:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set:     {X_test.shape[0]} samples")

# ================================================================
# STEP 6: SAVE PREPARED DATA FOR MODELLING
# ================================================================
# In production, we save these to disk so the modeling script can pick them up
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n" + "="*70)
print("PREPARATION COMPLETE - Ready for model_training.py")
print("="*70)