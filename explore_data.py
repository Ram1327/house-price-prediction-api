"""
Data exploration script for Cleaned_House_Data.csv
Senior Data Scientist - Statistical summary, feature correlation, and cleaning verification.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Change the path to point to your cleaned CSV output
CLEANED_DATA_PATH = Path("Cleaned_House_Data.csv")


def main():
    # 1. LOAD THE CLEANED DATA
    if not CLEANED_DATA_PATH.exists():
        print(f"Error: {CLEANED_DATA_PATH} not found. Please run cleandata.py first.")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)

    print("\n" + "=" * 60)
    print("1. CLEANING VERIFICATION")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Verify duplicates and missing values are gone
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Missing values remaining: {missing_count}")
    print(f"Duplicate rows remaining: {duplicate_count}")

    print("\n" + "=" * 60)
    print("2. FEATURE BREAKDOWN")
    print("=" * 60)
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numerical Features ({len(numerical)}): {numerical}")
    print(f"Categorical Features ({len(categorical)}): {categorical}")

    print("\n" + "=" * 60)
    print("3. STATISTICAL SUMMARY (Numerical)")
    print("=" * 60)
    # This gives us mean, median (50%), min, max, and std dev
    print(df.describe().T.round(2))

    print("\n" + "=" * 60)
    print("4. CATEGORICAL CARDINALITY")
    print("=" * 60)
    # Shows how many unique values are in each category (useful for encoding)
    for col in categorical:
        print(f"{col}: {df[col].nunique()} unique values (e.g., {df[col].unique()[:5]})")

    print("\n" + "=" * 60)
    print("5. TARGET VARIABLE ANALYSIS (Price)")
    print("=" * 60)
    print(f"Average Price: ${df['Price'].mean():,.2f}")
    print(f"Median Price:  ${df['Price'].median():,.2f}")
    print(f"Price Range:   ${df['Price'].min():,.2f} to ${df['Price'].max():,.2f}")

    print("\n" + "=" * 60)
    print("6. CORRELATION WITH PRICE")
    print("=" * 60)
    # Shows which numerical features have the strongest relationship with the price
    if 'Price' in df.columns:
        correlations = df[numerical].corr()['Price'].sort_values(ascending=False)
        print("Numerical correlation with Price:")
        print(correlations)

    print("\n" + "=" * 60)
    print("First 5 rows of Cleaned Data:")
    print("=" * 60)
    print(df.head().to_string())

    return df


if __name__ == "__main__":
    main()
