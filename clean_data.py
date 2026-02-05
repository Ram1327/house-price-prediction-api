"""
Data cleaning script for Case Study 1 Data.xlsx
Steps: remove duplicates, handle missing values, convert Date Sold to datetime,
       create Property_Age, drop non-informative columns.
"""
import pandas as pd
import numpy as np
from pathlib import Path

EXCEL_PATH = Path(r"C:\Users\ramra\Downloads\Case Study 1 Data.xlsx")


def _find_column(df, candidates, description="column"):
    """Find first column that exists (case-insensitive, stripped)."""
    cols_lower = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def load_data():
    """Load Excel from first sheet."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    print("Loaded data shape:", df.shape)
    return df


def step1_remove_duplicates(df):
    """
    STEP 1: Remove duplicates
    ----------------------------------------
    Duplicate rows are exact copies of another row. They add no new information
    and can bias counts and averages. We keep the first occurrence and drop
    all subsequent duplicates.
    """
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"\n1. REMOVE DUPLICATES")
    print(f"   Rows before: {n_before}  →  after: {len(df)}  (removed {n_removed} duplicate(s))")
    return df


def step2_handle_missing(df):
    """
    STEP 2: Handle missing values
    ----------------------------------------
    - Numeric columns: fill with median (robust to outliers).
    - Categorical columns: fill with mode (most frequent category) or 'Unknown'.
    - Columns that are mostly missing (>50%): we still impute so we keep rows;
      for critical analysis you might drop such columns or rows later.
    """
    print(f"\n2. HANDLE MISSING VALUES")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
            df[col] = df[col].fillna(val)
            print(f"   '{col}' (numeric): filled {missing} missing with median = {val}")
        else:
            mode_vals = df[col].mode()
            val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            df[col] = df[col].fillna(val)
            print(f"   '{col}' (categorical): filled {missing} missing with mode = '{val}'")
    return df


def step3_date_sold_to_datetime(df):
    """
    STEP 3: Convert Date Sold to datetime
    ----------------------------------------
    'Date Sold' may be stored as text or Excel serial number. Converting to
    proper datetime allows sorting, filtering by date, and extracting year/month
    for features like Property_Age.
    """
    date_col = _find_column(df, ["Date Sold", "DateSold", "date_sold", "Date sold"])
    if date_col is None:
        print(f"\n3. CONVERT DATE SOLD TO DATETIME")
        print(f"   No 'Date Sold' column found. Columns: {list(df.columns)}")
        return df
    print(f"\n3. CONVERT DATE SOLD TO DATETIME")
    print(f"   Column used: '{date_col}'")
    before_dtype = str(df[date_col].dtype)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Drop rows where date could not be parsed (optional; keeps analysis clean)
    na_dates = df[date_col].isna().sum()
    if na_dates > 0:
        df = df.dropna(subset=[date_col]).reset_index(drop=True)
        print(f"   Converted to datetime. Dropped {na_dates} row(s) with invalid dates.")
    else:
        print(f"   Converted to datetime (dtype was: {before_dtype}).")
    return df


def step4_create_property_age(df):
    """
    STEP 4: Create Property_Age
    ----------------------------------------
    Property age at time of sale = (Year of Date Sold) - (Year Built).
    If 'Year Built' is missing we use the column name that best matches
    (e.g. Year Built, Built, Construction Year). If no year-built column exists,
    we skip or use a placeholder and explain.
    """
    date_col = _find_column(df, ["Date Sold", "DateSold", "date_sold", "Date sold"])
    year_built_col = _find_column(
        df,
        ["Year Built", "YearBuilt", "year_built", "Built", "Year Built", "Construction Year", "Build Year"],
    )
    print(f"\n4. CREATE PROPERTY_AGE")
    if date_col is None:
        print("   Skipped: no date column for sale year.")
        return df
    sale_year = pd.to_datetime(df[date_col], errors="coerce").dt.year
    if year_built_col:
        year_built = pd.to_numeric(df[year_built_col], errors="coerce")
        df["Property_Age"] = (sale_year - year_built).clip(lower=0)
        df["Property_Age"] = df["Property_Age"].fillna(df["Property_Age"].median()).astype(int)
        print(f"   Created 'Property_Age' = (Year of '{date_col}') - ('{year_built_col}'), clipped to ≥0.")
    else:
        # Fallback: no year built -> use 0 or median age if we had it
        print(f"   No 'Year Built' column found. Creating Property_Age from '{date_col}' year only (age set to 0).")
        df["Property_Age"] = 0
    return df


def step5_drop_non_informative(df):
    """
    STEP 5: Drop non-informative columns
    ----------------------------------------
    - Constant columns: same value in every row (variance = 0).
    - All-null columns: no information.
    - Optional: columns that are purely IDs (e.g. 'ID', 'Index') with unique
      values per row add no predictive value for modeling; we drop if they
      look like row identifiers (e.g. name contains 'id' and unique count = rows).
    """
    print(f"\n5. DROP NON-INFORMATIVE COLUMNS")
    to_drop = []
    for col in df.columns:
        # Constant
        if df[col].nunique(dropna=False) <= 1:
            to_drop.append(col)
            print(f"   Dropping '{col}': constant (single value or all null).")
            continue
        # All null (after imputation this is rare)
        if df[col].isna().all():
            to_drop.append(col)
            print(f"   Dropping '{col}': all null.")
            continue
        # Purely ID-like: column name suggests ID and values are unique per row
        col_lower = col.lower()
        if ("id" in col_lower or "index" in col_lower) and df[col].nunique() == len(df):
            to_drop.append(col)
            print(f"   Dropping '{col}': row identifier (unique per row, no predictive value).")
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
    df = df.reset_index(drop=True)
    if not to_drop:
        print("   No non-informative columns identified; none dropped.")
    return df


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load and process data
    df = load_data()
    initial_shape = df.shape

    # 2. Initial cleaning steps
    df = step1_remove_duplicates(df)
    df = step2_handle_missing(df)
    df = step3_date_sold_to_datetime(df)
    df = step4_create_property_age(df)

    # 3. Drop ID columns
    # Removing these columns often reveals new duplicates that were only
    # "unique" because of their specific ID value.
    df = step5_drop_non_informative(df)

    # 4. SECOND DEDUPLICATION (The Fix)
    # This removes the 149 residual duplicates found in your exploration report.
    print("\nRE-CHECKING FOR DUPLICATES (Post-ID removal)")
    df = step1_remove_duplicates(df)

    # 5. Export final cleaned data
    output_path = "Cleaned_House_Data.csv"
    df.to_csv(output_path, index=False)

    final_shape = df.shape
    print("\n" + "=" * 60)
    print("FINAL CLEANED DATAFRAME")
    print("=" * 60)
    print(f"Shape: {final_shape[0]} rows, {final_shape[1]} columns")
    print(f"Change: {initial_shape[0]} → {final_shape[0]} rows, {initial_shape[1]} → {final_shape[1]} columns")

    print(f"\nSUCCESS: Cleaned data has been saved to: {output_path}")

    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    df_clean = main()
