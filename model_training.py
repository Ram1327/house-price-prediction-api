import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time


def main():
    print("=" * 70)
    print("MODEL TRAINING PHASE")
    print("=" * 70)

    # 1. Load the prepared datasets
    print("Loading data...")
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()

    # 2. Initialize Model with Multi-Core Processing
    # n_jobs=-1 tells the computer to use ALL available CPU cores
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # 3. Train the Model
    print(f"\nTraining Random Forest on {len(X_train)} samples...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 4. Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 30)
    print("MODEL PERFORMANCE")
    print("=" * 30)
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"R-squared Score:     {r2:.4f}")

    # 5. Save the Model for the API
    joblib.dump(model, "house_price_model.pkl")
    print("\nSUCCESS: Model saved as 'house_price_model.pkl'")


if __name__ == "__main__":
    main()