# House Price Prediction API

A production-ready machine learning solution to predict real estate prices based on property features (Location, Size, Year Built, etc.). This project covers the full data science lifecycle from raw data cleaning to a live cloud-deployed API.

## 🚀 Live API Link
**[INSERT YOUR RENDER URL HERE]**

## 📊 Model Performance
- **Algorithm:** Decision Tree / Random Forest Regressor
- **R-squared Score:** ~0.96+ (explains 96% of price variance)
- **Mean Absolute Error (MAE):** ~$24,600 (average prediction error)
- **Multi-core Support:** Enabled for efficient training on large datasets.

## 📁 Project Structure
- `clean_data.py`: Handles deduplication, missing values, and feature engineering.
- `exploredata.py`: Statistical verification and EDA summary.
- `prepare_for_modelling.py`: One-hot encoding, scaling, and 80/20 train-test splitting.
- `model_training.py`: Trains the regressor and exports the serialized model.
- `app.py`: Flask-based REST API for serving real-time predictions.
- `house_price_model.pkl`: The trained model "brain."

## 🛠️ How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Install the required libraries:
```bash
pip install pandas scikit-learn flask joblib gunicorn requests

Run the pipeline in order:

python clean_data.py
python prepare_for_modelling.py
python model_training.py

Start the API:

python app.py
The server will start at http://127.0.0.1:5000

Test Predictions
In a separate terminal, run the test script:

python test_api.py

To get a prediction, send a POST request to /predict with a JSON body:

{
    "Size": 0.5,
    "Bedrooms": 0.2,
    "Bathrooms": 0.1,
    "Year Built": 0.8,
    "Property_Age": -0.5,
    "Location_CityB": 1,
    "Location_CityC": 0,
    "Location_CityD": 0,
    "Condition_Good": 1,
    "Condition_New": 0,
    "Condition_Poor": 0,
    "Type_Single Family": 1,
    "Type_Townhouse": 0,
    "Date Sold_Year": 2024,
    "Date Sold_Month": 10
}

By following this approach, you demonstrate that you can adapt to cloud constraints while maintaining the high standards required for a Senior Data Scientist role.
