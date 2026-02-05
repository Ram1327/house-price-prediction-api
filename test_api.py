import requests

url = "https://house-price-prediction-api-small.onrender.com/predict"

# Updated sample data with the correct feature names
sample_property = {
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
    # FIX: These must match the columns in your training set exactly
    "Date Sold_Year": 2024,
    "Date Sold_Month": 10
}

response = requests.post(url, json=sample_property)
print(response.json())