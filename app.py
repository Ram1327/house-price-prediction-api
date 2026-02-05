import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.pkl")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "House Price Prediction API is Online",
        "usage": "Send a POST request to /predict with property features."
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Convert input into DataFrame
        input_df = pd.DataFrame([data])

        # Ensure the columns match exactly what the model expects
        prediction = model.predict(input_df)[0]

        return jsonify({
            'status': 'success',
            'predicted_price': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    import os

    # Use the port assigned by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
