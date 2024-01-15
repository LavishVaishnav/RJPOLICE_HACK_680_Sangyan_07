from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model
model_path = "path/to/your/model.pkl"
model = joblib.load(model_path)

# Create a StandardScaler object
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()

        # Create a DataFrame from the data
        data_df = pd.DataFrame([data])

        # Check if any values are zero
        any_zero = data_df.iloc[0, :-1].eq(0).any()

        if any_zero:
            prediction_label = 'The Transaction you made is Invalid, Enter Valid Data'
        else:
            # Scale the features using the previously created scaler
            data_scaled = scaler.transform(data_df)

            # Make the prediction
            prediction = model.predict(data_scaled)
            prediction_label = "Yes, it is fraud" if prediction[0] == 1 else "No, it is not fraud"

        result = {"prediction_label": prediction_label}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
