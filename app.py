# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessing information
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('model/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

with open('model/scaling_params.pkl', 'rb') as f:
    scaling_params = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', features=features, mappings=mappings)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {}
    for feature in features:
        if feature in mappings:
            # For categorical features, get the selected option
            input_data[feature] = request.form[feature]
        else:
            # For numerical features, convert to float
            input_data[feature] = float(request.form[feature])

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the same preprocessing as in the notebook
    # 1. Apply categorical mappings
    for col, mapping in mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping)

    # 2. Apply feature scaling
    for col in input_df.columns:
        if col in scaling_params:
            params = scaling_params[col]
            if params['method'] == 'mean_max':
                input_df[col] = (input_df[col] - params['mean']) / params['max']
            else:  # min_max
                input_df[col] = (input_df[col] - params['min']) / (params['max'] - params['min'])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of passing

    result = "Pass" if prediction == 1 else "Fail"
    confidence = probability if prediction == 1 else (1 - probability)

    return render_template('result.html', result=result, confidence=confidence * 100)


if __name__ == '__main__':
    app.run(debug=True)