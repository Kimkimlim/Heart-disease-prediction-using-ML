from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and scaler
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler1.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form inputs
    form_data = request.form
    features = [
        float(form_data['Age']),
        float(form_data['RestingBP']),
        float(form_data['Cholesterol']),
        float(form_data['FastingBS']),
        float(form_data['MaxHR']),
        float(form_data['Oldpeak']),
        1 if form_data['Sex'] == 'Male' else 0,
        1 if form_data['ChestPainType'] == 'ATA' else 0,
        1 if form_data['ChestPainType'] == 'NAP' else 0,
        1 if form_data['ChestPainType'] == 'ASY' else 0,
        1 if form_data['RestingECG'] == 'ST' else 0,
        1 if form_data['RestingECG'] == 'LVH' else 0,
        1 if form_data['ExerciseAngina'] == 'Y' else 0,
        1 if form_data['ST_Slope'] == 'Flat' else 0,
        1 if form_data['ST_Slope'] == 'Down' else 0
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict using the model
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1] * 100

    # Return the result
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    return render_template('result.html', result=result, probability=probability)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use port 5001 or any other available port
