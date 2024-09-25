from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from joblib import load  # Import joblib for loading the model

# Load the trained model
model_path = 'Isolation.pkl'
try:
    model = load(model_path)  # Use joblib to load the model
except Exception as e:
    print("Error loading model:", e)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Anomaly Detected' if prediction[0] == -1 else 'No Anomaly'

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
