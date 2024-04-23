from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)
CORS(app)

clf = None

def init_model():
    # Load your trained model
    global clf
    clf = joblib.load('diabetes_model.sav')

@app.route("/")
def hello_world():
    init_model()
    return "<p>Model Initialized</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if request.headers.get('Content-Type') == 'application/json':
        data = request.get_json()

        # Ensure that the keys in the JSON data match the features expected by your model
        features = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age']
        
        # Prepare the input data for prediction
        input_data = [float(data[feature]) for feature in features]

        # Make the prediction
        prediction = clf.predict([input_data])[0]

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction)})

    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
