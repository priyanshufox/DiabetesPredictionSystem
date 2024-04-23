from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import sys



def initModel():
    # load the iris dataset
    iris = load_iris()
    # separate the features (independent variables) and target (dependent variable)
    X = iris.data
    y = iris.target
    # create a Decision Tree classifier
    clf = DecisionTreeClassifier()
    # fit the model to the data
    clf.fit(X, y)
    return clf

from flask import Flask,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

clf = None

@app.route("/")
def hello_world():
    clf = initModel()
    return "<p>Model Initialised</p>"

@app.route('/predict', methods=['POST'])
def predict():
    print("request comes")
    if (request.headers.get('Content-Type') == 'application/json'):
        json = request.get_json()
        print(json)
        return "got"
    else:
        return 'Content-Type not supported!'


app.run(debug=True,port=5500)


# args = list(map(float,sys.argv[1:]))
# print(f"These are my arguments {args}")

# print("Model initialising")
# clf = initModel()
# print("Model loaded successfully")


# print("Now i am predicting ")

# predicted = clf.predict([args])

# print(f"This is the predicted value {predicted}")


# clf = initModel()
# new_observation = [[5.2, 3.1, 4.2, 1.5]] # a new observation to predict
# prediction = clf.predict(new_observation)

# print("Prediction for the new observation:", prediction)