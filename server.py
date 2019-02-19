import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("output.html")
# Load the model
model = pickle.load(open(r'C:\Users\I506660\Documents\GitHub\flask-salary-predictor\model.pkl','rb'))

@app.route('/result',methods=['POST'])
def predict():
    if request.form['submit_button'] == 'submit':
    # Get the data from the POST request.
        data = request.form['experience']
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data, dtype=np.float64)]])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port=5000, debug=True)
