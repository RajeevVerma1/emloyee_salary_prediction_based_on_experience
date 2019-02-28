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
        data = request.form['experience']
        if int(data) < 0:
            output = 'Enter Valid Year Experience'
        else:
            prediction = model.predict([[np.array(data, dtype=np.float64)]])
            output = prediction[0]
        return jsonify(output)

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port=5000, debug=True)
