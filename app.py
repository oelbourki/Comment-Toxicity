import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

with open('model.pkl','rb') as f:
	model = pickle.load(f)
with open('vectorizer.pkl','rb') as f:
	vectorizer = pickle.load(f)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
	print("prefict")
	data = request.json['data']
	new_data = vectorizer.transform(np.array(list(data.values())).reshape(1,-1))
	output = model.predict(new_data)
	return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
	data = [float(x) for x in request.form.values()]
	final_input = vectorizer.transform(np.array(data).reshape(1, -1))
	output = model.predict(final_input)[0]
	return render_template("home.html", prediction_text='The house price prediction is {}'.format(output))


if __name__ == "__main__":
	app.run(debug=True)