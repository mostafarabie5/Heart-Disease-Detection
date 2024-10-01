from app_ml import app, predictModel
from flask import render_template, request
import pandas as pd
import numpy as np

# @app.route('/')


def home_page():
    return render_template('home.html')


app.add_url_rule('/', 'home', home_page)


@app.route('/predict', methods=['POST'])
def predict():

    algorithm = request.form['algorithm']

    data = {
        "Age": int(request.form["Age"]),
        "Sex": request.form["Sex"],
        "ChestPainType": request.form["ChestPainType"],
        "RestingBP": int(request.form["RestingBP"]),
        "Cholesterol": int(request.form["Cholesterol"]),
        "FastingBS": int(request.form["FastingBS"]),
        "RestingECG": request.form["RestingECG"],
        "MaxHR": int(request.form["MaxHR"]),
        "ExerciseAngina": request.form["ExerciseAngina"],
        "Oldpeak": float(request.form["Oldpeak"]),
        "ST_Slope": request.form["ST_Slope"]
    }

    df = pd.DataFrame([data], columns=data.keys())
    value = predictModel.predictModel(algorithm, df)
    prediction = None
    result_color = None
    
    if (value[0] == 1):
        prediction = "With Heart Disease"
        result_color = "positive"
    else:
        prediction = "Healthy"
        result_color = "negative"

    return render_template('result.html', prediction=prediction, result=result_color, machine=algorithm)
