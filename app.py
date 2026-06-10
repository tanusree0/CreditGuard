from flask import Flask
from flask import render_template
from flask import request
from flask import send_file

import pandas as pd
import joblib
import numpy as np

from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/credit.csv")

# Load trained model
model = joblib.load(
    "artifacts/model.pkl"
)

scaler = joblib.load(
    "artifacts/scaler.pkl"
)

# Store latest prediction
latest_prediction = ""
latest_confidence = 0

@app.route("/")
def home():

    return render_template(
        "index.html",
        result=None
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    global latest_prediction
    global latest_confidence

    age = int(request.form["age"])

    income = float(
        request.form["income"]
    )

    loan = float(
        request.form["loan"]
    )

    score = int(
        request.form["score"]
    )

    years = int(
        request.form["years"]
    )

    data = np.array([
        [
            age,
            income,
            loan,
            score,
            years
        ]
    ])

    data = scaler.transform(data)

    result = model.predict(data)[0]

    confidence = max(
        model.predict_proba(data)[0]
    )

    if result == 0:
        prediction = "Safe Customer ✅"
    else:
        prediction = "Risky Customer ⚠️"

    latest_prediction = prediction

    latest_confidence = round(
        confidence * 100,
        2
    )

    return render_template(
        "index.html",
        result=prediction,
        confidence=latest_confidence
    )


@app.route("/download")
def download():

    filename = "Credit_Risk_Report.pdf"

    pdf = canvas.Canvas(
        filename
    )

    pdf.setTitle(
        "Credit Risk Report"
    )

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        150,
        800,
        "Credit Risk Analysis Report"
    )

    pdf.setFont(
        "Helvetica",
        14
    )

    pdf.drawString(
        100,
        740,
        f"Prediction: {latest_prediction}"
    )

    pdf.drawString(
        100,
        700,
        f"Confidence: {latest_confidence}%"
    )

    pdf.drawString(
        100,
        660,
        "Model: Random Forest Classifier"
    )

    pdf.drawString(
        100,
        620,
        "Generated Using Flask + Machine Learning"
    )

    pdf.save()

    return send_file(
        filename,
        as_attachment=True
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )