from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fraud_detection_model.pkl", "rb"))

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        step = float(request.form["step"])
        type_val = float(request.form["type"])
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])
        isFlaggedFraud = float(request.form["isFlaggedFraud"])

        features = np.array([[step, type_val, amount,
                              oldbalanceOrg, newbalanceOrig,
                              oldbalanceDest, newbalanceDest,
                              isFlaggedFraud]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Fraud"
        else:
            result = "Not Fraud"

        return render_template("submit.html", prediction_text=result)

    except Exception as e:
        return render_template("submit.html", prediction_text=str(e))


if __name__ == "__main__":
    app.run(debug=True)
