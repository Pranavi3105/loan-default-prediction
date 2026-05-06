from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")
features = joblib.load("feature_names.pkl")
purpose_map = {
    "credit_card": 0,
    "debt_consolidation": 1,
    "educational": 2,
    "home_improvement": 3,
    "major_purchase": 4,
    "small_business": 5
}
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    data["purpose"] = purpose_map[data["purpose"]]
    values = [float(data[f]) for f in features]
    final = pd.DataFrame([values], columns=features)
    prediction = model.predict(final)[0]
    probability = model.predict_proba(final)[0][1]
    importance = model.feature_importances_

# convert numpy int32 → python int
    feature_imp = {features[i]: int(importance[i]) for i in range(len(features))}
    if prediction == 1:
        result = f"Borrower likely to DEFAULT (Risk: {probability*100:.2f}%)"
    else:
        result = f"Borrower likely to REPAY (Risk: {probability*100:.2f}%)"
    return render_template(
        "index.html",
        prediction_text=result,
        importance=feature_imp
    )
if __name__ == "__main__":
    app.run(debug=True)