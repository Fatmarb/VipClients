from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load("vip_prediction.pkl")
scaler = joblib.load("scaler.pkl")

# Liste des features dans le bon ordre
features = ['Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        values = [float(request.form.get(f, 0)) for f in features]
        scaled = scaler.transform([values])
        proba = model.predict_proba(scaled)[0][1]
        prediction = int(proba >= 0.5)
        return render_template("index.html", proba=round(proba, 2), prediction=prediction)

    return render_template("index.html", proba=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
