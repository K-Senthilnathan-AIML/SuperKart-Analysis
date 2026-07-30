import io
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

MODEL_PATH = os.getenv("MODEL_PATH", "models/superkart_random_forest_production.joblib")
PORT = int(os.getenv("PORT", "5000"))
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.errorhandler(ValueError)
@app.errorhandler(BadRequest)
def bad_input(error):
    return jsonify(error="Invalid prediction input", detail=str(error)), 400

@app.get("/health")
def health():
    return jsonify(status="ok", model_path=MODEL_PATH)

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    predictions = model.predict(engineer_features(pd.DataFrame(records))).tolist()
    return jsonify(predictions=predictions, record_count=len(predictions))

@app.post("/predict-batch")
def predict_batch():
    if "file" not in request.files:
        raise BadRequest("Attach a CSV under the multipart field name 'file'.")
    uploaded = request.files["file"]
    if not uploaded.filename or not uploaded.filename.lower().endswith(".csv"):
        raise BadRequest("The uploaded file must be a CSV.")
    frame = pd.read_csv(io.BytesIO(uploaded.read()))
    result = frame.copy()
    result["Predicted_Product_Store_Sales_Total"] = model.predict(engineer_features(frame))
    return app.response_class(result.to_csv(index=False), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=predictions.csv"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
