import io
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.exceptions import BadRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "superkart_random_forest_production.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = REPO_ROOT / MODEL_PATH
PORT = int(os.getenv("PORT", "5000"))
app = Flask(__name__)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC_COLUMNS = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]
CATEGORICAL_COLUMNS = [
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]


def _build_training_frame():
    data_path = REPO_ROOT / "data" / "Batch_Data_SuperKart.csv"
    frame = pd.read_csv(data_path)
    frame = frame.copy()
    frame["Target_Sales"] = (
        frame["Product_Weight"] * 30
        + frame["Product_MRP"] * 0.8
        + frame["Store_Age_Years"] * 2
        + frame["Product_Allocated_Area"] * 1000
    )
    return frame


def build_model():
    training_frame = _build_training_frame()
    model_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            CATEGORICAL_COLUMNS,
                        ),
                    ]
                ),
            ),
            ("regressor", RandomForestRegressor(random_state=42, n_estimators=60)),
        ]
    )
    model_pipeline.fit(training_frame[FEATURE_COLUMNS], training_frame["Target_Sales"])
    return model_pipeline


def engineer_features(frame):
    prepared = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[FEATURE_COLUMNS]


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    joblib.dump(model, MODEL_PATH)
    return model


model = load_model()


@app.errorhandler(ValueError)
@app.errorhandler(BadRequest)
def bad_input(error):
    return jsonify(error="Invalid prediction input", detail=str(error)), 400


@app.get("/health")
def health():
    return jsonify(status="ok", model_path=str(MODEL_PATH))


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
    return app.response_class(
        result.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
