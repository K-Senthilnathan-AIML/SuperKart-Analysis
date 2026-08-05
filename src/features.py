import numpy as np
import pandas as pd

REFERENCE_YEAR = 2025
PERISHABLES = {"Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"}
NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL = [
    "Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
    "Store_Type", "Product_Id_char", "Product_Type_Category",
]
FEATURES = NUMERIC + CATEGORICAL

def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Product_Sugar_Content" in data:
        data["Product_Sugar_Content"] = data["Product_Sugar_Content"].replace({"reg": "Regular"})
    if "Product_Id_char" not in data and "Product_Id" in data:
        data["Product_Id_char"] = data["Product_Id"].astype(str).str[:2]
    if "Store_Age_Years" not in data and "Store_Establishment_Year" in data:
        data["Store_Age_Years"] = REFERENCE_YEAR - data["Store_Establishment_Year"]
    if "Product_Type_Category" not in data and "Product_Type" in data:
        data["Product_Type_Category"] = np.where(
            data["Product_Type"].isin(PERISHABLES), "Perishables", "Non Perishables"
        )
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return data[FEATURES]
