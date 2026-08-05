import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Predictor")
st.caption("Predict Product Store Sales Total using the deployed Random Forest pipeline.")

def show_api_error(response: requests.Response) -> None:
    """Show a useful backend/API error without exposing a raw traceback."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"Prediction request failed ({response.status_code}): {detail}")

with st.form("single_prediction_form"):
    st.subheader("Online inference — single prediction")
    product_id = st.text_input("Product ID", value="FD6114")
    product_type = st.selectbox(
        "Product type",
        ["Frozen Foods", "Dairy", "Snack Foods", "Fruits and Vegetables", "Household"],
    )
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.66, step=0.01)
    sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Allocated area", min_value=0.0, value=0.027, step=0.001, format="%.3f")
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, step=0.01)
    store_year = st.number_input("Store establishment year", min_value=1980, max_value=2025, value=2009)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city_tier = st.selectbox("Store location city type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox(
        "Store type",
        ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
    )
    submitted = st.form_submit_button("Predict sales")

if submitted:
    record = {
        "Product_Id": product_id,
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_tier,
        "Store_Type": store_type,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=record, timeout=20)
        if response.ok:
            prediction = response.json()["predictions"][0]
            st.metric("Predicted Product Store Sales Total", f"{prediction:,.2f}")
        else:
            show_api_error(response)
    except requests.RequestException as error:
        st.error(f"Cannot reach the prediction API at {API_URL}. Details: {error}")

st.divider()
st.subheader("Batch inference")
st.caption("Upload the supplied engineered batch schema or the raw source schema without the target column.")
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_file is not None and st.button("Run batch prediction"):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_URL}/predict-batch", files=files, timeout=60)
        if response.ok:
            st.success("Batch prediction completed.")
            st.download_button(
                label="Download predictions CSV",
                data=response.content,
                file_name="superkart_predictions.csv",
                mime="text/csv",
            )
        else:
            show_api_error(response)
    except requests.RequestException as error:
        st.error(f"Batch prediction failed. Details: {error}")

with st.sidebar:
    st.subheader("Configuration")
    st.code(f"API_URL={API_URL}")
    if st.button("Check backend health"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=10)
            st.json(health.json())
        except requests.RequestException as error:
            st.error(f"Backend health check failed: {error}")
