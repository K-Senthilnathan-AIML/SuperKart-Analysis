import os

os.makedirs("frontend", exist_ok=True)

app_code = import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000")
st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒")
st.title("SuperKart Sales Predictor")
st.caption("Estimated Product Store Sales Total")

def show_api_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"Prediction request failed ({response.status_code}): {detail}")

with st.form("single"):
    product_id = st.text_input("Product ID", "FD6114")
    product_type = st.selectbox("Product type", ["Frozen Foods", "Dairy", "Snack Foods", "Fruits and Vegetables", "Household"])
    weight = st.number_input("Product weight", min_value=0.0, value=12.66)
    sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular"])
    area = st.number_input("Allocated area", min_value=0.0, value=.027, format="%.3f")
    mrp = st.number_input("MRP", min_value=0.0, value=117.08)
    year = st.number_input("Store establishment year", min_value=1900, max_value=2025, value=2009)
    size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    submit = st.form_submit_button("Predict sales")
if submit:
    record = {"Product_Id": product_id, "Product_Type": product_type, "Product_Weight": weight,
              "Product_Sugar_Content": sugar, "Product_Allocated_Area": area, "Product_MRP": mrp,
              "Store_Establishment_Year": year, "Store_Size": size, "Store_Location_City_Type": city, "Store_Type": store_type}
    try:
        response = requests.post(f"{API_URL}/predict", json=record, timeout=20)
        if response.ok:
            st.metric("Predicted sales total", f"{response.json()['predictions'][0]:,.2f}")
        else:
            show_api_error(response)
    except requests.RequestException as error:
        st.error(f"Cannot reach the API at {API_URL}: {error}")

st.divider()
st.subheader("Batch inference")
file = st.file_uploader("Upload a source-format or engineered-format CSV", type="csv")
if file and st.button("Run batch prediction"):
    try:
        response = requests.post(f"{API_URL}/predict-batch", files={"file": (file.name, file.getvalue(), "text/csv")}, timeout=60)
        if response.ok:
            st.download_button("Download predictions", response.content, "predictions.csv", "text/csv")
        else:
            show_api_error(response)
    except requests.RequestException as error:
        st.error(f"Batch prediction failed: {error}")

with open("frontend/streamlit_app.py", "w") as f:
    f.write(app_code)

print("frontend/streamlit_app.py has been successfully written.")
