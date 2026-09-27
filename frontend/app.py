
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://host.docker.internal:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# =========================
# Online Prediction
# =========================

st.subheader("Online Prediction")

# Collect user input for SuperKart features
item_weight = st.number_input(
    "Item Weight",
    min_value=0.0,
    value=10.0
)

item_sugar_content = st.selectbox(
    "Item Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

item_visibility = st.number_input(
    "Item Visibility",
    min_value=0.0,
    value=0.05,
    format="%.4f"
)

item_type = st.selectbox(
    "Item Type",
    [
        "Dairy",
        "Soft Drinks",
        "Meat",
        "Fruits and Vegetables",
        "Household",
        "Baking Goods",
        "Snack Foods",
        "Frozen Foods",
        "Breakfast",
        "Health and Hygiene",
        "Hard Drinks",
        "Canned",
        "Breads",
        "Starchy Foods",
        "Others",
        "Seafood"
    ]
)

item_mrp = st.number_input(
    "Item MRP",
    min_value=0.0,
    value=100.0
)

outlet_size = st.selectbox(
    "Outlet Size",
    ["Small", "Medium", "High"]
)

outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

outlet_type = st.selectbox(
    "Outlet Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
        "Grocery Store"
    ]
)

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    "Item_Weight": item_weight,
    "Item_Sugar_Content": item_sugar_content,
    "Item_Visibility": item_visibility,
    "Item_Type": item_type,
    "Item_MRP": item_mrp,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location_type,
    "Outlet_Type": outlet_type
}])

# Show input data
st.write("Input Data")
st.dataframe(input_data)

# Make prediction when Predict button is clicked
if st.button("Predict", type="primary"):

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=input_data.to_dict(orient="records")[0]
        )

        if response.status_code == 200:

            prediction = response.json()

            st.success("Prediction completed successfully!")

            st.write(prediction)

        else:
            st.error(
                f"Prediction API returned error: "
                f"{response.status_code} - {response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")


# =========================
# Batch Prediction
# =========================

st.subheader("Batch Prediction")

st.write(
    "Upload a CSV file containing the SuperKart input features "
    "for multiple products."
)

# Allow users to upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction",
    type=["csv"]
)

if uploaded_file is not None:

    # Preview uploaded CSV
    try:
        batch_data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(batch_data)

        # Reset file pointer before sending it to backend
        uploaded_file.seek(0)

    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")

    # Make batch prediction
    if st.button("Predict Batch", type="primary"):

        try:
            uploaded_file.seek(0)

            response = requests.post(
                f"{BACKEND_URL}/v1/batch",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }
            )

            if response.status_code == 200:

                predictions = response.json()

                st.success("Batch predictions completed!")

                st.write(predictions)

            else:

                st.error(
                    f"Batch prediction API returned error: "
                    f"{response.status_code} - {response.text}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to the prediction API: {e}"
            )
