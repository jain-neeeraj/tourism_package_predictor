import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="neeraj-jain/turism-package-prediction", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Purchase Prediction
st.title("Wellness Tourism Package Purchase Prediction App")
st.write("This app predicts whether a customer will purchase the newly introduced Wellness Tourism Package.")
st.write("Please enter the customer details below to get a prediction.")

# Collect user input based on the problem statement features
Age = st.number_input("Age (customer's age)", min_value=18, max_value=90, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier (1, 2, or 3)", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government Sector"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating (1-5)", [1.0, 2.0, 3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (annually)", min_value=0, max_value=50, value=5)
Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # 0: No, 1: Yes
OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # 0: No, 1: Yes
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "CEO"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)

# Create DataFrame from inputs
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold (if used by the model)
classification_threshold = 0.45

# Predict button
if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Wellness Tourism Package" if prediction == 1 else "not purchase the Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
