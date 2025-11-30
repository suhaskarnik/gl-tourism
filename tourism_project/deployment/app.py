import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

model_repo = "sam-vimes/tourism_model"
# Download and load the trained model
model_path = hf_hub_download(
    repo_id=model_repo, filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the wellness tourism package.
Please enter the customer and interaction details below to get a revenue prediction.
""")

age = st.number_input("Age", min_value=18, max_value=80, value=44)
typeOfContact = st.selectbox(
    "Type of Contact", ["Self Enquiry", "Company Invited"], index=0
)
cityTier = st.selectbox("City Tier", (1, 2, 3), index=0)
durationOfPitch = st.number_input("Duration of Pitch", 3, 140)
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
    index=0,
)
gender = st.selectbox("Gender", ["Female", "Male"], index=0)
numberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting", min_value=1, max_value=6, value=3
)
numberOfFollowUps = st.number_input(
    "Number of Follow Ups", min_value=0, max_value=7, value=1
)
productPitched = st.selectbox(
    "Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"], index=2
)
preferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
maritalStatus = st.selectbox(
    "Marital Status", ["Single", "Divorced", "Married", "Unmarried"], index=2
)
numberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=24, value=2)
passport = st.checkbox("Has Passport?", value=True)  # convert to int in pandas
pitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=4
)
ownCar = st.checkbox("Owns a Car?", value=True)  # convert to int in pandas
numberOfChildren = st.number_input(
    "Number of Children Visiting", min_value=0, max_value=4, value=0
)
designation = st.selectbox(
    "Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"], index=2
)
monthlyIncome = st.number_input("Monthly Income", min_value=900.0, max_value=108546.0)

# Assemble input into DataFrame
input_data = pd.DataFrame(
    [
        {
            "Age": float(age),
            "TypeofContact": typeOfContact,
            "CityTier": cityTier,
            "DurationOfPitch": float(durationOfPitch),
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": numberOfPersonVisiting,
            "NumberOfFollowUps": float(numberOfFollowUps),
            "ProductPitched": productPitched,
            "PreferredPropertyStar": float(preferredPropertyStar),
            "MaritalStatus": maritalStatus,
            "NumberOfTrips": float(numberOfTrips),
            "Passport": passport,
            "PitchSatisfactionScore": pitchSatisfactionScore,
            "OwnCar": ownCar,
            "NumberOfChildrenVisiting": float(numberOfChildren),
            "Designation": designation,
            "MonthlyIncome": monthlyIncome,
        }
    ]
)

# Predict button
if st.button("Predict Purchase of Wellness Package"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Prediction: **{prediction}**")
