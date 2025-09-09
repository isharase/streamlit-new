import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("campaign_predictor.pkl")

st.title("📊 Campaign Performance Predictor")

st.markdown("Predict impressions, clicks, likes, shares, and purchases for a campaign based on its settings.")

# ----------------------------
# User Input
# ----------------------------
target_gender = st.selectbox("Target Gender", ["Male", "Female"])
target_age_group = st.selectbox("Target Age Group", ["18-24", "25-34", "35-44"])
target_interests = st.selectbox("Target Interests", ["Sports", "Fashion", "Tech", "Gaming", "Food"])
duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, value=10)
total_budget = st.number_input("Total Budget", min_value=100, max_value=1000000, value=5000)
ad_platform = st.selectbox("Ad Platform", ["Facebook", "Instagram"])
ad_type = st.selectbox("Ad Type", ["Image", "Video", "Carousel"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Performance"):
    # Prepare input for model
    input_df = pd.DataFrame([{
        "target_gender": target_gender,
        "target_age_group": target_age_group,
        "target_interests": target_interests,
        "duration_days": duration_days,
        "total_budget": total_budget,
        "ad_platform": ad_platform,
        "ad_type": ad_type,
        "time_of_day": time_of_day
    }])

    # Predict
    prediction = model.predict(input_df)

    # Convert to DataFrame
    results_df = pd.DataFrame(prediction, columns=["Impressions", "Clicks", "Comments", "Likes", "Shares", "Purchases"])

    # Display results
    st.subheader("Predicted Campaign Performance")
    st.dataframe(results_df)

    # Bar chart
    st.subheader("Visual Representation")
    st.bar_chart(results_df.T)
