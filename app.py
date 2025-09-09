import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Campaign Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom Dark Theme CSS
# ----------------------------
st.markdown("""
    <style>
    /* Main background */
    body, .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111;
    }
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954; /* green accent */
    }
    /* Dataframe text */
    .css-1d391kg, .css-1offfwp, .css-1n76uvr {
        color: #FFFFFF !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #14833b;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("campaign_predictor.pkl")

st.title("📊 Campaign Performance Predictor")
st.markdown("Predict **impressions, clicks, likes, shares, and purchases** for a campaign based on its settings.")

# ----------------------------
# User Input
# ----------------------------
st.sidebar.header("⚙️ Campaign Settings")
target_gender = st.sidebar.selectbox("Target Gender", ["Male", "Female"])
target_age_group = st.sidebar.selectbox("Target Age Group", ["18-24", "25-34", "35-44"])
target_interests = st.sidebar.selectbox("Target Interests", ["Sports", "Fashion", "Tech", "Gaming", "Food"])
duration_days = st.sidebar.number_input("Duration (days)", min_value=1, max_value=60, value=10)
total_budget = st.sidebar.number_input("Total Budget", min_value=100, max_value=1000000, value=5000)
ad_platform = st.sidebar.selectbox("Ad Platform", ["Facebook", "Instagram"])
ad_type = st.sidebar.selectbox("Ad Type", ["Image", "Video", "Carousel"])
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🚀 Predict Performance"):
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

    # Round results to integers (since counts should be whole numbers)
    results_df = results_df.round().astype(int)

    # Display results
    st.subheader("📌 Predicted Campaign Performance")
    st.dataframe(results_df)

    # Bar chart - sorted
    st.subheader("📈 Visual Representation")
    st.bar_chart(results_df.T.sort_values(by=0, ascending=False))

    # Normalized %
    st.subheader("📊 Contribution (Normalized %)")
    normalized = (results_df.T / results_df.T.sum()) * 100
    st.bar_chart(normalized)
