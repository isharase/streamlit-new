import streamlit as st
import pandas as pd
import joblib
import altair as alt

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("campaign_predictor.pkl")

st.set_page_config(page_title="Campaign Predictor", layout="wide")

st.title("📊 Campaign Performance Predictor")
st.markdown("Predict impressions, clicks, likes, shares, and purchases for a campaign based on its settings.")

# ----------------------------
# User Input
# ----------------------------
with st.sidebar:
    st.header("⚙️ Campaign Settings")
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

    results_df = results_df.round().astype(int)

    # ----------------------------
    # Show as Metric Cards
    # ----------------------------
    st.subheader("📌 Predicted Campaign Performance")

    cols = st.columns(3)
    for i, col in enumerate(results_df.columns):
        with cols[i % 3]:
            st.metric(label=col, value=f"{results_df.iloc[0, i]:.2f}")

    # ----------------------------
    # Horizontal Bar Chart
    # ----------------------------
    st.subheader("📈 Visual Representation")

    chart_data = results_df.T.reset_index()
    chart_data.columns = ["Metric", "Value"]

    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x="Value",
        y=alt.Y("Metric", sort="-x"),
        tooltip=["Metric", "Value"]
    ).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

    # ----------------------------
    # Normalized % View
    # ----------------------------
    st.subheader("📊 Contribution (Normalized %)")

    normalized = (results_df.T / results_df.T.sum()) * 100
    normalized_data = normalized.reset_index()
    normalized_data.columns = ["Metric", "Percentage"]

    chart2 = alt.Chart(normalized_data).mark_arc(innerRadius=50).encode(
        theta="Percentage",
        color="Metric",
        tooltip=["Metric", "Percentage"]
    ).properties(width=500, height=500)

    st.altair_chart(chart2, use_container_width=True)
