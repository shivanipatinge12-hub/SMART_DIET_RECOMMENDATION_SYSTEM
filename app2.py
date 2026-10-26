from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from features import RAW_FEATURES, add_engineered_features

st.set_page_config(
    page_title="Smart Diet Recommendation System",
    page_icon="🥗",
    layout="wide"
)

MODEL_PATH = Path("diet_model.joblib")
DATA_PATH = Path("diet_recommendation_dataset_1000.csv")

st.title("Smart Diet Recommendation System")
st.write(
    "An educational machine-learning demo that predicts a diet label from "
    "the input patterns learned from the supplied dataset."
)

st.warning(
    "This application is for educational demonstration only. It does not provide "
    "medical advice or validate whether a diet is safe for a person's health. "
    "Do not use its predictions to manage diabetes, cholesterol, weight, or other "
    "medical conditions. Consult a qualified healthcare professional for personal advice."
)

if not MODEL_PATH.exists():
    st.error("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Read available codes from the dataset where possible, rather than inventing
# meanings for encoded categorical values.
def get_options(column, fallback):
    if DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)
            if column in data.columns:
                vals = data[column].dropna().unique().tolist()
                vals = sorted(vals, key=lambda v: str(v))
                if vals:
                    return vals
        except Exception:
            pass
    return fallback

gender_options = get_options("Gender", [0, 1])
activity_options = get_options("Activity_Level", [0, 1, 2])
goal_options = get_options("Goal", [0, 1, 2])

st.subheader("Enter the input features")
st.caption(
    "For coded fields such as Gender, Activity_Level, and Goal, choose the code "
    "used in your dataset. Update the labels to their real meanings if you know them."
)

with st.form("diet_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender (dataset code)", gender_options)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=1.0)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=350.0, value=65.0, step=1.0)
        bmi = st.number_input("BMI (use dataset's definition)", min_value=5.0, max_value=100.0, value=23.9, step=0.1)
        activity = st.selectbox("Activity level (dataset code)", activity_options)
    with col3:
        sugar = st.number_input("Sugar level (use dataset units)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
        cholesterol = st.number_input("Cholesterol (use dataset units)", min_value=0.0, max_value=1000.0, value=180.0, step=1.0)
        goal = st.selectbox("Goal (dataset code)", goal_options)

    submitted = st.form_submit_button("Predict Diet Label", type="primary")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi,
        "Activity_Level": activity,
        "Sugar_Level": sugar,
        "Cholesterol": cholesterol,
        "Goal": goal
    }])

    engineered_df = add_engineered_features(input_df)
    prediction = model.predict(engineered_df)[0]

    st.subheader("Model output")
    st.success(f"Predicted dataset label: **{prediction}**")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(engineered_df)[0]
            probability_df = pd.DataFrame({
                "Diet label": model.classes_,
                "Model score": probabilities
            }).sort_values("Model score", ascending=False)
            probability_df["Model score (%)"] = (
                probability_df["Model score"] * 100
            ).round(1)
            st.write("Class scores (not a medical confidence or safety estimate):")
            st.dataframe(
                probability_df[["Diet label", "Model score (%)"]],
                use_container_width=True,
                hide_index=True
            )
            st.bar_chart(probability_df.set_index("Diet label")["Model score (%)"])
        except Exception:
            st.info("Class scores are not available for this fitted model.")

    st.download_button(
        "Download prediction as CSV",
        data=pd.DataFrame([{
            **input_df.iloc[0].to_dict(),
            "Predicted_Diet_Label": prediction
        }]).to_csv(index=False).encode("utf-8"),
        file_name="diet_prediction.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("About this model")
st.write(
    "The training workflow compares classification models using stratified cross-validation, "
    "performs feature selection within the training pipeline, applies SMOTE only to training "
    "folds, tunes the selected model, and evaluates it on a separate holdout test set."
)
st.write(
    "The model learns to reproduce the `Diet` labels in your CSV. If the dataset is synthetic, "
    "small, biased, or contains labels assigned by arbitrary rules, evaluation scores may not "
    "reflect real-world usefulness."
)
