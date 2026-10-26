
import pandas as pd

RAW_FEATURES = [
    "Age",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Activity_Level",
    "Sugar_Level",
    "Cholesterol",
    "Goal",
]

TARGET = "Diet"


def add_engineered_features(df):
    df = df.copy()

    # Age group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, float("inf")],
        labels=["Under 18", "18-30", "31-45", "46-60", "Above 60"],
        right=True,
    ).astype(str)

    # BMI category
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    ).astype(str)

    # Weight-to-height ratio
    height_m = df["Height_cm"] / 100
    df["Weight_Height_Ratio"] = df["Weight_kg"] / height_m.replace(0, float("nan"))

    # Age and BMI interaction
    df["Age_BMI_Interaction"] = df["Age"] * df["BMI"]

    # Activity and goal combination
    df["Activity_Goal_Combination"] = (
        df["Activity_Level"].astype(str)
        + "_"
        + df["Goal"].astype(str)
    )

    return df