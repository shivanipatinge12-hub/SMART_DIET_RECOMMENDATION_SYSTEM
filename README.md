# Smart Diet Recommendation System

## 📌 Project Overview
The Smart Diet Recommendation System is a machine learning-based application that predicts diet categories based on user-provided information such as age, gender, height, weight, BMI, activity level, sugar level, cholesterol level, and health goals.

The project uses Python, Scikit-learn, and Streamlit to build an interactive application that demonstrates data preprocessing, feature engineering, model training, evaluation, and prediction.

**Note:** This project is developed for educational purposes. Its predictions are not medically validated and should not be used as a substitute for professional dietary or medical advice.
## 🎯 Objectives

- Analyze user-related health and lifestyle features.
- Perform data cleaning and exploratory data analysis (EDA).
- Engineer additional features to support model training.
- Train and compare multiple machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Build an interactive interface using Streamlit.

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Imbalanced Data Handling:** Imbalanced-learn (SMOTE)
- **Model Persistence:** Joblib
- **Web Application:** Streamlit

## 📊 Dataset
The dataset contains input features related to user characteristics and lifestyle, along with a target column representing the diet category.

### Input Features
- Age
- Gender
- Height (cm)
- Weight (kg)
- BMI
- Activity Level
- Sugar Level
- Cholesterol
- Goal

### Target Variable
- **Diet:** The diet category predicted by the classification model.

## ⚙️ Machine Learning Workflow
The project follows these steps:

1. **Data Loading:** Load the dataset using Pandas.
2. **Data Cleaning:** Handle invalid values, missing target labels, and duplicate records.
3. **Exploratory Data Analysis:** Analyze diet category distribution, BMI distribution, and numerical feature correlations.
4. **Feature Engineering:** Create additional features such as age groups, BMI categories, weight-to-height ratio, age-BMI interaction, and activity-goal combinations.
5. **Train-Test Split:** Divide the data into training and testing sets using stratified sampling.
6. **Data Preprocessing:** Handle missing values, scale numerical features, and encode categorical features.
7. **Feature Selection:** Use mutual information to select a subset of features.
8. **Class Imbalance Handling:** Apply SMOTE to the training data within the model pipeline.
9. **Model Comparison:** Compare Logistic Regression, Random Forest, and Extra Trees classifiers using stratified cross-validation.
10. **Hyperparameter Tuning:** Use GridSearchCV to tune the selected classifier.
11. **Model Evaluation:** Evaluate predictions using accuracy, precision, recall, F1-score, classification report, and confusion matrix.
12. **Model Saving:** Save the trained pipeline using Joblib.
13. **Prediction:** Load the saved model in the Streamlit application to predict diet categories.

## 🤖 Machine Learning Algorithms
The project compares the following classification algorithms:

- **Logistic Regression:** A classification algorithm used as a baseline model.
- **Random Forest Classifier:** An ensemble learning algorithm that combines multiple decision trees.
- **Extra Trees Classifier:** An ensemble algorithm that introduces additional randomization when building decision trees.

The final model is selected based on cross-validation performance. Actual performance depends on the dataset and should be assessed using the evaluation results generated during training.

## 📁 Project Structure

SMART_DIET_RECOMMENDATION_SYSTEM/
│
├── app2.py
├── features.py
├── train_model.py
├── requirements.txt
├── README.md

### 2. Navigate to the Project Folder

```bash
cd SMART_DIET_RECOMMENDATION_SYSTEM
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Add the Dataset

Place your dataset CSV in the project folder and ensure its filename matches the `DATASET_PATH` configured in `train_model.py`.

If the dataset is not included in the repository, you must obtain it separately before training.

### 6. Train the Machine Learning Model
```bash
python train_model.py
``
This script trains and evaluates the candidate models and, if training completes successfully, saves the selected model as `diet_model.joblib`.

### 7. Launch the Streamlit Application
```bash
streamlit run app2.py
```
Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 📈 Model Evaluation
The training script reports the following metrics:

- **Accuracy:** The proportion of test predictions that are correct.
- **Precision:** How many predictions for a class are correct, calculated using a weighted average across classes.
- **Recall:** How many actual examples are correctly identified, calculated using a weighted average across classes.
- **F1-Score:** The harmonic mean of precision and recall, calculated using a weighted average across classes.
- **Confusion Matrix:** A visual comparison of actual and predicted classes.
The actual metric values are generated when the model is trained on the dataset. No performance score is claimed in this README.

## ⚠️ Limitations
- Model performance depends on dataset quality, size, and class distribution.
- Predictions reflect patterns learned from the training data and may not generalize to real-world users.
- The model is not clinically validated.
- Predicted diet categories are not personalized medical advice.
- Users should consult a qualified healthcare professional or registered dietitian for individual dietary guidance.

## 🔮 Future Improvements
- Improve validation using larger, higher-quality datasets.
- Evaluate additional classification algorithms.
- Improve feature selection and class imbalance handling.
- Add more detailed model evaluation and explainability.
- Deploy the application to a suitable hosting platform.
- Incorporate medically reviewed dietary information where appropriate.

## 👩‍💻 Author
Shivani Patinge
Engineering Graduate | Python | SQL | Data Analysis | Machine Learning | Deep Learning
