# Diabetes Classification Prediction using Machine Learning

A machine learning project that predicts whether a patient is at risk of diabetes based on health survey data. This project explores how well common health indicators such as BMI, blood pressure, and age, can be used to identify diabetic patients.

---

## About the Project

Diabetes is a chronic disease that is often hard to detect early because symptoms develop slowly. This project was built to test whether a machine learning model can assist early screening by learning patterns from existing patient health data.

**What this project covers:**
- Exploratory Data Analysis (EDA) — understanding the data before modeling
- Data preprocessing — preparing the data so models can learn from it effectively
- Training and comparing 3 machine learning models
- Identifying which health factors have the most influence on the prediction

---

## Built With

- Python
- pandas
- matplotlib
- scikit-learn
  
---

## Preprocessing & EDA

Before training, the dataset was explored and cleaned to ensure the models receive good-quality input.

During EDA, extreme values (outliers) were found.  For example, patients with very high BMI or unusually high blood pressure readings. To handle this, RobustScaler was used to normalize the data. Unlike standard scalers, RobustScaler is less sensitive to extreme values, so those data points don't disproportionately affect the model. Importantly, these outliers were kept rather than removed. Since this is medical data, extreme values most likely reflect real patient conditions not input errors. A patient with a very high BMI is a valid case the model should learn from, not discard.

---

## Model Comparison

Three models were trained and evaluated. All used RandomizedSearchCV, a method to automatically find the best settings for each model with n_jobs=2 to run efficiently on a standard CPU.

| No | Model               | Accuracy | Precision | Recall | F1-Score | 
|----|---------------------|----------|-----------|--------|----------|
| 1  | Linear SVC          | 74%      | 1.00 (class 1) | 76% | 0.73–0.76 |
| 2  | Logistic Regression | 74.5%    | 0.75      | 0.74   | 0.74     | 
| 3  | **Random Forest**   | **74.7%** | **0.75** | **0.75** | **0.75** | 
---

## Feature Importance

After training, the Random Forest model was analyzed to determine which health indicators contributed most to its predictions.

Top 3 most influential features:

1. **GenHlth** — The patient's self-reported general health score (scale 1–5). The strongest single predictor at ~23% importance.
2. **HighBP** — Whether the patient has high blood pressure (~21%). High blood pressure is a well-known risk factor closely associated with diabetes.
3. **BMI** — Body Mass Index (~14%). A standard measure of body weight relative to height, and a common diabetes risk indicator.

Lifestyle features such as smoking, fruit and vegetable intake, and physical activity ranked low in this dataset, not because they are medically irrelevant, but because they do not vary enough in this survey data to serve as strong predictors.

---

## Final Results (Random Forest)

| Metric    | No Diabetes (Class 0) | Diabetes (Class 1) |
|-----------|----------------------|---------------------|
| Precision | 0.77                 | 0.73                |
| Recall    | 0.69                 | 0.80                |
| F1-Score  | 0.73                 | 0.76                |
| **Overall Accuracy** | **74.7%** | |

The model shows higher recall for diabetic patients (Class 1), meaning it is better at correctly identifying patients who actually have diabetes. For a health screening context, this is the more critical direction to optimize.

---


### Dataset
 
This project uses the **Diabetes Health Indicators Dataset** published by Alex Teboul on Kaggle.
 
- **Source:** [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **File used:** diabetes_binary_5050split_health_indicators_BRFSS2015.csv
- **Origin:** CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey data
- **License:** CC0 — Public Domain (free to use without restriction)
> The dataset contains 253,680 survey responses with 21 health-related features derived from the CDC's annual telephone health survey across the United States.
