# 🩺 Heart Disease Prediction - Data Preprocessing Report

This project contains the preprocessing pipeline for heart disease classification. The goal was to clean the raw data and prepare it for machine learning models while maintaining medical data integrity.

---

## 🛠️ 1. Data Cleaning
*   **Removed ID**: Deleted the `id` column as it is a unique identifier and does not contribute to predictive power.
*   **Handling Missing Values**:
    *   **Numerical (`Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`)**: Imputed missing values using the **Median** to avoid the influence of extreme values.
    *   **Categorical (`Gender`, `work_type`, `smoking_status`, `Thallium`)**: Imputed missing values using the **Mode** (most frequent value).

## 📊 2. Outlier Management
*   **IQR Method**: Calculated the Interquartile Range (IQR) for continuous variables (`Blood Pressure`, `Cholesterol`, etc.).
*   **Clipping**: Extreme outliers were clipped to the lower and upper bounds. 
    *   *Note: Categorical data (like `Thallium`) was excluded from clipping to preserve its clinical categories.*

## 🔄 3. Feature Transformation & Scaling
*   **Label Encoding**: Converted the target variable `Heart Disease` into:
    *   `0`: No
    *   `1`: Yes
*   **One-Hot Encoding**: Transformed categorical features into binary columns with `handle_unknown='ignore'` for model stability.
*   **Standard Scaling**: Applied `StandardScaler` to numerical columns only.
    *   **Interpretation**: `0` = Average, `Positive` = Above Average, `Negative` = Below Average.

## ⚖️ 4. Data Balancing (SMOTE)
*   **Class Imbalance**: The original data had fewer disease cases.
*   **SMOTE Technique**: Applied to the training set to create synthetic examples of the minority class, ensuring the model is not biased toward "Healthy" results.

## 🎯 5. Feature Selection (RFE)
*   **Efficiency**: Used Recursive Feature Elimination (RFE) to select the **top 12 features**.
*   **Optimization**: This removed noise and focused the model on the most critical medical indicators for better accuracy.

## 📁 6. Final Output
*   **Data Splitting**: Ensured strict separation between Train and Test sets.
*   **Cleaned Files**:
    *   [`Modified_Cleaned_Train_Data.csv`](/Modified_Cleaned_Train_Data.csv) (Balanced & Optimized)
    *   [`Modified_Cleaned_Test_Data.csv`](/Modified_Cleaned_Test_Data.csv) (Ready for Evaluation)

---
> **Note for the Team**: Use the modified cleaned CSV files directly for model training. The preprocessing logic can be found in [`Preprocessing.py`](/Preprocessing.py).