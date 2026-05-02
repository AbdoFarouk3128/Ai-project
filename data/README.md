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
*   **Clipping**: Extreme outliers were clipped to the lower and upper bounds to preserve data volume while reducing noise.

## 🔄 3. Feature Transformation & Scaling
*   **Target Encoding**: Converted the target variable `Heart Disease` into `0` (No) and `1` (Yes).
*   **Hybrid Categorical Encoding**:
    *   **Label Encoding**: Applied to ordinal features (`Chest pain type`, `EKG results`, `Slope of ST`, `Thallium`) to maintain their logical medical rank.
    *   **One-Hot Encoding**: Transformed nominal features (`Gender`, `work_type`, `smoking_status`) into binary columns for model stability.
*   **Standard Scaling**: Applied `StandardScaler` to numerical columns after encoding to ensure all features are on the same scale.

## ⚖️ 4. Data Balancing (SMOTE) & Integrity
*   **Class Imbalance**: The original dataset (224 rows) was imbalanced, which could bias the model toward "Healthy" predictions.
*   **SMOTE Technique**: Applied **Synthetic Minority Over-sampling Technique** to the training set, increasing the sample size to **248 rows** to achieve a 50/50 balance.
*   **Data Integrity**: After the SMOTE process, the resulting **NumPy arrays** were explicitly converted back into **Pandas DataFrames** to retain feature names and structure.

## 🛡️ 5. Data Leakage Prevention
*   **Strict Separation**: Maintained a clear boundary between training and testing data to ensure unbiased evaluation.
*   **Fit vs. Transform**: We strictly performed `.fit()` only on the training data. The testing data was processed using `.transform()` only, utilizing parameters (means, medians, and scales) learned exclusively from the training set.
*   **Consistent Bounds**: Outlier clipping bounds were calculated from the training set and applied to the test set to maintain a consistent feature space.

## 📁 6. Final Output
*   **Cleaned Files**:
    *   [`Modified_Cleaned_Train_Data.csv`](./Modified_Cleaned_Train_Data.csv): Balanced, scaled, and ready for model training.
    *   [`Modified_Cleaned_Test_Data.csv`](./Modified_Cleaned_Test_Data.csv): Fully preprocessed for final evaluation.

---
> **Note**: The full preprocessing logic is implemented in [`Preprocessing.py`](./Preprocessing.py). Use the modified CSV files for training the Classification models.