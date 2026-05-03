# 🩺 Heart Disease Prediction - Data Preprocessing Report

This project contains an advanced preprocessing pipeline designed to clean, balance, and encode heart disease data while strictly maintaining **Medical Data Integrity**.

---

## 🛠️ 1. Data Cleaning & Imputation
*   **ID Removal**: The `id` column was dropped as it serves no predictive purpose.
*   **Numerical Imputation**: Used **Median** for `Age`, `BP`, `Cholesterol`, `Max HR`, and `ST depression` to stay robust against outliers.
*   **Categorical Imputation**: Used **Mode** (Most Frequent) for all categorical and ordinal features to maintain logical consistency[cite: 3].

## 📊 2. Outlier Management
*   **IQR Method**: Identified outliers using the Interquartile Range ($1.5 \times IQR$)[cite: 3].
*   **Feature Clipping**: Instead of removing rows, extreme values were **clipped** to the calculated upper and lower bounds. This preserves the sample size (224 rows) while neutralizing noise[cite: 3].

## ⚖️ 3. Advanced Balancing (SMOTENC)
*   **Class Imbalance**: Addressed the minority class by oversampling from **224 to 248 samples**[cite: 3].
*   **SMOTENC vs SMOTE**: We utilized **SMOTENC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous). This ensures that categorical features (like `Chest pain type`) remain as discrete categories rather than being averaged into nonsensical decimals[cite: 3].
*   **Integer Conversion**: Post-balancing, we applied **Rounding and Integer Casting** to `Age`, `BP`, `Cholesterol`, and `Max HR`. This ensures a realistic medical dataset (e.g., Age = 50 instead of 50.34).

## 🔄 4. Feature Encoding & Engineering
*   **Target Encoding**: Converted `Heart Disease` labels into `0` and `1` using `LabelEncoder`[cite: 3].
*   **Ordinal Management**: Features like `Chest pain type`, `EKG results`, and `Slope of ST` were kept as **Integers**. This allows the model to leverage the natural medical ranking (Order) of these features[cite: 3].
*   **One-Hot Encoding**: Transformed nominal features (`Gender`, `work_type`, `smoking_status`) into binary vectors to prevent the model from assuming an incorrect mathematical relationship between them[cite: 3].

## 🛡️ 5. Data Leakage Prevention
*   **Strict Separation**: All preprocessing parameters (Medians, Bounds, and Encoders) were **fitted on Training data only**[cite: 3].
*   **Test Alignment**: The test set was transformed using the training parameters and then **reindexed** to match the training feature space (22 columns) exactly, ensuring zero errors during model inference[cite: 3].

## 📁 6. Final Output
*   **[`Modified_Cleaned_Train_Data.csv`](./Cleaned_Train_Data.csv)**: Balanced, integer-aligned, and ready for model training[cite: 3].
*   **[`Modified_Cleaned_Test_Data.csv`](./Cleaned_Test_Data.csv)**: Preprocessed test set aligned with the training schema[cite: 3].

---
> **Developer Note**: The full preprocessing logic is implemented in [`ModPreprocessing.py`](./ModPreprocessing.py). The resulting data is optimized for high-performance classifiers such as Random Forest and XGBoost.