\# 🩺 Heart Disease Prediction - Data Preprocessing Report



This project contains the preprocessing pipeline for heart disease classification. The goal was to clean the raw data and prepare it for machine learning models while maintaining medical data integrity.



\---



## 🛠️ 1. Data Cleaning

\*   \*\*Removed ID\*\*: Deleted the `id` column as it is a unique identifier and does not contribute to predictive power.

\*   \*\*Handling Missing Values\*\*:

&#x20;   \*   \*\*Numerical (`Age`)\*\*: Imputed missing values using the \*\*Median\*\* to avoid the influence of extreme values.

&#x20;   \*   \*\*Categorical (`Gender`, `work\_type`, `Smoking\_status`)\*\*: Imputed missing values using the \*\*Mode\*\* (most frequent value).



## 📊 2. Outlier Management

\*   \*\*IQR Method\*\*: Calculated the Interquartile Range (IQR) for continuous variables (`Blood Pressure`, `Cholesterol`, etc.).

\*   \*\*Clipping\*\*: Extreme outliers were clipped to the lower and upper bounds. 

&#x20;   \*   \*Note: `Thallium` was excluded from clipping to preserve its specific clinical categories (3, 6, 7).\*



## 🔄 3. Feature Transformation \& Scaling

\*   \*\*Label Encoding\*\*: Converted the target variable `Heart Disease` into:

&#x20;   \*   `0`: No

&#x20;   \*   `1`: Yes

\*   \*\*One-Hot Encoding\*\*: Transformed categorical features (`Gender`, `work\_type`, `Smoking\_status`) into binary columns for model compatibility.

\*   \*\*Standard Scaling\*\*: Applied `StandardScaler` to numerical columns (`Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`).

&#x20;   \*   \*\*Interpretation\*\*: 

&#x20;       \*   `0` = Average value.

&#x20;       \*   `Positive` = Above Average.

&#x20;       \*   `Negative` = Below Average.



## 📁 4. Final Output
*   **Data Splitting**: Ensured strict separation between Train and Test sets.
*   **Cleaned Files** (Click to open):
    *   [Cleaned_Train_Data.csv](./data/Cleaned_Train_Data.csv)
    *   [Cleaned_Test_Data.csv](./data/Cleaned_Test_Data.csv)

---
> **Note for the Team**: Use the cleaned CSV files directly for model training. The preprocessing logic can be found in [Preprocessing.py](./data/Preprocessing.py).
