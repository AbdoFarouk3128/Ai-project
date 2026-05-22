import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Heart Disease Diagnostic", layout="wide")

# --- 1. Train Models on Startup ---
@st.cache_resource
def load_and_train_models():
    # Load Data
    train = pd.read_csv('data/Cleaned_Train_Data.csv')
    
    # Drop useless columns
    useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 
               'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 
               'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
    train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
    
    X_train = train.drop('Heart Disease', axis=1)
    y_train = train['Heart Disease']
    
    # Scale Numerical Columns (required for KNN and MLP)
    scaler = StandardScaler()
    num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Define Best Models (using your exact best hyperparameters)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=2, max_features='sqrt', criterion='gini', random_state=42)
    dt = DecisionTreeClassifier(max_depth=4, min_samples_split=15, criterion='gini', random_state=42)
    xgb = XGBClassifier(learning_rate=0.05, n_estimators=150, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', alpha=0.001, learning_rate_init=0.01, max_iter=1000, early_stopping=True, random_state=42)
    
    # Train Models
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    knn.fit(X_train_scaled, y_train)
    mlp.fit(X_train_scaled, y_train)
    
    # Train Logistic Regression
    lr = LogisticRegression(C=0.1, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Train SVM
    svm = SVC(C=1, kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    return rf, dt, xgb, knn, mlp, lr, svm, scaler, X_train.columns

models = load_and_train_models()
rf, dt, xgb, knn, mlp, lr, svm, scaler, feature_names = models

# --- 2. Build GUI ---
st.title("🫀 AI Heart Disease Diagnostic System")
st.markdown("Powered by Advanced Machine Learning Ensemble (7 Models)")

st.header("🩺 Patient Medical Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    gender = st.selectbox("Gender", options=["Female", "Male"])

with col2:
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
    cp = st.selectbox("Chest Pain Type (1-4)", options=[1, 2, 3, 4])
    restecg = st.selectbox("Resting EKG Results (0-2)", options=[0, 1, 2])
    vessels = st.selectbox("Number of Major Vessels Colored (0-3)", options=[0, 1, 2, 3])

with col3:
    exang = st.selectbox("Exercise Induced Angina?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (1-3)", options=[1, 2, 3])
    thal = st.selectbox("Thallium Stress Test Result", options=[3, 6, 7])

if st.button("🧪 Run AI Analysis", use_container_width=True):
    # Prepare Input Array
    gender_male = 1.0 if gender == "Male" else 0.0
    
    # Needs to perfectly match training columns order
    input_dict = {
        'Age': age, 'Chest pain type': cp, 'BP': bp, 'Cholesterol': chol, 
        'FBS over 120': fbs, 'EKG results': restecg, 'Max HR': max_hr, 
        'Exercise angina': exang, 'ST depression': oldpeak, 'Slope of ST': slope, 
        'Number of vessels fluro': vessels, 'Thallium': thal, 'Gender_Male': gender_male
    }
    
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    # Scaled version for KNN and MLP
    input_scaled = input_df.copy()
    num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    input_scaled[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    pred_rf = rf.predict(input_df)[0]
    pred_dt = dt.predict(input_df)[0]
    pred_xgb = xgb.predict(input_df)[0]
    pred_knn = knn.predict(input_scaled)[0]
    pred_mlp = mlp.predict(input_scaled)[0]
    pred_lr = lr.predict(input_scaled)[0]
    pred_svm = svm.predict(input_scaled)[0]
    
    predictions = [pred_rf, pred_dt, pred_xgb, pred_knn, pred_mlp, pred_lr, pred_svm]
    votes_for_disease = sum(predictions)
    
    st.divider()
    st.subheader(f"Final Diagnosis (Ensemble Majority Vote: {votes_for_disease}/7)")
    
    if votes_for_disease >= 4:
        st.error("⚠️ **High Risk:** The models detect significant signs of heart disease.")
    else:
        st.success("✅ **Low Risk:** The patient appears healthy. No significant signs detected.")
        
    with st.expander("🔍 View Individual Model Predictions"):
        def format_pred(p): return "Positive (Risk)" if p == 1 else "Negative (Healthy)"
        st.write(f"- **Random Forest:** {format_pred(pred_rf)}")
        st.write(f"- **Decision Tree:** {format_pred(pred_dt)}")
        st.write(f"- **XGBoost:** {format_pred(pred_xgb)}")
        st.write(f"- **KNN Model:** {format_pred(pred_knn)}")
        st.write(f"- **Neural Network (MLP):** {format_pred(pred_mlp)}")
        st.write(f"- **Support Vector Machine (SVM):** {format_pred(pred_svm)}")
        st.write(f"- **Logistic Regression:** {format_pred(pred_lr)}")