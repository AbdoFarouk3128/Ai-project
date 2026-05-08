import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load Data
train = pd.read_csv('data/Cleaned Train Data.csv')
test = pd.read_csv('data/Cleaned Test Data.csv')

# 2. Remove features identified as useless
useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 
           'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 
           'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
test.drop(columns=[c for c in useless if c in test.columns], inplace=True)

# 3. Separate Features and Target
X_train = train.drop('Heart Disease', axis=1)
y_train = train['Heart Disease']
X_test = test.drop('Heart Disease', axis=1)
y_test = test['Heart Disease']

# 4. Scale the Data (Necessary for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Setup the Model
model = LogisticRegression(solver='liblinear', random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# 6. Run GridSearch
grid = GridSearchCV(model, param_grid, cv=5, scoring='recall')
grid.fit(X_train_scaled, y_train)

# 7. Final Predictions
best = grid.best_estimator_
y_pred = best.predict(X_test_scaled)
y_train_pred = best.predict(X_train_scaled)

# 8. Simple Printing
print("--- LOGISTIC REGRESSION RESULTS ---")
print("Best Hyperparameters:")
print(grid.best_params_)

print("Accuracy (Test):")
print(accuracy_score(y_test, y_pred))

print("Accuracy (Train):")
print(accuracy_score(y_train, y_train_pred))

print("Recall (Test):")
print(recall_score(y_test, y_pred))

print("F1 Score (Test):")
print(f1_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Model Diagnosis (Your specific logic)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("\n--- DIAGNOSIS ---")
if train_acc > test_acc + 0.15:
    print("STATUS: Overfitting (Model is too complex)")
elif train_acc < 0.60:
    print("STATUS: Underfitting (Model is too simple)")
else:
    print("STATUS: Well-Fitted (Model is balanced)")