import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

train = pd.read_csv('data/Cleaned_Train_Data.csv')
test = pd.read_csv('data/Cleaned_Test_Data.csv')

useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
test.drop(columns=[c for c in useless if c in test.columns], inplace=True)

X_train, y_train = train.drop('Heart Disease', axis=1), train['Heart Disease']
X_test, y_test = test.drop('Heart Disease', axis=1), test['Heart Disease']

model = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)

print("--- RANDOM FOREST ---")
print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
