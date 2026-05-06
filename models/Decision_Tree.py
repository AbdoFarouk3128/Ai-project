import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('data/Cleaned_Train_Data.csv')
test = pd.read_csv('data/Cleaned_Test_Data.csv')

useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 
           'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 
           'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
test.drop(columns=[c for c in useless if c in test.columns], inplace=True)

X_train, y_train = train.drop('Heart Disease', axis=1), train['Heart Disease']
X_test, y_test = test.drop('Heart Disease', axis=1), test['Heart Disease']

model = DecisionTreeClassifier(random_state=42)

param_grid = {
   'max_depth': [3, 4, 5, 6],
    'min_samples_split': [15, 20, 30],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='recall')
grid.fit(X_train, y_train)

best = grid.best_estimator_
y_pred = best.predict(X_test)
y_train_pred = best.predict(X_train)

print("--- DECISION TREE ---")
print(f"Best Hyperparameters: {grid.best_params_}")
print("-" * 30)
print(f"Accuracy (Test):   {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy (Train):  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision (Test):  {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Test):     {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score (Test):   {f1_score(y_test, y_pred):.4f}")
print("-" * 30)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if accuracy_score(y_train, y_train_pred) > accuracy_score(y_test, y_pred) + 0.15:
    print("\n[NOTE] The model is OVERFITTING (Learning patterns too specific to training data)")
elif accuracy_score(y_train, y_train_pred) < 0.60:
    print("\n[NOTE] The model is UNDERFITTING (Too simple)")
else:
    print("\n[NOTE] The model is WELL-FITTED (Balanced)")