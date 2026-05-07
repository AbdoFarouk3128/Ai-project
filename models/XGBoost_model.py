import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


train = pd.read_csv('data/Cleaned_Train_Data.csv')
test = pd.read_csv('data/Cleaned_Test_Data.csv')

useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
cols_to_scale = ['Age' , 'BP' , 'Cholesterol' , 'Max HR' , 'ST depression']

train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
test.drop(columns=[c for c in useless if c in test.columns], inplace=True)

X_train, y_train = train.drop('Heart Disease', axis=1), train['Heart Disease']
X_test, y_test = test.drop('Heart Disease', axis=1), test['Heart Disease']


scaler = StandardScaler()

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


model = XGBClassifier(
    reg_alpha=0.5,
    reg_lambda=1.0
)

param_grid = {
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 150],
    'max_depth': [3, 4],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='recall')

grid.fit(X_train, y_train)


best = grid.best_estimator_
y_pred = best.predict(X_test)

y_train_pred = best.predict(X_train)


print("--- XGBOOST ---")
print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy Train:    {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall Test:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
