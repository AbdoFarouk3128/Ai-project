import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Load Data
train = pd.read_csv('data/Cleaned_Train_Data.csv')
test = pd.read_csv('data/Cleaned_Test_Data.csv')

# 2. Drop Noise
useless = ['Gender_Female', 'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
train.drop(columns=[c for c in useless if c in train.columns], inplace=True)
test.drop(columns=[c for c in useless if c in test.columns], inplace=True)

X_train, y_train = train.drop('Heart Disease', axis=1), train['Heart Disease']
X_test, y_test = test.drop('Heart Disease', axis=1), test['Heart Disease']

# 3. Neural Networks REQUIRE scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define the Model
model = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)

# 5. Hyperparameters to tune
param_grid = {
    'hidden_layer_sizes': [(16,), (32, 16), (64, 32, 16)],  # Number of neurons and layers
    'activation': ['relu', 'tanh'],                         # How the neurons fire
    'alpha': [0.001, 0.01, 0.1],                            # Regularization (stops overfitting)
    'learning_rate_init': [0.001, 0.01]                     # How fast it learns
}

# 6. Grid Search
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

# 7. Predict and Evaluate
best = grid.best_estimator_
y_train_pred = best.predict(X_train_scaled)
y_pred = best.predict(X_test_scaled)

print("--- MULTI-LAYER PERCEPTRON (NEURAL NETWORK) ---")
print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))