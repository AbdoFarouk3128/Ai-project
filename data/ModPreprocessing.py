import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTENC         

#Training Stage
#=======================================================================================
train_df = pd.read_csv("train_data.csv")
train_df.drop("id", axis=1, inplace=True)

X_train = train_df.drop('Heart Disease', axis=1)
y_train = train_df['Heart Disease']


Numerical_columns = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
Numerical_imputer = SimpleImputer(strategy='median')
X_train[Numerical_columns] = Numerical_imputer.fit_transform(X_train[Numerical_columns])


One_Hot_Categorical_Columns = ['Gender','work_type','smoking_status']
Ordinal_Columns = ['Chest pain type','EKG results','Slope of ST','Thallium']
All_Categorical = One_Hot_Categorical_Columns + Ordinal_Columns

Categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train[All_Categorical] = Categorical_imputer.fit_transform(X_train[All_Categorical])

outlier_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
bounds = {}
for col in outlier_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bounds[col] = (lower_bound, upper_bound)
    X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)


Cat_idx = [X_train.columns.get_loc(c) for c in All_Categorical]

sm = SMOTENC(categorical_features=Cat_idx,random_state=42)

X_Res, Y_Res = sm.fit_resample(X_train, y_train)
X_Res = pd.DataFrame(X_Res, columns=X_train.columns)
Y_Res = pd.Series(Y_Res)



for col in Ordinal_Columns :
    X_Res[col] = X_Res[col].astype(int)
    

cols_to_fix = ['Age', 'BP', 'Cholesterol', 'Max HR', 'FBS over 120', 'Exercise angina', 'Number of vessels fluro']
for col in cols_to_fix:
    if col in X_Res.columns:
        X_Res[col] = X_Res[col].round().astype(int)


One_Hot_Encoding=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
encoder=One_Hot_Encoding.fit_transform(X_Res[One_Hot_Categorical_Columns])
encoded_df = pd.DataFrame(encoder,columns=One_Hot_Encoding.get_feature_names_out(One_Hot_Categorical_Columns),index=X_Res.index)    

X_Res = X_Res.drop(One_Hot_Categorical_Columns, axis=1)
X_Res = pd.concat([X_Res, encoded_df], axis=1)


l_goal=LabelEncoder()
Y_Res = pd.Series(l_goal.fit_transform(Y_Res))

print(X_train.shape, y_train.shape)



print(X_Res.shape, Y_Res.shape)
print(X_Res.columns)
print(X_Res.head())
print(Y_Res.head())
print(X_Res)


Cleaned_Train_Data = X_Res.copy()
Cleaned_Train_Data['Heart Disease'] = Y_Res
Cleaned_Train_Data.to_csv("Modified_Cleaned_Train_Data.csv", index=False)

print("Training Data Saved Successfully")

#=======================================================================================
#Test Stage 
#=======================================================================================

test_df = pd.read_csv("test_data.csv")
test_df.drop("id", axis=1, inplace=True)

X_test = test_df.drop('Heart Disease', axis=1)
y_test = test_df['Heart Disease']

X_test[Numerical_columns] = Numerical_imputer.transform(X_test[Numerical_columns])
X_test[All_Categorical] = Categorical_imputer.transform(X_test[All_Categorical])

for col in outlier_cols :
    lower_bound,upper_bound=bounds[col]
    X_test[col]=np.clip(X_test[col],lower_bound,upper_bound)


for col in Ordinal_Columns:
    X_test[col]=X_test[col].astype(int)

encoded_test = One_Hot_Encoding.transform(X_test[One_Hot_Categorical_Columns])
encoded_test_df = pd.DataFrame(encoded_test, columns=One_Hot_Encoding.get_feature_names_out(One_Hot_Categorical_Columns), index=X_test.index)

X_test = X_test.drop(One_Hot_Categorical_Columns, axis=1)
X_test = pd.concat([X_test, encoded_test_df], axis=1)

X_test_final = X_test.reindex(columns=X_Res.columns , fill_value=0)

Cleaned_Test_Data=X_test_final.copy()
Cleaned_Test_Data['Heart Disease'] = l_goal.transform(y_test)

Cleaned_Test_Data.to_csv("Modified_Cleaned_Test_Data.csv",index=False)

print("Test Data Saved Successfully")