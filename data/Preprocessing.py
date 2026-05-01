import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#Training Stage
#=======================================================================================
train_df = pd.read_csv("train_data.csv")
train_df.drop("id", axis=1, inplace=True)

X_train = train_df.drop('Heart Disease', axis=1)
y_train = train_df['Heart Disease']


age_imputer = SimpleImputer(strategy='median')
X_train[['Age']] = age_imputer.fit_transform(X_train[['Age']])

mode_imputer = SimpleImputer(strategy='most_frequent')

categorical_missing = ['Gender', 'work_type', 'smoking_status']
X_train[categorical_missing] = mode_imputer.fit_transform(X_train[categorical_missing])

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



l_goal=LabelEncoder()
y_train=l_goal.fit_transform(y_train)

One_Hot_Encoding=OneHotEncoder(handle_unknown='ignore',sparse_output=False)


encoded=One_Hot_Encoding.fit_transform(X_train[['Gender','work_type','smoking_status']])

encoded_df = pd.DataFrame(encoded,columns=One_Hot_Encoding.get_feature_names_out(['Gender','work_type','smoking_status']),index=X_train.index)    


X_train = X_train.drop(['Gender','work_type', 'smoking_status'], axis=1)


X_train = pd.concat([X_train, encoded_df], axis=1)


scaler = StandardScaler()

scale_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])

print(X_train.shape, y_train.shape)


Cleaned_Train_Data = X_train.copy()

Cleaned_Train_Data['Heart Disease'] = y_train

Cleaned_Train_Data.to_csv("Cleaned_Train_Data.csv", index=False)

print("Successfully")

#=======================================================================================
#Test Stage 
#=======================================================================================

test_df = pd.read_csv("test_data.csv")
test_df.drop("id", axis=1, inplace=True)

X_test = test_df.drop('Heart Disease', axis=1)
y_test = test_df['Heart Disease']

X_test[['Age']] = age_imputer.transform(X_test[['Age']])
X_test[categorical_missing] = mode_imputer.transform(X_test[categorical_missing])

for col in outlier_cols :
    lower_bound,upper_bound=bounds[col]
    X_test[col]=np.clip(X_test[col],lower_bound,upper_bound)


encoded_test = One_Hot_Encoding.transform(X_test[['Gender','work_type','smoking_status']])
encoded_test_df = pd.DataFrame(encoded_test, columns=One_Hot_Encoding.get_feature_names_out(['Gender','work_type','smoking_status']), index=X_test.index)
X_test = X_test.drop(['Gender','work_type','smoking_status'], axis=1)
X_test = pd.concat([X_test, encoded_test_df], axis=1)
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

Cleaned_Test_Data=X_test.copy()

Cleaned_Test_Data['Heart Disease'] = l_goal.transform(y_test)

Cleaned_Test_Data.to_csv("Cleaned_Test_Data.csv",index=False)

print("Successfully")