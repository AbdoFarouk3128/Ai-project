import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE           
from sklearn.feature_selection import RFE   
from sklearn.tree import DecisionTreeClassifier
#Training Stage
#=======================================================================================
train_df = pd.read_csv("train_data.csv")
train_df.drop("id", axis=1, inplace=True)

X_train = train_df.drop('Heart Disease', axis=1)
y_train = train_df['Heart Disease']


Numerical_imputer = SimpleImputer(strategy='median')
Numerical_columns = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
X_train[Numerical_columns] = Numerical_imputer.fit_transform(X_train[Numerical_columns])


One_Hot_Categorical_Columns = ['Gender','work_type','smoking_status']
Lable_categorical_Columns = ['Chest pain type','EKG results','Slope of ST','Thallium']
All_Categorical = One_Hot_Categorical_Columns + Lable_categorical_Columns

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



l_goal=LabelEncoder()
y_train=l_goal.fit_transform(y_train)

le = LabelEncoder()
for col in Lable_categorical_Columns :
    X_train[col]=le.fit_transform(X_train[col].astype(str))


One_Hot_Encoding=OneHotEncoder(handle_unknown='ignore',sparse_output=False)

encoded=One_Hot_Encoding.fit_transform(X_train[One_Hot_Categorical_Columns])

encoded_df = pd.DataFrame(encoded,columns=One_Hot_Encoding.get_feature_names_out(One_Hot_Categorical_Columns),index=X_train.index)    

X_train = X_train.drop(One_Hot_Categorical_Columns, axis=1)
X_train = pd.concat([X_train, encoded_df], axis=1)


scaler = StandardScaler()

X_train[Numerical_columns] = scaler.fit_transform(X_train[Numerical_columns])

print(X_train.shape, y_train.shape)

sm = SMOTE(random_state=42)
X_Train_Res, Y_Train_Res = sm.fit_resample(X_train, y_train)

# selector = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=12)

# X_train_final = selector.fit_transform(X_Train_Res,Y_Train_Res )     

# print(X_train_final.shape, Y_Train_Res.shape)

X_Train_Res = pd.DataFrame(X_Train_Res, columns=X_train.columns)
Y_Train_Res = pd.Series(Y_Train_Res)

print(X_Train_Res.shape, Y_Train_Res.shape)
print(X_Train_Res.columns)
print(X_train.head())
print(Y_Train_Res.head())

Cleaned_Train_Data = X_Train_Res.copy()
Cleaned_Train_Data['Heart Disease'] = Y_Train_Res

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


encoded_test = One_Hot_Encoding.transform(X_test[One_Hot_Categorical_Columns])
encoded_test_df = pd.DataFrame(encoded_test, columns=One_Hot_Encoding.get_feature_names_out(One_Hot_Categorical_Columns), index=X_test.index)

X_test = X_test.drop(One_Hot_Categorical_Columns, axis=1)
X_test = pd.concat([X_test, encoded_test_df], axis=1)
X_test[Numerical_columns] = scaler.transform(X_test[Numerical_columns])

X_test_final = X_test[X_Train_Res.columns]

Cleaned_Test_Data=X_test_final.copy()
Cleaned_Test_Data['Heart Disease'] = l_goal.transform(y_test)

Cleaned_Test_Data.to_csv("Modified_Cleaned_Test_Data.csv",index=False)

print("Test Data Saved Successfully")