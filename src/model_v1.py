import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint

df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
# pprint(df.head(10))

# print(df.info())

# pprint(df.columns)

# pprint(df.shape)

####### Let's Check Outliers in our Columns ############

numerical_columns = df.select_dtypes(include=['number'])


plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns)
plt.title("Box Plot for Numerical Columns")


plt.xticks(rotation=45, ha="right")

plt.show()

############### Data Preprocessing #########################

df.columns = [col.replace('.', '_') for col in df.columns]

############## Feature Engineering #########################

df["Exp_Gap"] = df["Age"] - df["Experience"]
df["Income_per_Family"] = np.round(df["Income"] / (df["Family"].replace(0, 1)), 3)
df["CC_Spend_Ratio"] = df["CCAvg"] / (df["Income"] + 1)
df["Mortgage_Income_Ratio"] = df["Mortgage"] / (df["Income"] + 1)
df["Income_Mortgage_Ratio"] = df["Income"] / (df["Mortgage"] + 1)
df["Account_Score"] = df["Securities_Account"] + df["CD_Account"] + df["CreditCard"]
df["Digital_Score"] = df["Online"] + df["CreditCard"]
df["Income_Education"] = df["Income"] * df["Education"]
df["Exp_Education"] = df["Experience"] * df["Education"]
df["CC_per_Family"] = df["CCAvg"] / (df["Family"].replace(0, 1))

# pprint(df)

################# Baseline Model ########################

# assuming 'ZIP_Code' and 'Personal_Loan' are columns in the dataFrame
X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)  
y = df['Personal_Loan']  # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline with different classifiers
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# fitting the pipelines
pipeline_rf.fit(X_train, y_train)
pipeline_svm.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
pipeline_knn.fit(X_train, y_train)

# predictions
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_svm = pipeline_svm.predict(X_test)
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_knn = pipeline_knn.predict(X_test)

# performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Random Forest Accuracy:", accuracy_rf)
print("SVM Accuracy:", accuracy_svm)
print("Logistic Regression Accuracy:", accuracy_lr)
print("KNN Accuracy:", accuracy_knn)


######################## Hyperparameter Tuning ###########################
#Through this code we will use `GridSearchCV` and will print Best parameters can get Higher Performance

# hyperparameter grids for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}


pipeline_rf_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
])

# cross-validation and hyperparameter tuning
pipeline_rf_cv.fit(X_train, y_train)

# best hyperparameters and predictions
y_pred_rf_cv = pipeline_rf_cv.predict(X_test)


accuracy_rf_cv = accuracy_score(y_test, y_pred_rf_cv)
print("Random Forest Accuracy (with CV):", accuracy_rf_cv)


best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
print("\nBest Hyperparameters for RandomForestClassifier:")
print(best_params_rf)
