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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('model_resources/bankloan.csv')
df = df.sample(1000, random_state=42)
numerical_columns = df.select_dtypes(include=['number'])


plt.figure(figsize=(12, 8))
sns.boxplot(data=numerical_columns)
plt.title("Box Plot for Numerical Columns")


plt.xticks(rotation=45, ha="right")

plt.show()

############### Data Preprocessing #########################

# Features name
df.columns = [col.replace('.', '_') for col in df.columns]

############## Feature Engineering #########################
df["Exp_Gap"] = df["Age"] - df["Experience"]
df["Income_per_Family"] = np.round(df["Income"] / (df["Family"].replace(0, 2)), 4)
df["CC_Spend_Ratio"] = df["CCAvg"] / (df["Income"] + 2)
df["Mortgage_Income_Ratio"] = df["Mortgage"] / (df["Income"] + 2)
df["Income_Mortgage_Ratio"] = df["Income"] / (df["Mortgage"] + 2)
df["Account_Score"] = df["Securities_Account"] + df["CD_Account"] 
df["Digital_Score"] = df["Online"] + df["CreditCard"]
df["Income_Education"] = df["Income"] * df["Education"]
df["Exp_Education"] = df["Experience"] * df["Education"]
df["CC_per_Family"] = df["CCAvg"] / (df["Family"].replace(0, 1))

# pprint(df)


# assuming 'ZIP_Code' and 'Personal_Loan' are columns in the dataFrame
X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)  
y = df['Personal_Loan']  # target variable

# Train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create pipeline to train model
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Define hyperparameter grid according to whitepaper
param_grid = {
    'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'powell'],
    'classifier__max_iter': [100, 200, 500, 1000]
}

# Setup GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=pipeline_lr,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Predictions
y_pred_lr = grid_search.predict(X_test)

# Performance metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
balanced_acc = balanced_accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Precision:", precision_lr)
print("Recall:", recall)
print("F1-score:", f1)
print("Balanced Accuracy:", balanced_acc)

