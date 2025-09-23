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
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from datetime import datetime
import json


df = pd.read_csv('../model_resources/bankloan.csv')
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
    ('classifier', LogisticRegression())
])


# Fit logistic regression
pipeline_lr.fit(X_train, y_train)
# pipeline_knn.fit(X_train, y_train)

# predictions

y_pred_lr = pipeline_lr.predict(X_test)


# performance

accuracy_lr = accuracy_score(y_test, y_pred_lr)


print("Logistic Regression Accuracy:", accuracy_lr)


precision_lr = precision_score(y_test, y_pred_lr)
print("Logistic Regression Precision:", precision_lr)
# print("KNN Precision:", precision_knn)


######################## Hyperparameter Tuning ###########################
#Through this code we will use `GridSearchCV` and will print Best parameters can get Higher Performance

# hyperparameter grids for RandomForestClassifier
param_grid_lrg = {
    'penalty': ['l1', 'l2', 'elasticnet', None],   # Regularization type
    'C': [0.01, 0.1, 1.0, 10, 100],                # Inverse of regularization strength
    'solver': ['liblinear', 'saga', 'lbfgs'],      # Optimizers (note: not all support l1/elasticnet)
    'class_weight': [None, 'balanced'],            # Handle imbalance
    'fit_intercept': [True, False],                # Whether to fit intercept
    'max_iter': [100, 200, 500]                    # Iterations for convergence
}

pipeline_lrg_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GridSearchCV(LogisticRegression(), param_grid_lrg, cv=5))
])

# cross-validation and hyperparameter tuning
pipeline_lrg_cv.fit(X_train, y_train)

best_params_lr = pipeline_lrg_cv.named_steps['classifier'].best_params_

# best hyperparameters and predictions
y_pred_rf_cv = pipeline_lrg_cv.predict(X_test)


accuracy_rf_cv = accuracy_score(y_test, y_pred_rf_cv)
print("Random Forest Accuracy (with CV):", accuracy_rf_cv)


best_params_rf = pipeline_lrg_cv.named_steps['classifier'].best_params_
print("\nBest Hyperparameters for RandomForestClassifier:")
print(best_params_rf)


############### Model Evaluation ####################
classification_rep = classification_report(y_test, y_pred_rf_cv)
print("Classification Report:\n", classification_rep)


# ---------------- Store Results in JSON ---------------- #
results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "target_variable": "Personal_Loan",
        "excluded_features": ["ZIP_Code", "Personal_Loan", "ID"]
    },
    "features_used": list(X.columns),
    "preprocessing": {
        "scaler": "StandardScaler",
        "feature_engineering": [
            "Exp_Gap = Age - Experience",
            "Income_per_Family = Income / (Family, replace 0 with 2)",
            "CC_Spend_Ratio = CCAvg / (Income+2)",
            "Mortgage_Income_Ratio = Mortgage / (Income+2)",
            "Income_Mortgage_Ratio = Income / (Mortgage+2)",
            "Account_Score = Securities_Account + CD_Account",
            "Digital_Score = Online + CreditCard",
            "Income_Education = Income * Education",
            "Exp_Education = Experience * Education",
            "CC_per_Family = CCAvg / (Family, replace 0 with 1)"
        ]
    },
    "model": "LogisticRegression",
    "scores": {
        "accuracy_train": pipeline_lr.score(X_train, y_train),
        "accuracy_test": accuracy_lr,
        "precision_test": precision_lr
    },
    "hyperparameters": {
        "best_params": best_params_lr
    },
    "evaluation_metrics": classification_rep
}

# Save results to JSON
with open("../model_resources/loan_model_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to loan_model_results.json")


