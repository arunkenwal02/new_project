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