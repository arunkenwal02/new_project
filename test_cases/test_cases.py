import unittest
import pandas as pd
import requests
from io import StringIO
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class BankLoanModelTest(unittest.TestCase):
    # setUpClass to fetch dataset
    @classmethod
    def setUpClass(cls):
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)  # disables SSL verification safely for testing
        cls.df = pd.read_csv(StringIO(response.text))

    # Test Case 1: Test data preprocessing
    # This test checks if the data preprocessing steps are correctly applied.
    def test_data_preprocessing(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        self.assertIn('ZIP_Code', df.columns)
        self.assertIn('Personal_Loan', df.columns)
        self.assertIn('ID', df.columns)

    # Test Case 2: Test feature engineering
    # This test checks if the feature engineering steps are correctly applied.
    def test_feature_engineering(self):
        df = self.df.copy()
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
        self.assertIn("Exp_Gap", df.columns)
        self.assertIn("Income_per_Family", df.columns)

    # Test Case 3: Test model training and accuracy
    # This test checks if the models are trained and their accuracy is above a threshold.
    def test_model_training_and_accuracy(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        self.assertGreater(accuracy_rf, 0.7)

        # SVM
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(X_train, y_train)
        y_pred_svm = pipeline_svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        self.assertGreater(accuracy_svm, 0.7)

        # Logistic Regression
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(X_train, y_train)
        y_pred_lr = pipeline_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        self.assertGreater(accuracy_lr, 0.7)

        # KNN
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(X_train, y_train)
        y_pred_knn = pipeline_knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        self.assertGreater(accuracy_knn, 0.7)

    # Test Case 4: Test hyperparameter tuning
    # This test checks if the hyperparameter tuning improves the model accuracy.
    def test_hyperparameter_tuning(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest with GridSearchCV
        param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(X_train, y_train)
        y_pred_rf_cv = pipeline_rf_cv.predict(X_test)
        accuracy_rf_cv = accuracy_score(y_test, y_pred_rf_cv)
        self.assertGreater(accuracy_rf_cv, 0.7)

        # SVM with GridSearchCV
        param_grid_svm = {
            'C': [0.1, 1],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        pipeline_svm_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(SVC(), param_grid_svm, cv=5))
        ])
        pipeline_svm_cv.fit(X_train, y_train)
        y_pred_svm_cv = pipeline_svm_cv.predict(X_test)
        accuracy_svm_cv = accuracy_score(y_test, y_pred_svm_cv)
        self.assertGreater(accuracy_svm_cv, 0.7)

if __name__ == '__main__':
    unittest.main()