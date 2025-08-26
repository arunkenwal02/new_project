import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fetch the dataset from GitHub
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)
        cls.df = pd.read_csv(StringIO(response.text))

        # Preprocessing
        cls.df.columns = [col.replace('.', '_') for col in cls.df.columns]
        cls.df["Exp_Gap"] = cls.df["Age"] - cls.df["Experience"]
        cls.df["Income_per_Family"] = np.round(cls.df["Income"] / (cls.df["Family"].replace(0, 2)), 4)
        cls.df["CC_Spend_Ratio"] = cls.df["CCAvg"] / (cls.df["Income"] + 2)
        cls.df["Mortgage_Income_Ratio"] = cls.df["Mortgage"] / (cls.df["Income"] + 2)
        cls.df["Income_Mortgage_Ratio"] = cls.df["Income"] / (cls.df["Mortgage"] + 2)
        cls.df["Account_Score"] = cls.df["Securities_Account"] + cls.df["CD_Account"]
        cls.df["Digital_Score"] = cls.df["Online"] + cls.df["CreditCard"]
        cls.df["Income_Education"] = cls.df["Income"] * cls.df["Education"]
        cls.df["Exp_Education"] = cls.df["Experience"] * cls.df["Education"]
        cls.df["CC_per_Family"] = cls.df["CCAvg"] / (cls.df["Family"].replace(0, 1))

        # Prepare features and target variable
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']

        # Split the data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_random_forest_accuracy(self):
        # Test Random Forest Classifier accuracy
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        self.assertTrue(accuracy_rf > 0.7, "Random Forest accuracy is below 70%")

    def test_svm_accuracy(self):
        # Test SVM Classifier accuracy
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        self.assertTrue(accuracy_svm > 0.7, "SVM accuracy is below 70%")

    def test_logistic_regression_accuracy(self):
        # Test Logistic Regression Classifier accuracy
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        self.assertTrue(accuracy_lr > 0.7, "Logistic Regression accuracy is below 70%")

    def test_knn_accuracy(self):
        # Test KNN Classifier accuracy
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        self.assertTrue(accuracy_knn > 0.7, "KNN accuracy is below 70%")

    def test_random_forest_hyperparameter_tuning(self):
        # Test Random Forest with hyperparameter tuning
        param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        y_pred_rf_cv = pipeline_rf_cv.predict(self.X_test)
        accuracy_rf_cv = accuracy_score(self.y_test, y_pred_rf_cv)
        self.assertTrue(accuracy_rf_cv > 0.7, "Random Forest accuracy with CV is below 70%")

    def test_svm_hyperparameter_tuning(self):
        # Test SVM with hyperparameter tuning
        param_grid_svm = {
            'C': [0.1, 1],
            'kernel': ['linear', 'poly'],
            'gamma': ['scale', 'auto']
        }
        pipeline_svm_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(SVC(), param_grid_svm, cv=5))
        ])
        pipeline_svm_cv.fit(self.X_train, self.y_train)
        y_pred_svm_cv = pipeline_svm_cv.predict(self.X_test)
        accuracy_svm_cv = accuracy_score(self.y_test, y_pred_svm_cv)
        self.assertTrue(accuracy_svm_cv > 0.7, "SVM accuracy with CV is below 70%")

if __name__ == '__main__':
    unittest.main()