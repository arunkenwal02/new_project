import unittest
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class TestBankLoanModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset from GitHub
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url)
        cls.df = pd.read_csv(StringIO(response.text))

        # Preprocess the dataset
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

        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    # Test Case 1: Validate the shape of the dataset after preprocessing
    def test_shape_of_data(self):
        self.assertEqual(self.df.shape[1], 15)  # Check if there are 15 columns after feature engineering
        self.assertEqual(self.X_train.shape[0], 4000)  # Assuming the original dataset has 5000 rows

    # Test Case 2: Validate Random Forest Classifier accuracy
    def test_random_forest_accuracy(self):
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        self.assertGreater(accuracy_rf, 0.7)  # Assuming we expect > 70% accuracy

    # Test Case 3: Validate SVC Classifier accuracy
    def test_svc_accuracy(self):
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        self.assertGreater(accuracy_svm, 0.7)  # Assuming we expect > 70% accuracy

    # Test Case 4: Validate Logistic Regression accuracy
    def test_logistic_regression_accuracy(self):
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        self.assertGreater(accuracy_lr, 0.7)  # Assuming we expect > 70% accuracy

    # Test Case 5: Validate KNN Classifier accuracy
    def test_knn_accuracy(self):
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        self.assertGreater(accuracy_knn, 0.7)  # Assuming we expect > 70% accuracy

    # Test Case 6: Validate Random Forest hyperparameter tuning
    def test_random_forest_hyperparameter_tuning(self):
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
        self.assertIsInstance(best_params_rf, dict)  # Ensure best_params is a dictionary

if __name__ == '__main__':
    unittest.main()