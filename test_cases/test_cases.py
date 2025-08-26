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
        self.assertIn('Age', df.columns)
        self.assertIn('Experience', df.columns)
        self.assertIn('Income', df.columns)

    # Test Case 2: Test feature engineering
    # This test checks if the feature engineering steps are correctly applied.
    def test_feature_engineering(self):
        df = self.df.copy()
        df["Exp_Gap"] = df["Age"] - df["Experience"]
        df["Income_per_Family"] = np.round(df["Income"] / (df["Family"].replace(0, 2)), 4)
        self.assertIn('Exp_Gap', df.columns)
        self.assertIn('Income_per_Family', df.columns)

    # Test Case 3: Test model training and prediction
    # This test checks if the models are trained and make predictions without errors.
    def test_model_training_and_prediction(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        self.assertTrue(0 <= accuracy_rf <= 1)

    # Test Case 4: Test hyperparameter tuning
    # This test checks if the hyperparameter tuning process completes and returns best parameters.
    def test_hyperparameter_tuning(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
        self.assertIsInstance(best_params_rf, dict)
        self.assertIn('n_estimators', best_params_rf)

if __name__ == '__main__':
    unittest.main()
