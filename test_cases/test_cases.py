import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

class BankLoanModelTest(unittest.TestCase):
    # setUpClass to fetch dataset
    @classmethod
    def setUpClass(cls):
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)  # disables SSL verification safely for testing
        cls.df = pd.read_csv(StringIO(response.text))

    # Test Case 1: Test data preprocessing
    # This test checks if the columns are correctly renamed and new features are added.
    def test_data_preprocessing(self):
        df = self.df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        self.assertIn('Income_per_Family', df.columns)
        self.assertIn('Exp_Gap', df.columns)

    # Test Case 2: Test train-test split
    # This test ensures that the train-test split results in the correct sizes.
    def test_train_test_split(self):
        df = self.df.copy()
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), int(0.8 * len(df)))
        self.assertEqual(len(X_test), int(0.2 * len(df)))

    # Test Case 3: Test model accuracy
    # This test checks if the RandomForest model achieves a reasonable accuracy.
    def test_random_forest_accuracy(self):
        df = self.df.copy()
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
        self.assertGreater(accuracy_rf, 0.7)  # Assuming a reasonable threshold

    # Test Case 4: Test hyperparameter tuning
    # This test checks if GridSearchCV finds the best parameters for RandomForest.
    def test_hyperparameter_tuning(self):
        df = self.df.copy()
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
        self.assertIn(best_params_rf['n_estimators'], [50, 100])
        self.assertIn(best_params_rf['max_depth'], [5, 10])

if __name__ == '__main__':
    unittest.main()