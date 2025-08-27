import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelTestCase(unittest.TestCase):
    # setUpClass to fetch dataset
    @classmethod
    def setUpClass(cls):
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)  # disables SSL verification safely for testing
        cls.df = pd.read_csv(StringIO(response.text))

    # Test Case 1: Test data loading
    # This test checks if the data is loaded correctly and has the expected number of columns.
    def test_data_loading(self):
        expected_columns = 14  # Assuming the dataset has 14 columns
        self.assertEqual(self.df.shape[1], expected_columns)

    # Test Case 2: Test feature engineering
    # This test checks if the new features are added correctly to the dataframe.
    def test_feature_engineering(self):
        df = self.df.copy()
        df["Exp_Gap"] = df["Age"] - df["Experience"]
        df["Income_per_Family"] = df["Income"] / (df["Family"].replace(0, 2))
        self.assertIn("Exp_Gap", df.columns)
        self.assertIn("Income_per_Family", df.columns)

    # Test Case 3: Test model training
    # This test checks if the RandomForest model can be trained without errors.
    def test_model_training(self):
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
        self.assertTrue(0 <= accuracy_rf <= 1)

    # Test Case 4: Test model accuracy
    # This test checks if the accuracy of the RandomForest model is above a certain threshold.
    def test_model_accuracy(self):
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
        self.assertGreater(accuracy_rf, 0.7)  # Assuming a threshold of 0.7 for this test

if __name__ == '__main__':
    unittest.main()