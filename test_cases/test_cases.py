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
        self.assertEqual(self.df.shape[1], expected_columns, "Dataframe should have 14 columns")

    # Test Case 2: Test feature engineering
    # This test checks if the new features are added correctly.
    def test_feature_engineering(self):
        df = self.df.copy()
        df["Exp_Gap"] = df["Age"] - df["Experience"]
        df["Income_per_Family"] = df["Income"] / (df["Family"].replace(0, 2))
        self.assertIn("Exp_Gap", df.columns, "Exp_Gap column should be present in the dataframe")
        self.assertIn("Income_per_Family", df.columns, "Income_per_Family column should be present in the dataframe")

    # Test Case 3: Test model training
    # This test checks if the models can be trained without errors.
    def test_model_training(self):
        df = self.df.copy()
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        try:
            pipeline_rf.fit(X_train, y_train)
            self.assertTrue(True, "Model training should complete without errors")
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    # Test Case 4: Test model accuracy
    # This test checks if the model accuracy is within an expected range.
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
        self.assertGreaterEqual(accuracy_rf, 0.7, "Model accuracy should be at least 0.7")

if __name__ == '__main__':
    unittest.main()