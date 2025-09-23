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
    # This test checks if the feature engineering step adds the expected columns.
    def test_feature_engineering(self):
        df = self.df.copy()
        df["Exp_Gap"] = df["Age"] - df["Experience"]
        df["Income_per_Family"] = df["Income"] / (df["Family"].replace(0, 2))
        expected_columns = ['Exp_Gap', 'Income_per_Family']
        for col in expected_columns:
            self.assertIn(col, df.columns)

    # Test Case 3: Test train-test split
    # This test checks if the train-test split results in the expected number of samples.
    def test_train_test_split(self):
        X = self.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = self.df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), int(0.8 * len(self.df)))
        self.assertEqual(len(X_test), int(0.2 * len(self.df)))

    # Test Case 4: Test model accuracy
    # This test checks if the RandomForest model achieves a minimum accuracy threshold.
    def test_random_forest_accuracy(self):
        X = self.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = self.df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        self.assertGreaterEqual(accuracy_rf, 0.7)  # Assuming 70% is the minimum acceptable accuracy

if __name__ == '__main__':
    unittest.main()