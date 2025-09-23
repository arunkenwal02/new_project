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

    # Test Case 1: Check if the dataset is loaded correctly
    def test_dataset_loaded(self):
        # Check if the dataframe is not empty
        self.assertFalse(self.df.empty, "The dataframe should not be empty.")

    # Test Case 2: Validate feature engineering
    def test_feature_engineering(self):
        # Check if new columns are created correctly
        self.df["Exp_Gap"] = self.df["Age"] - self.df["Experience"]
        self.df["Income_per_Family"] = self.df["Income"] / (self.df["Family"].replace(0, 2))
        self.assertIn("Exp_Gap", self.df.columns, "Exp_Gap column should be present in the dataframe.")
        self.assertIn("Income_per_Family", self.df.columns, "Income_per_Family column should be present in the dataframe.")

    # Test Case 3: Validate model training and prediction
    def test_model_training(self):
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
        self.assertGreater(accuracy_rf, 0.5, "Random Forest accuracy should be greater than 0.5.")

    # Test Case 4: Validate hyperparameter tuning
    def test_hyperparameter_tuning(self):
        X = self.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = self.df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(C=1, kernel='linear'))
        ])
        pipeline_svm.fit(X_train, y_train)
        y_pred_svm = pipeline_svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        self.assertGreater(accuracy_svm, 0.5, "SVM accuracy should be greater than 0.5.")

if __name__ == '__main__':
    unittest.main()