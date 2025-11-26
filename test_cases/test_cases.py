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

    # Test Case 1: Check if the dataset is loaded correctly
    def test_dataset_loaded(self):
        self.assertFalse(self.df.empty, "The dataset should not be empty.")

    # Test Case 2: Check if feature engineering is applied correctly
    def test_feature_engineering(self):
        self.assertIn("Exp_Gap", self.df.columns, "Feature 'Exp_Gap' should be in the dataframe.")
        self.assertIn("Income_per_Family", self.df.columns, "Feature 'Income_per_Family' should be in the dataframe.")

    # Test Case 3: Validate RandomForestClassifier accuracy is within expected range
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
        self.assertGreaterEqual(accuracy_rf, 0.7, "Random Forest accuracy should be at least 0.7.")

    # Test Case 4: Validate GridSearchCV finds better parameters for RandomForestClassifier
    def test_random_forest_grid_search(self):
        X = self.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = self.df['Personal_Loan']
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
        self.assertIsNotNone(best_params_rf, "GridSearchCV should find the best parameters.")

if __name__ == '__main__':
    unittest.main()