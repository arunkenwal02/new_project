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
        # Fetch dataset from GitHub
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)  # disables SSL verification safely for testing
        cls.df = pd.read_csv(StringIO(response.text))

        # Preprocess the data as done in the original script
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

    def test_data_shape(self):
        # Test to check if the dataframe has the expected shape
        self.assertEqual(self.df.shape[0], 5000)  # Assuming there are 5000 rows
        self.assertEqual(self.df.shape[1], 14)    # Assuming there are 14 columns after processing

    def test_feature_engineering(self):
        # Test to check if new features are created correctly
        self.assertIn("Exp_Gap", self.df.columns)
        self.assertIn("Income_per_Family", self.df.columns)
        self.assertIn("CC_Spend_Ratio", self.df.columns)
        self.assertIn("Mortgage_Income_Ratio", self.df.columns)
        self.assertIn("Income_Mortgage_Ratio", self.df.columns)
        self.assertIn("Account_Score", self.df.columns)
        self.assertIn("Digital_Score", self.df.columns)

    def test_model_accuracy(self):
        # Test to check if the Random Forest model achieves a reasonable accuracy
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

        self.assertGreaterEqual(accuracy_rf, 0.7)  # Assuming we expect at least 70% accuracy

    def test_hyperparameter_tuning_rf(self):
        # Test to check if hyperparameter tuning for RandomForest works
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
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3))
        ])

        pipeline_rf_cv.fit(X_train, y_train)
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_

        self.assertIn('n_estimators', best_params_rf)
        self.assertIn('max_depth', best_params_rf)
        self.assertIn('min_samples_split', best_params_rf)

    def test_classification_report(self):
        # Test to check if classification report can be generated
        X = self.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = self.df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)

        report = classification_report(y_test, y_pred_rf)
        self.assertIsNotNone(report)  # Check if the report is generated

if __name__ == '__main__':
    unittest.main()