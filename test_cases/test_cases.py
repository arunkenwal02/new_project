import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import numpy as np

class ModelTest(unittest.TestCase):
    # setUpClass to fetch dataset
    @classmethod
    def setUpClass(cls):
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)  # disables SSL verification safely for testing
        cls.df = pd.read_csv(StringIO(response.text))
        cls.df = cls.df.sample(1000, random_state=42)
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

    # Test Case 1: Test data preprocessing
    # This test checks if the preprocessing steps are applied correctly.
    def test_data_preprocessing(self):
        self.assertIn("Exp_Gap", self.df.columns)
        self.assertIn("Income_per_Family", self.df.columns)
        self.assertIn("CC_Spend_Ratio", self.df.columns)

    # Test Case 2: Test train-test split
    # This test checks if the train-test split results in the correct number of samples.
    def test_train_test_split(self):
        self.assertEqual(len(self.X_train), 800)
        self.assertEqual(len(self.X_test), 200)
        self.assertEqual(len(self.y_train), 800)
        self.assertEqual(len(self.y_test), 200)

    # Test Case 3: Test model training and best parameters
    # This test checks if the model is trained and best parameters are found.
    def test_model_training(self):
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        param_grid = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
            'classifier__max_features': ['auto', 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(
            estimator=pipeline_lr,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        self.assertIsNotNone(best_params)

    # Test Case 4: Test model performance metrics
    # This test checks if the model achieves reasonable performance metrics.
    def test_model_performance(self):
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        param_grid = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
            'classifier__max_features': ['auto', 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(
            estimator=pipeline_lr,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        y_pred_lr = grid_search.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        precision_lr = precision_score(self.y_test, y_pred_lr)
        recall = recall_score(self.y_test, y_pred_lr)
        f1 = f1_score(self.y_test, y_pred_lr)
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred_lr)
        self.assertGreater(accuracy_lr, 0.7)
        self.assertGreater(precision_lr, 0.7)
        self.assertGreater(recall, 0.7)
        self.assertGreater(f1, 0.7)
        self.assertGreater(balanced_acc, 0.7)

if __name__ == '__main__':
    unittest.main()