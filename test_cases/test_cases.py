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
    def test_data_preprocessing(self):
        # Check if the columns have been renamed correctly
        self.assertIn('Exp_Gap', self.df.columns)
        self.assertIn('Income_per_Family', self.df.columns)

    # Test Case 2: Test feature engineering
    def test_feature_engineering(self):
        # Check if the new features are created
        self.assertIn('Exp_Gap', self.df.columns)
        self.assertIn('Income_per_Family', self.df.columns)
        self.assertIn('CC_Spend_Ratio', self.df.columns)

    # Test Case 3: Test train-test split
    def test_train_test_split(self):
        # Check if the train-test split is correct
        self.assertEqual(len(self.X_train), 800)
        self.assertEqual(len(self.X_test), 200)

    # Test Case 4: Test model training and prediction
    def test_model_training_and_prediction(self):
        # Create pipeline and grid search
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
        # Check if the accuracy is within a reasonable range
        self.assertGreater(accuracy_lr, 0.7)

if __name__ == '__main__':
    unittest.main()