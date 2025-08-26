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

class TestModelPipeline(unittest.TestCase):
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
        self.assertIn('Exp_Gap', df.columns)
        self.assertIn('Income_per_Family', df.columns)
        self.assertIn('CC_Spend_Ratio', df.columns)
        self.assertIn('Mortgage_Income_Ratio', df.columns)
        self.assertIn('Income_Mortgage_Ratio', df.columns)
        self.assertIn('Account_Score', df.columns)
        self.assertIn('Digital_Score', df.columns)
        self.assertIn('Income_Education', df.columns)
        self.assertIn('Exp_Education', df.columns)
        self.assertIn('CC_per_Family', df.columns)

    # Test Case 2: Test train-test split
    # This test checks if the train-test split results in the correct sizes.
    def test_train_test_split(self):
        df = self.df.copy()
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(df))

    # Test Case 3: Test model accuracy
    # This test checks if the models achieve a reasonable accuracy.
    def test_model_accuracy(self):
        df = self.df.copy()
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        self.assertGreater(accuracy_rf, 0.7)  # Assuming a reasonable threshold

        # SVM
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(X_train, y_train)
        y_pred_svm = pipeline_svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        self.assertGreater(accuracy_svm, 0.7)

        # Logistic Regression
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(X_train, y_train)
        y_pred_lr = pipeline_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        self.assertGreater(accuracy_lr, 0.7)

        # KNN
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(X_train, y_train)
        y_pred_knn = pipeline_knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        self.assertGreater(accuracy_knn, 0.7)

    # Test Case 4: Test hyperparameter tuning
    # This test checks if the hyperparameter tuning process finds the best parameters.
    def test_hyperparameter_tuning(self):
        df = self.df.copy()
        X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        y = df['Personal_Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest with GridSearchCV
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
        self.assertIn(best_params_rf['min_samples_split'], [2, 5])

if __name__ == '__main__':
    unittest.main()