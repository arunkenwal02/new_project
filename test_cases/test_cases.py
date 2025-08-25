import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class TestBankLoanModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the dataset for testing
        cls.df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

        # Initialize pipelines
        cls.pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        cls.pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        cls.pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        cls.pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])

    # Test Case 1: Check if the dataset loads correctly
    def test_dataset_shape(self):
        self.assertEqual(self.df.shape[1], 14)  # Assuming the dataset has 14 columns based on the provided code

    # Test Case 2: Ensure that the target variable is binary
    def test_target_variable(self):
        unique_values = self.y.unique()
        self.assertTrue(set(unique_values).issubset({0, 1}))  # Check if the target variable is binary

    # Test Case 3: Validate the Random Forest Classifier accuracy
    def test_random_forest_accuracy(self):
        self.pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = self.pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        self.assertGreater(accuracy_rf, 0.7)  # Check if accuracy is greater than 70%

    # Test Case 4: Validate the SVM Classifier accuracy
    def test_svm_accuracy(self):
        self.pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = self.pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        self.assertGreater(accuracy_svm, 0.7)  # Check if accuracy is greater than 70%

    # Test Case 5: Validate the Logistic Regression Classifier accuracy
    def test_logistic_regression_accuracy(self):
        self.pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = self.pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        self.assertGreater(accuracy_lr, 0.7)  # Check if accuracy is greater than 70%

    # Test Case 6: Validate the KNN Classifier accuracy
    def test_knn_accuracy(self):
        self.pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = self.pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        self.assertGreater(accuracy_knn, 0.7)  # Check if accuracy is greater than 70%

    # Test Case 7: Check if the RandomForest pipeline works with GridSearchCV
    def test_pipeline_rf_grid_search(self):
        param_grid_rf = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        self.assertIsNotNone(pipeline_rf_cv.named_steps['classifier'].best_params_)  # Ensure best params are found

if __name__ == '__main__':
    unittest.main()