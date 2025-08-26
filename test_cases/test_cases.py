```python
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset from the GitHub URL
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url)
        data = StringIO(response.text)
        cls.df = pd.read_csv(data)

        # Preprocess the data
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

        # Split the data
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    # Test Case 1: Test Random Forest Classifier Accuracy
    def test_random_forest_accuracy(self):
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_rf <= 1.0)

    # Test Case 2: Test SVM Classifier Accuracy
    def test_svm_accuracy(self):
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_svm <= 1.0)

    # Test Case 3: Test Logistic Regression Classifier Accuracy
    def test_logistic_regression_accuracy(self):
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_lr <= 1.0)

    # Test Case 4: Test KNN Classifier Accuracy
    def test_knn_accuracy(self):
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_knn <= 1.0)

    # Test Case 5: Test Random Forest with Cross-Validation
    def test_random_forest_cv(self):
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        y_pred_rf_cv = pipeline_rf_cv.predict(self.X_test)
        accuracy_rf_cv = accuracy_score(self.y_test, y_pred_rf_cv)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_rf_cv <= 1.0)

    # Test Case 6: Test SVM with Cross-Validation
    def test_svm_cv(self):
        param_grid_svm = {
            'C': [0.1, 1, 10, 50, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.001]
        }
        pipeline_svm_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(SVC(), param_grid_svm, cv=5))
        ])
        pipeline_svm_cv.fit(self.X_train, self.y_train)
        y_pred_svm_cv = pipeline_svm_cv.predict(self.X_test)
        accuracy_svm_cv = accuracy_score(self.y_test, y_pred_svm_cv)
        # Check if the accuracy is within a reasonable range
        self.assertTrue(0.7 <= accuracy_svm_cv <= 1.0)

    # Test Case 7: Test Classification Report for Random Forest with CV
    def test_classification_report_rf_cv(self):
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
        }
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        y_pred_rf_cv = pipeline_rf_cv.predict(self.X_test)
        classification_rep = classification_report(self.y_test, y_pred_rf_cv, output_dict=True)
        # Check if the classification report contains expected keys
        self.assertIn('0', classification_rep)
        self.assertIn('1', classification_rep)
        self.assertIn('accuracy', classification_rep)

if __name__ == '__main__':
    unittest.main()
```