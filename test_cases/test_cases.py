import unittest
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fetching the dataset from the GitHub raw URL
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url, verify=False)
        cls.df = pd.read_csv(StringIO(response.text))
        
        # Clean column names
        cls.df.columns = [col.replace('.', '_') for col in cls.df.columns]

        # Feature engineering
        cls.df["Exp_Gap"] = cls.df["Age"] - cls.df["Experience"]
        cls.df["Income_per_Family"] = np.round(cls.df["Income"] / (cls.df["Family"].replace(0, 2)), 4)
        cls.df["CC_Spend_Ratio"] = cls.df["CCAvg"] / (cls.df["Income"] + 2)
        cls.df["Income_Education"] = cls.df["Income"] * cls.df["Education"]
        cls.df["Exp_Education"] = cls.df["Experience"] * cls.df["Education"]
        cls.df["CC_per_Family"] = cls.df["CCAvg"] / (cls.df["Family"].replace(0, 1))

        # Prepare data for training
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_random_forest_accuracy(self):
        # Test the accuracy of Random Forest Classifier
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)

        # Check if accuracy is greater than a threshold, e.g., 0.7
        self.assertGreater(accuracy_rf, 0.7, "Random Forest accuracy is below the expected threshold.")

    def test_svm_accuracy(self):
        # Test the accuracy of SVM Classifier
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)

        # Check if accuracy is greater than a threshold, e.g., 0.7
        self.assertGreater(accuracy_svm, 0.7, "SVM accuracy is below the expected threshold.")

    def test_logistic_regression_accuracy(self):
        # Test the accuracy of Logistic Regression Classifier
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)

        # Check if accuracy is greater than a threshold, e.g., 0.7
        self.assertGreater(accuracy_lr, 0.7, "Logistic Regression accuracy is below the expected threshold.")

    def test_knn_accuracy(self):
        # Test the accuracy of KNeighbors Classifier
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)

        # Check if accuracy is greater than a threshold, e.g., 0.7
        self.assertGreater(accuracy_knn, 0.7, "KNN accuracy is below the expected threshold.")

    def test_random_forest_hyperparameter_tuning(self):
        # Test the performance after hyperparameter tuning for Random Forest
        param_grid_rf = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10],
            'classifier__min_samples_split': [2, 5],
        }

        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
        ])
        
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        y_pred_rf_cv = pipeline_rf_cv.predict(self.X_test)
        accuracy_rf_cv = accuracy_score(self.y_test, y_pred_rf_cv)

        # Check if accuracy is greater than a threshold, e.g., 0.7
        self.assertGreater(accuracy_rf_cv, 0.7, "Random Forest accuracy with CV is below the expected threshold.")

    def test_classification_report(self):
        # Test the classification report for Random Forest Classifier
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        classification_rep = classification_report(self.y_test, y_pred_rf, output_dict=True)

        # Ensure the report contains expected metrics
        self.assertIn('0', classification_rep, "Classification report is missing class 0 metrics.")
        self.assertIn('1', classification_rep, "Classification report is missing class 1 metrics.")

if __name__ == '__main__':
    unittest.main()