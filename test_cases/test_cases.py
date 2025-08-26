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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fetching dataset
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
        # Creating and fitting the Random Forest model
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        
        # Predictions and accuracy
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_rf, 0.7)

    def test_svm_accuracy(self):
        # Creating and fitting the SVM model
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        
        # Predictions and accuracy
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_svm, 0.7)

    def test_logistic_regression_accuracy(self):
        # Creating and fitting the Logistic Regression model
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        
        # Predictions and accuracy
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_lr, 0.7)

    def test_knn_accuracy(self):
        # Creating and fitting the KNN model
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        
        # Predictions and accuracy
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_knn, 0.7)

    def test_random_forest_cv(self):
        # Hyperparameter tuning for Random Forest
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
        
        # Predictions and accuracy
        y_pred_rf_cv = pipeline_rf_cv.predict(self.X_test)
        accuracy_rf_cv = accuracy_score(self.y_test, y_pred_rf_cv)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_rf_cv, 0.7)

    def test_svm_cv(self):
        # Hyperparameter tuning for SVM
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
        
        # Predictions and accuracy
        y_pred_svm_cv = pipeline_svm_cv.predict(self.X_test)
        accuracy_svm_cv = accuracy_score(self.y_test, y_pred_svm_cv)
        
        # Assert accuracy is greater than a threshold (e.g., 0.7)
        self.assertGreater(accuracy_svm_cv, 0.7)

if __name__ == '__main__':
    unittest.main()