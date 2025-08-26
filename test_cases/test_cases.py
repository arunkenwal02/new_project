import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class TestModelPerformance(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the dataset from the GitHub URL
        url = 'https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv'
        response = requests.get(url)
        data = StringIO(response.text)
        cls.df = pd.read_csv(data)
        
        # Preprocess the data similar to the original script
        cls.df.columns = [col.replace('.', '_') for col in cls.df.columns]
        cls.df["Exp_Gap"] = cls.df["Age"] - cls.df["Experience"]
        cls.df["Income_per_Family"] = cls.df["Income"] / (cls.df["Family"].replace(0, 2))
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

    def test_random_forest_accuracy(self):
        # Test the accuracy of the Random Forest model
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        
        # Assert that the accuracy is greater than a threshold
        self.assertGreater(accuracy_rf, 0.7, "Random Forest accuracy is below the expected threshold.")

    def test_svm_accuracy(self):
        # Test the accuracy of the SVM model
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        
        # Assert that the accuracy is greater than a threshold
        self.assertGreater(accuracy_svm, 0.7, "SVM accuracy is below the expected threshold.")

    def test_logistic_regression_accuracy(self):
        # Test the accuracy of the Logistic Regression model
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        
        # Assert that the accuracy is greater than a threshold
        self.assertGreater(accuracy_lr, 0.7, "Logistic Regression accuracy is below the expected threshold.")

    def test_knn_accuracy(self):
        # Test the accuracy of the KNN model
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        
        # Assert that the accuracy is greater than a threshold
        self.assertGreater(accuracy_knn, 0.7, "KNN accuracy is below the expected threshold.")

    def test_random_forest_cv_best_params(self):
        # Test hyperparameter tuning for the Random Forest model
        param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
        }
        
        pipeline_rf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=2))
        ])
        pipeline_rf_cv.fit(self.X_train, self.y_train)
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
        
        # Assert that the best parameters are correctly identified
        self.assertIn('n_estimators', best_params_rf, "Best parameters do not include 'n_estimators'.")
        self.assertIn('max_depth', best_params_rf, "Best parameters do not include 'max_depth'.")

    def test_svm_cv_best_params(self):
        # Test hyperparameter tuning for the SVM model
        param_grid_svm = {
            'C': [0.1, 1],
            'kernel': ['linear', 'rbf'],
        }
        
        pipeline_svm_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GridSearchCV(SVC(), param_grid_svm, cv=2))
        ])
        pipeline_svm_cv.fit(self.X_train, self.y_train)
        best_params_svm = pipeline_svm_cv.named_steps['classifier'].best_params_
        
        # Assert that the best parameters are correctly identified
        self.assertIn('C', best_params_svm, "Best parameters do not include 'C'.")
        self.assertIn('kernel', best_params_svm, "Best parameters do not include 'kernel'.")

if __name__ == '__main__':
    unittest.main()