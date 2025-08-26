import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class TestBankLoanModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset from GitHub raw URL
        cls.df = pd.read_csv('https://github.com/arunkenwal02/new_project/raw/main/model_resources/bankloan.csv')
        cls.df.columns = [col.replace('.', '_') for col in cls.df.columns]
        
        # Feature engineering
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
        
        # Prepare data for training
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_random_forest_model_accuracy(self):
        # Test Random Forest Classifier accuracy
        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        
        # Check if accuracy is greater than a minimum threshold
        self.assertGreater(accuracy_rf, 0.7, "Random Forest accuracy should be greater than 0.7")

    def test_svm_model_accuracy(self):
        # Test SVM Classifier accuracy
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        
        # Check if accuracy is greater than a minimum threshold
        self.assertGreater(accuracy_svm, 0.7, "SVM accuracy should be greater than 0.7")

    def test_logistic_regression_model_accuracy(self):
        # Test Logistic Regression Classifier accuracy
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='liblinear'))
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        
        # Check if accuracy is greater than a minimum threshold
        self.assertGreater(accuracy_lr, 0.7, "Logistic Regression accuracy should be greater than 0.7")

    def test_knn_model_accuracy(self):
        # Test K-Neighbors Classifier accuracy
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        
        # Check if accuracy is greater than a minimum threshold
        self.assertGreater(accuracy_knn, 0.7, "KNN accuracy should be greater than 0.7")

    def test_random_forest_hyperparameter_tuning(self):
        # Test Random Forest Classifier with GridSearchCV
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
        
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
        
        # Ensure best_params_rf is not empty
        self.assertTrue(bool(best_params_rf), "Best parameters for Random Forest should not be empty")

if __name__ == '__main__':
    unittest.main()