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

class TestModeling(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load data for testing
        cls.df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
        
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

        # Set X and y
        cls.X = cls.df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
        cls.y = cls.df['Personal_Loan']

        # Split the data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

        # Create pipelines
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

    def test_random_forest_accuracy(self):
        # Test Random Forest Classifier accuracy
        self.pipeline_rf.fit(self.X_train, self.y_train)
        y_pred_rf = self.pipeline_rf.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        
        # Assert that the accuracy is greater than a certain threshold
        self.assertGreater(accuracy_rf, 0.7, "Random Forest accuracy is below expected threshold")

    def test_svm_accuracy(self):
        # Test SVM accuracy
        self.pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = self.pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        
        # Assert that the accuracy is greater than a certain threshold
        self.assertGreater(accuracy_svm, 0.7, "SVM accuracy is below expected threshold")

    def test_logistic_regression_accuracy(self):
        # Test Logistic Regression accuracy
        self.pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = self.pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        
        # Assert that the accuracy is greater than a certain threshold
        self.assertGreater(accuracy_lr, 0.7, "Logistic Regression accuracy is below expected threshold")

    def test_knn_accuracy(self):
        # Test KNN accuracy
        self.pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = self.pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        
        # Assert that the accuracy is greater than a certain threshold
        self.assertGreater(accuracy_knn, 0.7, "KNN accuracy is below expected threshold")

if __name__ == '__main__':
    unittest.main()