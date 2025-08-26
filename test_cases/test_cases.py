import unittest
import pandas as pd
import requests
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class BankLoanModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset from the provided GitHub URL
        url = 'https://raw.githubusercontent.com/arunkenwal02/new_project/main/model_resources/bankloan.csv'
        response = requests.get(url)
        cls.df = pd.read_csv(StringIO(response.content.decode('utf-8')))

        # Preprocess the data as per the original script
        cls.df.columns = [col.replace('.', '_') for col in cls.df.columns]
        cls.df["Exp_Gap"] = cls.df["Age"] - cls.df["Experience"]
        cls.df["Income_per_Family"] = (cls.df["Income"] / (cls.df["Family"].replace(0, 2))).round(4)
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

        # Check if the accuracy is above a certain threshold
        self.assertGreater(accuracy_rf, 0.7, "Random Forest accuracy should be above 0.7")

    def test_svm_accuracy(self):
        # Test the accuracy of the SVM model
        pipeline_svm = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ])
        pipeline_svm.fit(self.X_train, self.y_train)
        y_pred_svm = pipeline_svm.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)

        # Check if the accuracy is above a certain threshold
        self.assertGreater(accuracy_svm, 0.7, "SVM accuracy should be above 0.7")

    def test_logistic_regression_accuracy(self):
        # Test the accuracy of the Logistic Regression model
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        pipeline_lr.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline_lr.predict(self.X_test)
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)

        # Check if the accuracy is above a certain threshold
        self.assertGreater(accuracy_lr, 0.7, "Logistic Regression accuracy should be above 0.7")

    def test_knn_accuracy(self):
        # Test the accuracy of the KNN model
        pipeline_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        pipeline_knn.fit(self.X_train, self.y_train)
        y_pred_knn = pipeline_knn.predict(self.X_test)
        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)

        # Check if the accuracy is above a certain threshold
        self.assertGreater(accuracy_knn, 0.7, "KNN accuracy should be above 0.7")

    def test_random_forest_hyperparameter_tuning(self):
        # Test the hyperparameter tuning for Random Forest
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

        # Validate best parameters found
        best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
        self.assertIn('n_estimators', best_params_rf, "Best parameters for Random Forest should include 'n_estimators'")
        self.assertIn('max_depth', best_params_rf, "Best parameters for Random Forest should include 'max_depth'")

    def test_classification_report(self):
        # Test the classification report for Random Forest after hyperparameter tuning
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

        # Validate the classification report
        report = classification_report(self.y_test, y_pred_rf_cv, output_dict=True)
        self.assertIn('1', report, "Classification report should contain class '1' (positive class)")
        self.assertGreater(report['1']['f1-score'], 0.6, "F1 score for class '1' should be above 0.6")

if __name__ == '__main__':
    unittest.main()