```python
# Test Case 1: Data Import Validation
# Description: Validates that the data is imported correctly and the DataFrame is not empty.
# How to Perform:
import pandas as pd

df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
assert not df.empty, "DataFrame is empty after import"

# Test Case 2: Column Name Replacement
# Description: Ensures that column names are correctly replaced with underscores instead of periods.
# How to Perform:
df.columns = [col.replace('.', '_') for col in df.columns]
assert all('.' not in col for col in df.columns), "Column names still contain periods"

# Test Case 3: Feature Engineering Validation
# Description: Validates that new features are correctly added to the DataFrame.
# How to Perform:
expected_features = ["Exp_Gap", "Income_per_Family", "CC_Spend_Ratio", "Mortgage_Income_Ratio", 
                     "Income_Mortgage_Ratio", "Account_Score", "Digital_Score", 
                     "Income_Education", "Exp_Education", "CC_per_Family"]
for feature in expected_features:
    assert feature in df.columns, f"{feature} not found in DataFrame"

# Test Case 4: Train-Test Split Validation
# Description: Ensures that the train-test split results in the correct number of samples.
# How to Perform:
from sklearn.model_selection import train_test_split

X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
y = df['Personal_Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
assert len(X_train) == 0.8 * len(X), "Train set size is incorrect"
assert len(X_test) == 0.2 * len(X), "Test set size is incorrect"

# Test Case 5: Model Training Validation
# Description: Validates that models are trained without errors.
# How to Perform:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipeline_rf.fit(X_train, y_train)

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])
pipeline_svm.fit(X_train, y_train)

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline_lr.fit(X_train, y_train)

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])
pipeline_knn.fit(X_train, y_train)

# Test Case 6: Prediction Validation
# Description: Ensures that predictions are made without errors and have the correct length.
# How to Perform:
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_svm = pipeline_svm.predict(X_test)
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_knn = pipeline_knn.predict(X_test)

assert len(y_pred_rf) == len(y_test), "Random Forest predictions have incorrect length"
assert len(y_pred_svm) == len(y_test), "SVM predictions have incorrect length"
assert len(y_pred_lr) == len(y_test), "Logistic Regression predictions have incorrect length"
assert len(y_pred_knn) == len(y_test), "KNN predictions have incorrect length"

# Test Case 7: Accuracy Calculation Validation
# Description: Validates that accuracy scores are calculated and are within a reasonable range.
# How to Perform:
from sklearn.metrics import accuracy_score

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

assert 0 <= accuracy_rf <= 1, "Random Forest accuracy is out of bounds"
assert 0 <= accuracy_svm <= 1, "SVM accuracy is out of bounds"
assert 0 <= accuracy_lr <= 1, "Logistic Regression accuracy is out of bounds"
assert 0 <= accuracy_knn <= 1, "KNN accuracy is out of bounds"

# Test Case 8: Hyperparameter Tuning Validation
# Description: Ensures that GridSearchCV finds the best parameters without errors.
# How to Perform:
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
}

pipeline_rf_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5))
])
pipeline_rf_cv.fit(X_train, y_train)

best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
assert best_params_rf is not None, "Best parameters for RandomForestClassifier not found"

# Test Case 9: Classification Report Validation
# Description: Validates that the classification report is generated without errors.
# How to Perform:
from sklearn.metrics import classification_report

classification_rep = classification_report(y_test, y_pred_rf_cv)
assert isinstance(classification_rep, str) and len(classification_rep) > 0, "Classification report is empty or not a string"
```