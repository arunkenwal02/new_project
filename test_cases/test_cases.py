```python
# Test Case 1: Data Import Validation
# This test case validates that the data is imported correctly and the DataFrame is not empty.
# How to Perform:
import pandas as pd

df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
assert not df.empty, "DataFrame is empty after import"

# Test Case 2: Column Name Replacement Validation
# This test case checks if the column names have been correctly replaced to remove periods.
# How to Perform:
expected_columns = [col.replace('.', '_') for col in df.columns]
df.columns = expected_columns
assert all('.' not in col for col in df.columns), "Column names still contain periods"

# Test Case 3: Feature Engineering Validation
# This test case validates that new features are correctly added to the DataFrame.
# How to Perform:
df["Exp_Gap"] = df["Age"] - df["Experience"]
df["Income_per_Family"] = np.round(df["Income"] / (df["Family"].replace(0, 2)), 4)
assert "Exp_Gap" in df.columns and "Income_per_Family" in df.columns, "Feature engineering failed"

# Test Case 4: Train-Test Split Validation
# This test case ensures that the train-test split results in the correct number of samples.
# How to Perform:
from sklearn.model_selection import train_test_split

X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
y = df['Personal_Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
assert len(X_train) == 0.8 * len(X) and len(X_test) == 0.2 * len(X), "Train-test split incorrect"

# Test Case 5: Model Training Validation
# This test case checks if the models are trained without errors.
# How to Perform:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipeline_rf.fit(X_train, y_train)
assert pipeline_rf.named_steps['classifier'].n_estimators > 0, "Model training failed"

# Test Case 6: Prediction Validation
# This test case ensures that predictions are made without errors and have the correct length.
# How to Perform:
y_pred_rf = pipeline_rf.predict(X_test)
assert len(y_pred_rf) == len(y_test), "Prediction length mismatch"

# Test Case 7: Accuracy Calculation Validation
# This test case validates that the accuracy is calculated and is within a reasonable range.
# How to Perform:
from sklearn.metrics import accuracy_score

accuracy_rf = accuracy_score(y_test, y_pred_rf)
assert 0 <= accuracy_rf <= 1, "Accuracy score out of bounds"

# Test Case 8: Hyperparameter Tuning Validation
# This test case checks if hyperparameter tuning is performed and best parameters are found.
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
assert best_params_rf is not None, "Hyperparameter tuning failed"

# Test Case 9: Classification Report Validation
# This test case ensures that the classification report is generated without errors.
# How to Perform:
from sklearn.metrics import classification_report

classification_rep = classification_report(y_test, y_pred_rf)
assert isinstance(classification_rep, str) and "precision" in classification_rep, "Classification report generation failed"
```