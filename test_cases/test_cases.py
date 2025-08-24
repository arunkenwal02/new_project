- # Test Case 1: Data Import Validation
  # This test case validates that the data is correctly imported from the CSV file.
  # It checks for the presence of essential columns and verifies the shape of the DataFrame.
  # 
  # How to Perform:
  df = pd.read_csv('/Users/arunkenwal/Desktop/new_project/model_resources/bankloan.csv')
  assert 'Personal_Loan' in df.columns, "Personal_Loan column is missing"
  assert 'ZIP_Code' in df.columns, "ZIP_Code column is missing"
  assert df.shape[0] > 0, "DataFrame is empty"

- # Test Case 2: Feature Engineering Validation
  # This test case validates that the feature engineering process correctly creates new features.
  # It checks if the new features are added to the DataFrame and verifies their values.
  #
  # How to Perform:
  df['Exp_Gap'] = df['Age'] - df['Experience']
  assert 'Exp_Gap' in df.columns, "Exp_Gap feature is missing"
  assert df['Exp_Gap'].notnull().all(), "Exp_Gap contains null values"

- # Test Case 3: Data Preprocessing Validation
  # This test case validates the preprocessing step where column names are modified.
  # It checks if the columns have been correctly renamed by replacing '.' with '_'.
  #
  # How to Perform:
  original_columns = df.columns.tolist()
  df.columns = [col.replace('.', '_') for col in df.columns]
  assert all(col.replace('.', '_') in df.columns for col in original_columns), "Column renaming failed"

- # Test Case 4: Model Training Validation
  # This test case validates that the models are trained successfully and do not raise any errors.
  #
  # How to Perform:
  from sklearn.model_selection import train_test_split
  X = df.drop(['ZIP_Code', 'Personal_Loan', 'ID'], axis=1)
  y = df['Personal_Loan']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  pipeline_rf = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', RandomForestClassifier())
  ])
  
  pipeline_rf.fit(X_train, y_train)

- # Test Case 5: Prediction Validation
  # This test case validates that predictions are generated correctly after model training.
  # It checks if the predictions have the correct shape and data type.
  #
  # How to Perform:
  y_pred_rf = pipeline_rf.predict(X_test)
  assert y_pred_rf.shape == y_test.shape, "Prediction shape does not match test labels"
  assert isinstance(y_pred_rf, np.ndarray), "Predictions are not in numpy array format"

- # Test Case 6: Model Evaluation Validation
  # This test case validates that the model evaluation metrics are computed successfully.
  # It checks if the accuracy score and classification report can be generated without errors.
  #
  # How to Perform:
  accuracy_rf = accuracy_score(y_test, y_pred_rf)
  assert isinstance(accuracy_rf, float), "Accuracy score is not a float"
  
  classification_rep = classification_report(y_test, y_pred_rf)
  assert isinstance(classification_rep, str), "Classification report is not a string"

- # Test Case 7: Hyperparameter Tuning Validation
  # This test case validates that the hyperparameter tuning process executes without errors.
  # It checks if the GridSearchCV finds the best parameters for the model.
  #
  # How to Perform:
  param_grid_rf = {
      'n_estimators': [50, 100],
      'max_depth': [5, 10],
      'min_samples_split': [2, 5],
  }
  
  pipeline_rf_cv = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=2))
  ])
  
  pipeline_rf_cv.fit(X_train, y_train)
  best_params_rf = pipeline_rf_cv.named_steps['classifier'].best_params_
  assert isinstance(best_params_rf, dict), "Best parameters are not a dictionary"

- # Test Case 8: Cross-Validation Accuracy Validation
  # This test case validates that the accuracy from the hyperparameter tuning process is computed correctly.
  # It checks if the accuracy score after cross-validation is a valid float.
  #
  # How to Perform:
  y_pred_rf_cv = pipeline_rf_cv.predict(X_test)
  accuracy_rf_cv = accuracy_score(y_test, y_pred_rf_cv)
  assert isinstance(accuracy_rf_cv, float), "Cross-validated accuracy score is not a float"