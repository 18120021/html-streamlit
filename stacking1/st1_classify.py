def classify():
   import pandas as pd
   import warnings
   from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_val_score
   from sklearn.svm import SVC
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import make_pipeline
   # Suppress UndefinedMetricWarning
   warnings.filterwarnings("ignore", category=UserWarning)

   # Load dataset
   data = pd.read_csv('network_traffic_5G_final_data.csv')

   # Select features and target variable
   features = ['variation_traffic_from', 'protocol', 'packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
   target = 'traffic_type'

   # Handle categorical columns (traffic_type, variation_traffic_from, protocol)
   categorical_features = ['variation_traffic_from', 'protocol']
   numeric_features = [feature for feature in features if feature not in categorical_features]

   # Preprocessing pipeline
   numeric_transformer = StandardScaler()
   categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

   preprocessor = ColumnTransformer(
      transformers=[
         ('num', numeric_transformer, numeric_features),
         ('cat', categorical_transformer, categorical_features)])

   # Define hyperparameter grids for each model
   param_grid_svm = {
      'C': [1, 10, 100],
      'gamma': [0.1, 1, 10],
      'kernel': ['rbf']
   }

   param_grid_knn = {
      'n_neighbors': [5, 7, 9],
      'weights': ['uniform', 'distance'],
      'metric': ['euclidean', 'manhattan']
   }

   # Initialize base models with hyperparameter tuning
   svm_model = SVC(probability=True)  # probability=True to enable predict_proba for stacking
   knn_model = KNeighborsClassifier()

   # Initialize meta-model
   meta_model = LogisticRegression()

   # Define K-Fold cross-validation with fewer folds
   kfold = KFold(n_splits=2, shuffle=True, random_state=42)

   # Initialize empty arrays to store meta-features
   meta_features_train = []

   # Define scoring metrics
   scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
   cv_scores = {metric: [] for metric in scoring}

   # Iterate through base models
   for model, param_grid in [(svm_model, param_grid_svm), (knn_model, param_grid_knn)]:
      # Create pipeline with preprocessor and model
      pipeline = make_pipeline(preprocessor, GridSearchCV(model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1))

      # Perform cross-validation predictions
      cv_pred = cross_val_predict(pipeline, data[features], data[target], cv=kfold, method='predict_proba', n_jobs=-1)

      # Append predictions as meta-features
      meta_features_train.append(pd.DataFrame(cv_pred))

      # Calculate cross-validation scores
      for metric in scoring:
         scores = cross_val_score(pipeline, data[features], data[target], cv=kfold, scoring=metric, n_jobs=-1)
         cv_scores[metric].append(scores.mean())

   # Concatenate meta-features across base models
   meta_features_train = pd.concat(meta_features_train, axis=1)

   # Fit the meta-model (logistic regression) with the meta-features
   meta_model.fit(meta_features_train, data[target])

   # Inverse transform predicted labels to original class labels using the stacked ensemble model
   predicted_traffic_type = meta_model.predict(meta_features_train)

   # Assign predicted values to the entire DataFrame
   data['Predicted Traffic Type'] = predicted_traffic_type
   predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

   # Assign predicted values to the entire DataFrame
   return predicted_traffic_type_df