def classify():
   import pandas as pd
   from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
   from sklearn.neighbors import KNeighborsClassifier
   from lightgbm import LGBMClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import make_pipeline

   # Load dataset
   data = pd.read_csv('network_traffic_5G_final_data.csv')

   # Preprocessing
   data['traffic_type'] = data['traffic_type'].str.strip()

   # Replace unexpected values in 'traffic_type'
   expected_values = ['Voice', 'Video', 'Game']
   data['traffic_type'] = data['traffic_type'].apply(lambda x: x if x in expected_values else 'Unknown')

   # Select features and target variable
   features = ['variation_traffic_from', 'protocol', 'packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
   target = 'traffic_type'

   # Handle categorical columns (variation_traffic_from, protocol)
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
   param_grid_knn = {
      'n_neighbors': [5],
      'weights': ['uniform'],
      'metric': ['manhattan']
   }

   param_grid_lgbm = {
      'learning_rate': [0.1],
      'n_estimators': [100],
      'max_depth': [5]
   }

   param_grid_svm = {
      'C': [1],
      'kernel': ['rbf'],
      'gamma': ['scale']
   }

   # Initialize base models with hyperparameter tuning
   knn_model = KNeighborsClassifier()
   lgbm_model = LGBMClassifier()
   svm_model = SVC(probability=True)

   # Define K-Fold cross-validation with fewer folds
   skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

   # Initialize empty arrays to store meta-features
   meta_features_train = []

   # Iterate through base models
   for model, param_grid in [(knn_model, param_grid_knn), (lgbm_model, param_grid_lgbm), (svm_model, param_grid_svm)]:
      # Create pipeline with preprocessor and model
      pipeline = make_pipeline(preprocessor, GridSearchCV(model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1))

      # Perform cross-validation predictions
      cv_pred = cross_val_predict(pipeline, data[features], data[target], cv=skf, method='predict_proba', n_jobs=-1)

      # Append predictions as meta-features
      meta_features_train.append(pd.DataFrame(cv_pred))

   # Concatenate meta-features across base models
   meta_features_train = pd.concat(meta_features_train, axis=1)

   # Fit logistic regression on meta-features
   meta_model = LogisticRegression()
   meta_model.fit(meta_features_train, data[target])

   # Inverse transform predicted labels to original class labels using the stacked ensemble model
   predicted_traffic_type = meta_model.predict(meta_features_train)

   # Assign predicted values to the entire DataFrame
   data['Predicted Traffic Type'] = predicted_traffic_type
   predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

   # Assign predicted values to the entire DataFrame
   return predicted_traffic_type_df