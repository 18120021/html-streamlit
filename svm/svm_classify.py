def classify():
   import pandas as pd
   import warnings
   warnings.filterwarnings("ignore", category=Warning)
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.svm import SVC
   from sklearn.model_selection import GridSearchCV

   # Load dataset
   traffic_data_df = pd.read_csv('network_traffic_5G_final_data.csv')

   # Drop rows with missing values
   traffic_data_df.dropna(inplace=True)

   # Shuffle dataset to introduce randomness
   traffic_data_df = traffic_data_df.sample(frac=1, random_state=42).reset_index(drop=True)

   # Select features and target variable
   features = ['packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
   target = 'traffic_type'

   # Encode categorical target variable 'traffic_type' using LabelEncoder
   label_encoder = LabelEncoder()
   traffic_data_df[target] = label_encoder.fit_transform(traffic_data_df[target])

   # Prepare the feature matrix and target vector
   X = traffic_data_df[features]
   y = traffic_data_df[target]

   # Scale numerical features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Define parameter grid for hyperparameter tuning
   param_grid = {
      'C': [0.1, 1, 10, 100, 1000],
      'gamma': [0.001, 0.01, 0.1, 1],
      'kernel': ['rbf']
   }

   # Initialize SVM classifier
   svm_model = SVC()

   # Perform grid search with cross-validation
   grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
   grid_search.fit(X_scaled, y)

   # Get the best hyperparameters
   best_params = grid_search.best_params_

   # Initialize SVM classifier with the best hyperparameters
   best_svm_model = SVC(**best_params)

   # Fit the model
   best_svm_model.fit(X_scaled, y)

   # Predictions using the voting classifier
   y_pred_voting = best_svm_model.predict(X_scaled)

   # Inverse transform predicted labels to original class labels
   predicted_traffic_type = label_encoder.inverse_transform(y_pred_voting)

   traffic_data_df['Predicted Traffic Type'] = predicted_traffic_type

   predicted_traffic_type_df = pd.DataFrame(traffic_data_df.iloc[1:1018])

   # Assign predicted values to the entire DataFrame

   return predicted_traffic_type_df