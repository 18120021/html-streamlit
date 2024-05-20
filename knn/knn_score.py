def score():
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

    # Load dataset
    data = pd.read_csv('network_traffic_5G_final_data.csv')

    # Select features and target variable
    features = ['variation_traffic_from', 'protocol', 'packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
    target = 'traffic_type'

    # Encode categorical target variable 'traffic_type' using LabelEncoder
    label_encoder = LabelEncoder()
    data[target] = label_encoder.fit_transform(data[target])

    # Handle categorical columns (traffic_type, variation_traffic_from, protocol)
    categorical_features = ['traffic_type', 'variation_traffic_from', 'protocol']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = pd.concat([data[numerical_features] for numerical_features in features if numerical_features not in categorical_features],
                            axis=1,
                            join='inner')
    encoded_data = pd.concat([encoded_data, pd.DataFrame(encoder.fit_transform(data[categorical_features]))], axis=1)

    # Feature scaling for numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(encoded_data[['packet_loss', 'throughput', 'latency', 'jitter', 'ping']])

    # Define K-Fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Initialize KNN classifier
    knn_model = KNeighborsClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(scaled_features, data[target])

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Initialize KNN classifier with the best hyperparameters
    best_knn_model = KNeighborsClassifier(**best_params)

    # Perform cross-validation with the best model
    cv_scores = cross_val_score(best_knn_model, scaled_features, data[target], cv=kfold, scoring='accuracy')

    # Calculate mean cross-validation accuracy
    mean_cv_accuracy = cv_scores.mean()

    # Fit the model
    best_knn_model.fit(scaled_features, data[target])

    # Make predictions
    y_pred = best_knn_model.predict(scaled_features)

    # Evaluate model performance
    accuracy = accuracy_score(data[target], y_pred)
    precision = precision_score(data[target], y_pred, average='weighted')
    recall = recall_score(data[target], y_pred, average='weighted')
    f1 = f1_score(data[target], y_pred, average='weighted')

    # Create a DataFrame with the metrics data
    metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
    "Value": [accuracy, precision, recall, f1, mean_cv_accuracy]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df