def classify():
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KNeighborsClassifier
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

    # Fit the model
    best_knn_model.fit(scaled_features, data[target])

    # Make predictions
    y_pred = best_knn_model.predict(scaled_features)

    # Inverse transform predicted labels to original class labels
    predicted_traffic_type = label_encoder.inverse_transform(y_pred)

    # Assign predicted values to the entire DataFrame
    data['Predicted Traffic Type'] = predicted_traffic_type
    predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

    return predicted_traffic_type_df