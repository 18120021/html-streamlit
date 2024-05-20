def score():
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from lightgbm import LGBMClassifier
    import warnings
    warnings.filterwarnings("ignore")

    # Load dataset
    data = pd.read_csv('network_traffic_5G_final_data.csv')

    # Select features and target variable
    features = ['packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
    target = 'traffic_type'

    # Encode categorical target variable 'traffic_type' using LabelEncoder
    label_encoder = LabelEncoder()
    data[target] = label_encoder.fit_transform(data[target])

    # Prepare data
    X = data[features]
    y = data[target]

    # Define categorical features
    categorical_features = ['variation_traffic_from', 'protocol']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = pd.concat([data[numerical_features] for numerical_features in features if numerical_features not in categorical_features],
                            axis=1,
                            join='inner')
    encoded_data = pd.concat([encoded_data, pd.DataFrame(encoder.fit_transform(data[categorical_features]))], axis=1)

    # Feature scaling for numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded_data[features])

    # Initialize K-Fold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize LightGBM classifier with verbosity set to -1
    lgbm_model = LGBMClassifier(verbosity=0)

    # Define a smaller parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'max_depth': [3, 5],
        'min_child_samples': [10, 20]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Initialize LGBM classifier with the best hyperparameters
    best_lgbm_model = LGBMClassifier(**best_params)

    # Perform cross-validation with the best model
    cv_scores = cross_val_score(best_lgbm_model, X_scaled, y, cv=kfold, scoring='accuracy')

    # Calculate mean cross-validation accuracy
    mean_cv_accuracy = cv_scores.mean()

    # Fit the best model on the entire dataset
    best_lgbm_model.fit(X_scaled, y)

    # Make predictions on the entire dataset
    y_pred = best_lgbm_model.predict(X_scaled)

    # Evaluate model performance on the entire dataset
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    # Create a DataFrame with the metrics data
    metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
    "Value": [accuracy, precision, recall, f1, mean_cv_accuracy]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df