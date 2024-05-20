def classify():
    import pandas as pd
    from sklearn.model_selection import KFold, GridSearchCV
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

    # Fit the best model on the entire dataset
    best_lgbm_model.fit(X_scaled, y)

    # Make predictions on the entire dataset
    y_pred = best_lgbm_model.predict(X_scaled)

    # Inverse transform predicted labels to original class labels
    predicted_traffic_type = label_encoder.inverse_transform(y_pred)

    # Assign predicted values to the entire DataFrame
    data['Predicted Traffic Type'] = predicted_traffic_type

    predicted_traffic_type_df = pd.DataFrame(data.iloc[1:1018])

    return predicted_traffic_type_df